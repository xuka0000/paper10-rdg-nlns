"""Target-task adapters for pinned NLNS, GRLOS and DAC-ALNS networks.

Authors' neural modules are executed directly. Inspection actions, rewards,
feasibility projection and bounds are explicit task adaptations, not CVRP runs.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import random
import time
import os
from collections import Counter, defaultdict, deque

import numpy as np
import torch
from Env.snapshot import uav_schedulable

from Agent.baselines.common import initial_solution, result_validator, select_policy_action
from Agent.evaluation_runtime import RuntimeDeadline
from Agent.hybrid.rdg_nlns.operators import DestroyOperator, destroy_neighborhood
from Agent.hybrid.rdg_nlns.initialization import build_route_feasible_neighborhood_plan, _initialization_route_options
from Agent.hybrid.rdg_nlns.projection import project_candidate
from Agent.hybrid.rdg_nlns.repair import RepairMode, deterministic_insertion_repair
from Agent.hybrid.rdg_nlns.solution import CandidatePlan
from experiments_3d.native_policies import AuthorsNLNSRepair, AuthorsNLNSCritic, authors_dac, authors_grlos, pyg

ADAPTATION_VERSION = "inspection-public-v5-incomplete-destroy-projection"
DESTROY_RATIOS = (.1,.2,.3,.4)


class ServiceCatalog:
    """One immutable coverage/index build per decision; no quadratic rescans."""
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.columns = snapshot.service_columns
        self.by_id = {c.object_id: c for c in self.columns}
        residual = {s.column_id: s for s in snapshot.airborne_residual_states}
        self.coverage = {c.object_id: set(residual[c.object_id].remaining_checkpoint_ids if c.object_id in residual else c.covered_checkpoint_ids)
                         for c in self.columns}
        self.required = {t.checkpoint_id for t in snapshot.tasks} - set(snapshot.completed_checkpoint_ids)
        self.by_checkpoint = defaultdict(list)
        self.available = {u.uav_id: uav_schedulable(u) for u in snapshot.uav_states}
        self.duration_scale = max((c.dynamic_duration for c in self.columns), default=1.)
        self.energy_scale = max((c.energy_quantile for c in self.columns), default=1.)
        self.previous = set(snapshot.previous_selected_column_ids)
        for c in self.columns:
            for k in self.coverage[c.object_id]:
                self.by_checkpoint[k].append(c)
        groups = defaultdict(list)
        for i, c in enumerate(self.columns):
            for group in [("uav", c.uav_id), ("launch", c.launch_stop_id), ("recover", c.recovery_stop_id),
                          *(("checkpoint", k) for k in sorted(self.coverage[c.object_id]))]:
                groups[group].append(i)
        edges = set()
        # Star expansion retains every candidate in each shared-resource group.
        # This task graph is explicitly different from the authors' CVRP graph.
        for group in groups.values():
            for i in group[1:]:
                edges.add((group[0], i)); edges.add((i, group[0]))
        if not edges:
            edges = {(i, i) for i in range(len(self.columns))}
        self.edges = torch.tensor(sorted(edges), dtype=torch.long).T.contiguous()

    def features(self, candidates, selected, uncovered, removed):
        chosen = [self.by_id[k] for k in selected]
        uav = Counter(c.uav_id for c in chosen)
        recovery = Counter(c.recovery_stop_id for c in chosen)
        count = max(1, len(chosen))
        return torch.tensor([(len(self.coverage[c.object_id] & uncovered) / max(1, len(uncovered)),
            c.dynamic_duration / max(self.duration_scale, 1e-9), c.energy_quantile / max(self.energy_scale, 1e-9),
            float(c.launch_stop_id == c.recovery_stop_id), uav[c.uav_id] / count, recovery[c.launch_stop_id] / count,
            float(c.object_id in self.previous), float(c.object_id in removed)) for c in candidates], dtype=torch.float32)

    def graph(self, selected, removed=(), ratio=0., relative_cost=0.):
        covered = set().union(*(self.coverage[k] for k in selected)) if selected else set()
        uncovered = self.required - covered
        Data = pyg()["Data"]
        selected_ids = set(selected)
        membership = torch.tensor([[float(c.object_id in selected_ids)] for c in self.columns])
        features = torch.cat((self.features(self.columns, selected, uncovered, set(removed)),membership),dim=1)
        return Data(x=features, edge_index=self.edges,
            edge_attr=torch.ones(self.edges.shape[1], 1), edge_weight=torch.ones(self.edges.shape[1]),
            global_feats=torch.tensor([len(selected)/max(1,len(self.columns)), len(uncovered)/max(1,len(self.required)), ratio, relative_cost]),
            state_num_nodes=torch.tensor([len(self.columns)]))

    def insert(self, action, rng, *, actor=None, training=False, generator=None, mode="random"):
        locked = (*self.snapshot.locked_column_ids, *self.snapshot.committed_column_ids,
                  *(s.column_id for s in self.snapshot.airborne_residual_states))
        selected = list(dict.fromkeys((*locked, *action.retained_column_ids)))
        removed = set(action.removed_column_ids)
        logs, actions = [], []
        first_features = None
        while True:
            covered = set().union(*(self.coverage[k] for k in selected)) if selected else set()
            uncovered = self.required - covered
            options = {}
            for checkpoint in sorted(uncovered):
                available = [c for c in self.by_checkpoint[checkpoint] if c.object_id not in selected
                    and not (self.coverage[c.object_id] & covered) and self.available[c.uav_id]]
                if available:
                    options[checkpoint] = available
            if not options:
                break
            if actor is not None or mode == "random":
                candidates = options[rng.choice(sorted(options))]
                if actor is None:
                    chosen = rng.randrange(len(candidates))
                else:
                    features = self.features(candidates, selected, uncovered, removed)
                    if first_features is None:
                        first_features = features
                    index, logp, _ = select_policy_action(actor(features), stochastic=training, generator=generator)
                    chosen = index
                    logs.append(logp)
                column = candidates[chosen]
            else:
                cost = lambda c: (c.energy_quantile + c.dynamic_duration, c.object_id)
                ranked = {k: sorted(v, key=cost) for k,v in options.items()}
                if mode == "regret":
                    key = min(ranked, key=lambda k: (-(cost(ranked[k][1])[0] - cost(ranked[k][0])[0]) if len(ranked[k])>1 else -math.inf, k))
                elif mode == "greedy":
                    key = min(ranked, key=lambda k: cost(ranked[k][0]))
                else:
                    raise ValueError("unknown insertion mode")
                column = ranked[key][0]
            selected.append(column.object_id)
            actions.append(column.object_id)
        covered = set().union(*(self.coverage[k] for k in selected)) if selected else set()
        return CandidatePlan(tuple(selected), tuple(sorted(self.required-covered)), tuple(actions)), logs, first_features


class PublicLearningSearch:
    def __init__(self, method, *, seed=11, hidden_dim=128):
        if method not in {"nlns_public", "grlos_public", "dac_alns_public"}:
            raise ValueError("unknown public learning baseline")
        self.method, self.seed = method, seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.updates = self.interactions = 0
        self.repair_counts = [0,0,0]
        if method == "nlns_public":
            self.policy, self.critic = AuthorsNLNSRepair(hidden_dim), AuthorsNLNSCritic(hidden_dim)
            self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
            self.source = self.policy.source_manifest
        elif method == "dac_alns_public":
            self.policy, self.source = authors_dac(64, len(DestroyOperator)*len(DESTROY_RATIOS))
        else:
            self.policy, self.source = authors_grlos(32)
            self.target = copy.deepcopy(self.policy).eval()
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
            self.replay = [deque(maxlen=512), deque(maxlen=512)]

    def _project(self, snapshot, plan, solver, deadline):
        remaining = max(.001, deadline.remaining_seconds())
        completion = build_route_feasible_neighborhood_plan(snapshot, required_column_ids=plan.selected_column_ids,
            time_limit_seconds=min(10.,remaining), deadline=deadline)
        if completion is None or deadline.expired():
            return None, {"projection_valid": False, "issues": ["route_or_time_limit"]}
        ids, roads, _ = completion
        if set(ids) != set(plan.selected_column_ids):
            return None, {"projection_valid": False, "issues": ["fixed_column_set_changed"]}
        from Agent.hybrid.rdg_nlns.initialization import solver_with_time_limit
        bounded = solver_with_time_limit(solver, min(10., max(.001, deadline.remaining_seconds())))
        # The timing projection locks this sequence, not only the selected set.
        # Insertion order can revisit an earlier stop after the vehicle has left.
        route_plan = dataclasses.replace(plan, selected_column_ids=tuple(ids))
        outcome = project_candidate(dataclasses.replace(snapshot, road_options=roads), route_plan, bounded, result_validator(snapshot),
            deadline=deadline, route_projection_time_limit_seconds=min(10., remaining))
        valid = outcome.candidate is not None and outcome.validation.is_valid and outcome.candidate.is_feasible
        return outcome.candidate if valid else None, {"projection_valid": valid, "issues": list(outcome.issue_codes),
            "corrections": list(outcome.corrections), "before_projection_ids": list(plan.selected_column_ids),
            "route_ordered_projection_ids": list(ids),
            "after_projection_ids": list(outcome.candidate.selected_column_ids) if valid else []}

    def _project_partial(self, snapshot, plan, solver, deadline):
        """Score the fixed destroyed set with original tasks and missed penalties."""
        fixed = tuple(dict.fromkeys(plan.selected_column_ids))
        if deadline.expired():
            return None, {"partial_valid":False,"partial_issues":["runtime_deadline"]}
        roads = _initialization_route_options(snapshot, fixed)
        if roads is None or deadline.expired():
            return None, {"partial_valid":False,"partial_issues":["route_or_time_limit"]}
        from Agent.hybrid.rdg_nlns.initialization import solver_with_time_limit
        remaining = max(.001,deadline.remaining_seconds())
        bounded = solver_with_time_limit(solver,min(10.,remaining))
        fixed_plan = dataclasses.replace(plan,selected_column_ids=fixed)
        outcome = project_candidate(dataclasses.replace(snapshot,road_options=roads),fixed_plan,bounded,
            result_validator(snapshot),deadline=deadline,
            route_projection_time_limit_seconds=min(10.,remaining))
        candidate = outcome.candidate
        valid = (candidate is not None and outcome.validation.is_valid and candidate.is_feasible
            and set(candidate.selected_column_ids)==set(fixed))
        issues = list(outcome.issue_codes)
        if candidate is not None and set(candidate.selected_column_ids)!=set(fixed):
            issues.append("partial_fixed_column_set_changed")
        return candidate if valid else None, {"partial_valid":valid,"partial_issues":issues,
            "partial_column_count":len(fixed),"partial_selected_column_count":len(candidate.selected_column_ids) if valid else None}

    def _grlos_update(self):
        if min(map(len,self.replay)) < 2:
            return
        stage = self.rng.randrange(2)
        batch = self.rng.sample(list(self.replay[stage]), min(16,len(self.replay[stage])))
        states, actions, rewards, next_states, terminals = zip(*batch)
        with torch.no_grad():
            following = self.target((stage+1)%2, list(next_states)).max(1).values
            # Authors' two-stage finite-horizon DQN: nonterminal target is
            # next-stage max Q; terminal target is the observed terminal reward.
            target = torch.where(torch.tensor(terminals), torch.tensor(rewards), following)
        predicted = self.policy(stage, list(states)).gather(1, torch.tensor(actions)[:,None]).flatten()
        loss = torch.nn.functional.mse_loss(predicted, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def solve(self, snapshot, solver, *, iterations=1000, wall_s=300., training=False, trace_callback=None):
        if iterations<1 or wall_s<=0:
            raise ValueError("positive neighborhood and wall bounds required")
        started = time.perf_counter()
        deadline = RuntimeDeadline.start(wall_s)
        current = initial_solution(snapshot, solver, "greedy", deadline=deadline, route_projection_time_limit_seconds=min(30.,wall_s))
        best = current
        scale = max(1.,abs(current.objective_value))
        catalog = ServiceCatalog(snapshot)
        self.policy.train(training)
        completed = 0
        with torch.set_grad_enabled(training):
            for iteration in range(iterations):
                if deadline.expired(): break
                before = current.objective_value
                best_before = best.objective_value
                ratios = DESTROY_RATIOS
                if self.method == "nlns_public":
                    d = iteration % 12
                else:
                    graph = catalog.graph(current.selected_column_ids, relative_cost=before/scale)
                    if self.method == "dac_alns_public":
                        state = self.policy.build_state(graph, 0., 0)
                        d = self.policy.act_destroy(state, eps=.1 if training else 0.)
                    else:
                        q = self.policy(0,[graph])[0]
                        d = self.rng.randrange(12) if training and self.rng.random()<.1 else int(q.argmax())
                operator = DestroyOperator(d//len(ratios))
                ratio = ratios[d%len(ratios)]
                action = destroy_neighborhood(snapshot,current,operator,ratio,self.rng)
                if not action.removed_column_ids: break
                destroyed_cost = None
                partial_diagnostics = {"partial_valid":None,"partial_issues":[]}
                repair_cost = destroy_reward = repair_reward = None
                if training and self.method in {"nlns_public", "dac_alns_public"}:
                    # Evaluate the incomplete repaired schedule with its missed-
                    # task penalty removed, analogous to incomplete-route cost.
                    partial, partial_diagnostics = self._project_partial(snapshot,
                        CandidatePlan(action.retained_column_ids,(),()),solver,deadline)
                    if partial is not None:
                        destroyed_cost = partial.objective_value-dict(partial.result.objective_decomposition).get("missed",0.)
                if self.method == "nlns_public":
                    plan, logs, features = catalog.insert(action,self.rng,actor=self.policy,training=training,generator=self.generator)
                    r = None
                else:
                    destroyed = catalog.graph(action.retained_column_ids, action.removed_column_ids, ratio, before/scale)
                    if self.method == "dac_alns_public":
                        repair_state = self.policy.build_state(destroyed,ratio,d)
                        r = self.policy.act_repair(repair_state,eps=.1 if training else 0.)
                    else:
                        q = self.policy(1,[destroyed])[0]
                        r = self.rng.randrange(3) if training and self.rng.random()<.1 else int(q.argmax())
                    plan = deterministic_insertion_repair(snapshot,action,RepairMode(r),deadline=deadline)
                candidate, projection = self._project(snapshot,plan,solver,deadline)
                cost_delta = (candidate.objective_value-before)/scale if candidate else 1.
                temperature = max(.0001, .05*(.0001/.05)**(iteration/max(1,iterations-1)))
                accept = candidate is not None and (cost_delta<=0 or self.rng.random()<math.exp(-min(700.,cost_delta/temperature)))
                if accept: current = candidate
                if candidate and candidate.objective_value<best.objective_value: best=candidate
                if training:
                    if self.method == "nlns_public" and logs:
                        value = self.critic(features)
                        repair_cost = ((candidate.objective_value-destroyed_cost)/scale if candidate is not None and destroyed_cost is not None else 1.)
                        advantage = value.new_tensor(repair_cost)-value
                        loss = advantage.detach()*torch.stack(logs).sum()
                        self.actor_optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.); self.actor_optimizer.step()
                        self.critic_optimizer.zero_grad(); advantage.square().backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.); self.critic_optimizer.step()
                        self.updates += 1
                    elif self.method == "dac_alns_public":
                        next_graph = catalog.graph(current.selected_column_ids,relative_cost=current.objective_value/scale)
                        with torch.no_grad(): next_state = self.policy.build_state(next_graph,0.,0)
                        self.repair_counts[r] += 1
                        improvement = (best_before-destroyed_cost)/max(1e-6,best_before) if destroyed_cost is not None else 0.
                        destroy_reward = ((1. if destroyed_cost<=best_before else 0.) + improvement
                            if destroyed_cost is not None and destroyed_cost<before else 0.)
                        bonus = .1*(1.-self.repair_counts[r]/max(1,max(self.repair_counts)))
                        repair_reward = (20. if candidate is not None and candidate.objective_value<best_before else
                            10. if candidate is not None and candidate.objective_value<before else -.1)+bonus
                        # Native two-stage update, including encoder gradients.
                        self.policy.update({"states":[state],"actions":[d],"rewards":[destroy_reward],"next_states":[repair_state.detach()],"dones":[False]},
                            {"states":[repair_state],"actions":[r],"rewards":[repair_reward],"next_states":[next_state],"dones":[True]})
                        self.updates += 1
                    elif self.method == "grlos_public":
                        next_graph = catalog.graph(current.selected_column_ids,relative_cost=current.objective_value/scale)
                        self.replay[0].append((graph,d,0.,destroyed,False))
                        self.replay[1].append((destroyed,r,-cost_delta,next_graph,True))
                        self._grlos_update()
                completed += 1
                self.interactions += int(training)
                if trace_callback:
                    trace_callback({"iteration":completed,"destroy":operator.name,"ratio":ratio,"repair_action":r,
                        "repair_mode":RepairMode(r).name if r is not None else "NEURAL_INSERTION",
                        "adaptation_version":ADAPTATION_VERSION,
                        "before_objective":before,"candidate_objective":candidate.objective_value if candidate else None,
                        "accepted":accept,"best_objective":best.objective_value,"policy_updates_total":self.updates,
                        "removed_column_ids":list(action.removed_column_ids),
                        "destroyed_cost":destroyed_cost,"repair_cost":repair_cost,
                        "destroy_reward":destroy_reward,"repair_reward":repair_reward,
                        **partial_diagnostics,**projection})
        diagnostics = (*best.result.diagnostics,("algorithm",self.method),("upstream_commit",self.source["commit"]),
            ("completed_iterations",str(completed)),("training",str(training)),("native_policy_updates_total",str(self.updates)),
            ("terminal_reason","wall_clock_limit" if deadline.expired() else "iteration_or_action_limit"))
        return dataclasses.replace(best.result,runtime_seconds=time.perf_counter()-started,diagnostics=diagnostics)

    def save(self,path,identity):
        payload={"method":self.method,"source_commit":self.source["commit"],"seed":self.seed,"identity":identity,
            "adaptation_version":ADAPTATION_VERSION,
            "policy":self.policy.state_dict(),"interactions":self.interactions,"updates":self.updates}
        if self.method=="nlns_public": payload["critic"]=self.critic.state_dict()
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,path)

    def resume_state(self, identity, lineage):
        """Complete training state; inference-only save() is deliberately separate."""
        state = {"format": 1, "identity": identity, "lineage": lineage,
            "method": self.method, "seed": self.seed, "source_commit": self.source["commit"],
            "adaptation_version": ADAPTATION_VERSION, "policy": self.policy.state_dict(),
            "interactions": self.interactions, "updates": self.updates,
            "repair_counts": list(self.repair_counts), "rng": self.rng.getstate(),
            "generator": self.generator.get_state(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
        if self.method == "nlns_public":
            state.update(critic=self.critic.state_dict(),
                actor_optimizer=self.actor_optimizer.state_dict(),
                critic_optimizer=self.critic_optimizer.state_dict())
        elif self.method == "dac_alns_public":
            state["dac_optimizers"] = {name:getattr(self.policy,name).state_dict()
                for name in ("opt_enc","opt_critic","opt_ad","opt_ar")}
        elif self.method == "grlos_public":
            state.update(target=self.target.state_dict(), optimizer=self.optimizer.state_dict(),
                replay=[list(q) for q in self.replay])
        return state

    def restore_resume_state(self, state, identity, *, expected_interactions=None):
        if (state.get("format") != 1 or state.get("identity") != identity or
                state.get("method") != self.method or state.get("seed") != self.seed or
                state.get("source_commit") != self.source["commit"] or
                state.get("adaptation_version") != ADAPTATION_VERSION):
            raise ValueError("resume task/source identity mismatch")
        if expected_interactions is not None and state.get("interactions") != expected_interactions:
            raise ValueError("resume interaction counter mismatch")
        if not 0 <= state["interactions"] <= 10000 or state["updates"] < 0:
            raise ValueError("resume counter outside approved budget")
        lineage = state.get("lineage", {})
        segments = lineage.get("segments", [])
        if not segments or segments[0].get("first_interaction") != 1 or segments[-1].get("adaptation_version") != ADAPTATION_VERSION or segments[-1].get("last_interaction") is not None:
            raise ValueError("resume lineage is incomplete")
        for earlier, later in zip(segments, segments[1:]):
            if not isinstance(earlier.get("last_interaction"), int) or later.get("first_interaction") != earlier["last_interaction"] + 1:
                raise ValueError("resume lineage segments are not contiguous")
        if len(segments)>1 and (lineage.get("prefix_interactions") != segments[-2]["last_interaction"] or
            not lineage.get("prefix_journal_sha256") or not lineage.get("prefix_adapter_sha256")):
            raise ValueError("resume prefix lineage is incomplete")
        self.policy.load_state_dict(state["policy"])
        if self.method == "nlns_public":
            self.critic.load_state_dict(state["critic"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        elif self.method == "dac_alns_public":
            for name in ("opt_enc","opt_critic","opt_ad","opt_ar"):
                getattr(self.policy,name).load_state_dict(state["dac_optimizers"][name])
        elif self.method == "grlos_public":
            self.target.load_state_dict(state["target"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.replay = [deque(q, maxlen=512) for q in state["replay"]]
        self.interactions, self.updates = state["interactions"], state["updates"]
        self.repair_counts = list(state["repair_counts"])
        self.rng.setstate(state["rng"])
        self.generator.set_state(state["generator"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])

    def save_resume(self, path, identity, lineage):
        path = __import__("pathlib").Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        if temporary.exists():
            raise FileExistsError(temporary)
        torch.save(self.resume_state(identity, lineage), temporary)
        os.replace(temporary, path)

    def load(self,path,identity):
        payload=torch.load(path,map_location="cpu",weights_only=True)
        if payload.get("adaptation_version")!=ADAPTATION_VERSION or payload["identity"]!=identity or payload["method"]!=self.method or payload["source_commit"]!=self.source["commit"]:
            raise ValueError("checkpoint task/source identity mismatch")
        self.policy.load_state_dict(payload["policy"])
        if self.method=="nlns_public": self.critic.load_state_dict(payload["critic"])
        self.interactions,self.updates=payload["interactions"],payload["updates"]
