"""Learned sequential repair policy and projected NLNS search. 中文说明。"""

# 中文说明。
import math
# 中文说明。
import random
# 中文说明。
import time
# 中文说明。
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

# 中文说明。
import torch
# 中文说明。
from torch import nn

# 中文说明。
from Agent.base_agent import BaseAgent
# 中文说明。
from Agent.baselines.common import (
    BaselineTrainingEpisode,
    initial_solution,
    normalized_objective_reward,
    result_validator,
    select_policy_action,
    with_diagnostics,
)
from Agent.evaluation_runtime import EvaluationRuntimePolicy, RuntimeDeadline
# 中文说明。
from Agent.hybrid.rdg_nlns.operators import DestroyOperator, destroy_neighborhood
# 中文说明。
from Agent.hybrid.rdg_nlns.initialization import build_route_feasible_neighborhood_plan
from Agent.hybrid.rdg_nlns.projection import project_candidate
# 中文说明。
from Agent.hybrid.rdg_nlns.solution import CandidatePlan, SearchSolution
# 中文说明。
from Agent.results import AgentResult
# 中文说明。
from Env.snapshot import SceneSnapshot


FeasibilityOracle = Callable[[tuple[str, ...]], bool]


# 中文说明。
class NLNSRepairPolicy(nn.Module):
    """Adapt the NLNS attention encoder and pointer repair to service columns."""

    # 中文说明。
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64) -> None:
        """中文说明。"""
        # 中文说明。
        super().__init__()
        # 中文说明。
        if min(feature_dim, hidden_dim) <= 0:
            # 中文说明。
            raise ValueError("NLNS dimensions must be positive")
        # 中文说明。
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.origin_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention_projection = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(hidden_dim))
        self.pointer_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pointer_vector = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the Xavier initialization adopted by the source actor."""
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter, -1.0 / math.sqrt(self.hidden_dim), 1.0 / math.sqrt(self.hidden_dim))

    # 中文说明。
    def forward(
        self,
        candidate_features: torch.Tensor,
        origin_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """中文说明。"""
        # 中文说明。
        if candidate_features.ndim != 2 or candidate_features.shape[1] != self.feature_dim:
            # 中文说明。
            raise ValueError("candidate features must have shape [candidate, feature_dim]")
        # 中文说明。
        if candidate_features.shape[0] == 0:
            raise ValueError("NLNS attention repair requires at least one candidate")
        if origin_features is None:
            origin_features = candidate_features.mean(dim=0)
        if origin_features.ndim == 2 and origin_features.shape[0] == 1:
            origin_features = origin_features.squeeze(0)
        if origin_features.ndim != 1 or origin_features.shape[0] != self.feature_dim:
            raise ValueError("origin features must have shape [feature_dim]")

        candidate_hidden = self.candidate_encoder(candidate_features)
        origin_hidden = self.origin_encoder(origin_features)
        expanded_origin = origin_hidden.unsqueeze(0).expand_as(candidate_hidden)
        attention_hidden = torch.tanh(
            self.attention_projection(
                torch.cat((candidate_hidden, expanded_origin), dim=-1)
            )
        )
        attention_logits = attention_hidden @ self.attention_vector
        attention_weights = torch.softmax(attention_logits, dim=0)
        context = torch.sum(attention_weights.unsqueeze(-1) * candidate_hidden, dim=0)
        pointer_query = self.pointer_projection(torch.cat((origin_hidden, context), dim=-1))
        return torch.tanh(candidate_hidden + pointer_query.unsqueeze(0)) @ self.pointer_vector


@dataclass(frozen=True, slots=True)
# 中文说明。
class NLNSConfig:
    """中文说明。"""
    # 中文说明。
    iterations: int
    # 中文说明。
    destroy_ratios: tuple[float, ...]
    # 中文说明。
    seed: int = 0
    # 中文说明。
    initial_solution_mode: str = "greedy"
    # 中文说明。
    entropy_coefficient: float = 0.01
    initial_temperature_ratio: float = 0.05
    final_temperature_ratio: float = 0.001

    # 中文说明。
    def __post_init__(self) -> None:
        """中文说明。"""
        # 中文说明。
        if self.iterations <= 0 or not self.destroy_ratios:
            # 中文说明。
            raise ValueError("NLNS requires positive iterations and destroy ratios")
        # 中文说明。
        if any(not 0.0 < ratio <= 1.0 for ratio in self.destroy_ratios):
            # 中文说明。
            raise ValueError("NLNS destroy ratios must lie in (0, 1]")
        # 中文说明。
        if self.initial_solution_mode not in {"greedy", "exact"}:
            # 中文说明。
            raise ValueError("NLNS initial solution mode is invalid")
        # 中文说明。
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0.0:
            # 中文说明。
            raise ValueError("NLNS entropy coefficient must be finite and nonnegative")
        if not math.isfinite(self.initial_temperature_ratio) or self.initial_temperature_ratio <= 0.0:
            raise ValueError("NLNS initial temperature ratio must be finite and positive")
        if not math.isfinite(self.final_temperature_ratio) or self.final_temperature_ratio <= 0.0:
            raise ValueError("NLNS final temperature ratio must be finite and positive")
        if self.final_temperature_ratio > self.initial_temperature_ratio:
            raise ValueError("NLNS final temperature ratio cannot exceed its initial ratio")


# 中文说明。
def _effective_checkpoint_ids(snapshot: SceneSnapshot, column: object) -> tuple[str, ...]:
    """中文说明。"""
    # 中文说明。
    if hasattr(snapshot, "column_state"):
        # 中文说明。
        return tuple(snapshot.column_state(column.object_id).effective_checkpoint_ids)
    # 中文说明。
    return tuple(column.covered_checkpoint_ids)


# 中文说明。
def _candidate_features(
    snapshot: SceneSnapshot,
    column: object,
    selected_ids: tuple[str, ...],
    uncovered: set[str],
    removed_ids: set[str],
) -> tuple[float, ...]:
    """中文说明。"""
    # 中文说明。
    columns = tuple(snapshot.service_columns)
    # 中文说明。
    duration_scale = max((float(item.dynamic_duration) for item in columns), default=1.0)
    # 中文说明。
    energy_scale = max((float(item.energy_quantile) for item in columns), default=1.0)
    # 中文说明。
    selected = tuple(item for item in columns if item.object_id in set(selected_ids))
    # 中文说明。
    new_coverage = len(set(_effective_checkpoint_ids(snapshot, column)) & uncovered)
    # 中文说明。
    uav_load = sum(item.uav_id == column.uav_id for item in selected)
    # 中文说明。
    stop_load = sum(item.recovery_stop_id == column.launch_stop_id for item in selected)
    # 中文说明。
    denominator = max(1, len(selected))
    # 中文说明。
    return (
        new_coverage / max(1, len(uncovered)),
        float(column.dynamic_duration) / max(duration_scale, 1.0e-9),
        float(column.energy_quantile) / max(energy_scale, 1.0e-9),
        float(column.launch_stop_id == column.recovery_stop_id),
        uav_load / denominator,
        stop_load / denominator,
        float(column.object_id in set(snapshot.previous_selected_column_ids)),
        float(column.object_id in removed_ids),
    )


# 中文说明。
def learned_insertion_plan(
    snapshot: SceneSnapshot,
    retained_column_ids: tuple[str, ...],
    removed_column_ids: tuple[str, ...],
    policy: NLNSRepairPolicy,
    *,
    stochastic: bool,
    generator: torch.Generator,
    rng: random.Random,
    feasibility_oracle: FeasibilityOracle | None = None,
) -> tuple[CandidatePlan, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[str, ...]]:
    """Sequentially reconnect destroyed columns through feasible policy actions."""
    locked = tuple(
        dict.fromkeys(
            (
                *snapshot.locked_column_ids,
                *snapshot.committed_column_ids,
                *(state.column_id for state in snapshot.airborne_residual_states),
            )
        )
    )
    selected = list(dict.fromkeys((*locked, *retained_column_ids)))
    required = {
        task.checkpoint_id
        for task in snapshot.tasks
        if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids)
    }
    removed = set(removed_column_ids)
    coverage_by_id = {
        column.object_id: set(_effective_checkpoint_ids(snapshot, column))
        for column in snapshot.service_columns
    }
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    actions: list[str] = []

    while True:
        covered = {
            checkpoint_id
            for column_id in selected
            for checkpoint_id in coverage_by_id[column_id]
        }
        uncovered = required - covered
        if not uncovered:
            break
        origin_checkpoint = rng.choice(tuple(sorted(uncovered)))
        candidates: list[object] = []
        proposals: list[tuple[str, ...]] = []
        for column in snapshot.service_columns:
            candidate_coverage = coverage_by_id[column.object_id]
            if column.object_id in selected:
                continue
            if origin_checkpoint not in candidate_coverage:
                continue
            if candidate_coverage & covered:
                continue
            if not snapshot.uav_state(column.uav_id).available:
                continue
            proposal = tuple((*selected, column.object_id))
            if feasibility_oracle is not None and not feasibility_oracle(proposal):
                continue
            candidates.append(column)
            proposals.append(proposal)
        if not candidates:
            break
        features = torch.tensor(
            [
                _candidate_features(snapshot, column, tuple(selected), uncovered, removed)
                for column in candidates
            ],
            dtype=torch.float32,
        )
        logits = policy(features, features.mean(dim=0))
        action, log_probability, entropy = select_policy_action(
            logits,
            stochastic=stochastic,
            generator=generator,
        )
        selected = list(proposals[action])
        actions.append(candidates[action].object_id)
        log_probabilities.append(log_probability)
        entropies.append(entropy)

    final_coverage = {
        checkpoint_id
        for column in snapshot.service_columns
        if column.object_id in selected
        for checkpoint_id in _effective_checkpoint_ids(snapshot, column)
    }
    deferred = tuple(sorted(required - final_coverage))
    return (
        CandidatePlan(tuple(selected), deferred, tuple(actions)),
        tuple(log_probabilities),
        tuple(entropies),
        tuple(actions),
    )


# 中文说明。
class NLNSAgent(BaseAgent):
    """Run NLNS with a learned sequential service-column repair policy. 中文说明。"""

    # 中文说明。
    def __init__(
        self,
        projection_solver: BaseAgent,
        policy: NLNSRepairPolicy,
        config: NLNSConfig,
        *,
        runtime_policy: EvaluationRuntimePolicy | None = None,
    ) -> None:
        """中文说明。"""
        # 中文说明。
        self._solver = projection_solver
        # 中文说明。
        self.policy = policy
        # 中文说明。
        self._config = config
        self._runtime_policy = runtime_policy
        # 中文说明。
        self._rng = random.Random(config.seed)
        # 中文说明。
        self._torch_generator = torch.Generator(device="cpu").manual_seed(config.seed)

    # 中文说明。
    def _run(self, snapshot: SceneSnapshot, *, training: bool) -> BaselineTrainingEpisode:
        """中文说明。"""
        # 中文说明。
        started = time.perf_counter()
        deadline = (
            RuntimeDeadline.start(self._runtime_policy.wall_clock_limit_seconds)
            if self._runtime_policy is not None
            else None
        )
        route_limit = (
            self._runtime_policy.projection_time_limit_seconds
            if self._runtime_policy is not None
            else 10.0
        )
        initialization_limit = (
            self._runtime_policy.initialization_projection_time_limit_seconds
            if self._runtime_policy is not None
            else 10.0
        )
        # 中文说明。
        current = initial_solution(
            snapshot,
            self._solver,
            self._config.initial_solution_mode,
            deadline=deadline,
            route_projection_time_limit_seconds=initialization_limit,
        )
        # 中文说明。
        best = current
        objective_scale = max(abs(current.objective_value), 1.0)
        initial_temperature = self._config.initial_temperature_ratio * objective_scale
        final_temperature = self._config.final_temperature_ratio * objective_scale
        # 中文说明。
        attempted = feasible = accepted = rejected = 0
        # 中文说明。
        action_log: list[str] = []
        # 中文说明。
        losses: list[torch.Tensor] = []
        # 中文说明。
        reward_sum = 0.0
        # 中文说明。
        policy_action_samples = 0
        feasibility_mask_checks = 0
        feasibility_mask_rejections = 0
        completed_iterations = 0
        terminal_reason = "iteration_limit"
        validator = result_validator(snapshot)

        # 中文说明。
        self.policy.train(training)
        # 中文说明。
        for iteration in range(self._config.iterations):
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break
            # 中文说明。
            operator = DestroyOperator(iteration % len(DestroyOperator))
            # 中文说明。
            ratio = self._config.destroy_ratios[iteration % len(self._config.destroy_ratios)]
            # 中文说明。
            destroy = destroy_neighborhood(snapshot, current, operator, ratio, self._rng)
            # 中文说明。
            if not destroy.removed_column_ids:
                # 中文说明。
                terminal_reason = "empty_action_space"
                break
            # 中文说明。
            attempted += 1

            def feasibility_oracle(column_ids: tuple[str, ...]) -> bool:
                nonlocal feasibility_mask_checks
                nonlocal feasibility_mask_rejections
                feasibility_mask_checks += 1
                candidate_state = snapshot.column_state(column_ids[-1])
                is_feasible = bool(
                    snapshot.uav_state(candidate_state.uav_id).available
                    and candidate_state.effective_protected_energy
                    <= candidate_state.usable_battery_energy + 1.0e-9
                )
                if not is_feasible:
                    feasibility_mask_rejections += 1
                return is_feasible

            # 中文说明。
            plan, log_probabilities, entropies, insertion_actions = learned_insertion_plan(
                snapshot,
                destroy.retained_column_ids,
                destroy.removed_column_ids,
                self.policy,
                stochastic=training,
                generator=self._torch_generator,
                rng=self._rng,
                feasibility_oracle=feasibility_oracle,
            )
            # 中文说明。
            action_log.append(f"{operator.name}:{ratio:.3g}:{'|'.join(insertion_actions)}")
            if set(plan.selected_column_ids) == set(current.selected_column_ids):
                candidate = current
            else:
                completion = build_route_feasible_neighborhood_plan(
                    snapshot,
                    required_column_ids=plan.selected_column_ids,
                    time_limit_seconds=route_limit,
                    deadline=deadline,
                )
                if deadline is not None and deadline.expired():
                    terminal_reason = "wall_clock_limit"
                    break
                if completion is None:
                    candidate = None
                else:
                    completed_ids, road_options, _ = completion
                    if set(completed_ids) != set(plan.selected_column_ids):
                        candidate = None
                    else:
                        ordered_plan = CandidatePlan(
                            tuple(completed_ids),
                            plan.deferred_checkpoint_ids,
                            plan.insertion_order,
                        )
                        routed_snapshot = dataclasses.replace(
                            snapshot,
                            road_options=road_options,
                        )
                        outcome = project_candidate(
                            routed_snapshot,
                            ordered_plan,
                            self._solver,
                            validator,
                            deadline=deadline,
                            route_projection_time_limit_seconds=route_limit,
                        )
                        if "runtime_deadline" in outcome.issue_codes:
                            terminal_reason = "wall_clock_limit"
                            break
                        candidate = outcome.candidate
            # 中文说明。
            reward = -1.0
            # 中文说明。
            if candidate is None or not candidate.is_feasible:
                # 中文说明。
                rejected += 1
            else:
                # 中文说明。
                feasible += 1
                # 中文说明。
                reward = normalized_objective_reward(
                    current.objective_value,
                    candidate.objective_value,
                )
                # 中文说明。
                progress = iteration / max(1, self._config.iterations - 1)
                temperature = initial_temperature * (
                    final_temperature / initial_temperature
                ) ** progress
                delta = candidate.objective_value - current.objective_value
                accept = delta <= 0.0 or self._rng.random() < math.exp(
                    -delta / max(temperature, 1.0e-12)
                )
                if accept:
                    # 中文说明。
                    current = candidate
                    # 中文说明。
                    accepted += 1
                else:
                    # 中文说明。
                    rejected += 1
                # 中文说明。
                if candidate.objective_value < best.objective_value:
                    # 中文说明。
                    best = candidate
            # 中文说明。
            reward_sum += reward
            # 中文说明。
            policy_action_samples += len(log_probabilities)
            # 中文说明。
            if training and log_probabilities:
                # 中文说明。
                policy_term = -torch.stack(log_probabilities).sum() * reward
                # 中文说明。
                entropy_term = torch.stack(entropies).sum()
                # 中文说明。
                losses.append(policy_term - self._config.entropy_coefficient * entropy_term)
            completed_iterations += 1
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break

        # 中文说明。
        diagnostics = (
            ("algorithm", "NLNS"),
            ("algorithm_source", "Hottung_Tierney_ECAI_2020_official_repository"),
            ("repair_policy_architecture", "attention_encoder_pointer"),
            ("repair_task_adapter", "service_column_origin_and_connection_candidates"),
            ("locked_sequence_order", "vehicle_route_uav_canonical"),
            ("feasibility_mask", "coverage_availability_and_battery"),
            ("acceptance_mechanism", "simulated_annealing"),
            ("temperature_schedule", "iteration_normalized_exponential_task_adapter"),
            ("attempted_neighborhoods", str(attempted)),
            ("feasible_repairs", str(feasible)),
            ("accepted_candidates", str(accepted)),
            ("projection_rejections", str(rejected)),
            ("feasibility_mask_checks", str(feasibility_mask_checks)),
            ("feasibility_mask_rejections", str(feasibility_mask_rejections)),
            ("policy_action_samples", str(policy_action_samples)),
            ("policy_actions", ";".join(action_log)),
            ("completed_iterations", str(completed_iterations)),
            ("terminal_reason", terminal_reason),
            ("iteration_limit", str(self._config.iterations)),
        )
        # 中文说明。
        result = with_diagnostics(best.result, time.perf_counter() - started, diagnostics)
        # 中文说明。
        loss = torch.stack(losses).mean() if losses else None
        # 中文说明。
        return BaselineTrainingEpisode(result, loss, reward_sum, attempted)

    # 中文说明。
    def solve(self, snapshot: SceneSnapshot) -> AgentResult:
        """中文说明。"""
        # 中文说明。
        with torch.no_grad():
            # 中文说明。
            return self._run(snapshot, training=False).result

    # 中文说明。
    def training_episode(self, snapshot: SceneSnapshot) -> BaselineTrainingEpisode:
        """中文说明。"""
        # 中文说明。
        return self._run(snapshot, training=True)
