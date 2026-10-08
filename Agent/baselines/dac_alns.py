"""Expected-return dual-actor graph policy and projected DAC-ALNS search. 中文说明。"""

# 中文说明。
import math
# 中文说明。
import random
# 中文说明。
import time
# 中文说明。
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
from Agent.graph_neural_network.heterogeneous_service_graph import HeterogeneousServiceGraph, build_service_graph
# 中文说明。
from Agent.hybrid.rdg_nlns.operators import DestroyOperator, destroy_neighborhood
# 中文说明。
from Agent.hybrid.rdg_nlns.repair import RepairMode, repair_neighborhood
# 中文说明。
from Agent.reinforcement_learning.rdg_dac import StaticDynamicGraphEncoder
# 中文说明。
from Agent.results import AgentResult
# 中文说明。
from Env.snapshot import SceneSnapshot


# 中文说明。
class DACALNSPolicy(nn.Module):
    """Use graph features, conditional actors, and a scalar joint critic. 中文说明。"""

    # 中文说明。
    def __init__(self, hidden_dim: int = 64, destroy_actions: int = 9, repair_actions: int = 3) -> None:
        """中文说明。"""
        # 中文说明。
        super().__init__()
        # 中文说明。
        if min(hidden_dim, destroy_actions, repair_actions) <= 0:
            # 中文说明。
            raise ValueError("DAC-ALNS dimensions must be positive")
        # 中文说明。
        self.destroy_actions = destroy_actions
        # 中文说明。
        self.repair_actions = repair_actions
        # 中文说明。
        self.encoder = StaticDynamicGraphEncoder(hidden_dim)
        # 中文说明。
        self.destroy_actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, destroy_actions))
        # 中文说明。
        self.destroy_embedding = nn.Embedding(destroy_actions, hidden_dim)
        # 中文说明。
        self.repair_embedding = nn.Embedding(repair_actions, hidden_dim)
        # 中文说明。
        self.repair_actor = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, repair_actions))
        # 中文说明。
        self.critic = nn.Sequential(nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, 1))

    # 中文说明。
    def encode(self, graph: HeterogeneousServiceGraph) -> torch.Tensor:
        """中文说明。"""
        # 中文说明。
        return self.encoder(graph)

    # 中文说明。
    def destroy_logits(self, state: torch.Tensor) -> torch.Tensor:
        """中文说明。"""
        # 中文说明。
        return self.destroy_actor(state)

    # 中文说明。
    def repair_logits(self, state: torch.Tensor, destroy_action: torch.Tensor) -> torch.Tensor:
        """中文说明。"""
        # 中文说明。
        destroy = self.destroy_embedding(destroy_action.to(state.device))
        # 中文说明。
        return self.repair_actor(torch.cat((state, destroy), dim=-1))

    # 中文说明。
    def action_values(self, state: torch.Tensor) -> torch.Tensor:
        """中文说明。"""
        # 中文说明。
        destroy_ids = torch.arange(self.destroy_actions, device=state.device)
        # 中文说明。
        repair_ids = torch.arange(self.repair_actions, device=state.device)
        # 中文说明。
        destroy_grid = destroy_ids[:, None].expand(self.destroy_actions, self.repair_actions).reshape(-1)
        # 中文说明。
        repair_grid = repair_ids[None, :].expand(self.destroy_actions, self.repair_actions).reshape(-1)
        # 中文说明。
        state_grid = state.reshape(1, -1).expand(destroy_grid.shape[0], -1)
        # 中文说明。
        values = self.critic(
            torch.cat(
                (state_grid, self.destroy_embedding(destroy_grid), self.repair_embedding(repair_grid)),
                dim=-1,
            )
        )
        # 中文说明。
        return values.reshape(self.destroy_actions, self.repair_actions)


@dataclass(frozen=True, slots=True)
# 中文说明。
class DACALNSConfig:
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
    # 中文说明。
    value_coefficient: float = 0.5

    # 中文说明。
    def __post_init__(self) -> None:
        """中文说明。"""
        # 中文说明。
        if self.iterations <= 0 or not self.destroy_ratios:
            # 中文说明。
            raise ValueError("DAC-ALNS requires positive iterations and destroy ratios")
        # 中文说明。
        if any(not 0.0 < ratio <= 1.0 for ratio in self.destroy_ratios):
            # 中文说明。
            raise ValueError("DAC-ALNS destroy ratios must lie in (0, 1]")
        # 中文说明。
        if self.initial_solution_mode not in {"greedy", "exact"}:
            # 中文说明。
            raise ValueError("DAC-ALNS initial solution mode is invalid")
        # 中文说明。
        if any(not math.isfinite(value) or value < 0.0 for value in (self.entropy_coefficient, self.value_coefficient)):
            # 中文说明。
            raise ValueError("DAC-ALNS loss coefficients must be finite and nonnegative")


# 中文说明。
class DACALNSAgent(BaseAgent):
    """Select destroy and conditional repair actions from a service graph. 中文说明。"""

    # 中文说明。
    def __init__(
        self,
        projection_solver: BaseAgent,
        policy: DACALNSPolicy,
        config: DACALNSConfig,
        *,
        runtime_policy: EvaluationRuntimePolicy | None = None,
    ) -> None:
        """中文说明。"""
        # 中文说明。
        expected_destroy_actions = len(DestroyOperator) * len(config.destroy_ratios)
        # 中文说明。
        if policy.destroy_actions != expected_destroy_actions:
            # 中文说明。
            raise ValueError("DAC-ALNS destroy action count does not match operators and ratios")
        # 中文说明。
        if policy.repair_actions != len(RepairMode):
            # 中文说明。
            raise ValueError("DAC-ALNS repair action count does not match repair modes")
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
    def _run(
        self,
        snapshot: SceneSnapshot,
        *,
        training: bool,
        accumulate_gradients: bool = False,
    ) -> BaselineTrainingEpisode:
        """中文说明。"""
        if accumulate_gradients and not training:
            raise ValueError("gradient accumulation requires training mode")
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
        # 中文说明。
        attempted = feasible = accepted = rejected = 0
        # 中文说明。
        reward_sum = 0.0
        # 中文说明。
        losses: list[torch.Tensor] = []
        # 中文说明。
        action_log: list[str] = []
        # 中文说明。
        ratio_count = len(self._config.destroy_ratios)
        completed_iterations = 0
        terminal_reason = "iteration_limit"
        # 中文说明。
        self.policy.train(training)

        # 中文说明。
        for _ in range(self._config.iterations):
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break
            # 中文说明。
            graph = build_service_graph(snapshot, current.selected_column_ids, current.result)
            # 中文说明。
            state = self.policy.encode(graph)
            # 中文说明。
            destroy_logits = self.policy.destroy_logits(state)
            # 中文说明。
            destroy_action, destroy_log_probability, destroy_entropy = select_policy_action(
                destroy_logits,
                stochastic=training,
                generator=self._torch_generator,
            )
            # 中文说明。
            repair_logits = self.policy.repair_logits(state, torch.tensor(destroy_action))
            # 中文说明。
            repair_action, repair_log_probability, repair_entropy = select_policy_action(
                repair_logits,
                stochastic=training,
                generator=self._torch_generator,
            )
            # 中文说明。
            operator = DestroyOperator(destroy_action // ratio_count)
            # 中文说明。
            ratio = self._config.destroy_ratios[destroy_action % ratio_count]
            # 中文说明。
            destroy = destroy_neighborhood(snapshot, current, operator, ratio, self._rng)
            # 中文说明。
            if not destroy.removed_column_ids:
                # 中文说明。
                terminal_reason = "empty_action_space"
                break
            # 中文说明。
            attempted += 1
            # 中文说明。
            outcome = repair_neighborhood(
                snapshot,
                destroy,
                RepairMode(repair_action),
                self._solver,
                result_validator(snapshot),
                deadline=deadline,
                route_projection_time_limit_seconds=route_limit,
            )
            if "runtime_deadline" in outcome.validation.issue_codes:
                terminal_reason = "wall_clock_limit"
                break
            # 中文说明。
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
                if candidate.objective_value <= current.objective_value:
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
            action_log.append(f"{operator.name}:{RepairMode(repair_action).name}:{ratio:.3g}")
            # 中文说明。
            if training:
                # 中文说明。
                value = self.policy.action_values(state)[destroy_action, repair_action]
                # 中文说明。
                advantage = torch.tensor(reward, dtype=value.dtype) - value
                # 中文说明。
                actor_loss = -(destroy_log_probability + repair_log_probability) * advantage.detach()
                # 中文说明。
                critic_loss = advantage.square()
                # 中文说明。
                entropy = destroy_entropy + repair_entropy
                # 中文说明。
                step_loss = (
                    actor_loss
                    + self._config.value_coefficient * critic_loss
                    - self._config.entropy_coefficient * entropy
                )
                if accumulate_gradients:
                    step_loss.backward()
                    losses.append(step_loss.detach())
                else:
                    losses.append(step_loss)
            completed_iterations += 1
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break

        # 中文说明。
        diagnostics = (
            ("algorithm", "DAC_ALNS"),
            ("attempted_neighborhoods", str(attempted)),
            ("feasible_repairs", str(feasible)),
            ("accepted_candidates", str(accepted)),
            ("projection_rejections", str(rejected)),
            ("policy_actions", ";".join(action_log)),
            ("completed_iterations", str(completed_iterations)),
            ("terminal_reason", terminal_reason),
            ("iteration_limit", str(self._config.iterations)),
        )
        # 中文说明。
        result = with_diagnostics(best.result, time.perf_counter() - started, diagnostics)
        # 中文说明。
        if accumulate_gradients and losses:
            scale = float(len(losses))
            for parameter in self.policy.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(scale)
        loss = torch.stack(losses).mean() if losses else None
        # 中文说明。
        return BaselineTrainingEpisode(
            result,
            loss,
            reward_sum,
            attempted,
            gradients_accumulated=bool(accumulate_gradients and losses),
        )

    # 中文说明。
    def solve(self, snapshot: SceneSnapshot) -> AgentResult:
        """中文说明。"""
        # 中文说明。
        with torch.no_grad():
            # 中文说明。
            return self._run(snapshot, training=False).result

    # 中文说明。
    def training_episode(
        self,
        snapshot: SceneSnapshot,
        *,
        accumulate_gradients: bool = False,
    ) -> BaselineTrainingEpisode:
        """中文说明。"""
        # 中文说明。
        return self._run(
            snapshot,
            training=True,
            accumulate_gradients=accumulate_gradients,
        )
