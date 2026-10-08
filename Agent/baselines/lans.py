"""CNN policies and projected Learning-Aided Neighborhood Search. 中文说明。"""

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
from Agent.hybrid.rdg_nlns.operators import DestroyOperator, destroy_neighborhood
# 中文说明。
from Agent.hybrid.rdg_nlns.repair import RepairMode, repair_neighborhood
# 中文说明。
from Agent.results import AgentResult
# 中文说明。
from Env.snapshot import SceneSnapshot


# 中文说明。
class LaNSPolicy(nn.Module):
    """Encode an image-like route tensor for two independent operator agents. 中文说明。"""

    # 中文说明。
    def __init__(self, channels: int, shaking_actions: int, improvement_actions: int, hidden_dim: int = 64) -> None:
        """中文说明。"""
        # 中文说明。
        super().__init__()
        # 中文说明。
        if min(channels, shaking_actions, improvement_actions, hidden_dim) <= 0:
            # 中文说明。
            raise ValueError("LaNS dimensions must be positive")
        # 中文说明。
        self.channels = channels
        # 中文说明。
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # 中文说明。
        self.shaking_head = nn.Linear(hidden_dim, shaking_actions)
        # 中文说明。
        self.improvement_head = nn.Linear(hidden_dim, improvement_actions)
        # 中文说明。
        self.value_head = nn.Linear(hidden_dim, 1)

    # 中文说明。
    def forward(self, route_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """中文说明。"""
        # 中文说明。
        if route_tensor.ndim != 4 or route_tensor.shape[1] != self.channels:
            # 中文说明。
            raise ValueError("route tensor must have shape [batch, channels, height, width]")
        # 中文说明。
        state = self.encoder(route_tensor)
        # 中文说明。
        return self.shaking_head(state), self.improvement_head(state), self.value_head(state).squeeze(-1)


@dataclass(frozen=True, slots=True)
# 中文说明。
class LaNSConfig:
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
    tensor_size: int = 8

    # 中文说明。
    def __post_init__(self) -> None:
        """中文说明。"""
        # 中文说明。
        if self.iterations <= 0 or not self.destroy_ratios:
            # 中文说明。
            raise ValueError("LaNS requires positive iterations and destroy ratios")
        # 中文说明。
        if any(not 0.0 < ratio <= 1.0 for ratio in self.destroy_ratios):
            # 中文说明。
            raise ValueError("LaNS destroy ratios must lie in (0, 1]")
        # 中文说明。
        if self.initial_solution_mode not in {"greedy", "exact"}:
            # 中文说明。
            raise ValueError("LaNS initial solution mode is invalid")
        # 中文说明。
        if self.tensor_size <= 0:
            # 中文说明。
            raise ValueError("LaNS tensor size must be positive")
        # 中文说明。
        if any(not math.isfinite(value) or value < 0.0 for value in (self.entropy_coefficient, self.value_coefficient)):
            # 中文说明。
            raise ValueError("LaNS loss coefficients must be finite and nonnegative")


# 中文说明。
def build_route_tensor(snapshot: SceneSnapshot, selected_column_ids: tuple[str, ...], size: int = 8) -> torch.Tensor:
    """Encode selected service-column order as the image-like LaNS state. 中文说明。"""
    # 中文说明。
    if size <= 0:
        # 中文说明。
        raise ValueError("route tensor size must be positive")
    # 中文说明。
    tensor = torch.zeros((1, 6, size, size), dtype=torch.float32)
    # 中文说明。
    selected = [column for column in snapshot.service_columns if column.object_id in set(selected_column_ids)]
    # 中文说明。
    selected.sort(key=lambda column: selected_column_ids.index(column.object_id))
    # 中文说明。
    duration_scale = max((float(column.dynamic_duration) for column in snapshot.service_columns), default=1.0)
    # 中文说明。
    energy_scale = max((float(column.energy_quantile) for column in snapshot.service_columns), default=1.0)
    # 中文说明。
    task_scale = max((len(column.covered_checkpoint_ids) for column in snapshot.service_columns), default=1)
    # 中文说明。
    uav_counts: dict[str, int] = {}
    # 中文说明。
    for index, column in enumerate(selected[: size * size]):
        # 中文说明。
        row, column_index = divmod(index, size)
        # 中文说明。
        uav_counts[column.uav_id] = uav_counts.get(column.uav_id, 0) + 1
        # 中文说明。
        previous = selected[index - 1] if index > 0 else None
        # 中文说明。
        tensor[0, 0, row, column_index] = 1.0
        # 中文说明。
        tensor[0, 1, row, column_index] = float(column.dynamic_duration) / max(duration_scale, 1.0e-9)
        # 中文说明。
        tensor[0, 2, row, column_index] = float(column.energy_quantile) / max(energy_scale, 1.0e-9)
        # 中文说明。
        tensor[0, 3, row, column_index] = len(column.covered_checkpoint_ids) / max(task_scale, 1)
        # 中文说明。
        tensor[0, 4, row, column_index] = float(previous is not None and previous.recovery_stop_id != column.launch_stop_id)
        # 中文说明。
        tensor[0, 5, row, column_index] = uav_counts[column.uav_id] / max(1, len(selected))
    # 中文说明。
    return tensor


# 中文说明。
class LaNSAgent(BaseAgent):
    """Use independent CNN shaking and improvement policies for neighborhood search. 中文说明。"""

    # 中文说明。
    def __init__(
        self,
        projection_solver: BaseAgent,
        policy: LaNSPolicy,
        config: LaNSConfig,
        *,
        runtime_policy: EvaluationRuntimePolicy | None = None,
    ) -> None:
        """中文说明。"""
        # 中文说明。
        if policy.shaking_head.out_features != len(DestroyOperator):
            # 中文说明。
            raise ValueError("LaNS shaking action count must match destroy operators")
        # 中文说明。
        if policy.improvement_head.out_features != len(RepairMode):
            # 中文说明。
            raise ValueError("LaNS improvement action count must match repair modes")
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
        # 中文说明。
        attempted = feasible = accepted = rejected = 0
        # 中文说明。
        reward_sum = 0.0
        # 中文说明。
        losses: list[torch.Tensor] = []
        # 中文说明。
        action_log: list[str] = []
        completed_iterations = 0
        terminal_reason = "iteration_limit"
        # 中文说明。
        self.policy.train(training)

        # 中文说明。
        for iteration in range(self._config.iterations):
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break
            # 中文说明。
            route_tensor = build_route_tensor(snapshot, current.selected_column_ids, self._config.tensor_size)
            # 中文说明。
            shaking_logits, improvement_logits, value = self.policy(route_tensor)
            # 中文说明。
            shaking_action, shaking_log_probability, shaking_entropy = select_policy_action(
                shaking_logits.squeeze(0),
                stochastic=training,
                generator=self._torch_generator,
            )
            # 中文说明。
            improvement_action, improvement_log_probability, improvement_entropy = select_policy_action(
                improvement_logits.squeeze(0),
                stochastic=training,
                generator=self._torch_generator,
            )
            # 中文说明。
            ratio = self._config.destroy_ratios[iteration % len(self._config.destroy_ratios)]
            # 中文说明。
            destroy = destroy_neighborhood(
                snapshot,
                current,
                DestroyOperator(shaking_action),
                ratio,
                self._rng,
            )
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
                RepairMode(improvement_action),
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
            action_log.append(f"{DestroyOperator(shaking_action).name}:{RepairMode(improvement_action).name}:{ratio:.3g}")
            # 中文说明。
            if training:
                # 中文说明。
                advantage = torch.tensor(reward, dtype=value.dtype) - value.squeeze(0)
                # 中文说明。
                actor_loss = -(shaking_log_probability + improvement_log_probability) * advantage.detach()
                # 中文说明。
                critic_loss = advantage.square()
                # 中文说明。
                entropy = shaking_entropy + improvement_entropy
                # 中文说明。
                losses.append(
                    actor_loss
                    + self._config.value_coefficient * critic_loss
                    - self._config.entropy_coefficient * entropy
                )
            completed_iterations += 1
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break

        # 中文说明。
        diagnostics = (
            ("algorithm", "LANS"),
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
