"""Classical score-adaptive operator selection for ALNS. 中文说明。"""

# 中文说明。
import dataclasses
# 中文说明。
import math
# 中文说明。
import random
# 中文说明。
import time
# 中文说明。
from dataclasses import dataclass

# 中文说明。
from Agent.base_agent import BaseAgent
# 中文说明。
from Agent.baselines.common import initial_solution, result_validator, with_diagnostics
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
class AdaptiveOperatorWeights:
    """Maintain segment scores and reaction-factor weight updates. 中文说明。"""

    # 中文说明。
    def __init__(self, action_count: int, reaction: float = 0.2, minimum_weight: float = 1.0e-6) -> None:
        """中文说明。"""
        # 中文说明。
        if type(action_count) is not int or action_count <= 0:
            # 中文说明。
            raise ValueError("action_count must be a positive integer")
        # 中文说明。
        if not math.isfinite(reaction) or not 0.0 < reaction <= 1.0:
            # 中文说明。
            raise ValueError("reaction must lie in (0, 1]")
        # 中文说明。
        if not math.isfinite(minimum_weight) or minimum_weight <= 0.0:
            # 中文说明。
            raise ValueError("minimum_weight must be positive")
        # 中文说明。
        self._weights = [1.0] * action_count
        # 中文说明。
        self._scores = [0.0] * action_count
        # 中文说明。
        self._uses = [0] * action_count
        # 中文说明。
        self._reaction = reaction
        # 中文说明。
        self._minimum = minimum_weight

    @property
    # 中文说明。
    def values(self) -> tuple[float, ...]:
        """中文说明。"""
        # 中文说明。
        return tuple(self._weights)

    # 中文说明。
    def sample(self, rng: random.Random) -> int:
        """中文说明。"""
        # 中文说明。
        return int(rng.choices(range(len(self._weights)), weights=self._weights, k=1)[0])

    # 中文说明。
    def record(self, action_id: int, score: float) -> None:
        """中文说明。"""
        # 中文说明。
        if not 0 <= action_id < len(self._weights):
            # 中文说明。
            raise IndexError("operator action is outside the configured range")
        # 中文说明。
        if not math.isfinite(score) or score < 0.0:
            # 中文说明。
            raise ValueError("operator score must be finite and nonnegative")
        # 中文说明。
        self._scores[action_id] += score
        # 中文说明。
        self._uses[action_id] += 1

    # 中文说明。
    def end_segment(self) -> None:
        """中文说明。"""
        # 中文说明。
        for action_id, uses in enumerate(self._uses):
            # 中文说明。
            if uses:
                # 中文说明。
                average = self._scores[action_id] / uses
                # 中文说明。
                updated = (1.0 - self._reaction) * self._weights[action_id] + self._reaction * average
                # 中文说明。
                self._weights[action_id] = max(self._minimum, updated)
        # 中文说明。
        self._scores = [0.0] * len(self._scores)
        # 中文说明。
        self._uses = [0] * len(self._uses)


@dataclass(frozen=True, slots=True)
# 中文说明。
class ALNSConfig:
    """中文说明。"""
    # 中文说明。
    iterations: int
    # 中文说明。
    destroy_ratios: tuple[float, ...]
    # 中文说明。
    reaction: float = 0.2
    # 中文说明。
    segment_length: int = 10
    # 中文说明。
    initial_temperature_ratio: float = 0.05
    # 中文说明。
    cooling_rate: float = 0.995
    # 中文说明。
    seed: int = 0
    # 中文说明。
    initial_solution_mode: str = "greedy"

    # 中文说明。
    def __post_init__(self) -> None:
        """中文说明。"""
        # 中文说明。
        if self.iterations <= 0 or not self.destroy_ratios:
            # 中文说明。
            raise ValueError("ALNS requires positive iterations and destroy ratios")
        # 中文说明。
        if any(not 0.0 < value <= 1.0 for value in self.destroy_ratios):
            # 中文说明。
            raise ValueError("ALNS destroy ratios must lie in (0, 1]")
        # 中文说明。
        if not 0.0 < self.reaction <= 1.0 or self.segment_length <= 0:
            # 中文说明。
            raise ValueError("ALNS adaptive-weight controls are invalid")
        # 中文说明。
        if self.initial_temperature_ratio < 0.0 or not 0.0 < self.cooling_rate <= 1.0:
            # 中文说明。
            raise ValueError("ALNS annealing controls are invalid")
        # 中文说明。
        if self.initial_solution_mode not in {"greedy", "exact"}:
            # 中文说明。
            raise ValueError("ALNS initial solution mode is invalid")


# 中文说明。
class ALNSAgent(BaseAgent):
    """Run adaptive destroy-repair search with simulated-annealing acceptance. 中文说明。"""

    # 中文说明。
    def __init__(
        self,
        projection_solver: BaseAgent,
        config: ALNSConfig,
        *,
        runtime_policy: EvaluationRuntimePolicy | None = None,
    ) -> None:
        """中文说明。"""
        # 中文说明。
        self._solver = projection_solver
        # 中文说明。
        self._config = config
        self._runtime_policy = runtime_policy
        # 中文说明。
        self._rng = random.Random(config.seed)

    # 中文说明。
    def solve(self, snapshot: SceneSnapshot) -> AgentResult:
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
        destroy_count = len(DestroyOperator) * len(self._config.destroy_ratios)
        # 中文说明。
        destroy_weights = AdaptiveOperatorWeights(destroy_count, self._config.reaction)
        # 中文说明。
        repair_weights = AdaptiveOperatorWeights(len(RepairMode), self._config.reaction)
        # 中文说明。
        temperature = self._config.initial_temperature_ratio * max(abs(current.objective_value), 1.0)
        # 中文说明。
        attempted = feasible = accepted = rejected = 0
        completed_iterations = 0
        terminal_reason = "iteration_limit"

        # 中文说明。
        for iteration in range(self._config.iterations):
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break
            # 中文说明。
            destroy_id = destroy_weights.sample(self._rng)
            # 中文说明。
            repair_id = repair_weights.sample(self._rng)
            # 中文说明。
            ratio_count = len(self._config.destroy_ratios)
            # 中文说明。
            operator = DestroyOperator(destroy_id // ratio_count)
            # 中文说明。
            ratio = self._config.destroy_ratios[destroy_id % ratio_count]
            # 中文说明。
            action = destroy_neighborhood(snapshot, current, operator, ratio, self._rng)
            # 中文说明。
            if not action.removed_column_ids:
                # 中文说明。
                terminal_reason = "empty_action_space"
                break
            # 中文说明。
            attempted += 1
            # 中文说明。
            outcome = repair_neighborhood(
                snapshot,
                action,
                RepairMode(repair_id),
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
            score = 0.0
            # 中文说明。
            if candidate is None or not candidate.is_feasible:
                # 中文说明。
                rejected += 1
            else:
                # 中文说明。
                feasible += 1
                # 中文说明。
                delta = candidate.objective_value - current.objective_value
                # 中文说明。
                accept = delta <= 0.0
                # 中文说明。
                if not accept and temperature > 0.0:
                    # 中文说明。
                    accept = self._rng.random() < math.exp(-delta / max(temperature, 1.0e-12))
                # 中文说明。
                if candidate.objective_value < best.objective_value:
                    # 中文说明。
                    best = candidate
                    # 中文说明。
                    score = 8.0
                # 中文说明。
                elif delta < 0.0:
                    # 中文说明。
                    score = 4.0
                # 中文说明。
                elif accept:
                    # 中文说明。
                    score = 2.0
                else:
                    # 中文说明。
                    score = 1.0
                # 中文说明。
                if accept:
                    # 中文说明。
                    current = candidate
                    # 中文说明。
                    accepted += 1
                else:
                    # 中文说明。
                    rejected += 1
            # 中文说明。
            destroy_weights.record(destroy_id, score)
            # 中文说明。
            repair_weights.record(repair_id, score)
            # 中文说明。
            if (iteration + 1) % self._config.segment_length == 0:
                # 中文说明。
                destroy_weights.end_segment()
                # 中文说明。
                repair_weights.end_segment()
            # 中文说明。
            temperature *= self._config.cooling_rate
            completed_iterations += 1
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break

        # 中文说明。
        destroy_weights.end_segment()
        # 中文说明。
        repair_weights.end_segment()
        # 中文说明。
        diagnostics = (
            ("algorithm", "ALNS"),
            ("attempted_neighborhoods", str(attempted)),
            ("feasible_repairs", str(feasible)),
            ("accepted_candidates", str(accepted)),
            ("projection_rejections", str(rejected)),
            ("destroy_weights", ",".join(f"{value:.8g}" for value in destroy_weights.values)),
            ("repair_weights", ",".join(f"{value:.8g}" for value in repair_weights.values)),
            ("final_temperature", f"{temperature:.8g}"),
            ("completed_iterations", str(completed_iterations)),
            ("terminal_reason", terminal_reason),
            ("iteration_limit", str(self._config.iterations)),
        )
        # 中文说明。
        return with_diagnostics(best.result, time.perf_counter() - started, diagnostics)
