"""Adapted two-stage heuristic with the source TSHA neighborhood search."""

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from enum import Enum
import math
import random
import time

from Agent.base_agent import BaseAgent
from Agent.baselines.common import result_validator, with_diagnostics
from Agent.evaluation_runtime import EvaluationRuntimePolicy, RuntimeDeadline
from Agent.hybrid.rdg_nlns.initialization import (
    build_greedy_initial_solution,
    _route_order_selected_columns,
    solver_with_time_limit,
)
from Agent.hybrid.rdg_nlns.projection import project_candidate
from Agent.hybrid.rdg_nlns.solution import CandidatePlan
from Agent.results import AgentResult
from Env.snapshot import SceneSnapshot
from Env.validation import ResultValidation
from Mask.inspection import ObservationPose


class TSHAOperator(str, Enum):
    """The four task-sequence transformations defined by Xia et al."""

    SWAP = "swap"
    PAIR_SWAP = "pair_swap"
    REVERSE = "reverse"
    NEAREST_INSERT = "nearest_insert"


@dataclass(frozen=True, slots=True)
class TSHAConfig:
    """Bound the adapted simulated-annealing stage under a common budget."""

    iterations: int = 100
    seed: int = 0
    initial_temperature: float = 90.0
    cooling_rate: float = 0.98
    iterations_per_temperature: int = 150
    maximum_stagnant_temperatures: int = 11
    minimum_temperature: float = 0.01
    weight_memory: float = 0.8
    history_retry_limit: int = 8
    route_aware_initialization_fallback: bool = False

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("TSHA iterations must be nonnegative")
        if not math.isfinite(self.initial_temperature) or self.initial_temperature <= 0.0:
            raise ValueError("TSHA initial temperature must be finite and positive")
        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError("TSHA cooling rate must be in (0, 1)")
        if self.iterations_per_temperature <= 0:
            raise ValueError("TSHA iterations per temperature must be positive")
        if self.maximum_stagnant_temperatures <= 0:
            raise ValueError("TSHA stagnant-temperature limit must be positive")
        if not math.isfinite(self.minimum_temperature) or self.minimum_temperature <= 0.0:
            raise ValueError("TSHA minimum temperature must be finite and positive")
        if not 0.0 < self.weight_memory < 1.0:
            raise ValueError("TSHA weight memory must be in (0, 1)")
        if self.history_retry_limit <= 0:
            raise ValueError("TSHA history retry limit must be positive")


@dataclass(frozen=True, slots=True)
class TwoStagePlan:
    """Store the ground-stop stage and the UAV assignment stage."""

    candidate: CandidatePlan
    stop_sequence: tuple[str, ...]
    task_order: tuple[str, ...]


class AdaptiveNeighborhoodWeights:
    """Maintain source-style roulette weights over the four transformations."""

    def __init__(self, memory: float) -> None:
        if not 0.0 < memory < 1.0:
            raise ValueError("weight memory must be in (0, 1)")
        self._memory = float(memory)
        self._weights = [1.0] * len(TSHAOperator)
        self._scores = [0.0] * len(TSHAOperator)
        self._uses = [0] * len(TSHAOperator)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._weights)

    def sample(self, rng: random.Random) -> TSHAOperator:
        threshold = rng.random() * sum(self._weights)
        cumulative = 0.0
        for index, weight in enumerate(self._weights):
            cumulative += weight
            if threshold <= cumulative:
                return tuple(TSHAOperator)[index]
        return tuple(TSHAOperator)[-1]

    def record(self, operator: TSHAOperator, score: float) -> None:
        index = tuple(TSHAOperator).index(operator)
        self._scores[index] += max(0.0, float(score))
        self._uses[index] += 1

    def end_temperature(self) -> None:
        for index, uses in enumerate(self._uses):
            if uses > 0:
                average_score = self._scores[index] / uses
                self._weights[index] = max(
                    1.0e-9,
                    self._memory * self._weights[index]
                    + (1.0 - self._memory) * average_score,
                )
            self._scores[index] = 0.0
            self._uses[index] = 0


def apply_tsha_operator(
    task_order: Sequence[str],
    operator: TSHAOperator,
    rng: random.Random,
    positions: Mapping[str, tuple[float, float, float]] | None = None,
) -> tuple[str, ...]:
    """Apply one source TSHA task-sequence transformation."""
    transformed = list(task_order)
    count = len(transformed)
    if count < 2:
        return tuple(transformed)

    if operator is TSHAOperator.SWAP:
        first, second = rng.sample(range(count), 2)
        transformed[first], transformed[second] = transformed[second], transformed[first]
    elif operator is TSHAOperator.PAIR_SWAP:
        pair_starts = [
            (first, second)
            for first in range(count - 1)
            for second in range(first + 2, count - 1)
        ]
        if not pair_starts:
            return tuple(transformed)
        first, second = rng.choice(pair_starts)
        left = transformed[first : first + 2]
        right = transformed[second : second + 2]
        transformed[first : first + 2] = right
        transformed[second : second + 2] = left
    elif operator is TSHAOperator.REVERSE:
        first, second = sorted(rng.sample(range(count), 2))
        transformed[first : second + 1] = reversed(transformed[first : second + 1])
    elif operator is TSHAOperator.NEAREST_INSERT:
        moved_index = rng.randrange(count)
        moved = transformed.pop(moved_index)
        if positions is None or moved not in positions:
            return tuple(task_order)
        moved_position = positions[moved]

        def squared_distance(checkpoint_id: str) -> float:
            target = positions.get(checkpoint_id)
            if target is None:
                return math.inf
            return sum((left - right) ** 2 for left, right in zip(moved_position, target))

        nearest_index = min(
            range(len(transformed)),
            key=lambda index: (squared_distance(transformed[index]), transformed[index]),
        )
        if not math.isfinite(squared_distance(transformed[nearest_index])):
            return tuple(task_order)
        transformed.insert(nearest_index + 1, moved)
    else:
        raise ValueError(f"unsupported TSHA operator: {operator}")
    return tuple(transformed)


def _stop_access_cost(snapshot: SceneSnapshot) -> dict[str, float]:
    stop_node = {stop.object_id: stop.road_node_id for stop in snapshot.stop_sites}
    costs: dict[str, float] = {}
    for stop_id, node_id in stop_node.items():
        adjacent = [
            option.travel_time
            for option in snapshot.road_options
            if option.accessible and node_id in {option.source_node_id, option.target_node_id}
        ]
        costs[stop_id] = min(adjacent, default=snapshot.time_horizon)
    return costs


def _candidate_columns_by_checkpoint(snapshot: SceneSnapshot) -> dict[str, list[object]]:
    completed = set(snapshot.completed_checkpoint_ids)
    return {
        task.checkpoint_id: [
            column
            for column in snapshot.service_columns
            if column.admitted
            and task.checkpoint_id in snapshot.column_state(column.object_id).effective_checkpoint_ids
            and (
                snapshot.uav_state(column.uav_id).available
                or snapshot.column_state(column.object_id).is_committed
            )
        ]
        for task in snapshot.tasks
        if task.checkpoint_id not in completed
    }


def _build_plan_for_task_order(
    snapshot: SceneSnapshot,
    task_order: Sequence[str],
) -> TwoStagePlan:
    completed = set(snapshot.completed_checkpoint_ids)
    selected = list(dict.fromkeys((*snapshot.locked_column_ids, *snapshot.committed_column_ids)))
    column_by_id = {column.object_id: column for column in snapshot.service_columns}
    covered = set(completed)
    for column_id in selected:
        if column_id in column_by_id:
            covered.update(snapshot.column_state(column_id).effective_checkpoint_ids)

    access_cost = _stop_access_cost(snapshot)
    candidates_by_checkpoint = _candidate_columns_by_checkpoint(snapshot)
    stop_sequence: list[str] = []
    uav_load: dict[str, int] = {}
    deferred: list[str] = []
    for checkpoint_id in task_order:
        if checkpoint_id in covered:
            continue
        candidates = [
            column
            for column in candidates_by_checkpoint.get(checkpoint_id, ())
            if column.object_id not in selected
        ]
        if not candidates:
            deferred.append(checkpoint_id)
            continue
        chosen = min(
            candidates,
            key=lambda column: (
                access_cost.get(column.launch_stop_id, snapshot.time_horizon)
                + access_cost.get(column.recovery_stop_id, snapshot.time_horizon)
                + float(snapshot.column_state(column.object_id).effective_duration)
                + 0.001 * float(snapshot.column_state(column.object_id).effective_protected_energy)
                + uav_load.get(column.uav_id, 0) * snapshot.time_horizon,
                -len(snapshot.column_state(column.object_id).effective_checkpoint_ids),
                column.object_id,
            ),
        )
        selected.append(chosen.object_id)
        uav_load[chosen.uav_id] = uav_load.get(chosen.uav_id, 0) + 1
        covered.update(snapshot.column_state(chosen.object_id).effective_checkpoint_ids)
        for stop_id in (chosen.launch_stop_id, chosen.recovery_stop_id):
            if stop_id not in stop_sequence:
                stop_sequence.append(stop_id)

    required = {task.checkpoint_id for task in snapshot.tasks if task.checkpoint_id not in completed}
    deferred.extend(sorted(required - covered - set(deferred)))
    ordered_selected = _route_order_selected_columns(
        snapshot,
        tuple(dict.fromkeys(selected)),
    )
    order = tuple(task_order)
    candidate = CandidatePlan(ordered_selected, tuple(dict.fromkeys(deferred)), order)
    return TwoStagePlan(candidate, tuple(stop_sequence), order)


def build_two_stage_plan(snapshot: SceneSnapshot) -> TwoStagePlan:
    """Construct the first-stage stop plan and initial coded task sequence."""
    access_cost = _stop_access_cost(snapshot)
    candidates_by_checkpoint = _candidate_columns_by_checkpoint(snapshot)
    priority_by_checkpoint = {task.checkpoint_id: float(task.priority) for task in snapshot.tasks}
    task_order = tuple(
        sorted(
            candidates_by_checkpoint,
            key=lambda checkpoint_id: (
                -priority_by_checkpoint[checkpoint_id],
                min(
                    (
                        access_cost.get(column.launch_stop_id, snapshot.time_horizon)
                        + access_cost.get(column.recovery_stop_id, snapshot.time_horizon)
                        for column in candidates_by_checkpoint[checkpoint_id]
                    ),
                    default=2.0 * snapshot.time_horizon,
                ),
                checkpoint_id,
            ),
        )
    )
    return _build_plan_for_task_order(snapshot, task_order)


def _checkpoint_positions(snapshot: SceneSnapshot) -> dict[str, tuple[float, float, float]]:
    samples: dict[str, list[tuple[float, float, float]]] = {}
    for element in snapshot.elements:
        if isinstance(element, ObservationPose):
            samples.setdefault(element.checkpoint_id, []).append(
                (element.position.x, element.position.y, element.position.z)
            )
    return {
        checkpoint_id: tuple(
            sum(point[axis] for point in points) / len(points)
            for axis in range(3)
        )
        for checkpoint_id, points in samples.items()
    }


def _source_score(current_objective: float, candidate_objective: float, accepted: bool) -> float:
    if candidate_objective < current_objective:
        return 1.0 + (current_objective - candidate_objective) / max(abs(current_objective), 1.0e-12)
    if accepted:
        return 1.0 + (candidate_objective - current_objective) / max(abs(candidate_objective), 1.0e-12)
    return max(
        0.0,
        1.0 + (current_objective - candidate_objective) / max(abs(candidate_objective), 1.0e-12),
    )


class TSHAAgent(BaseAgent):
    """Run an adapted TSHA construction and multi-neighborhood SA search."""

    def __init__(
        self,
        projection_solver: BaseAgent,
        config: TSHAConfig | None = None,
        *,
        runtime_policy: EvaluationRuntimePolicy | None = None,
    ) -> None:
        self._projection_solver = projection_solver
        self._config = config or TSHAConfig()
        self._runtime_policy = runtime_policy
        self._rng = random.Random(self._config.seed)

    def solve(self, snapshot: SceneSnapshot) -> AgentResult:
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
        initial_plan = build_two_stage_plan(snapshot)
        initial_projection = project_candidate(
            snapshot,
            initial_plan.candidate,
            solver_with_time_limit(self._projection_solver, initialization_limit),
            result_validator(snapshot),
            deadline=deadline,
            route_projection_time_limit_seconds=initialization_limit,
        )
        initialization_mode = "two_stage_fixed_columns"
        if initial_projection.candidate is None and self._config.route_aware_initialization_fallback:
            original_issues = initial_projection.issue_codes or initial_projection.validation.issue_codes
            try:
                fallback = build_greedy_initial_solution(
                    snapshot,
                    solver_with_time_limit(self._projection_solver, initialization_limit),
                    result_validator(snapshot),
                    deadline=deadline,
                    route_projection_time_limit_seconds=initialization_limit,
                )
            except Exception:
                raise RuntimeError(f"TSHA projection failed: {original_issues}")
            initial_projection = dataclasses.replace(
                initial_projection,
                candidate=fallback,
                validation=ResultValidation(True, ()),
                issue_codes=(),
                corrections=(*initial_projection.corrections, "route_aware_initialization_fallback"),
            )
            initialization_mode = "route_aware_greedy_fallback"
        if initial_projection.candidate is None:
            raise RuntimeError(
                f"TSHA projection failed: "
                f"{initial_projection.issue_codes or initial_projection.validation.issue_codes}"
            )

        current = best = initial_projection.candidate
        current_order = best_order = initial_plan.task_order
        best_deferred_count = len(initial_plan.candidate.deferred_checkpoint_ids)
        history = {current_order}
        positions = _checkpoint_positions(snapshot)
        weights = AdaptiveNeighborhoodWeights(self._config.weight_memory)
        operator_counts = {operator: 0 for operator in TSHAOperator}
        attempted = feasible = accepted = rejected = duplicate_retries = 0
        projection_rejections = completed_iterations = 0
        temperature = self._config.initial_temperature
        stagnant_temperatures = 0
        best_at_temperature_start = best.objective_value
        iterations_at_temperature = 0
        terminal_reason = "iteration_limit"

        for _ in range(self._config.iterations):
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break
            if temperature <= self._config.minimum_temperature:
                terminal_reason = "temperature_limit"
                break
            if stagnant_temperatures >= self._config.maximum_stagnant_temperatures:
                terminal_reason = "stagnation_limit"
                break

            operator = weights.sample(self._rng)
            candidate_order = apply_tsha_operator(current_order, operator, self._rng, positions)
            retries = 0
            while candidate_order in history and retries < self._config.history_retry_limit:
                duplicate_retries += 1
                retries += 1
                operator = weights.sample(self._rng)
                candidate_order = apply_tsha_operator(current_order, operator, self._rng, positions)

            attempted += 1
            completed_iterations += 1
            iterations_at_temperature += 1
            operator_counts[operator] += 1
            score = 0.0
            if candidate_order in history:
                rejected += 1
            else:
                candidate_plan = _build_plan_for_task_order(snapshot, candidate_order)
                outcome = project_candidate(
                    snapshot,
                    candidate_plan.candidate,
                    solver_with_time_limit(self._projection_solver, route_limit),
                    result_validator(snapshot),
                    deadline=deadline,
                    route_projection_time_limit_seconds=route_limit,
                )
                if "runtime_deadline" in (*outcome.issue_codes, *outcome.validation.issue_codes):
                    terminal_reason = "wall_clock_limit"
                    completed_iterations -= 1
                    attempted -= 1
                    iterations_at_temperature -= 1
                    break
                candidate = outcome.candidate
                if candidate is None or not candidate.is_feasible:
                    projection_rejections += 1
                    rejected += 1
                else:
                    feasible += 1
                    delta = candidate.objective_value - current.objective_value
                    accept = delta <= 0.0 or self._rng.random() < math.exp(
                        -delta / max(temperature, 1.0e-12)
                    )
                    score = _source_score(current.objective_value, candidate.objective_value, accept)
                    if candidate.objective_value < best.objective_value:
                        best = candidate
                        best_order = candidate_order
                        best_deferred_count = len(candidate_plan.candidate.deferred_checkpoint_ids)
                    if accept:
                        current = candidate
                        current_order = candidate_order
                        history.add(candidate_order)
                        accepted += 1
                    else:
                        rejected += 1
            weights.record(operator, score)

            if iterations_at_temperature >= self._config.iterations_per_temperature:
                weights.end_temperature()
                temperature *= self._config.cooling_rate
                if best.objective_value < best_at_temperature_start:
                    stagnant_temperatures = 0
                else:
                    stagnant_temperatures += 1
                best_at_temperature_start = best.objective_value
                iterations_at_temperature = 0

        if iterations_at_temperature:
            weights.end_temperature()

        diagnostics = (
            ("algorithm", "TSHA"),
            ("algorithm_source", "Xia_et_al_2025_10.1049/itr2.70028"),
            ("stage_1_stop_count", str(len(initial_plan.stop_sequence))),
            ("stage_2_task_count", str(len(initial_plan.task_order))),
            ("stage_2_mechanism", "adaptive_multineighborhood_simulated_annealing"),
            ("initialization_mode", initialization_mode),
            ("task_adapter", "checkpoint_sequence_to_certified_service_columns"),
            ("deferred_checkpoints", str(best_deferred_count)),
            ("projection_corrections", str(len(initial_projection.corrections))),
            ("attempted_neighborhoods", str(attempted)),
            ("feasible_neighborhoods", str(feasible)),
            ("accepted_neighborhoods", str(accepted)),
            ("rejected_neighborhoods", str(rejected)),
            ("projection_rejections", str(projection_rejections)),
            ("duplicate_retries", str(duplicate_retries)),
            ("history_size", str(len(history))),
            ("best_task_order", ";".join(best_order)),
            ("final_temperature", f"{temperature:.12g}"),
            ("stagnant_temperatures", str(stagnant_temperatures)),
            ("weight_memory", f"{self._config.weight_memory:.12g}"),
            ("weight_memory_source", "local_task_adapter"),
            ("operator_counts", ";".join(f"{operator.value}={operator_counts[operator]}" for operator in TSHAOperator)),
            ("operator_weights", ";".join(f"{operator.value}={weight:.12g}" for operator, weight in zip(TSHAOperator, weights.values))),
            ("completed_iterations", str(completed_iterations)),
            ("terminal_reason", terminal_reason),
            ("iteration_limit", str(self._config.iterations)),
        )
        return with_diagnostics(best.result, time.perf_counter() - started, diagnostics)
