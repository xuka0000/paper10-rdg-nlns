"""Deterministic MILP neighborhood repair for RDG-NLNS. 中文说明。"""

from Env.snapshot import uav_schedulable

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import dataclasses
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from dataclasses import dataclass
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from enum import IntEnum
import math

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.base_agent import BaseAgent
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.evaluation_runtime import RuntimeDeadline
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.operators import DestroyAction
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.initialization import build_route_feasible_neighborhood_plan
from Agent.hybrid.rdg_nlns.projection import ValidationFunction, project_candidate
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.solution import CandidatePlan, SearchSolution
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Env.snapshot import SceneSnapshot
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Env.validation import ResultValidation


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RepairMode(IntEnum):
    """Stable action identifiers used by the repair actor. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    FEASIBLE_INSERTION = 0
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    REGRET_INSERTION = 1
    SYNCHRONIZATION_AWARE = 2  # 中文说明。
    RESTRICTED = FEASIBLE_INSERTION  # 中文说明。
    RELAXED = REGRET_INSERTION  # 中文说明。


@dataclass(frozen=True, slots=True)
class InsertionWeights:
    """Explicit weights for the manuscript insertion-cost equation."""

    energy: float = 1.0
    duration: float = 1.0
    shared_uav: float = 1000.0
    stop_discontinuity: float = 10.0

    def __post_init__(self) -> None:
        values = (self.energy, self.duration, self.shared_uav, self.stop_discontinuity)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Insertion weights must be finite and non-negative")


def _locked_column_ids(snapshot: SceneSnapshot) -> tuple[str, ...]:  # 中文说明。
    """中文说明。"""
    airborne = tuple(state.column_id for state in getattr(snapshot, "airborne_residual_states", ()))  # 中文说明。
    values = (*snapshot.locked_column_ids, *getattr(snapshot, "committed_column_ids", ()), *airborne)  # 中文说明。
    return tuple(dict.fromkeys(values))  # 中文说明。


def deterministic_insertion_repair(
    snapshot: SceneSnapshot,
    action: DestroyAction,
    mode: RepairMode,
    weights: InsertionWeights = InsertionWeights(),
    *,
    deadline: RuntimeDeadline | None = None,
) -> CandidatePlan:  # 中文说明。
    """Reinsert uncovered checkpoints with stable, solver-free tie breaking. 中文说明。"""
    if mode not in (RepairMode.FEASIBLE_INSERTION, RepairMode.REGRET_INSERTION, RepairMode.SYNCHRONIZATION_AWARE):  # 中文说明。
        raise ValueError(f"Unsupported repair mode: {mode}")  # 中文说明。
    locked = _locked_column_ids(snapshot)  # 中文说明。
    columns = tuple(snapshot.service_columns)
    column_by_id = {column.object_id: column for column in columns}
    effective_by_id = {
        column.object_id: (
            snapshot.column_state(column.object_id)
            if hasattr(snapshot, "column_state")
            else column
        )
        for column in columns
    }
    admitted_column_ids = {
        column.object_id
        for column in columns
        if (
            getattr(effective_by_id[column.object_id], "is_committed", False)
            or not hasattr(snapshot, "uav_state")
            or uav_schedulable(snapshot.uav_state(column.uav_id))
        )
    }
    selected = list(
        dict.fromkeys(
            (
                *locked,
                *(
                    column_id
                    for column_id in action.retained_column_ids
                    if column_id in admitted_column_ids
                ),
            )
        )
    )
    selected_set = set(selected)
    insertion_order: list[str] = []  # 中文说明。
    completed = set(snapshot.completed_checkpoint_ids)
    all_checkpoint_ids = tuple(
        dict.fromkeys(
            task.checkpoint_id
            for task in snapshot.tasks
            if task.checkpoint_id not in completed
        )
    )
    checkpoint_ids_by_column = {
        column.object_id: tuple(
            getattr(
                effective_by_id[column.object_id],
                "effective_checkpoint_ids",
                column.covered_checkpoint_ids,
            )
        )
        for column in columns
    }
    base_cost_by_column = {
        column.object_id: weights.energy * float(
            getattr(
                effective_by_id[column.object_id],
                "effective_protected_energy",
                column.energy_quantile,
            )
        )
        + weights.duration * float(
            getattr(
                effective_by_id[column.object_id],
                "effective_duration",
                column.dynamic_duration,
            )
        )
        for column in columns
        if column.object_id in admitted_column_ids
    }
    candidate_ids_by_checkpoint: dict[str, list[str]] = {
        checkpoint_id: [] for checkpoint_id in all_checkpoint_ids
    }
    for column in columns:
        if column.object_id not in admitted_column_ids:
            continue
        for checkpoint_id in checkpoint_ids_by_column[column.object_id]:
            if checkpoint_id in candidate_ids_by_checkpoint:
                candidate_ids_by_checkpoint[checkpoint_id].append(column.object_id)

    covered_ids: set[str] = set()
    selected_uav_counts: dict[str, int] = {}
    selected_recovery_counts: dict[str, int] = {}
    selected_known_count = 0
    for column_id in selected:
        column = column_by_id.get(column_id)
        if column is None:
            continue
        covered_ids.update(checkpoint_ids_by_column[column_id])
        selected_known_count += 1
        selected_uav_counts[column.uav_id] = selected_uav_counts.get(column.uav_id, 0) + 1
        selected_recovery_counts[column.recovery_stop_id] = selected_recovery_counts.get(column.recovery_stop_id, 0) + 1

    def insertion_cost(column_id: str) -> float:
        column = column_by_id[column_id]
        cost = base_cost_by_column[column_id]
        if mode == RepairMode.SYNCHRONIZATION_AWARE:
            shared_uav = selected_uav_counts.get(column.uav_id, 0)
            changed_stop = selected_known_count - selected_recovery_counts.get(
                column.launch_stop_id, 0
            )
            cost += (
                shared_uav * weights.shared_uav
                + changed_stop * weights.stop_discontinuity
            )
        return cost

    while True:  # 中文说明。
        uncovered = [
            checkpoint_id
            for checkpoint_id in all_checkpoint_ids
            if checkpoint_id not in covered_ids
        ]
        options: dict[str, list[tuple[float, str]]] = {}  # 中文说明。
        for checkpoint_id in uncovered:  # 中文说明。
            if deadline is not None and deadline.expired():
                return CandidatePlan(tuple(selected), tuple(uncovered), tuple(insertion_order))
            candidates: list[tuple[float, str]] = []  # 中文说明。
            for column_id in candidate_ids_by_checkpoint.get(checkpoint_id, ()):  # 中文说明。
                if column_id in selected_set:  # 中文说明。
                    continue  # 中文说明。
                if mode == RepairMode.FEASIBLE_INSERTION and column_id in action.removed_column_ids:  # 中文说明。
                    continue  # 中文说明。
                if set(checkpoint_ids_by_column[column_id]) & covered_ids:  # 中文说明。
                    continue  # 中文说明。
                candidates.append((insertion_cost(column_id), column_id))  # 中文说明。
            options[checkpoint_id] = sorted(candidates, key=lambda item: (item[0], item[1]))  # 中文说明。
        feasible_tasks = [checkpoint_id for checkpoint_id, candidates in options.items() if candidates]  # 中文说明。
        if not feasible_tasks:  # 中文说明。
            return CandidatePlan(tuple(selected), tuple(uncovered), tuple(insertion_order))  # 中文说明。
        if mode == RepairMode.REGRET_INSERTION:  # 中文说明。
            def regret(checkpoint_id: str) -> tuple[float, str]:  # 中文说明。
                """中文说明。"""
                values = options[checkpoint_id]  # 中文说明。
                value = values[1][0] - values[0][0] if len(values) > 1 else float("inf")  # 中文说明。
                return (-value, checkpoint_id)  # 中文说明。
            chosen_checkpoint = min(feasible_tasks, key=regret)  # 中文说明。
        else:
            chosen_checkpoint = min(feasible_tasks, key=lambda checkpoint_id: (options[checkpoint_id][0][0], checkpoint_id))  # 中文说明。
        chosen_column_id = options[chosen_checkpoint][0][1]  # 中文说明。
        selected.append(chosen_column_id)  # 中文说明。
        selected_set.add(chosen_column_id)
        insertion_order.append(chosen_checkpoint)  # 中文说明。
        chosen_column = column_by_id[chosen_column_id]
        covered_ids.update(checkpoint_ids_by_column[chosen_column_id])
        selected_known_count += 1
        selected_uav_counts[chosen_column.uav_id] = selected_uav_counts.get(
            chosen_column.uav_id, 0
        ) + 1
        selected_recovery_counts[chosen_column.recovery_stop_id] = (
            selected_recovery_counts.get(chosen_column.recovery_stop_id, 0) + 1
        )


@dataclass(frozen=True, slots=True)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RepairOutcome:
    """Store the repaired candidate and its model-based projection evidence. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    candidate: SearchSolution | None
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    validation: ResultValidation
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    mode: RepairMode
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    removed_column_ids: tuple[str, ...]
    corrections: tuple[str, ...] = ()  # 中文说明。
    used_milp_fallback: bool = False  # 中文说明。


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def _neighborhood_snapshot(snapshot: SceneSnapshot, action: DestroyAction, mode: RepairMode) -> SceneSnapshot:
    """说明当前算法或测试步骤。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    retained_locks = tuple(dict.fromkeys((*snapshot.locked_column_ids, *action.retained_column_ids)))
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    service_columns = snapshot.service_columns
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if mode == RepairMode.RESTRICTED:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        removed = set(action.removed_column_ids)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        service_columns = tuple(column for column in service_columns if column.object_id not in removed)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    elif mode != RepairMode.RELAXED:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError(f"Unsupported repair mode: {mode}")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return dataclasses.replace(snapshot, service_columns=service_columns, locked_column_ids=retained_locks)


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def repair_neighborhood(
    snapshot: SceneSnapshot,
    action: DestroyAction,
    mode: RepairMode,
    solver: BaseAgent,
    validator: ValidationFunction,
    *,
    weights: InsertionWeights = InsertionWeights(),
    deadline: RuntimeDeadline | None = None,
    route_projection_time_limit_seconds: float = 10.0,
    plan_callback=None,
) -> RepairOutcome:
    """Repair one neighborhood and project the candidate through all environment checks. 中文说明。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    plan = deterministic_insertion_repair(snapshot, action, mode, weights, deadline=deadline)
    if plan_callback is not None:
        plan_callback(plan)
    projection = project_candidate(
        snapshot,
        plan,
        solver,
        validator,
        deadline=deadline,
        route_projection_time_limit_seconds=route_projection_time_limit_seconds,
    )  # 中文说明。
    validation = projection.validation  # 中文说明。
    candidate = projection.candidate  # 中文说明。
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if candidate is not None:
        return RepairOutcome(
            candidate,
            validation,
            mode,
            action.removed_column_ids,
            projection.corrections,
            projection.used_milp_fallback,
        )

    excluded_column_ids = (
        action.removed_column_ids
        if mode == RepairMode.FEASIBLE_INSERTION
        else ()
    )
    route_plan = build_route_feasible_neighborhood_plan(
        snapshot,
        required_column_ids=action.retained_column_ids,
        excluded_column_ids=excluded_column_ids,
        time_limit_seconds=route_projection_time_limit_seconds,
        deadline=deadline,
    )
    if route_plan is None:
        return RepairOutcome(
            None,
            validation,
            mode,
            action.removed_column_ids,
            projection.corrections,
            projection.used_milp_fallback,
        )
    selected_column_ids, road_options, route_planner = route_plan
    routed_snapshot = dataclasses.replace(snapshot, road_options=road_options)
    routed_projection = project_candidate(
        routed_snapshot,
        CandidatePlan(selected_column_ids, ()),
        solver,
        validator,
        deadline=deadline,
        route_projection_time_limit_seconds=route_projection_time_limit_seconds,
    )
    corrections = (
        *projection.corrections,
        f"route_repair:{route_planner}",
        *routed_projection.corrections,
    )
    used_milp_fallback = bool(
        projection.used_milp_fallback
        or routed_projection.used_milp_fallback
        or route_planner == "route_aware_milp"
    )
    return RepairOutcome(
        routed_projection.candidate,
        routed_projection.validation,
        mode,
        action.removed_column_ids,
        corrections,
        used_milp_fallback,
    )
