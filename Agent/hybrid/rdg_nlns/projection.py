"""Model-based feasibility projection boundary for RDG-NLNS. 中文说明。"""

from Env.snapshot import uav_schedulable

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from collections.abc import Callable
import dataclasses  # 中文说明。
from dataclasses import dataclass  # 中文说明。
import heapq
import math

import numpy as np
from scipy.optimize import milp

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.evaluation_runtime import RuntimeDeadline
from Agent.results import AgentResult
from Agent.base_agent import BaseAgent  # 中文说明。
from Agent.hybrid.rdg_nlns.solution import CandidatePlan, SearchSolution  # 中文说明。
from Agent.optimization.linear_model import LinearModelBuilder
from Env.snapshot import SceneSnapshot  # 中文说明。
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Env.validation import ResultValidation


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
ValidationFunction = Callable[[AgentResult], ResultValidation]


def has_finite_incumbent(result: AgentResult) -> bool:
    """Return whether a solver result contains a complete numerical candidate."""
    return bool(
        result.objective_value is not None
        and math.isfinite(float(result.objective_value))
        and result.variable_values
    )


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def project_result(result: AgentResult, validator: ValidationFunction) -> ResultValidation:
    """Validate a repaired result through the environment-owned constraint checker. 中文说明。"""
    validation_view = (
        dataclasses.replace(result, success=True)
        if has_finite_incumbent(result)
        else result
    )
    return validator(validation_view)


@dataclass(frozen=True, slots=True)
class ProjectionOutcome:  # 中文说明。
    """A corrected, timed candidate and auditable projection evidence. 中文说明。"""

    candidate: SearchSolution | None  # 中文说明。
    corrections: tuple[str, ...]  # 中文说明。
    validation: ResultValidation  # 中文说明。
    used_milp_fallback: bool  # 中文说明。
    issue_codes: tuple[str, ...] = ()  # 中文说明。
    deferred_checkpoint_ids: tuple[str, ...] = ()


def _covered_checkpoint_ids(snapshot: SceneSnapshot, column_ids: list[str]) -> set[str]:
    """Return the effective checkpoint coverage of the selected columns."""
    covered: set[str] = set()
    for column_id in column_ids:
        covered.update(snapshot.column_state(column_id).effective_checkpoint_ids)
    return covered


# Legacy diagnostic only. The manuscript-aligned production path below must
# preserve the selected joint action and never call this route-subset projector.
def _shortest_road_times(
    road_options: tuple[object, ...],
    source_node_id: str,
) -> dict[str, float]:
    """Compute directed shortest travel times on one vehicle's road graph."""
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for option in road_options:
        adjacency.setdefault(option.source_node_id, []).append(
            (option.target_node_id, float(option.travel_time))
        )
    distances = {source_node_id: 0.0}
    queue = [(0.0, source_node_id)]
    while queue:
        distance, node_id = heapq.heappop(queue)
        if distance > distances.get(node_id, math.inf):
            continue
        for target_id, travel_time in adjacency.get(node_id, ()):
            candidate = distance + travel_time
            if candidate < distances.get(target_id, math.inf):
                distances[target_id] = candidate
                heapq.heappush(queue, (candidate, target_id))
    return distances


def _single_column_route_time_lower_bound(
    snapshot: SceneSnapshot,
    column_id: str,
    start_node_id: str,
    end_node_id: str,
    road_options: tuple[object, ...],
    stop_node_by_id: dict[str, str],
    shortest_time_cache: dict[str, dict[str, float]] | None = None,
) -> float:
    """Return the earliest possible depot completion time for one column."""
    cache = shortest_time_cache if shortest_time_cache is not None else {}

    def shortest_from(node_id: str) -> dict[str, float]:
        if node_id not in cache:
            cache[node_id] = _shortest_road_times(road_options, node_id)
        return cache[node_id]

    column = next(
        item for item in snapshot.service_columns if item.object_id == column_id
    )
    effective = snapshot.column_state(column_id)
    recovery_node_id = stop_node_by_id[column.recovery_stop_id]
    from_start = shortest_from(start_node_id)
    from_recovery = shortest_from(recovery_node_id)
    recovery_to_end = from_recovery.get(end_node_id, math.inf)
    duration = float(effective.effective_duration)
    if not effective.requires_launch_stop:
        return max(
            from_start.get(recovery_node_id, math.inf),
            duration,
        ) + recovery_to_end
    launch_node_id = stop_node_by_id[column.launch_stop_id]
    from_launch = shortest_from(launch_node_id)
    return (
        from_start.get(launch_node_id, math.inf)
        + max(from_launch.get(recovery_node_id, math.inf), duration)
        + recovery_to_end
    )


def _legacy_vehicle_route_projected_column_ids(
    snapshot: SceneSnapshot,
    vehicle_id: str,
    column_ids: tuple[str, ...],
    locked_ids: set[str],
    *,
    deadline: RuntimeDeadline | None = None,
    route_projection_time_limit_seconds: float = 10.0,
) -> set[str] | None:
    """Select the maximum-value column subset supported by one simple route."""
    if not column_ids:
        return set()

    road_options = tuple(
        option
        for option in snapshot.road_options
        if option.vehicle_id == vehicle_id and option.accessible
    )
    start_node_id = snapshot.vehicle_state(vehicle_id).current_road_node_id
    end_node_id = next(
        node.object_id for node in snapshot.road_nodes if node.role == "depot_end"
    )
    column_by_id = {
        column.object_id: column for column in snapshot.service_columns
    }
    stop_node_by_id = {
        stop.object_id: stop.road_node_id for stop in snapshot.stop_sites
    }
    time_horizon = float(getattr(snapshot, "time_horizon", math.inf))
    shortest_time_cache: dict[str, dict[str, float]] = {}
    time_lower_bound = {
        column_id: _single_column_route_time_lower_bound(
            snapshot,
            column_id,
            start_node_id,
            end_node_id,
            road_options,
            stop_node_by_id,
            shortest_time_cache,
        )
        for column_id in column_ids
    }

    def deterministic_seed_subset() -> set[str]:
        retained_locked = set(column_ids) & locked_ids
        if retained_locked:
            return retained_locked
        feasible_ids = tuple(
            column_id
            for column_id in column_ids
            if time_lower_bound[column_id] <= time_horizon
        )
        if not feasible_ids:
            return set()
        selected = min(
            feasible_ids,
            key=lambda column_id: (
                time_lower_bound[column_id],
                column_id,
            ),
        )
        return {selected}

    road_node_ids = tuple(node.object_id for node in snapshot.road_nodes)
    if not road_options or start_node_id == end_node_id:
        return deterministic_seed_subset()

    priority_by_checkpoint = {
        task.checkpoint_id: float(task.priority) for task in snapshot.tasks
    }
    builder = LinearModelBuilder()
    total_travel_time = sum(float(option.travel_time) for option in road_options) + 1.0
    road_use = {
        option.road_leg_id: builder.add_binary(
            f"route_use[{vehicle_id},{option.road_leg_id}]",
            float(option.travel_time) / total_travel_time,
        )
        for option in road_options
    }
    visit = {
        node_id: builder.add_binary(f"route_visit[{vehicle_id},{node_id}]")
        for node_id in road_node_ids
        if node_id not in {start_node_id, end_node_id}
    }
    order = {
        node_id: builder.add_continuous(
            f"route_order[{vehicle_id},{node_id}]",
            0.0,
            float(len(road_node_ids)),
        )
        for node_id in road_node_ids
    }
    retain = {}
    for column_id in column_ids:
        effective = snapshot.column_state(column_id)
        retention_value = (
            1.0
            + sum(
                priority_by_checkpoint.get(checkpoint_id, 0.0)
                for checkpoint_id in effective.effective_checkpoint_ids
            )
            + 0.01 * len(effective.effective_checkpoint_ids)
        )
        retain[column_id] = builder.add_binary(
            f"route_retain[{vehicle_id},{column_id}]",
            -retention_value,
            upper=(
                1.0
                if column_id in locked_ids
                or time_lower_bound[column_id] <= time_horizon
                else 0.0
            ),
        )

    builder.add_constraint(
        f"route_time_horizon[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], float(option.travel_time))
            for option in road_options
        ),
        -math.inf,
        time_horizon,
    )
    uav_ids = tuple(
        dict.fromkeys(column_by_id[column_id].uav_id for column_id in column_ids)
    )
    for uav_id in uav_ids:
        builder.add_constraint(
            f"route_uav_time_horizon[{vehicle_id},{uav_id}]",
            (
                (
                    retain[column_id],
                    float(snapshot.column_state(column_id).effective_duration),
                )
                for column_id in column_ids
                if column_by_id[column_id].uav_id == uav_id
            ),
            -math.inf,
            time_horizon,
        )

    builder.add_constraint(
        f"route_start_out[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.source_node_id == start_node_id
        ),
        1.0,
        1.0,
    )
    builder.add_constraint(
        f"route_start_in[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.target_node_id == start_node_id
        ),
        0.0,
        0.0,
    )
    builder.add_constraint(
        f"route_end_in[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.target_node_id == end_node_id
        ),
        1.0,
        1.0,
    )
    builder.add_constraint(
        f"route_end_out[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.source_node_id == end_node_id
        ),
        0.0,
        0.0,
    )
    builder.add_constraint(
        f"route_start_order[{vehicle_id}]",
        ((order[start_node_id], 1.0),),
        0.0,
        0.0,
    )
    for node_id, visited in visit.items():
        incoming = tuple(
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.target_node_id == node_id
        )
        outgoing = tuple(
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.source_node_id == node_id
        )
        builder.add_constraint(
            f"route_visit_in[{vehicle_id},{node_id}]",
            (*incoming, (visited, -1.0)),
            0.0,
            0.0,
        )
        builder.add_constraint(
            f"route_visit_out[{vehicle_id},{node_id}]",
            (*outgoing, (visited, -1.0)),
            0.0,
            0.0,
        )

    order_big_m = float(len(road_node_ids) + 1)
    for option in road_options:
        if option.target_node_id == start_node_id or option.source_node_id == end_node_id:
            continue
        builder.add_constraint(
            f"route_order_link[{vehicle_id},{option.road_leg_id}]",
            (
                (order[option.target_node_id], 1.0),
                (order[option.source_node_id], -1.0),
                (road_use[option.road_leg_id], -order_big_m),
            ),
            1.0 - order_big_m,
            math.inf,
        )

    for column_id, retained in retain.items():
        column = column_by_id[column_id]
        effective = snapshot.column_state(column_id)
        required_node_ids = {stop_node_by_id[column.recovery_stop_id]}
        if effective.requires_launch_stop:
            required_node_ids.add(stop_node_by_id[column.launch_stop_id])
        for node_id in required_node_ids - {start_node_id, end_node_id}:
            builder.add_constraint(
                f"route_column_visit[{vehicle_id},{column_id},{node_id}]",
                ((retained, 1.0), (visit[node_id], -1.0)),
                -math.inf,
                0.0,
            )
        if column_id in locked_ids:
            builder.add_constraint(
                f"route_locked_column[{vehicle_id},{column_id}]",
                ((retained, 1.0),),
                1.0,
                1.0,
            )

    model = builder.build()
    if deadline is not None and deadline.expired():
        return None
    time_limit = float(route_projection_time_limit_seconds)
    if deadline is not None:
        time_limit = min(time_limit, deadline.remaining_seconds())
        if time_limit <= 0.0:
            return None
    result = milp(
        c=model.objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=model.linear_constraint,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True},
    )
    if deadline is not None and deadline.expired():
        return None
    if result.x is None or not np.isfinite(result.x).all():
        return deterministic_seed_subset()
    value_map = {
        name: float(result.x[index])
        for index, name in enumerate(model.variable_names)
    }
    return {
        column_id
        for column_id, variable in retain.items()
        if value_map.get(variable.name, 0.0) > 0.5
    }


def _legacy_route_projected_column_ids(
    snapshot: SceneSnapshot,
    selected_ids: list[str],
    locked_ids: tuple[str, ...],
    *,
    deadline: RuntimeDeadline | None = None,
    route_projection_time_limit_seconds: float = 10.0,
) -> tuple[str, ...] | None:
    """Project selected columns onto independently routable vehicle subsets."""
    if not selected_ids:
        return ()
    carrier_by_uav = {
        uav.object_id: uav.carrier_vehicle_id for uav in snapshot.uavs
    }
    column_by_id = {
        column.object_id: column for column in snapshot.service_columns
    }
    retained: set[str] = set()
    locked_set = set(locked_ids)
    for vehicle in snapshot.vehicles:
        if deadline is not None and deadline.expired():
            return None
        vehicle_column_ids = tuple(
            column_id
            for column_id in selected_ids
            if carrier_by_uav[column_by_id[column_id].uav_id] == vehicle.object_id
        )
        vehicle_retained = _legacy_vehicle_route_projected_column_ids(
            snapshot,
            vehicle.object_id,
            vehicle_column_ids,
            locked_set,
            deadline=deadline,
            route_projection_time_limit_seconds=route_projection_time_limit_seconds,
        )
        if vehicle_retained is None:
            return None
        retained.update(vehicle_retained)
    return tuple(column_id for column_id in selected_ids if column_id in retained)


def project_candidate(
    snapshot: SceneSnapshot,
    plan: CandidatePlan,
    solver: BaseAgent,
    validator: ValidationFunction,
    *,
    deadline: RuntimeDeadline | None = None,
    route_projection_time_limit_seconds: float = 10.0,
) -> ProjectionOutcome:  # 中文说明。
    """Correct a column plan before solving its restricted timing projection. 中文说明。"""
    column_by_id = {column.object_id: column for column in snapshot.service_columns}  # 中文说明。
    airborne = tuple(state.column_id for state in snapshot.airborne_residual_states)  # 中文说明。
    locked_ids = tuple(dict.fromkeys((*snapshot.locked_column_ids, *snapshot.committed_column_ids, *airborne)))  # 中文说明。
    selected = [column_id for column_id in dict.fromkeys(plan.selected_column_ids) if column_id in column_by_id]  # 中文说明。
    requested_selected = list(selected)
    corrections: list[str] = []  # 中文说明。
    deferred = list(dict.fromkeys(plan.deferred_checkpoint_ids))

    def runtime_deadline_outcome() -> ProjectionOutcome:
        validation = ResultValidation(False, ("runtime_deadline",))
        return ProjectionOutcome(
            None,
            tuple(corrections),
            validation,
            False,
            ("runtime_deadline",),
            tuple(deferred),
        )

    if deadline is not None and deadline.expired():
        return runtime_deadline_outcome()
    for column_id in plan.selected_column_ids:  # 中文说明。
        if column_id not in column_by_id:  # 中文说明。
            corrections.append(f"remove_unknown:{column_id}")  # 中文说明。
    for column_id in locked_ids:  # 中文说明。
        if column_id not in selected:  # 中文说明。
            selected.append(column_id)  # 中文说明。
            requested_selected.append(column_id)
            corrections.append(f"restore_locked:{column_id}")  # 中文说明。

    locked_coverage: dict[str, list[str]] = {}  # 中文说明。
    for column_id in locked_ids:  # 中文说明。
        for checkpoint_id in snapshot.column_state(column_id).effective_checkpoint_ids:  # 中文说明。
            locked_coverage.setdefault(checkpoint_id, []).append(column_id)  # 中文说明。
    if any(len(column_ids) > 1 for column_ids in locked_coverage.values()):  # 中文说明。
        validation = ResultValidation(False, ())  # 中文说明。
        return ProjectionOutcome(None, tuple(corrections), validation, False, ("locked_conflict",))  # 中文说明。

    for column_id in tuple(selected):  # 中文说明。
        effective = snapshot.column_state(column_id)  # 中文说明。
        if not uav_schedulable(snapshot.uav_state(effective.uav_id)) and not effective.is_committed:  # 中文说明。
            selected.remove(column_id)  # 中文说明。
            corrections.append(f"remove_unavailable:{column_id}")  # 中文说明。

    owners: dict[str, list[str]] = {}  # 中文说明。
    for column_id in selected:  # 中文说明。
        for checkpoint_id in snapshot.column_state(column_id).effective_checkpoint_ids:  # 中文说明。
            owners.setdefault(checkpoint_id, []).append(column_id)  # 中文说明。
    for checkpoint_id, column_ids in owners.items():  # 中文说明。
        if len(column_ids) <= 1:  # 中文说明。
            continue  # 中文说明。
        locked_owner = next((column_id for column_id in column_ids if column_id in locked_ids), None)  # 中文说明。
        keep = locked_owner or min(  # 中文说明。
            column_ids,
            key=lambda column_id: (
                snapshot.column_state(column_id).effective_protected_energy
                + snapshot.column_state(column_id).effective_duration,
                column_id,
            ),
        )
        for column_id in column_ids:  # 中文说明。
            if column_id != keep and column_id in selected:  # 中文说明。
                selected.remove(column_id)  # 中文说明。
                corrections.append(f"remove_duplicate:{checkpoint_id}:{column_id}")  # 中文说明。

    remaining_checkpoint_ids = {
        task.checkpoint_id
        for task in snapshot.tasks
        if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids)
    }
    initially_lost = (
        _covered_checkpoint_ids(snapshot, requested_selected)
        - _covered_checkpoint_ids(snapshot, selected)
    ) & remaining_checkpoint_ids
    deferred.extend(sorted(initially_lost - set(deferred)))
    if deadline is not None and deadline.expired():
        return runtime_deadline_outcome()
    restricted_columns = tuple(column_by_id[column_id] for column_id in selected)
    restricted_snapshot = dataclasses.replace(
        snapshot,
        service_columns=restricted_columns,
        locked_column_ids=tuple(selected),
        committed_column_ids=tuple(
            column_id
            for column_id in snapshot.committed_column_ids
            if column_id in selected
        ),
        airborne_residual_states=tuple(
            state
            for state in snapshot.airborne_residual_states
            if state.column_id in selected
        ),
    )
    bounded_solve = getattr(solver, 'solve_with_deadline', None)
    result = (bounded_solve(restricted_snapshot, deadline=deadline)
              if deadline is not None and callable(bounded_solve) else solver.solve(restricted_snapshot))
    if deadline is not None and deadline.expired():
        return runtime_deadline_outcome()
    validation = project_result(result, validator)
    fixed_set_preserved = set(result.selected_column_ids) == set(selected)
    if has_finite_incumbent(result) and validation.is_valid and fixed_set_preserved:
        return ProjectionOutcome(
            SearchSolution.from_result(result, True),
            tuple(corrections),
            validation,
            False,
            (),
            tuple(deferred),
        )
    issue_codes = tuple(validation.issue_codes)
    if has_finite_incumbent(result) and validation.is_valid and not fixed_set_preserved:
        issue_codes = (*issue_codes, "fixed_column_set_changed")
    if not issue_codes:
        issue_codes = ("restricted_projection_failed",)
    return ProjectionOutcome(
        None,
        tuple(corrections),
        validation,
        False,
        issue_codes,
        tuple(deferred),
    )
