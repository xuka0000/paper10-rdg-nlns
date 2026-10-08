"""为 RDG-NLNS 构造保留改进空间的确定性贪婪初始解。"""

from Env.snapshot import uav_schedulable

from collections import Counter
from collections.abc import Callable  # 导入验证器调用协议类型。
import dataclasses
from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy.optimize import milp

from Agent.base_agent import BaseAgent  # 导入统一求解器接口用于计划投影。
from Agent.evaluation_runtime import RuntimeDeadline
from Agent.hybrid.rdg_nlns.projection import project_candidate
from Agent.hybrid.rdg_nlns.solution import CandidatePlan, SearchSolution  # 导入搜索计划和解封装类型。
from Agent.optimization.linear_model import LinearModelBuilder
from Agent.results import AgentResult  # 导入环境验证器接收的结果类型。
from Env.snapshot import SceneSnapshot, VehicleRoadOption  # 导入滚动场景快照类型。


def solver_with_time_limit(
    solver: BaseAgent,
    time_limit_seconds: float,
) -> BaseAgent:
    """Copy budget-aware solvers without mutating the neighborhood solver."""
    factory = getattr(solver, "with_time_limit", None)
    return factory(time_limit_seconds) if callable(factory) else solver


def _select_greedy_column_ids(snapshot: SceneSnapshot) -> tuple[str, ...]:
    """Select a deterministic, load-balanced set of admitted service columns."""
    completed = set(snapshot.completed_checkpoint_ids)
    covered = set(completed)
    airborne = tuple(
        state.column_id for state in snapshot.airborne_residual_states
    )
    chosen = list(
        dict.fromkeys(
            (
                *snapshot.locked_column_ids,
                *snapshot.committed_column_ids,
                *airborne,
            )
        )
    )
    column_by_id = {
        column.object_id: column for column in snapshot.service_columns
    }
    unknown_fixed = tuple(
        column_id for column_id in chosen if column_id not in column_by_id
    )
    if unknown_fixed:
        raise RuntimeError(
            f"rolling state references unknown fixed service columns: {unknown_fixed}"
        )
    uav_load = Counter(
        column_by_id[column_id].uav_id
        for column_id in chosen
    )
    for column_id in chosen:
        covered.update(
            snapshot.column_state(column_id).effective_checkpoint_ids
        )
    remaining_tasks = sorted(
        (
            task
            for task in snapshot.tasks
            if task.checkpoint_id not in completed
        ),
        key=lambda task: (-float(task.priority), task.object_id),
    )
    admitted_columns = tuple(
        (column, snapshot.column_state(column.object_id))
        for column in snapshot.service_columns
        if column.admitted and column.object_id not in chosen
    )
    for task in remaining_tasks:
        if task.checkpoint_id in covered:
            continue
        candidates = tuple(
            (column, effective)
            for column, effective in admitted_columns
            if task.checkpoint_id in effective.effective_checkpoint_ids
            and not covered.intersection(effective.effective_checkpoint_ids)
            and uav_schedulable(snapshot.uav_state(effective.uav_id))
            and effective.effective_protected_energy
            <= effective.usable_battery_energy + 1e-9
        )
        if not candidates:
            raise RuntimeError(
                f"检查点缺少准入服务列: {task.checkpoint_id}"
            )
        selected, selected_state = min(
            candidates,
            key=lambda item: (
                uav_load[item[0].uav_id],
                float(item[1].effective_duration)
                + 0.001 * float(item[1].effective_protected_energy),
                len(item[1].effective_checkpoint_ids),
                item[0].object_id,
            ),
        )
        chosen.append(selected.object_id)
        uav_load[selected.uav_id] += 1
        covered.update(selected_state.effective_checkpoint_ids)
    return tuple(dict.fromkeys(chosen))


def _shortest_path_options(
    options: tuple[VehicleRoadOption, ...],
    source_node_id: str,
    target_node_id: str,
) -> tuple[VehicleRoadOption, ...] | None:
    """Return one deterministic accessible shortest path."""
    if source_node_id == target_node_id:
        return ()
    adjacency: dict[str, list[VehicleRoadOption]] = {}
    for option in options:
        if option.accessible:
            adjacency.setdefault(option.source_node_id, []).append(option)
    for outgoing in adjacency.values():
        outgoing.sort(key=lambda option: (option.target_node_id, option.road_leg_id))

    distances = {source_node_id: 0.0}
    predecessors: dict[str, tuple[str, VehicleRoadOption]] = {}
    queue = [(0.0, source_node_id)]
    while queue:
        distance, node_id = heapq.heappop(queue)
        if distance > distances.get(node_id, float("inf")):
            continue
        if node_id == target_node_id:
            break
        for option in adjacency.get(node_id, ()):
            candidate = distance + float(option.travel_time)
            previous = distances.get(option.target_node_id, float("inf"))
            if candidate >= previous:
                continue
            distances[option.target_node_id] = candidate
            predecessors[option.target_node_id] = (node_id, option)
            heapq.heappush(queue, (candidate, option.target_node_id))
    if target_node_id not in distances:
        return None

    reversed_path: list[VehicleRoadOption] = []
    node_id = target_node_id
    while node_id != source_node_id:
        predecessor = predecessors.get(node_id)
        if predecessor is None:
            return None
        node_id, option = predecessor
        reversed_path.append(option)
    return tuple(reversed(reversed_path))


def _route_order_selected_columns(
    snapshot: SceneSnapshot,
    selected_column_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Order a fixed greedy column set along each carrier's directed road corridor."""
    column_by_id = {column.object_id: column for column in snapshot.service_columns}
    carrier_by_uav = {
        uav.object_id: uav.carrier_vehicle_id for uav in snapshot.uavs
    }
    stop_node_by_id = {
        stop.object_id: stop.road_node_id for stop in snapshot.stop_sites
    }
    options_by_vehicle = {
        vehicle.object_id: tuple(
            option
            for option in snapshot.road_options
            if option.vehicle_id == vehicle.object_id and option.accessible
        )
        for vehicle in snapshot.vehicles
    }
    start_by_vehicle = {
        vehicle.object_id: snapshot.vehicle_state(vehicle.object_id).current_road_node_id
        for vehicle in snapshot.vehicles
    }

    distance_cache: dict[tuple[str, str], float] = {}

    def distance(vehicle_id: str, node_id: str) -> float:
        key = (vehicle_id, node_id)
        if key not in distance_cache:
            path = _shortest_path_options(
                options_by_vehicle[vehicle_id], start_by_vehicle[vehicle_id], node_id
            )
            distance_cache[key] = (
                float("inf")
                if path is None
                else sum(float(option.travel_time) for option in path)
            )
        return distance_cache[key]

    def route_key(column_id: str) -> tuple[object, ...]:
        column = column_by_id[column_id]
        vehicle_id = carrier_by_uav[column.uav_id]
        effective = snapshot.column_state(column_id)
        launch_node = stop_node_by_id[column.launch_stop_id]
        recovery_node = stop_node_by_id[column.recovery_stop_id]
        first_node = launch_node if effective.requires_launch_stop else recovery_node
        return (
            vehicle_id,
            distance(vehicle_id, first_node),
            distance(vehicle_id, recovery_node),
            column.uav_id,
            column_id,
        )

    return tuple(sorted(selected_column_ids, key=route_key))


@dataclass(frozen=True, slots=True)
class _VehicleRouteCandidate:
    vehicle_id: str
    options: tuple[VehicleRoadOption, ...]
    node_ids: tuple[str, ...]
    coverable_checkpoint_ids: frozenset[str]


def _fixed_initialization_column_ids(snapshot: SceneSnapshot) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *snapshot.locked_column_ids,
                *snapshot.committed_column_ids,
                *(state.column_id for state in snapshot.airborne_residual_states),
            )
        )
    )


def _column_route_nodes(
    snapshot: SceneSnapshot,
    column_id: str,
    stop_node_by_id: dict[str, str],
) -> tuple[str | None, str]:
    effective = snapshot.column_state(column_id)
    column = getattr(effective, 'column', None)
    if column is None:  # Compatibility with lightweight external state views.
        column = next(item for item in snapshot.service_columns if item.object_id == column_id)
    launch_node_id = (
        stop_node_by_id[column.launch_stop_id]
        if effective.requires_launch_stop
        else None
    )
    return launch_node_id, stop_node_by_id[column.recovery_stop_id]


def _column_fits_route(
    snapshot: SceneSnapshot,
    column_id: str,
    stop_node_by_id: dict[str, str],
    route_rank: dict[str, int],
) -> bool:
    launch_node_id, recovery_node_id = _column_route_nodes(
        snapshot, column_id, stop_node_by_id
    )
    if recovery_node_id not in route_rank:
        return False
    if launch_node_id is None:
        return True
    return (
        launch_node_id in route_rank
        and route_rank[launch_node_id] <= route_rank[recovery_node_id]
    )


def _path_nodes_from_options(
    snapshot: SceneSnapshot,
    vehicle_id: str,
    options: tuple[VehicleRoadOption, ...],
) -> tuple[str, ...] | None:
    vehicle_options = tuple(
        option for option in options if option.vehicle_id == vehicle_id
    )
    start_node_id = snapshot.vehicle_state(vehicle_id).current_road_node_id
    end_node_id = next(
        (node.object_id for node in snapshot.road_nodes if node.role == "depot_end"),
        None,
    )
    if end_node_id is None or not vehicle_options:
        return None
    outgoing: dict[str, VehicleRoadOption] = {}
    incoming: dict[str, VehicleRoadOption] = {}
    for option in vehicle_options:
        if (
            option.source_node_id in outgoing
            or option.target_node_id in incoming
        ):
            return None
        outgoing[option.source_node_id] = option
        incoming[option.target_node_id] = option
    if start_node_id in incoming or end_node_id in outgoing:
        return None
    node_ids = [start_node_id]
    seen = {start_node_id}
    current_node_id = start_node_id
    while current_node_id != end_node_id:
        option = outgoing.get(current_node_id)
        if option is None or option.target_node_id in seen:
            return None
        current_node_id = option.target_node_id
        node_ids.append(current_node_id)
        seen.add(current_node_id)
    if len(node_ids) - 1 != len(vehicle_options):
        return None
    return tuple(node_ids)


def _route_options_form_simple_paths(
    snapshot: SceneSnapshot,
    options: tuple[VehicleRoadOption, ...],
) -> bool:
    return all(
        _path_nodes_from_options(snapshot, vehicle.object_id, options) is not None
        for vehicle in snapshot.vehicles
    )


def _solve_covering_vehicle_route(
    snapshot: SceneSnapshot,
    vehicle_id: str,
    target_checkpoint_ids: set[str],
    *,
    time_limit_seconds: float,
    deadline: RuntimeDeadline | None,
    require_target_coverage: bool = False,
) -> _VehicleRouteCandidate | None:
    if deadline is not None and deadline.expired():
        return None
    road_options = tuple(
        option
        for option in snapshot.road_options
        if option.vehicle_id == vehicle_id and option.accessible
    )
    road_node_ids = tuple(node.object_id for node in snapshot.road_nodes)
    start_node_id = snapshot.vehicle_state(vehicle_id).current_road_node_id
    end_node_id = next(
        (node.object_id for node in snapshot.road_nodes if node.role == "depot_end"),
        None,
    )
    if end_node_id is None or not road_options:
        return None

    carrier_by_uav = {
        uav.object_id: uav.carrier_vehicle_id for uav in snapshot.uavs
    }
    stop_node_by_id = {
        stop.object_id: stop.road_node_id for stop in snapshot.stop_sites
    }
    fixed_ids = set(_fixed_initialization_column_ids(snapshot))
    eligible_columns = tuple(
        column
        for column in snapshot.service_columns
        if column.admitted
        and carrier_by_uav[column.uav_id] == vehicle_id
        and (
            uav_schedulable(snapshot.uav_state(column.uav_id))
            or snapshot.column_state(column.object_id).is_committed
        )
        and snapshot.column_state(column.object_id).effective_protected_energy
        <= snapshot.column_state(column.object_id).usable_battery_energy + 1e-9
    )
    eligible_ids = {column.object_id for column in eligible_columns}
    if any(
        column_id not in eligible_ids
        for column_id in fixed_ids
        if next(
            (
                carrier_by_uav[column.uav_id]
                for column in snapshot.service_columns
                if column.object_id == column_id
            ),
            None,
        )
        == vehicle_id
    ):
        return None

    builder = LinearModelBuilder()
    total_road_time = math.fsum(
        float(option.travel_time) for option in road_options
    ) + 1.0
    road_use = {
        option.road_leg_id: builder.add_binary(
            f"initial_route_use[{vehicle_id},{option.road_leg_id}]",
            0.1 * float(option.travel_time) / total_road_time,
        )
        for option in road_options
    }
    visit = {
        node_id: builder.add_binary(
            f"initial_route_visit[{vehicle_id},{node_id}]"
        )
        for node_id in road_node_ids
        if node_id not in {start_node_id, end_node_id}
    }
    order = {
        node_id: builder.add_continuous(
            f"initial_route_order[{vehicle_id},{node_id}]",
            0.0,
            float(len(road_node_ids)),
        )
        for node_id in road_node_ids
    }
    route_nodes_by_column = {
        column.object_id: _column_route_nodes(
            snapshot, column.object_id, stop_node_by_id
        )
        for column in eligible_columns
    }
    single_node_by_column = {
        column_id: recovery_node_id
        for column_id, (launch_node_id, recovery_node_id) in route_nodes_by_column.items()
        if launch_node_id is None or launch_node_id == recovery_node_id
    }
    cross_stop_columns = tuple(
        column
        for column in eligible_columns
        if column.object_id not in single_node_by_column
    )
    support = {
        column.object_id: builder.add_binary(
            f"initial_route_column[{vehicle_id},{column.object_id}]"
        )
        for column in cross_stop_columns
    }
    remaining_tasks = tuple(
        task
        for task in snapshot.tasks
        if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids)
    )
    coverage = {
        task.checkpoint_id: builder.add_binary(
            f"initial_route_coverage[{vehicle_id},{task.checkpoint_id}]",
            -(
                1000.0 + float(task.priority)
                if task.checkpoint_id in target_checkpoint_ids
                else 0.01
            ),
        )
        for task in remaining_tasks
    }

    builder.add_constraint(
        f"initial_route_start_out[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.source_node_id == start_node_id
        ),
        1.0,
        1.0,
    )
    builder.add_constraint(
        f"initial_route_start_in[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.target_node_id == start_node_id
        ),
        0.0,
        0.0,
    )
    builder.add_constraint(
        f"initial_route_end_in[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.target_node_id == end_node_id
        ),
        1.0,
        1.0,
    )
    builder.add_constraint(
        f"initial_route_end_out[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], 1.0)
            for option in road_options
            if option.source_node_id == end_node_id
        ),
        0.0,
        0.0,
    )
    builder.add_constraint(
        f"initial_route_start_order[{vehicle_id}]",
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
            f"initial_route_visit_in[{vehicle_id},{node_id}]",
            (*incoming, (visited, -1.0)),
            0.0,
            0.0,
        )
        builder.add_constraint(
            f"initial_route_visit_out[{vehicle_id},{node_id}]",
            (*outgoing, (visited, -1.0)),
            0.0,
            0.0,
        )
    order_big_m = float(len(road_node_ids) + 1)
    for option in road_options:
        if (
            option.target_node_id == start_node_id
            or option.source_node_id == end_node_id
        ):
            continue
        builder.add_constraint(
            f"initial_route_order_link[{vehicle_id},{option.road_leg_id}]",
            (
                (order[option.target_node_id], 1.0),
                (order[option.source_node_id], -1.0),
                (road_use[option.road_leg_id], -order_big_m),
            ),
            1.0 - order_big_m,
            math.inf,
        )
    builder.add_constraint(
        f"initial_route_horizon[{vehicle_id}]",
        (
            (road_use[option.road_leg_id], float(option.travel_time))
            for option in road_options
        ),
        -math.inf,
        float(snapshot.time_horizon),
    )

    for column in cross_stop_columns:
        column_id = column.object_id
        selected = support[column_id]
        launch_node_id, recovery_node_id = route_nodes_by_column[column_id]
        required_node_ids = {recovery_node_id}
        if launch_node_id is not None:
            required_node_ids.add(launch_node_id)
        for node_id in required_node_ids - {start_node_id, end_node_id}:
            builder.add_constraint(
                f"initial_route_column_visit[{vehicle_id},{column_id},{node_id}]",
                ((selected, 1.0), (visit[node_id], -1.0)),
                -math.inf,
                0.0,
            )
        if launch_node_id is not None and launch_node_id != recovery_node_id:
            builder.add_constraint(
                f"initial_route_column_order[{vehicle_id},{column_id}]",
                (
                    (order[recovery_node_id], 1.0),
                    (order[launch_node_id], -1.0),
                    (selected, -order_big_m),
                ),
                1.0 - order_big_m,
                math.inf,
            )
        if column_id in fixed_ids:
            builder.add_constraint(
                f"initial_route_fixed_column[{vehicle_id},{column_id}]",
                ((selected, 1.0),),
                1.0,
                1.0,
            )
    for column_id, node_id in single_node_by_column.items():
        if column_id not in fixed_ids or node_id in {start_node_id, end_node_id}:
            continue
        builder.add_constraint(
            f"initial_route_fixed_column_visit[{vehicle_id},{column_id}]",
            ((visit[node_id], 1.0),),
            1.0,
            1.0,
        )
    for task in remaining_tasks:
        checkpoint_id = task.checkpoint_id
        single_nodes = {
            single_node_by_column[column.object_id]
            for column in eligible_columns
            if column.object_id in single_node_by_column
            if checkpoint_id
            in snapshot.column_state(column.object_id).effective_checkpoint_ids
        }
        terminal_support = bool(single_nodes & {start_node_id, end_node_id})
        covering = tuple(
            (visit[node_id], 1.0)
            for node_id in sorted(single_nodes - {start_node_id, end_node_id})
        ) + tuple(
            (support[column.object_id], 1.0)
            for column in cross_stop_columns
            if checkpoint_id
            in snapshot.column_state(column.object_id).effective_checkpoint_ids
        )
        if terminal_support:
            builder.add_constraint(
                f"initial_route_terminal_coverage[{vehicle_id},{checkpoint_id}]",
                ((coverage[checkpoint_id], 1.0),),
                1.0,
                1.0,
            )
            continue
        builder.add_constraint(
            f"initial_route_coverage_link[{vehicle_id},{checkpoint_id}]",
            (*covering, (coverage[checkpoint_id], -1.0)),
            0.0,
            math.inf,
        )
    if require_target_coverage:
        for checkpoint_id in sorted(target_checkpoint_ids):
            builder.add_constraint(
                f"initial_route_required_coverage[{vehicle_id},{checkpoint_id}]",
                ((coverage[checkpoint_id], 1.0),),
                1.0,
                1.0,
            )

    model = builder.build()
    solve_limit = float(time_limit_seconds)
    if deadline is not None:
        solve_limit = min(solve_limit, deadline.remaining_seconds())
    if solve_limit <= 0.0:
        return None
    result = milp(
        c=model.objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=model.linear_constraint,
        options={
            "time_limit": solve_limit,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    if (
        result.x is None
        or not np.isfinite(result.x).all()
        or (deadline is not None and deadline.expired())
    ):
        return None
    value_map = {
        name: float(result.x[index])
        for index, name in enumerate(model.variable_names)
    }
    selected_options = tuple(
        option
        for option in road_options
        if value_map.get(road_use[option.road_leg_id].name, 0.0) > 0.5
    )
    node_ids = _path_nodes_from_options(snapshot, vehicle_id, selected_options)
    if node_ids is None:
        return None
    route_rank = {node_id: index for index, node_id in enumerate(node_ids)}
    coverable = frozenset(
        task.checkpoint_id
        for task in remaining_tasks
        if any(
            task.checkpoint_id
            in snapshot.column_state(column.object_id).effective_checkpoint_ids
            and _column_fits_route(
                snapshot, column.object_id, stop_node_by_id, route_rank
            )
            for column in eligible_columns
        )
    )
    if any(
        column_id in eligible_ids
        and not _column_fits_route(
            snapshot, column_id, stop_node_by_id, route_rank
        )
        for column_id in fixed_ids
    ):
        return None
    option_by_pair = {
        (option.source_node_id, option.target_node_id): option
        for option in selected_options
    }
    ordered_options = tuple(
        option_by_pair[(source_node_id, target_node_id)]
        for source_node_id, target_node_id in zip(node_ids[:-1], node_ids[1:])
    )
    return _VehicleRouteCandidate(
        vehicle_id,
        ordered_options,
        node_ids,
        coverable,
    )


def _select_columns_for_routes(
    snapshot: SceneSnapshot,
    routes: dict[str, _VehicleRouteCandidate],
    *, deadline: RuntimeDeadline | None = None,
) -> tuple[tuple[str, ...], tuple[VehicleRoadOption, ...]] | None:
    remaining_checkpoint_ids = {
        task.checkpoint_id
        for task in snapshot.tasks
        if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids)
    }
    if set(routes) != {vehicle.object_id for vehicle in snapshot.vehicles}:
        return None
    column_by_id = {
        column.object_id: column for column in snapshot.service_columns
    }
    carrier_by_uav = {
        uav.object_id: uav.carrier_vehicle_id for uav in snapshot.uavs
    }
    stop_node_by_id = {
        stop.object_id: stop.road_node_id for stop in snapshot.stop_sites
    }
    route_rank_by_vehicle = {
        vehicle_id: {
            node_id: index for index, node_id in enumerate(route.node_ids)
        }
        for vehicle_id, route in routes.items()
    }
    fixed_ids = _fixed_initialization_column_ids(snapshot)
    if any(column_id not in column_by_id for column_id in fixed_ids):
        return None
    chosen = list(fixed_ids)
    covered = set(snapshot.completed_checkpoint_ids)
    uav_load = Counter()
    node_uav_load = Counter()
    node_dwell = Counter()

    def same_stop_key(column_id: str) -> tuple[str, str, str] | None:
        column = column_by_id[column_id]
        vehicle_id = carrier_by_uav[column.uav_id]
        launch_node_id, recovery_node_id = _column_route_nodes(
            snapshot, column_id, stop_node_by_id
        )
        if launch_node_id is None or launch_node_id != recovery_node_id:
            return None
        return vehicle_id, recovery_node_id, column.uav_id

    def record_column_load(column_id: str) -> None:
        effective = snapshot.column_state(column_id)
        duration = float(effective.effective_duration)
        column = column_by_id[column_id]
        uav_load[column.uav_id] += duration
        key = same_stop_key(column_id)
        if key is None:
            return
        vehicle_id, node_id, uav_id = key
        node_uav_load[key] += duration
        node_dwell[(vehicle_id, node_id)] = max(
            node_dwell[(vehicle_id, node_id)], node_uav_load[key]
        )

    for column_id in chosen:
        column = column_by_id[column_id]
        vehicle_id = carrier_by_uav[column.uav_id]
        if not _column_fits_route(
            snapshot,
            column_id,
            stop_node_by_id,
            route_rank_by_vehicle[vehicle_id],
        ):
            return None
        effective = snapshot.column_state(column_id)
        if covered.intersection(effective.effective_checkpoint_ids):
            return None
        covered.update(effective.effective_checkpoint_ids)
        record_column_load(column_id)

    route_admitted_columns = []
    for column in snapshot.service_columns:
        if deadline is not None and deadline.expired():
            return None
        if column.admitted and column.object_id not in chosen and _column_fits_route(
                snapshot, column.object_id, stop_node_by_id,
                route_rank_by_vehicle[carrier_by_uav[column.uav_id]]):
            route_admitted_columns.append((column, snapshot.column_state(column.object_id)))
    remaining_tasks = sorted(
        (
            task
            for task in snapshot.tasks
            if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids)
        ),
        key=lambda task: (-float(task.priority), task.object_id),
    )
    for task in remaining_tasks:
        if deadline is not None and deadline.expired():
            return None
        if task.checkpoint_id in covered:
            continue
        candidates = tuple(
            (column, effective)
            for column, effective in route_admitted_columns
            if task.checkpoint_id in effective.effective_checkpoint_ids
            and not covered.intersection(effective.effective_checkpoint_ids)
            and uav_schedulable(snapshot.uav_state(effective.uav_id))
            and effective.effective_protected_energy
            <= effective.usable_battery_energy + 1e-9
        )
        if not candidates:
            return None
        same_stop_candidates = tuple(
            item for item in candidates if same_stop_key(item[0].object_id) is not None
        )
        if same_stop_candidates:
            candidates = same_stop_candidates

        def selection_key(item: tuple[object, object]) -> tuple[object, ...]:
            column, effective = item
            duration = float(effective.effective_duration)
            key = same_stop_key(column.object_id)
            if key is None:
                return (
                    1,
                    uav_load[column.uav_id] + duration,
                    duration + 0.001 * float(effective.effective_protected_energy),
                    column.object_id,
                )
            vehicle_id, node_id, _ = key
            current_dwell = node_dwell[(vehicle_id, node_id)]
            updated_uav_load = node_uav_load[key] + duration
            updated_dwell = max(current_dwell, updated_uav_load)
            return (
                0,
                updated_dwell - current_dwell,
                updated_dwell,
                duration + 0.001 * float(effective.effective_protected_energy),
                route_rank_by_vehicle[vehicle_id][node_id],
                column.object_id,
            )

        selected, selected_state = min(
            candidates,
            key=selection_key,
        )
        chosen.append(selected.object_id)
        record_column_load(selected.object_id)
        covered.update(selected_state.effective_checkpoint_ids)
    if not remaining_checkpoint_ids <= covered:
        return None

    vehicle_rank = {
        vehicle.object_id: index for index, vehicle in enumerate(snapshot.vehicles)
    }

    def column_order_key(column_id: str) -> tuple[object, ...]:
        column = column_by_id[column_id]
        vehicle_id = carrier_by_uav[column.uav_id]
        route_rank = route_rank_by_vehicle[vehicle_id]
        launch_node_id, recovery_node_id = _column_route_nodes(
            snapshot, column_id, stop_node_by_id
        )
        return (
            vehicle_rank[vehicle_id],
            route_rank[
                recovery_node_id if launch_node_id is None else launch_node_id
            ],
            route_rank[recovery_node_id],
            column.uav_id,
            column_id,
        )

    selected_ids = tuple(sorted(dict.fromkeys(chosen), key=column_order_key))
    route_options = tuple(
        option
        for vehicle in snapshot.vehicles
        for option in routes[vehicle.object_id].options
    )
    return selected_ids, route_options


def _certified_route_initialization_plan(
    snapshot: SceneSnapshot,
    *, deadline: RuntimeDeadline | None = None,
) -> tuple[tuple[str, ...], tuple[VehicleRoadOption, ...]] | None:
    """Use each vehicle's remaining suffix of the certified GIS corridor.

    Coverage is checked jointly across vehicles. A vehicle that has already
    advanced must not be sent through its travelled prefix again.
    """
    ordered_node_ids = tuple(
        node.object_id
        for node in sorted(
            snapshot.road_nodes,
            key=lambda node: (
                float(node.position.x),
                float(node.position.y),
                node.object_id,
            ),
        )
    )
    if len(ordered_node_ids) < 2:
        return None
    end_node_id = next(
        (node.object_id for node in snapshot.road_nodes if node.role == "depot_end"),
        None,
    )
    if end_node_id != ordered_node_ids[-1]:
        return None
    stop_node_by_id = {
        stop.object_id: stop.road_node_id for stop in snapshot.stop_sites
    }
    carrier_by_uav = {
        uav.object_id: uav.carrier_vehicle_id for uav in snapshot.uavs
    }
    fixed_ids = set(_fixed_initialization_column_ids(snapshot))
    routes: dict[str, _VehicleRouteCandidate] = {}
    for vehicle in snapshot.vehicles:
        vehicle_id = vehicle.object_id
        current_node_id = snapshot.vehicle_state(vehicle_id).current_road_node_id
        if current_node_id not in ordered_node_ids:
            return None
        vehicle_node_ids = ordered_node_ids[ordered_node_ids.index(current_node_id):]
        option_by_pair = {
            (option.source_node_id, option.target_node_id): option
            for option in snapshot.road_options
            if option.vehicle_id == vehicle_id and option.accessible
        }
        try:
            options = tuple(
                option_by_pair[(source_node_id, target_node_id)]
                for source_node_id, target_node_id in zip(
                    vehicle_node_ids[:-1], vehicle_node_ids[1:]
                )
            )
        except KeyError:
            return None
        if math.fsum(float(option.travel_time) for option in options) > float(
            snapshot.time_horizon
        ) + 1.0e-9:
            return None
        route_rank = {
            node_id: index for index, node_id in enumerate(vehicle_node_ids)
        }
        eligible_columns = tuple(
            column
            for column in snapshot.service_columns
            if column.admitted
            and carrier_by_uav[column.uav_id] == vehicle_id
            and (
                uav_schedulable(snapshot.uav_state(column.uav_id))
                or snapshot.column_state(column.object_id).is_committed
            )
            and snapshot.column_state(column.object_id).effective_protected_energy
            <= snapshot.column_state(column.object_id).usable_battery_energy + 1.0e-9
        )
        if any(
            column.object_id in fixed_ids
            and not _column_fits_route(
                snapshot,
                column.object_id,
                stop_node_by_id,
                route_rank,
            )
            for column in eligible_columns
        ):
            return None
        # The existential task/column test is exactly the union of effective
        # coverage of route-admitted columns. Evaluate each column once.
        covered_by_route = set()
        for column in eligible_columns:
            if deadline is not None and deadline.expired():
                return None
            if _column_fits_route(snapshot, column.object_id, stop_node_by_id, route_rank):
                covered_by_route.update(snapshot.column_state(column.object_id).effective_checkpoint_ids)
        coverable = frozenset(task.checkpoint_id for task in snapshot.tasks
            if task.checkpoint_id not in snapshot.completed_checkpoint_ids and task.checkpoint_id in covered_by_route)
        routes[vehicle_id] = _VehicleRouteCandidate(
            vehicle_id,
            options,
            vehicle_node_ids,
            coverable,
        )
    remaining_checkpoint_ids = {
        task.checkpoint_id
        for task in snapshot.tasks
        if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids)
    }
    if not remaining_checkpoint_ids <= set().union(
        *(route.coverable_checkpoint_ids for route in routes.values())
    ):
        return None
    return _select_columns_for_routes(snapshot, routes, deadline=deadline)


def _route_aware_initialization_plan(
    snapshot: SceneSnapshot,
    *,
    time_limit_seconds: float,
    deadline: RuntimeDeadline | None,
) -> tuple[tuple[str, ...], tuple[VehicleRoadOption, ...]] | None:
    remaining_checkpoint_ids = {
        task.checkpoint_id
        for task in snapshot.tasks
        if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids)
    }
    uncovered = set(remaining_checkpoint_ids)
    routes: dict[str, _VehicleRouteCandidate] = {}
    vehicles = tuple(snapshot.vehicles)
    for vehicle_index, vehicle in enumerate(vehicles):
        route = _solve_covering_vehicle_route(
            snapshot,
            vehicle.object_id,
            uncovered,
            time_limit_seconds=time_limit_seconds,
            deadline=deadline,
            require_target_coverage=vehicle_index == len(vehicles) - 1,
        )
        if route is None:
            return None
        routes[vehicle.object_id] = route
        uncovered.difference_update(route.coverable_checkpoint_ids)
    if uncovered:
        return None
    return _select_columns_for_routes(snapshot, routes, deadline=deadline)


def build_route_feasible_neighborhood_plan(
    snapshot: SceneSnapshot,
    *,
    required_column_ids: tuple[str, ...],
    excluded_column_ids: tuple[str, ...] = (),
    time_limit_seconds: float = 10.0,
    deadline: RuntimeDeadline | None = None,
) -> tuple[tuple[str, ...], tuple[VehicleRoadOption, ...], str] | None:
    """Complete a destroyed neighborhood while preserving retained columns."""
    if deadline is not None and deadline.expired():
        return None
    if not math.isfinite(float(time_limit_seconds)) or time_limit_seconds <= 0.0:
        raise ValueError("Neighborhood route-repair time limit must be positive")
    column_ids = {column.object_id for column in snapshot.service_columns}
    required = tuple(dict.fromkeys(required_column_ids))
    excluded = set(excluded_column_ids)
    unknown_required = set(required) - column_ids
    if unknown_required:
        raise ValueError(
            "Neighborhood route repair references unknown required columns: "
            + ", ".join(sorted(unknown_required))
        )
    if set(required) & excluded:
        raise ValueError("Required neighborhood columns cannot also be excluded")
    available_columns = tuple(
        column
        for column in snapshot.service_columns
        if column.object_id not in excluded
    )
    if not available_columns:
        return None
    fixed_ids = tuple(dict.fromkeys((*snapshot.locked_column_ids, *required)))
    neighborhood = dataclasses.replace(
        snapshot,
        service_columns=available_columns,
        locked_column_ids=fixed_ids,
    )
    plan = _certified_route_initialization_plan(neighborhood, deadline=deadline)
    planner = "certified_corridor"
    if deadline is not None and deadline.expired():
        return None
    if plan is None:
        plan = _route_aware_initialization_plan(
            neighborhood,
            time_limit_seconds=float(time_limit_seconds),
            deadline=deadline,
        )
        planner = "route_aware_milp"
    if plan is None:
        return None
    selected_column_ids, road_options = plan
    if not set(required) <= set(selected_column_ids):
        raise RuntimeError("Neighborhood route repair dropped a retained column")
    if set(selected_column_ids) & excluded:
        raise RuntimeError("Neighborhood route repair selected an excluded column")
    return selected_column_ids, road_options, planner


def _initialization_route_options(
    snapshot: SceneSnapshot,
    selected_column_ids: tuple[str, ...],
) -> tuple[VehicleRoadOption, ...] | None:
    """Build a deterministic original-arc corridor for a fixed initialization set."""
    column_by_id = {
        column.object_id: column for column in snapshot.service_columns
    }
    carrier_by_uav = {
        uav.object_id: uav.carrier_vehicle_id for uav in snapshot.uavs
    }
    stop_node_by_id = {
        stop.object_id: stop.road_node_id for stop in snapshot.stop_sites
    }
    end_node_id = next(
        (node.object_id for node in snapshot.road_nodes if node.role == "depot_end"),
        None,
    )
    if end_node_id is None:
        return None
    airborne = tuple(
        state.column_id for state in snapshot.airborne_residual_states
    )
    fixed_column_ids = tuple(
        dict.fromkeys(
            (
                *selected_column_ids,
                *snapshot.locked_column_ids,
                *snapshot.committed_column_ids,
                *airborne,
            )
        )
    )

    retained: list[VehicleRoadOption] = []
    retained_keys: set[tuple[str, str]] = set()
    for vehicle in snapshot.vehicles:
        vehicle_id = vehicle.object_id
        required_node_ids: list[str] = []
        for column_id in fixed_column_ids:
            column = column_by_id.get(column_id)
            if column is None or carrier_by_uav.get(column.uav_id) != vehicle_id:
                continue
            effective = snapshot.column_state(column_id)
            if effective.requires_launch_stop:
                required_node_ids.append(stop_node_by_id[column.launch_stop_id])
            required_node_ids.append(stop_node_by_id[column.recovery_stop_id])
        ordered_targets = tuple(
            dict.fromkeys((*required_node_ids, end_node_id))
        )
        current_node_id = snapshot.vehicle_state(
            vehicle_id
        ).current_road_node_id
        vehicle_options = tuple(
            option
            for option in snapshot.road_options
            if option.vehicle_id == vehicle_id and option.accessible
        )
        for target_node_id in ordered_targets:
            path = _shortest_path_options(
                vehicle_options,
                current_node_id,
                target_node_id,
            )
            if path is None:
                return None
            for option in path:
                key = (option.vehicle_id, option.road_leg_id)
                if key not in retained_keys:
                    retained.append(option)
                    retained_keys.add(key)
            current_node_id = target_node_id
    return tuple(retained)


def build_greedy_initial_solution(  # 定义服务列贪婪选择与完整 M5 投影入口。
    snapshot: SceneSnapshot,
    projection_solver: BaseAgent,
    validator: Callable[[AgentResult], object],
    *,
    deadline: RuntimeDeadline | None = None,
    route_projection_time_limit_seconds: float = 30.0,
) -> SearchSolution:
    """Select one complete greedy column set and project that fixed set once."""
    selected_ids = _route_order_selected_columns(
        snapshot, _select_greedy_column_ids(snapshot)
    )
    if deadline is not None and deadline.expired():
        raise RuntimeError(
            "RDG-NLNS initialization failed: ('runtime_deadline',)"
        )
    attempt_limit = float(route_projection_time_limit_seconds)
    if deadline is not None:
        attempt_limit = min(attempt_limit, deadline.remaining_seconds())
    if attempt_limit <= 0.0:
        raise RuntimeError(
            "RDG-NLNS initialization failed: ('runtime_deadline',)"
        )
    route_options = _initialization_route_options(snapshot, selected_ids)
    if route_options is None or not _route_options_form_simple_paths(
        snapshot, route_options
    ):
        route_plan = _certified_route_initialization_plan(snapshot, deadline=deadline)
        if route_plan is None:
            route_plan = _route_aware_initialization_plan(
                snapshot,
                time_limit_seconds=attempt_limit,
                deadline=deadline,
            )
        if route_plan is not None:
            selected_ids, route_options = route_plan
    if route_options is None or not _route_options_form_simple_paths(
        snapshot, route_options
    ):
        raise RuntimeError(
            "RDG-NLNS initialization failed: ('initialization_route_unreachable',)"
        )
    projected_snapshot = dataclasses.replace(snapshot, road_options=route_options)
    attempt = project_candidate(
        projected_snapshot,
        CandidatePlan(selected_ids, ()),
        solver_with_time_limit(projection_solver, attempt_limit),
        validator,
        deadline=deadline,
        route_projection_time_limit_seconds=attempt_limit,
    )
    candidate = attempt.candidate
    if candidate is None or set(candidate.selected_column_ids) != set(selected_ids):
        issue_codes = (
            attempt.issue_codes
            or attempt.validation.issue_codes
            or ("initialization_no_feasible_fixed_set",)
        )
        raise RuntimeError(
            f"RDG-NLNS initialization failed for the complete fixed column set: {issue_codes}"
        )
    if not candidate.selected_column_ids:
        raise RuntimeError(
            "RDG-NLNS initialization requires a nonempty projected service column set"
        )
    return candidate
