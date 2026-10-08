"""定义不依赖具体 Agent 类型的环境结果复核对象和基础检查。"""  # 说明本文件的环境职责。

import math  # 导入数学函数以执行独立时序和数值容差检查。
from dataclasses import dataclass  # 导入数据类工具以保存不可变复核结果。
from typing import Protocol  # 导入协议类型以避免 Env 反向依赖 Agent。

from Env.snapshot import SceneSnapshot  # 导入不可变场景快照以读取准入对象。


class ResultLike(Protocol):  # 定义 Env 可以复核的最小 Agent 结果协议。
    """要求结果暴露选择对象、命名变量和检查点完成时间。"""  # 解释结果协议边界。

    success: bool  # 声明结果是否包含可用决策。
    variable_values: tuple[tuple[str, float], ...]  # 声明结果全部稳定命名变量值。
    selected_road_leg_ids: tuple[str, ...]  # 声明结果选择的车辆道路弧标识。
    selected_stop_ids: tuple[str, ...]  # 声明结果选择的车辆停靠点标识。
    selected_column_ids: tuple[str, ...]  # 声明结果选择的服务列标识。
    checkpoint_completion_times: tuple[tuple[str, float], ...]  # 声明结果报告的检查点完成时间。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义环境结果复核报告。
class ResultValidation:  # 定义与具体求解算法无关的结果复核输出。
    """记录结果有效状态和全部违反环境约束的问题代码。"""  # 解释结果复核内容。

    is_valid: bool  # 保存结果是否通过当前环境复核。
    issue_codes: tuple[str, ...]  # 保存全部复核问题代码。


def _add_issue(issues: list[str], issue_code: str) -> None:  # 定义保持首次出现顺序的问题代码登记函数。
    """仅在问题代码尚未出现时将其加入复核结果。"""  # 解释问题代码去重规则。
    if issue_code not in issues:  # 判断当前问题代码是否尚未登记。
        issues.append(issue_code)  # 将当前问题代码加入复核列表。


def _value_map(result: ResultLike) -> dict[str, float]:  # 定义命名变量结果映射函数。
    """将稳定变量名称和值转换为便于独立复核的字典。"""  # 解释命名变量映射用途。
    return {name: float(value) for name, value in result.variable_values}  # 返回保持最后一次值的变量映射。


def _binary_selected(value_map: dict[str, float], variable_name: str) -> bool:  # 定义二元决策读取函数。
    """按照二分之一阈值读取 SciPy 返回的二元决策状态。"""  # 解释二元数值容差规则。
    return value_map.get(variable_name, 0.0) > 0.5  # 返回当前命名二元变量是否被选择。


def validate_selected_columns(snapshot: SceneSnapshot, result: ResultLike) -> ResultValidation:  # 定义完整 M5 结果独立复核函数。
    """独立检查准入、道路流、覆盖、时序、能量、不重叠和滚动锁定。"""  # 解释完整复核范围。
    issues: list[str] = []  # 创建结果复核问题代码列表。
    tolerance = 1.0e-6  # 定义连续时间和能量比较使用的数值容差。
    if not result.success:  # 判断 Agent 结果是否包含可用决策。
        _add_issue(issues, "agent_result_unsuccessful")  # 记录 Agent 结果不可用问题。
    value_map = _value_map(result)  # 将全部命名变量结果转换为稳定映射。
    admitted_column_by_id = {column.object_id: column for column in snapshot.service_columns}  # 按稳定标识索引环境准入服务列。
    admitted_ids = set(admitted_column_by_id)  # 提取环境准入服务列标识集合。
    if any(column_id not in admitted_ids for column_id in result.selected_column_ids):  # 判断结果是否选择环境未准入服务列。
        _add_issue(issues, "selected_unadmitted_column")  # 记录选择未准入服务列问题。
    if len(result.selected_column_ids) != len(set(result.selected_column_ids)):  # 判断结果是否重复选择同一服务列。
        _add_issue(issues, "duplicate_selected_column")  # 记录重复选择服务列问题。
    uav_by_id = {uav.object_id: uav for uav in snapshot.uavs}  # 按稳定标识索引车载无人机。
    stop_by_id = {stop.object_id: stop for stop in snapshot.stop_sites}  # 按稳定标识索引候选停靠点。
    road_option_by_key = {(option.vehicle_id, option.road_leg_id): option for option in snapshot.road_options if option.accessible}  # 按车辆和道路弧索引可通行道路系数。
    stop_option_by_key = {(option.vehicle_id, option.stop_site_id): option for option in snapshot.stop_options if option.accessible}  # 按车辆和停靠点索引准入停靠组合。
    variable_road_pairs = {key for key in road_option_by_key if _binary_selected(value_map, f"road_use[{key[0]},{key[1]}]")}  # 从命名变量重构选择车辆道路弧组合集合。
    variable_road_ids = {road_id for _, road_id in variable_road_pairs}  # 将车辆道路组合投影为既有结果摘要使用的道路弧集合。
    variable_stop_pairs = {key for key in stop_option_by_key if _binary_selected(value_map, f"stop_visit[{key[0]},{key[1]}]")}  # 从命名变量重构选择车辆停靠组合集合。
    variable_stop_ids = {stop_id for _, stop_id in variable_stop_pairs}  # 将车辆停靠组合投影为既有结果摘要使用的停靠点集合。
    variable_column_ids = {column.object_id for column in snapshot.service_columns if column.uav_id in uav_by_id and _binary_selected(value_map, f"column_select[{uav_by_id[column.uav_id].carrier_vehicle_id},{column.object_id}]")}  # 从命名变量重构选择服务列集合。
    if variable_road_ids != set(result.selected_road_leg_ids):  # 判断结果摘要道路弧是否与命名变量一致。
        _add_issue(issues, "vehicle_route_flow")  # 记录车辆道路摘要或流状态不一致问题。
    if variable_stop_ids != set(result.selected_stop_ids):  # 判断结果摘要停靠点是否与命名变量一致。
        _add_issue(issues, "stop_visit_link")  # 记录车辆停靠摘要或联结状态不一致问题。
    if variable_column_ids != set(result.selected_column_ids):  # 判断结果摘要服务列是否与命名变量一致。
        _add_issue(issues, "result_variable_mismatch")  # 记录服务列摘要与命名变量不一致问题。
    road_node_by_id = {node.object_id: node for node in snapshot.road_nodes}  # 按稳定标识索引道路节点。
    start_nodes = tuple(node for node in snapshot.road_nodes if node.role == "depot_start")  # 提取起点仓库道路节点。
    end_nodes = tuple(node for node in snapshot.road_nodes if node.role == "depot_end")  # 提取终点仓库道路节点。
    if len(start_nodes) != 1 or len(end_nodes) != 1:  # 判断快照是否具有唯一的起终点仓库副本。
        _add_issue(issues, "vehicle_route_flow")  # 记录无法独立检查的仓库节点结构问题。
    else:  # 处理具有唯一仓库副本的常规场景。
        default_start_node_id = start_nodes[0].object_id  # 读取无滚动状态时使用的唯一起点仓库节点标识。
        end_node_id = end_nodes[0].object_id  # 读取唯一终点仓库节点标识。
        for vehicle in snapshot.vehicles:  # 逐一独立复核每辆地面车辆路径。
            start_node_id = snapshot.vehicle_state(vehicle.object_id).current_road_node_id if hasattr(snapshot, "vehicle_state") else default_start_node_id  # 使用车辆实时道路节点作为当前滚动路径起点。
            if start_node_id not in road_node_by_id:  # 判断滚动状态是否引用未知道路节点。
                _add_issue(issues, "vehicle_route_flow")  # 记录无法建立当前路径边界的问题。
                continue  # 跳过当前车辆后续路径检查。
            selected_options = tuple(option for key, option in road_option_by_key.items() if key in variable_road_pairs and key[0] == vehicle.object_id)  # 按命名二元变量提取当前车辆实际选择的道路选项。
            outgoing_start = sum(option.source_node_id == start_node_id for option in selected_options)  # 计算当前车辆起点仓库出流数量。
            incoming_start = sum(option.target_node_id == start_node_id for option in selected_options)  # 计算当前车辆起点仓库入流数量。
            incoming_end = sum(option.target_node_id == end_node_id for option in selected_options)  # 计算当前车辆终点仓库入流数量。
            outgoing_end = sum(option.source_node_id == end_node_id for option in selected_options)  # 计算当前车辆终点仓库出流数量。
            if (outgoing_start, incoming_start, incoming_end, outgoing_end) != (1, 0, 1, 0):  # 判断当前车辆是否满足仓库流边界。
                _add_issue(issues, "vehicle_route_flow")  # 记录当前车辆仓库流违反。
            for node in snapshot.road_nodes:  # 逐一检查当前车辆中间道路节点流守恒。
                if node.object_id in {start_node_id, end_node_id}:  # 判断当前节点是否属于仓库副本。
                    continue  # 跳过已经由专门边界检查覆盖的仓库节点。
                incoming_count = sum(option.target_node_id == node.object_id for option in selected_options)  # 计算当前中间节点车辆入流数量。
                outgoing_count = sum(option.source_node_id == node.object_id for option in selected_options)  # 计算当前中间节点车辆出流数量。
                if incoming_count != outgoing_count:  # 判断当前中间节点是否违反车辆流守恒。
                    _add_issue(issues, "vehicle_route_flow")  # 记录当前中间节点车辆流违反。
            for node in snapshot.road_nodes:  # 逐一检查车辆到达和离开时间顺序。
                arrival = value_map.get(f"vehicle_arrival[{vehicle.object_id},{node.object_id}]", math.nan)  # 读取当前车辆节点到达时间。
                departure = value_map.get(f"vehicle_departure[{vehicle.object_id},{node.object_id}]", math.nan)  # 读取当前车辆节点离开时间。
                if not math.isfinite(arrival) or not math.isfinite(departure) or departure + tolerance < arrival:  # 判断当前节点时序是否缺失或违反非负停留要求。
                    _add_issue(issues, "road_time")  # 记录车辆节点时序问题。
            for option in selected_options:  # 逐一检查当前车辆选择道路弧的时间传递。
                source_departure = value_map.get(f"vehicle_departure[{vehicle.object_id},{option.source_node_id}]", math.nan)  # 读取当前道路弧源节点离开时间。
                target_arrival = value_map.get(f"vehicle_arrival[{vehicle.object_id},{option.target_node_id}]", math.nan)  # 读取当前道路弧目标节点到达时间。
                if not math.isfinite(source_departure) or not math.isfinite(target_arrival) or target_arrival + tolerance < source_departure + option.travel_time:  # 判断当前道路弧时间是否覆盖物理行驶时间。
                    _add_issue(issues, "road_time")  # 记录车辆道路时间传递问题。
            vehicle_stop_ids = {stop_id for vehicle_id, stop_id in stop_option_by_key if vehicle_id == vehicle.object_id}  # 提取当前车辆通过能力认证的停靠点标识。
            for stop in snapshot.stop_sites:  # 逐一检查当前车辆准入停靠摘要与道路流联结。
                if stop.object_id not in vehicle_stop_ids:  # 判断当前物理停靠点是否未对当前车辆开放。
                    continue  # 允许车辆通过关联道路节点但不将其解释为停靠访问。
                incoming_count = sum(option.target_node_id == stop.road_node_id for option in selected_options)  # 计算停靠点道路节点车辆入流数量。
                outgoing_count = sum(option.source_node_id == stop.road_node_id for option in selected_options)  # 计算停靠点道路节点车辆出流数量。
                if stop.road_node_id == start_node_id:  # 判断当前停靠点是否位于滚动路径起点。
                    expected_visit = outgoing_count == 1  # 起点停靠由唯一出流表明已访问。
                elif stop.road_node_id == end_node_id:  # 判断当前停靠点是否位于终点仓库。
                    expected_visit = incoming_count == 1  # 终点停靠由唯一入流表明已访问。
                else:  # 处理中间道路节点上的常规停靠点。
                    expected_visit = incoming_count == 1 and outgoing_count == 1  # 中间停靠仍要求一进一出。
                declared_visit = (vehicle.object_id, stop.object_id) in variable_stop_pairs  # 读取命名变量中的车辆级停靠访问状态。
                if expected_visit != declared_visit:  # 判断停靠点摘要是否与道路流一致。
                    _add_issue(issues, "stop_visit_link")  # 记录停靠点访问联结问题。
    selected_columns = tuple(admitted_column_by_id[column_id] for column_id in result.selected_column_ids if column_id in admitted_column_by_id)  # 提取结果选择的环境准入服务列对象。
    selected_checkpoint_count: dict[str, int] = {}  # 创建检查点被选择服务列覆盖次数映射。
    for column in selected_columns:  # 逐一处理结果选择的服务列。
        effective = snapshot.column_state(column.object_id)  # 读取当前执行阶段的有效服务列属性。
        for checkpoint_id in effective.effective_checkpoint_ids:  # 逐一处理当前服务列覆盖的检查点。
            selected_checkpoint_count[checkpoint_id] = selected_checkpoint_count.get(checkpoint_id, 0) + 1  # 累加当前检查点服务列覆盖次数。
    remaining_tasks = tuple(task for task in snapshot.tasks if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids))  # 提取当前滚动时刻尚未完成的检查点任务。
    for task in remaining_tasks:  # 逐一独立检查剩余检查点覆盖或漏检状态。
        missed = _binary_selected(value_map, f"checkpoint_missed[{task.checkpoint_id}]")  # 读取当前检查点漏检二元变量。
        coverage_count = selected_checkpoint_count.get(task.checkpoint_id, 0)  # 读取当前检查点被选择服务列覆盖次数。
        if coverage_count + int(missed) != 1:  # 判断当前检查点是否恰好覆盖一次或标记漏检。
            _add_issue(issues, "checkpoint_coverage")  # 记录检查点覆盖或漏检问题。
    reported_completion = {checkpoint_id: float(value) for checkpoint_id, value in result.checkpoint_completion_times}  # 将结果摘要检查点完成时间转换为映射。
    for column in selected_columns:  # 逐一独立检查选择服务列的同步、能量和完成时间。
        if column.uav_id not in uav_by_id or column.launch_stop_id not in stop_by_id or column.recovery_stop_id not in stop_by_id:  # 判断当前服务列基础引用是否缺失。
            _add_issue(issues, "column_time_sync")  # 记录不能独立复核的服务列引用问题。
            continue  # 跳过缺失基础引用的当前服务列后续检查。
        uav = uav_by_id[column.uav_id]  # 读取当前服务列执行无人机。
        effective = snapshot.column_state(column.object_id)  # 读取残余或完整列的统一属性。
        uav_state = snapshot.uav_state(column.uav_id)  # 读取无人机滚动资源状态。
        vehicle_id = uav.carrier_vehicle_id  # 读取当前服务列所属地面车辆标识。
        if (vehicle_id, column.launch_stop_id) not in stop_option_by_key or (vehicle_id, column.recovery_stop_id) not in stop_option_by_key:  # 判断结果是否包含引用未准入停靠组合的服务列。
            _add_issue(issues, "selected_unadmitted_stop_pair")  # 记录选择未准入车辆停靠组合问题。
            continue  # 跳过不能建立合法同步关系的当前服务列。
        launch = value_map.get(f"column_launch[{vehicle_id},{column.object_id}]", math.nan)  # 读取当前服务列起飞时间。
        recovery = value_map.get(f"column_recovery[{vehicle_id},{column.object_id}]", math.nan)  # 读取当前服务列回收时间。
        launch_stop = stop_by_id[column.launch_stop_id]  # 读取当前服务列起飞停靠点。
        recovery_stop = stop_by_id[column.recovery_stop_id]  # 读取当前服务列回收停靠点。
        vehicle_launch_arrival = value_map.get(f"vehicle_arrival[{vehicle_id},{launch_stop.road_node_id}]", math.nan)  # 读取车辆在起飞道路节点的到达时间。
        vehicle_launch_departure = value_map.get(f"vehicle_departure[{vehicle_id},{launch_stop.road_node_id}]", math.nan)  # 读取车辆在起飞道路节点的离开时间。
        vehicle_recovery_arrival = value_map.get(f"vehicle_arrival[{vehicle_id},{recovery_stop.road_node_id}]", math.nan)  # 读取车辆在回收道路节点的到达时间。
        vehicle_recovery_departure = value_map.get(f"vehicle_departure[{vehicle_id},{recovery_stop.road_node_id}]", math.nan)  # 读取车辆在回收道路节点的离开时间。
        timing_values = (launch, recovery, vehicle_launch_arrival, vehicle_launch_departure, vehicle_recovery_arrival, vehicle_recovery_departure)  # 汇总当前服务列同步检查所需时间。
        if not all(math.isfinite(value) for value in timing_values):  # 判断当前服务列同步时间是否全部存在且有限。
            _add_issue(issues, "column_time_sync")  # 记录服务列同步时间缺失问题。
        elif (effective.requires_launch_stop and (launch + tolerance < vehicle_launch_arrival or vehicle_launch_departure + tolerance < launch)) or (effective.is_airborne and abs(launch) > tolerance) or recovery + tolerance < launch + effective.effective_duration or recovery + tolerance < vehicle_recovery_arrival or vehicle_recovery_departure + tolerance < recovery:  # 判断当前服务列是否违反起飞点离站、动力学、回收或回收点离站同步关系。
            _add_issue(issues, "column_time_sync")  # 记录服务列时空同步违反。
        waiting = value_map.get(f"recovery_wait[{vehicle_id},{column.object_id}]", math.nan)  # 读取恢复等待时间。
        if not math.isfinite(waiting) or effective.effective_protected_energy + uav.hover_power * waiting > effective.usable_battery_energy + tolerance:  # 判断当前服务列概率安全分位能耗是否超过可用电池。
            _add_issue(issues, "energy_limit")  # 记录服务列概率能量安全违反。
        if not uav_state.available and getattr(uav_state,'available_after_seconds',0.)<=0 and not effective.is_committed:
            _add_issue(issues, "uav_unavailable")  # 记录无人机可用性违反。
        if not effective.is_airborne and launch+tolerance<getattr(uav_state,'available_after_seconds',0.):
            _add_issue(issues,'uav_ready_time')
        completion_offset_by_checkpoint = dict(effective.effective_completion_offsets)  # 将当前服务列检查点完成偏移转换为映射。
        for checkpoint_id, completion_offset in completion_offset_by_checkpoint.items():  # 逐一检查当前服务列覆盖检查点的完成时间。
            completion = value_map.get(f"checkpoint_completion[{checkpoint_id}]", math.nan)  # 读取当前检查点命名完成时间变量。
            expected_completion = launch + completion_offset  # 根据当前服务列起飞时间和偏移重构预期完成时间。
            if not math.isfinite(completion) or not math.isfinite(expected_completion) or abs(completion - expected_completion) > tolerance:  # 判断当前检查点完成时间是否与服务列偏移一致。
                _add_issue(issues, "checkpoint_completion")  # 记录检查点完成时间联结问题。
            if checkpoint_id not in reported_completion or abs(reported_completion[checkpoint_id] - completion) > tolerance:  # 判断结果摘要完成时间是否与命名变量一致。
                _add_issue(issues, "checkpoint_completion")  # 记录检查点完成时间摘要不一致问题。
    for uav in snapshot.uavs:  # 逐一独立检查同一无人机选择服务列时间不重叠。
        intervals: list[tuple[float, float]] = []  # 创建当前无人机选择服务列时间区间列表。
        for column in selected_columns:  # 逐一处理结果选择的服务列。
            if column.uav_id != uav.object_id:  # 判断当前服务列是否由其他无人机执行。
                continue  # 跳过不属于当前无人机的服务列。
            vehicle_id = uav.carrier_vehicle_id  # 读取当前无人机所属车辆标识。
            launch = value_map.get(f"column_launch[{vehicle_id},{column.object_id}]", math.nan)  # 读取当前服务列起飞时间。
            recovery = value_map.get(f"column_recovery[{vehicle_id},{column.object_id}]", math.nan)  # 读取当前服务列回收时间。
            intervals.append((launch, recovery))  # 将当前服务列时间区间加入列表。
        intervals.sort(key=lambda interval: interval[0])  # 按起飞时间排列当前无人机服务列区间。
        for first_interval, second_interval in zip(intervals[:-1], intervals[1:]):  # 逐一检查相邻服务列时间区间。
            turnaround = snapshot.uav_state(uav.object_id).turnaround_time  # 读取相邻航次周转时间。
            if not all(math.isfinite(value) for value in (*first_interval, *second_interval)) or second_interval[0] + tolerance < first_interval[1] + turnaround:  # 判断相邻服务列是否存在数值缺失或时间重叠。
                _add_issue(issues, "uav_nonoverlap")  # 记录同一无人机服务列时间重叠问题。
    locked_ids = (*snapshot.locked_column_ids, *getattr(snapshot, "committed_column_ids", ()), *(state.column_id for state in getattr(snapshot, "airborne_residual_states", ())))  # 合并全部不可修改服务列。
    if any(column_id not in set(result.selected_column_ids) for column_id in locked_ids):  # 判断滚动时刻锁定服务列是否全部保持选择。
        _add_issue(issues, "rolling_lock")  # 记录滚动服务列锁定违反。
    return ResultValidation(not issues, tuple(issues))  # 返回不可变完整结果复核报告。
