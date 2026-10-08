"""从数学模型快照构建 RDG-NLNS 异构静态动态图。"""

from Env.snapshot import uav_schedulable

import math  # 导入数学函数以验证数值字段。
from dataclasses import dataclass, field  # 导入数据类以保存不可变图张量。
from enum import IntEnum  # 导入整数枚举以稳定节点和边类型编号。
from typing import Any  # 导入通用类型以兼容正式快照协议。

import torch  # 导入张量库以生成 GNN 输入。


class NodeType(IntEnum):  # 定义四类论文服务图节点。
    """保存车辆、停靠点、服务列和任务节点类型。"""

    VEHICLE = 0  # 标记地面车辆节点。
    STOP = 1  # 标记候选停靠点节点。
    SERVICE_COLUMN = 2  # 标记无人机服务列节点。
    TASK = 3  # 标记构件巡检任务节点。


class EdgeType(IntEnum):  # 定义五类服务图结构关系。
    """保存车辆、停靠点、服务列和任务之间的关系类型。"""

    VEHICLE_STOP = 0  # 表示车辆可以访问停靠点。
    VEHICLE_COLUMN = 1  # 表示车辆搭载执行服务列的无人机。
    COLUMN_LAUNCH_STOP = 2  # 表示服务列起飞停靠点。
    COLUMN_RECOVERY_STOP = 3  # 表示服务列回收停靠点。
    COLUMN_TASK = 4  # 表示服务列覆盖巡检任务。


class GraphFeature(IntEnum):  # 中文说明。
    """中文说明。"""
    SELECTED = 0  # 中文说明。
    LOCKED = 1  # 中文说明。
    COMPLETED = 2  # 中文说明。
    PREVIOUSLY_SELECTED = 3  # 中文说明。
    ROLLING_EPOCH = 4  # 中文说明。
    ACTIVE = 5  # 中文说明。
    STOP_SELECTED = 6  # 中文说明。
    ROUTE_TIME = 7  # 中文说明。
    STOP_RATIO = 8  # 中文说明。
    LAUNCH_TIME = 9  # 中文说明。
    RECOVERY_TIME = 10  # 中文说明。
    COMPLETION_OR_MISSED = 11  # 中文说明。
    COLUMN_IS_AIRBORNE = 12  # 中文说明。
    RESIDUAL_RATIO = 13  # 中文说明。
    BATTERY_MARGIN = 14  # 中文说明。
    UAV_AVAILABLE = 15  # 中文说明。
    WAITING_ENERGY = 16  # 中文说明。
    TURNAROUND_CONFLICT = 17  # 中文说明。
    VEHICLE_AT_CURRENT_LOCATION = 18  # 中文说明。
    LOCK_SUBTYPE = 19  # 中文说明。


STATIC_FEATURE_WIDTH = 8  # 中文说明。
DYNAMIC_FEATURE_WIDTH = len(GraphFeature)  # 中文说明。
GLOBAL_FEATURE_WIDTH = 10  # 中文说明。


@dataclass(frozen=True, slots=True)  # 定义不可变异构服务图张量。
class HeterogeneousServiceGraph:  # 保存稳定对象顺序及静态、动态和全局状态。
    """绑定节点标识、节点特征、边关系和 M5 全局搜索状态。"""

    node_ids: tuple[str, ...]  # 保存与张量行一致的对象标识。
    node_types: torch.Tensor  # 保存每个节点的类型编号。
    static_features: torch.Tensor  # 保存 M1 至 M4 的静态物理特征。
    dynamic_features: torch.Tensor  # 保存当前 M5 解和滚动状态特征。
    global_features: torch.Tensor  # 保存七项目标分解和滚动上下文。
    edge_index: torch.Tensor  # 保存双向图边端点索引。
    edge_types: torch.Tensor  # 保存每条边的关系类型。
    destroy_action_mask: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )


def _number(obj: Any, name: str, default: float = 0.0) -> float:  # 安全读取有限数值字段。
    """读取存在且有限的对象数值，否则返回显式默认值。"""
    value = getattr(obj, name, default)  # 读取指定对象字段。
    if isinstance(value, (int, float)) and math.isfinite(float(value)):  # 判断字段是否为有限数值。
        return float(value)  # 返回统一浮点表示。
    return default  # 返回缺失或无效字段的默认值。


def _bounded(value: float) -> float:  # 将无界滚动标量压缩到有限区间。
    """使用保序有界变换编码滚动时刻和时间上界。"""
    return value / (1.0 + abs(value))  # 返回符号保持的有界数值。


def _raw_static_features(node_type: NodeType, obj: Any, effective: Any | None = None) -> tuple[float, ...]:  # 提取与数学模型字段一致的静态特征。
    """按照节点类型读取 M1 至 M4 中真实存在的物理字段。"""
    if node_type == NodeType.VEHICLE:  # 处理地面车辆能力字段。
        values = (obj.road_speed, obj.maximum_road_grade_ratio, obj.road_slope_sensitivity, obj.maximum_stop_grade_ratio, obj.minimum_stop_area)  # 对齐 GroundVehicle 字段。
    elif node_type == NodeType.STOP:  # 处理停靠点三维位置和面积代理字段。
        values = (obj.position.x, obj.position.y, obj.position.z, obj.footprint_radius, math.pi * obj.footprint_radius**2)  # 对齐 StopSite.position 和半径。
    elif node_type == NodeType.SERVICE_COLUMN:  # 处理 M4 服务列系数。
        if effective is not None and effective.is_airborne:
            values = (effective.effective_duration, effective.effective_protected_energy, 0.0, effective.effective_protected_energy, obj.minimum_clearance_margin, float(len(effective.effective_checkpoint_ids)))
        else:
            values = (obj.dynamic_duration, obj.energy_mean, obj.energy_standard_deviation, obj.energy_quantile, obj.minimum_clearance_margin, float(len(obj.covered_checkpoint_ids)))  # 对齐式 28 至式 30。
    else:  # 处理构件级任务参数。
        values = (obj.priority, obj.service_duration)  # 对齐任务优先权重和服务时间。
    return (*values[:8], *((0.0,) * (8 - len(values))))  # 补齐固定八维输入。


def _normalize_rows(rows: list[tuple[float, ...]], node_types: list[NodeType]) -> list[tuple[float, ...]]:  # 按类型归一化静态字段。
    """逐节点类型和字段缩放，避免能量大数值压缩掉列间差异。"""
    normalized = [list(row) for row in rows]  # 创建可写特征副本。
    for node_type in NodeType:  # 逐一处理每类节点。
        indices = [index for index, current in enumerate(node_types) if current == node_type]  # 收集当前类型行索引。
        for feature_index in range(8):  # 逐一处理固定八个特征维度。
            scale = max((abs(rows[index][feature_index]) for index in indices), default=0.0)  # 计算同类字段最大绝对值。
            if scale > 0.0:  # 仅缩放含有效信号的字段。
                for index in indices:  # 逐一更新当前类型节点。
                    normalized[index][feature_index] = rows[index][feature_index] / scale  # 保留同类对象相对差异。
    return [tuple(row) for row in normalized]  # 返回不可变行特征。


def build_service_graph(snapshot: Any, selected_column_ids: tuple[str, ...] = (), result: Any | None = None, include_dynamic: bool = True) -> HeterogeneousServiceGraph:  # 构建当前滚动时刻服务图并支持静态消融。
    """从快照和可选 M5 解构建静态物理图、动态调度状态和目标上下文。"""
    if not isinstance(include_dynamic, bool):  # 判断动态通道开关是否保持布尔语义。
        raise ValueError("include_dynamic 必须为布尔值")  # 拒绝隐式真值造成的消融漂移。
    completed = set(snapshot.completed_checkpoint_ids)  # 读取已经完成检查点。
    remaining_tasks = tuple(
        task
        for task in sorted(snapshot.tasks, key=lambda item: item.object_id)
        if task.checkpoint_id not in completed
    )
    remaining_checkpoint_ids = {task.checkpoint_id for task in remaining_tasks}
    locked_column_ids = {
        *snapshot.locked_column_ids,
        *getattr(snapshot, "committed_column_ids", ()),
        *(state.column_id for state in getattr(snapshot, "airborne_residual_states", ())),
    }
    effective_by_id = {
        column.object_id: snapshot.column_state(column.object_id)
        for column in snapshot.service_columns
    }
    effective_columns = tuple(
        column
        for column in sorted(snapshot.service_columns, key=lambda item: item.object_id)
        if (
            column.object_id in locked_column_ids
            or (
                uav_schedulable(snapshot.uav_state(column.uav_id))
                and bool(
                    set(effective_by_id[column.object_id].effective_checkpoint_ids)
                    & remaining_checkpoint_ids
                )
            )
        )
    )
    groups = (  # 按稳定类型顺序组织图节点。
        (NodeType.VEHICLE, tuple(sorted(snapshot.vehicles, key=lambda item: item.object_id))),
        (NodeType.STOP, tuple(sorted(snapshot.stop_sites, key=lambda item: item.object_id))),
        (NodeType.SERVICE_COLUMN, effective_columns),
        (NodeType.TASK, remaining_tasks),
    )
    records = tuple((node_type, obj) for node_type, objects in groups for obj in objects)  # 展开稳定节点记录。
    node_ids = tuple(obj.object_id for _, obj in records)  # 提取张量行对应对象标识。
    node_types = [node_type for node_type, _ in records]  # 提取节点类型序列。
    index_by_id = {object_id: index for index, object_id in enumerate(node_ids)}  # 建立对象到行索引映射。
    selected = set(selected_column_ids)  # 读取当前解选择服务列。
    locked = locked_column_ids  # 使用锁定、已承诺和空中残余列的完整不可变并集。
    previous = set(snapshot.previous_selected_column_ids)  # 读取上一窗口选择服务列。
    epoch_feature = _bounded(float(snapshot.rolling_epoch))  # 编码滚动窗口编号。
    static_rows = _normalize_rows([
        _raw_static_features(
            node_type,
            obj,
            effective_by_id.get(obj.object_id) if node_type == NodeType.SERVICE_COLUMN else None,
        )
        for node_type, obj in records
    ], node_types)  # 生成归一化静态特征。

    value_map = dict(result.variable_values) if result is not None else {}  # 读取 M5 命名变量解。
    selected_roads = set(result.selected_road_leg_ids) if result is not None else set()  # 读取选择道路弧。
    selected_stops = set(result.selected_stop_ids) if result is not None else set()  # 读取选择停靠点。
    completion_by_checkpoint = dict(result.checkpoint_completion_times) if result is not None else {}  # 读取检查点完成时间。
    road_time_by_vehicle = {vehicle.object_id: sum(option.travel_time for option in snapshot.road_options if option.vehicle_id == vehicle.object_id and option.road_leg_id in selected_roads) for vehicle in snapshot.vehicles}  # 汇总车辆当前路线时间。
    carrier_by_uav = {uav.object_id: uav.carrier_vehicle_id for uav in snapshot.uavs}  # 建立无人机到车辆映射。
    dynamic_rows: list[tuple[float, ...]] = []  # 创建十二维动态特征行。
    for node_type, obj in records:  # 逐一编码当前搜索状态。
        is_column = node_type == NodeType.SERVICE_COLUMN  # 判断服务列节点。
        is_task = node_type == NodeType.TASK  # 判断任务节点。
        checkpoint_id = getattr(obj, "checkpoint_id", "")  # 读取任务对应检查点。
        is_completed = is_task and checkpoint_id in completed  # 判断任务是否已经执行。
        selected_stop = node_type == NodeType.STOP and obj.object_id in selected_stops  # 判断停靠点是否在当前路线中。
        route_time = road_time_by_vehicle.get(obj.object_id, 0.0) / snapshot.time_horizon if node_type == NodeType.VEHICLE else 0.0  # 编码车辆路线时间。
        stop_ratio = (sum(option.stop_site_id in selected_stops for option in snapshot.stop_options if option.vehicle_id == obj.object_id) / max(1, sum(option.vehicle_id == obj.object_id for option in snapshot.stop_options))) if node_type == NodeType.VEHICLE else 0.0  # 编码车辆停靠选择比例。
        launch = 0.0  # 初始化服务列起飞时间特征。
        recovery = 0.0  # 初始化服务列回收时间特征。
        if is_column:  # 处理已选择或候选服务列时序。
            carrier = carrier_by_uav.get(obj.uav_id, "")  # 读取服务列所属车辆。
            launch = value_map.get(f"column_launch[{carrier},{obj.object_id}]", 0.0) / snapshot.time_horizon  # 编码起飞时间。
            recovery = value_map.get(f"column_recovery[{carrier},{obj.object_id}]", 0.0) / snapshot.time_horizon  # 编码回收时间。
        completion = completion_by_checkpoint.get(checkpoint_id, 0.0) / snapshot.time_horizon if is_task else 0.0  # 编码任务完成时间。
        missed = value_map.get(f"checkpoint_missed[{checkpoint_id}]", 0.0) if is_task else 0.0  # 编码任务漏检状态。
        aligned = [0.0] * 8  # 中文说明。
        if is_column:  # 中文说明。
            effective = effective_by_id[obj.object_id]  # 中文说明。
            uav_state = snapshot.uav_state(obj.uav_id)  # 中文说明。
            physical_uav = next(uav for uav in snapshot.uavs if uav.object_id == obj.uav_id)  # 中文说明。
            aligned[0] = float(effective.is_airborne)  # 中文说明。
            aligned[1] = effective.effective_duration / max(obj.dynamic_duration, 1.0e-9)  # 中文说明。
            aligned[2] = (effective.usable_battery_energy - effective.effective_protected_energy) / max(effective.usable_battery_energy, 1.0)  # 中文说明。
            aligned[3] = float(uav_state.available)  # 中文说明。
            carrier = carrier_by_uav.get(obj.uav_id, "")  # 中文说明。
            waiting = value_map.get(f"recovery_wait[{carrier},{obj.object_id}]", 0.0)  # 中文说明。
            aligned[4] = physical_uav.hover_power * waiting / max(effective.usable_battery_energy, 1.0)  # 中文说明。
            aligned[5] = float(uav_state.turnaround_time > 0.0 and sum(column.uav_id == obj.uav_id and column.object_id in selected for column in snapshot.service_columns) > 1)  # 中文说明。
            aligned[7] = 2.0 if effective.is_airborne else float(effective.is_committed)  # 中文说明。
        elif node_type == NodeType.VEHICLE:  # 中文说明。
            aligned[6] = float(bool(snapshot.vehicle_state(obj.object_id).current_road_node_id))  # 中文说明。
        dynamic_rows.append((float(is_column and obj.object_id in selected), float(is_column and obj.object_id in locked), float(is_completed), float(is_column and obj.object_id in previous), epoch_feature, float(not is_completed), float(selected_stop), route_time, stop_ratio, launch, recovery, max(completion, float(missed)), *aligned))  # 写入完整动态状态。

    objective_components = dict(result.objective_decomposition) if result is not None else {}  # 读取式 50 七项目标分解。
    component_names = ("completion", "makespan", "vehicle_time", "energy", "missed", "waiting", "plan_change")  # 固定目标分项顺序。
    component_values = [float(objective_components.get(name, 0.0)) for name in component_names]  # 提取当前目标分项。
    objective_scale = max(1.0, sum(abs(value) for value in component_values))  # 计算目标归一化尺度。
    global_features = (*(value / objective_scale for value in component_values), (float(result.objective_value) / objective_scale if result is not None and result.objective_value is not None else 0.0), epoch_feature, _bounded(float(snapshot.time_horizon)))  # 组成十维全局搜索上下文。
    if not include_dynamic:  # 判断是否启用只保留静态物理输入的消融。
        dynamic_rows = [(0.0,) * DYNAMIC_FEATURE_WIDTH for _ in records]  # 清零全部节点级调度和滚动状态通道。
        global_features = (0.0,) * GLOBAL_FEATURE_WIDTH  # 清零全部目标分解和图级滚动上下文。

    edges: list[tuple[int, int]] = []  # 创建双向边端点列表。
    edge_types: list[int] = []  # 创建边类型列表。

    def add_edge(source_id: str, target_id: str, edge_type: EdgeType) -> None:  # 添加存在端点的双向关系边。
        """将一个物理关系编码为共享类型的双向消息边。"""
        if source_id not in index_by_id or target_id not in index_by_id:  # 跳过图外端点。
            return  # 返回当前边添加调用。
        source = index_by_id[source_id]  # 读取源节点索引。
        target = index_by_id[target_id]  # 读取目标节点索引。
        edges.extend(((source, target), (target, source)))  # 添加双向端点。
        edge_types.extend((int(edge_type), int(edge_type)))  # 添加对应关系类型。

    for option in snapshot.stop_options:  # 编码车辆停靠准入关系。
        if option.accessible:  # 仅使用通过 M1 准入的组合。
            add_edge(option.vehicle_id, option.stop_site_id, EdgeType.VEHICLE_STOP)  # 添加车辆停靠边。
    task_by_checkpoint = {task.checkpoint_id: task.object_id for task in snapshot.tasks}  # 建立检查点到任务映射。
    for column in effective_columns:  # 编码当前有效服务列结构关系。
        vehicle_id = carrier_by_uav.get(column.uav_id)  # 读取执行车辆。
        if vehicle_id:  # 判断车辆映射是否存在。
            add_edge(vehicle_id, column.object_id, EdgeType.VEHICLE_COLUMN)  # 添加车辆服务列边。
        effective = effective_by_id[column.object_id]  # 中文说明。
        if effective.requires_launch_stop:  # 中文说明。
            add_edge(column.object_id, column.launch_stop_id, EdgeType.COLUMN_LAUNCH_STOP)  # 添加起飞停靠边。
        add_edge(column.object_id, column.recovery_stop_id, EdgeType.COLUMN_RECOVERY_STOP)  # 添加回收停靠边。
        for checkpoint_id in effective.effective_checkpoint_ids:  # 逐一编码服务列覆盖任务。
            task_id = task_by_checkpoint.get(checkpoint_id)  # 查找对应任务节点。
            if task_id:  # 判断任务节点存在。
                add_edge(column.object_id, task_id, EdgeType.COLUMN_TASK)  # 添加服务列任务边。

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)  # 构造边端点张量。
    return HeterogeneousServiceGraph(node_ids=node_ids, node_types=torch.tensor([int(node_type) for node_type in node_types], dtype=torch.long), static_features=torch.tensor(static_rows, dtype=torch.float32), dynamic_features=torch.tensor(dynamic_rows, dtype=torch.float32), global_features=torch.tensor(global_features, dtype=torch.float32), edge_index=edge_index, edge_types=torch.tensor(edge_types, dtype=torch.long))  # 返回完整图张量。
