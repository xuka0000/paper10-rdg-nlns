"""定义环境拒绝记录、预求解报告和不可变场景快照。"""  # 说明本文件的环境职责。

import math  # 导入数学函数以验证道路认证系数。
from dataclasses import dataclass  # 导入数据类工具以保存不可变环境输出。

from Mask.base import SceneElement, SceneRelation  # 导入通用场景对象类型以保留完整快照。
from Mask.flight_relation import FlightPrimitive  # 导入飞行原语类型以暴露动力学系数。
from Mask.infrastructure import RoadNode, StopAssessment, StopSite  # 导入道路节点、停靠点评估和停靠点类型以暴露地面网络。
from Mask.inspection import InspectionTask  # 导入巡检任务类型以暴露任务状态。
from Mask.mobility import GroundVehicle, UnmannedAerialVehicle  # 导入车辆和无人机类型以暴露资源状态。
from Mask.observation_relation import ObservationAssessment  # 导入候选观测位姿评估以暴露式 10 至式 15 证据。
from Mask.road_relation import RoadLeg  # 导入道路弧关系类型以暴露车辆可选路径。
from Mask.service_column import ServiceColumn  # 导入服务列关系类型以暴露调度候选。
from Mask.weather import WeatherState  # 导入气象状态类型以暴露滚动预测信息。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义物理拒绝记录。
class RejectionRecord:  # 定义未进入准入集合的对象审计记录。
    """保存对象标识、对象类型、拒绝原因、相关字段和 Data 来源。"""  # 解释拒绝记录内容。

    object_id: str  # 保存被拒绝对象的稳定标识。
    object_type: str  # 保存被拒绝对象的类别名称。
    reason: str  # 保存稳定拒绝原因代码。
    related_fields: tuple[str, ...]  # 保存导致拒绝的相关物理或配置字段。
    source_path: str  # 保存被拒绝对象对应的 Data 来源路径。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义预求解可行性报告。
class FeasibilityReport:  # 定义环境是否能够交给 Agent 的预检查结果。
    """记录准入服务列数量、拒绝数量、就绪状态和阻断原因。"""  # 解释预求解报告内容。

    ready_for_agent: bool  # 保存环境是否具备调用 Agent 的基本条件。
    reason: str  # 保存未就绪时的稳定阻断原因或就绪标记。
    admitted_column_count: int  # 保存准入服务列数量。
    rejection_count: int  # 保存物理和配置拒绝记录数量。


@dataclass(frozen=True, slots=True)
class RollingVehicleState:  # 中文说明。
    """Current road-network state of one ground vehicle. 中文说明。"""

    vehicle_id: str  # 中文说明。
    current_road_node_id: str  # 中文说明。


@dataclass(frozen=True, slots=True)
class RollingUAVState:  # 中文说明。
    """Availability and battery resources visible at a rolling epoch. 中文说明。"""

    uav_id: str  # 中文说明。
    available: bool  # 中文说明。
    measured_battery_energy: float  # 中文说明。
    replacement_pack_energy: float  # 中文说明。
    turnaround_time: float  # 中文说明。
    available_after_seconds: float = 0.0  # Measured/predicted earliest future launch offset; zero preserves legacy semantics.

    def __post_init__(self):
        if not math.isfinite(self.available_after_seconds) or self.available_after_seconds < 0:
            raise ValueError('UAV readiness offset must be finite and nonnegative')


def uav_schedulable(state) -> bool:
    """Unavailable resources may enter future plans only with a timed release."""
    return state.available or getattr(state,'available_after_seconds',0.)>0


@dataclass(frozen=True, slots=True)
class AirborneResidualState:  # 中文说明。
    """Remaining execution state of a service column already airborne. 中文说明。"""

    column_id: str  # 中文说明。
    remaining_checkpoint_ids: tuple[str, ...]  # 中文说明。
    residual_completion_offsets: tuple[tuple[str, float], ...]  # 中文说明。
    residual_duration: float  # 中文说明。
    residual_protected_energy: float  # 中文说明。
    remaining_battery_energy: float  # 剩余总电量，尚未扣除本航次固定预留电量。


@dataclass(frozen=True, slots=True)
class EffectiveColumnState:  # 中文说明。
    """Scheduling coefficients after applying the current execution state. 中文说明。"""

    column: ServiceColumn  # 中文说明。
    is_committed: bool  # 中文说明。
    is_airborne: bool  # 中文说明。
    effective_checkpoint_ids: tuple[str, ...]  # 中文说明。
    effective_completion_offsets: tuple[tuple[str, float], ...]  # 中文说明。
    effective_duration: float  # 中文说明。
    effective_protected_energy: float  # 中文说明。
    usable_battery_energy: float  # 中文说明。
    requires_launch_stop: bool  # 中文说明。

    @property
    def column_id(self) -> str:  # 中文说明。
        """中文说明。"""
        return self.column.object_id  # 中文说明。

    @property
    def uav_id(self) -> str:  # 中文说明。
        """中文说明。"""
        return self.column.uav_id  # 中文说明。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义车辆道路认证选项。
class VehicleRoadOption:  # 定义 Env 提供给 Agent 的车辆道路物理系数。
    """绑定车辆、道路弧、端点、时间、长度和通行性认证结果。"""  # 解释车辆道路选项内容。

    vehicle_id: str  # 保存适用地面车辆标识。
    road_leg_id: str  # 保存道路弧稳定标识。
    source_node_id: str  # 保存道路弧起点道路节点标识。
    target_node_id: str  # 保存道路弧终点道路节点标识。
    travel_time: float  # 保存地形和路面修正后的车辆行驶时间。
    length_3d: float  # 保存道路高程剖面的三维长度。
    accessible: bool  # 保存该车辆是否通过道路弧通行认证。

    def __post_init__(self) -> None:  # 定义车辆道路选项初始化后的参数验证方法。
        """验证稳定标识以及通行道路的时间和长度系数。"""  # 解释车辆道路选项输入约束。
        identifiers = (self.vehicle_id, self.road_leg_id, self.source_node_id, self.target_node_id)  # 汇总车辆道路选项引用标识。
        if any(not identifier.strip() for identifier in identifiers):  # 判断车辆道路选项是否包含空白标识。
            raise ValueError("车辆道路选项引用标识不能为空")  # 抛出明确的车辆道路标识异常。
        if self.accessible and (not math.isfinite(self.travel_time) or self.travel_time < 0.0):  # 判断可通行道路时间是否为非负有限数值。
            raise ValueError("可通行道路时间必须为非负有限数值")  # 抛出明确的车辆道路时间异常。
        if self.accessible and (not math.isfinite(self.length_3d) or self.length_3d < 0.0):  # 判断可通行道路长度是否为非负有限数值。
            raise ValueError("可通行道路长度必须为非负有限数值")  # 抛出明确的车辆道路长度异常。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义车辆停靠认证选项。
class VehicleStopOption:  # 定义 Env 提供给 Agent 的车辆级停靠准入参数。
    """绑定车辆、停靠点、地形坡比、几何面积和准入结果。"""  # 解释车辆停靠选项内容。

    vehicle_id: str  # 保存适用地面车辆标识。
    stop_site_id: str  # 保存候选停靠点稳定标识。
    maximum_local_grade_ratio: float  # 保存停靠范围最大局部坡比。
    footprint_area: float  # 保存停靠范围圆形几何面积。
    accessible: bool  # 保存该车辆是否通过当前停靠点准入认证。

    def __post_init__(self) -> None:  # 定义车辆停靠选项初始化后的参数验证方法。
        """验证引用标识以及坡比和面积物理值。"""  # 解释车辆停靠选项输入约束。
        if not self.vehicle_id.strip() or not self.stop_site_id.strip():  # 判断车辆和停靠点标识是否为空白。
            raise ValueError("车辆停靠选项引用标识不能为空")  # 抛出明确的车辆停靠引用异常。
        if math.isnan(self.maximum_local_grade_ratio) or self.maximum_local_grade_ratio < 0.0:  # 判断坡比是否为非负数或地形边界拒绝使用的正无穷标记。
            raise ValueError("车辆停靠选项坡比必须为非负数或正无穷拒绝标记")  # 抛出明确的车辆停靠坡比异常。
        if not math.isfinite(self.footprint_area) or self.footprint_area < 0.0:  # 判断圆形几何面积是否为非负有限值。
            raise ValueError("车辆停靠选项面积必须为非负有限数值")  # 抛出明确的车辆停靠面积异常。
        if math.isinf(self.maximum_local_grade_ratio) and self.accessible:  # 判断地形边界外停靠范围是否被错误标为可准入。
            raise ValueError("正无穷坡比停靠选项必须标记为不可准入")  # 阻止越界停靠范围进入 Agent 准入集合。

    @classmethod  # 声明从 Mask 物理评估创建车辆级选项的命名构造器。
    def from_assessment(cls, vehicle: GroundVehicle, stop_site_id: str, assessment: StopAssessment) -> "VehicleStopOption":  # 定义车辆级停靠准入计算方法。
        """根据车辆坡比和面积能力返回对应停靠点准入结果。"""  # 解释命名构造器计算边界。
        accessible = assessment.maximum_local_grade_ratio <= vehicle.maximum_stop_grade_ratio and assessment.footprint_area >= vehicle.minimum_stop_area  # 同时检查车辆停靠坡比和最小面积能力。
        return cls(vehicle.object_id, stop_site_id, assessment.maximum_local_grade_ratio, assessment.footprint_area, accessible)  # 返回包含物理证据的车辆停靠选项。


class _SnapshotLookupStorage:
    # Derived indexes are not dataclass fields: equality, serialization and
    # dataclasses.replace retain the original scientific state representation.
    __slots__ = ('_lookup_cache',)


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义环境快照。
class SceneSnapshot(_SnapshotLookupStorage):  # 定义所有 Agent 共同消费的版本化只读场景状态。
    """按类型暴露场景对象、滚动状态、准入服务列和拒绝证据。"""  # 解释环境快照职责。

    scene_id: str  # 保存场景系统稳定标识。
    scene_name: str  # 保存场景系统可读名称。
    rolling_epoch: int  # 保存当前滚动决策时刻。
    time_horizon: float  # 保存当前调度时间上界。
    elements: tuple[SceneElement, ...]  # 保存全部场景元素。
    relations: tuple[SceneRelation, ...]  # 保存全部场景关系。
    vehicles: tuple[GroundVehicle, ...]  # 保存全部地面车辆。
    uavs: tuple[UnmannedAerialVehicle, ...]  # 保存全部车载无人机。
    road_nodes: tuple[RoadNode, ...]  # 保存全部道路节点。
    road_legs: tuple[RoadLeg, ...]  # 保存全部道路弧关系。
    road_options: tuple[VehicleRoadOption, ...]  # 保存按车辆认证的可用道路时间和长度系数。
    stop_sites: tuple[StopSite, ...]  # 保存全部候选停靠点。
    stop_options: tuple[VehicleStopOption, ...]  # 保存全部车辆停靠组合的物理准入参数。
    tasks: tuple[InspectionTask, ...]  # 保存全部构件级巡检任务。
    observation_assessments: tuple[ObservationAssessment, ...]  # 保存全部候选位姿的几何、净空和视线准入证据。
    flight_primitives: tuple[FlightPrimitive, ...]  # 保存全部地形认证飞行原语。
    service_columns: tuple[ServiceColumn, ...]  # 保存全部准入服务列。
    weather_states: tuple[WeatherState, ...]  # 保存全部滚动气象时段状态。
    completed_checkpoint_ids: tuple[str, ...]  # 保存已经完成的检查点标识。
    locked_column_ids: tuple[str, ...]  # 保存滚动计划中必须保持的服务列标识。
    previous_selected_column_ids: tuple[str, ...]  # 保存上一滚动方案选择的服务列标识。
    previous_selected_road_leg_ids: tuple[str, ...]  # 保存上一滚动方案选择的道路弧标识。
    previous_selected_stop_ids: tuple[str, ...]  # 保存上一滚动方案选择的停靠点标识。
    rejection_records: tuple[RejectionRecord, ...]  # 保存全部未进入准入集合的审计记录。
    committed_column_ids: tuple[str, ...] = ()  # 中文说明。
    airborne_residual_states: tuple[AirborneResidualState, ...] = ()  # 中文说明。
    vehicle_states: tuple[RollingVehicleState, ...] = ()  # 中文说明。
    uav_states: tuple[RollingUAVState, ...] = ()  # 中文说明。
    state_default_diagnostics: tuple[str, ...] = ()  # 中文说明。
    rolling_window_seconds: float = 0.0  # 保存滚动重优化与提交前缀窗口，旧手工快照可保持默认值。

    def _lookups(self):
        try:
            return self._lookup_cache
        except AttributeError:
            def first_by(items, key):
                result = {}
                for item in items:
                    result.setdefault(getattr(item, key), item)
                return result
            cache = {
                'columns': first_by(self.service_columns, 'object_id'),
                'uavs': first_by(self.uavs, 'object_id'),
                'uav_states': first_by(self.uav_states, 'uav_id'),
                'vehicle_states': first_by(self.vehicle_states, 'vehicle_id'),
                'residuals': first_by(self.airborne_residual_states, 'column_id'),
                'effective': {},
            }
            object.__setattr__(self, '_lookup_cache', cache)
            return cache

    def vehicle_state(self, vehicle_id: str) -> RollingVehicleState:  # 中文说明。
        """Return the current state for a known vehicle. 中文说明。"""
        try:  # 中文说明。
            return self._lookups()['vehicle_states'][vehicle_id]
        except KeyError as error:
            raise KeyError(f"vehicle state not found: {vehicle_id}") from error  # 中文说明。

    def uav_state(self, uav_id: str) -> RollingUAVState:  # 中文说明。
        """Return the current state for a known UAV. 中文说明。"""
        try:  # 中文说明。
            return self._lookups()['uav_states'][uav_id]
        except KeyError as error:
            raise KeyError(f"UAV state not found: {uav_id}") from error  # 中文说明。

    def column_state(self, column_id: str) -> EffectiveColumnState:  # 中文说明。
        """Return one column's immutable coefficients at this rolling epoch. 中文说明。"""
        cache = self._lookups()
        if column_id in cache['effective']:
            return cache['effective'][column_id]
        try:  # 中文说明。
            column = cache['columns'][column_id]
        except KeyError as error:
            raise KeyError(f"service column not found: {column_id}") from error  # 中文说明。
        residual = cache['residuals'].get(column_id)
        uav_state = self.uav_state(column.uav_id)  # 中文说明。
        uav = cache['uavs'][column.uav_id]
        # Replacement packs have the same mission-start energy in the current model.
        # Keep this reserve fixed as telemetry decreases during an airborne mission.
        reserve_energy = uav.battery_reserve_ratio * uav_state.replacement_pack_energy
        if residual is not None:  # 中文说明。
            state = EffectiveColumnState(  # 中文说明。
                column=column,
                is_committed=True,
                is_airborne=True,
                effective_checkpoint_ids=residual.remaining_checkpoint_ids,
                effective_completion_offsets=residual.residual_completion_offsets,
                effective_duration=residual.residual_duration,
                effective_protected_energy=residual.residual_protected_energy,
                usable_battery_energy=residual.remaining_battery_energy - reserve_energy,
                requires_launch_stop=False,
            )
            cache['effective'][column_id] = state
            return state
        state = EffectiveColumnState(  # 中文说明。
            column=column,
            is_committed=column_id in self.committed_column_ids or column_id in self.locked_column_ids,
            is_airborne=False,
            effective_checkpoint_ids=column.covered_checkpoint_ids,
            effective_completion_offsets=column.completion_offsets,
            effective_duration=column.dynamic_duration,
            effective_protected_energy=column.energy_quantile,
            usable_battery_energy=(1.0 - uav.battery_reserve_ratio) * uav_state.replacement_pack_energy,
            requires_launch_stop=True,
        )
        cache['effective'][column_id] = state
        return state
