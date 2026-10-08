"""定义连接完整 Mask 场景、滚动状态和统一 Agent 快照的巡检环境。"""  # 说明本文件的环境职责。

import math  # 导入数学函数以验证环境时间上界。
from dataclasses import replace
from pathlib import Path  # 导入路径工具以定位 Data 场景和配置文件。

from Env.base_env import BaseEnvironment  # 导入环境抽象基类以实现统一接口。
from Env.data_loader import load_project_data  # 导入不含决策逻辑的严格 Data 读取接口。
from Env.scene_builder import SceneBuilder  # 导入负责实例化和连接 Mask 对象的场景构建器。
from Env.snapshot import AirborneResidualState, FeasibilityReport, RejectionRecord, RollingUAVState, RollingVehicleState, SceneSnapshot, VehicleRoadOption, VehicleStopOption  # 导入环境快照、车辆道路停靠选项和预求解报告类型。
from Env.validation import ResultLike, ResultValidation, validate_selected_columns  # 导入不依赖 Agent 实现的结果复核接口。
from Mask.base import SceneSystem  # 导入完整场景系统类型。
from Mask.flight_relation import FlightPrimitive  # 导入飞行原语类型以构造分类快照。
from Mask.infrastructure import RoadNode, StopSite  # 导入道路节点和停靠点类型以构造分类快照。
from Mask.inspection import InspectionTask  # 导入巡检任务类型以构造分类快照。
from Mask.mobility import GroundVehicle, UnmannedAerialVehicle  # 导入车辆和无人机类型以构造分类快照。
from Mask.observation_relation import ObservationAssessment  # 导入候选观测位姿评估以保留 M2 准入证据。
from Mask.road_relation import RoadLeg  # 导入道路弧类型以构造分类快照。
from Mask.service_column import ServiceColumn  # 导入服务列类型以构造准入候选快照。
from Mask.trajectory_optimization import TrajectoryOptimizer  # 导入算法无关连续求解协议以保持 Env 不依赖 Agent。
from Mask.weather import EnergyModelParameters, WeatherState  # 导入气象状态类型以构造分类快照。


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:  # 定义保持输入顺序的字符串去重函数。
    """返回删除重复值但保留首次出现顺序的字符串元组。"""  # 解释滚动状态去重规则。
    return tuple(dict.fromkeys(values))  # 利用有序字典键删除重复字符串。


class InspectionEnv(BaseEnvironment):  # 定义三维车载无人机巡检环境。
    """管理完整场景、准入服务列、拒绝证据和滚动决策状态。"""  # 解释巡检环境职责。

    def __init__(self, system: SceneSystem, time_horizon: float, service_columns: tuple[ServiceColumn, ...] = (), rejection_records: tuple[RejectionRecord, ...] = (), road_options: tuple[VehicleRoadOption, ...] = (), stop_options: tuple[VehicleStopOption, ...] = (), observation_assessments: tuple[ObservationAssessment, ...] = (), vehicle_states: tuple[RollingVehicleState, ...] = (), uav_states: tuple[RollingUAVState, ...] = (), state_default_diagnostics: tuple[str, ...] = (), rolling_window_seconds: float | None = None, energy_parameters: EnergyModelParameters | None = None) -> None:  # 定义环境初始化方法。
        """保存经过连接的场景并初始化滚动状态。"""  # 解释巡检环境初始状态。
        if not math.isfinite(time_horizon) or time_horizon <= 0.0:  # 判断调度时间上界是否为正有限数值。
            raise ValueError("环境调度时间上界必须为正有限数值")  # 抛出明确的环境时间上界异常。
        resolved_rolling_window = time_horizon if rolling_window_seconds is None else rolling_window_seconds  # 旧直接构造调用将滚动窗口解释为原单时域。
        if not math.isfinite(resolved_rolling_window) or resolved_rolling_window <= 0.0 or resolved_rolling_window > time_horizon or not math.isclose(time_horizon / resolved_rolling_window, round(time_horizon / resolved_rolling_window), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("滚动窗口必须为可整除任务总时域的正有限数值")
        scene_issues = system.validate()  # 汇总输入场景系统验证问题。
        if scene_issues:  # 判断输入场景是否存在未闭合引用。
            issue_codes = ",".join(issue.code for issue in scene_issues)  # 汇总输入场景验证问题代码。
            raise ValueError(f"环境不能接收未闭合场景: {issue_codes}")  # 拒绝使用未闭合场景构建环境。
        relation_columns = tuple(relation for relation in system.relations() if isinstance(relation, ServiceColumn) and relation.admitted)  # 提取场景系统中已经登记的准入服务列。
        combined_columns = (*relation_columns, *(column for column in service_columns if column.admitted))  # 合并场景关系和显式传入的准入服务列。
        column_by_id = {column.object_id: column for column in combined_columns}  # 按稳定标识去重并索引准入服务列。
        self._system = system  # 保存经过引用验证的完整场景系统。
        self._time_horizon = time_horizon  # 保存当前调度时间上界。
        self._rolling_window_seconds = resolved_rolling_window  # 保存独立滚动更新窗口。
        self._base_service_columns = tuple(column_by_id.values())
        self._service_columns = self._base_service_columns  # 保存不可变准入服务列集合。
        self._energy_parameters = energy_parameters
        self._initial_weather_states = tuple(
            element for element in system.elements() if isinstance(element, WeatherState)
        )
        self._weather_states = self._initial_weather_states
        self._rejection_records = tuple(rejection_records)  # 保存不可变物理和配置拒绝记录集合。
        self._road_options = tuple(option for option in road_options if option.accessible)  # 保存通过车辆通行认证的道路选项。
        self._stop_options = tuple(stop_options)  # 保存包含通过和拒绝状态的全部车辆停靠选项。
        self._observation_assessments = tuple(observation_assessments)  # 保存全部候选位姿的不可变准入评估证据。
        self._rolling_epoch = 0  # 初始化滚动决策时刻为零。
        self._completed_checkpoint_ids: tuple[str, ...] = ()  # 初始化已完成检查点集合为空。
        self._locked_column_ids: tuple[str, ...] = ()  # 初始化锁定服务列集合为空。
        self._committed_column_ids: tuple[str, ...] = ()  # 中文说明。
        self._airborne_residual_states: tuple[AirborneResidualState, ...] = ()  # 中文说明。
        self._initial_vehicle_states = tuple(vehicle_states)  # 中文说明。
        self._initial_uav_states = tuple(uav_states)  # 中文说明。
        self._vehicle_states: tuple[RollingVehicleState, ...] = tuple(vehicle_states)  # 中文说明。
        self._uav_states: tuple[RollingUAVState, ...] = tuple(uav_states)  # 中文说明。
        self._state_default_diagnostics = tuple(state_default_diagnostics)  # 中文说明。
        self._previous_selected_column_ids: tuple[str, ...] = ()  # 初始化上一方案服务列集合为空。
        self._previous_selected_road_leg_ids: tuple[str, ...] = ()  # 初始化上一方案道路弧集合为空。
        self._previous_selected_stop_ids: tuple[str, ...] = ()  # 初始化上一方案停靠点集合为空。

    @classmethod  # 声明从 Data 目录构建完整环境的替代构造方法。
    def from_directory(cls, scenario_directory: Path, config_path: Path, trajectory_optimizer: TrajectoryOptimizer | None = None, weather_scenario_path: Path | None = None, weather_split: str | None = None) -> "InspectionEnv":  # 定义完整 Data 环境构建入口并允许注入连续求解器和正式天气场景。
        """读取 Data 并通过 SceneBuilder 实例化全部 Mask 对象和认证关系。"""  # 解释外部环境构建流程。
        project_data = load_project_data(scenario_directory, config_path)  # 严格读取场景、服务列候选和物理算法配置。
        build_result = SceneBuilder.from_project_data(project_data, trajectory_optimizer=trajectory_optimizer, weather_scenario_path=weather_scenario_path, weather_split=weather_split)  # 按元素到系统顺序实例化并通过可选内层求解器和天气发布连接全部 Mask 对象。
        return cls(build_result.system, build_result.time_horizon, build_result.service_columns, build_result.rejection_records, build_result.road_options, build_result.stop_options, build_result.observation_assessments, build_result.vehicle_states, build_result.uav_states, build_result.state_default_diagnostics, rolling_window_seconds=build_result.rolling_window_seconds, energy_parameters=build_result.energy_parameters)  # 返回包含准入列、滚动资源和两个时间控制的完整环境。

    def reset(self) -> SceneSnapshot:  # 实现环境重置接口。
        """清除滚动状态并返回恢复到初始时刻的环境快照。"""  # 解释环境重置行为。
        self._rolling_epoch = 0  # 将滚动决策时刻恢复为零。
        self._completed_checkpoint_ids = ()  # 清空已完成检查点集合。
        self._locked_column_ids = ()  # 清空锁定服务列集合。
        self._committed_column_ids = ()  # 中文说明。
        self._airborne_residual_states = ()  # 中文说明。
        self._vehicle_states = self._initial_vehicle_states  # 中文说明。
        self._uav_states = self._initial_uav_states  # 中文说明。
        self._service_columns = self._base_service_columns
        self._weather_states = self._initial_weather_states
        self._previous_selected_column_ids = ()  # 清空上一方案服务列集合。
        self._previous_selected_road_leg_ids = ()  # 清空上一方案道路弧集合。
        self._previous_selected_stop_ids = ()  # 清空上一方案停靠点集合。
        return self.snapshot()  # 返回重置后的不可变环境快照。

    def advance_epoch(self, completed_checkpoint_ids: tuple[str, ...] = (), locked_column_ids: tuple[str, ...] = (), previous_selected_column_ids: tuple[str, ...] = (), previous_selected_road_leg_ids: tuple[str, ...] = (), previous_selected_stop_ids: tuple[str, ...] = (), committed_column_ids: tuple[str, ...] = (), airborne_residual_states: tuple[AirborneResidualState, ...] = (), vehicle_states: tuple[RollingVehicleState, ...] = (), uav_states: tuple[RollingUAVState, ...] = (), weather_state: WeatherState | None = None) -> SceneSnapshot:  # 定义滚动决策时刻推进方法。
        """推进一个滚动时刻并合并完成、锁定和上一方案状态。"""  # 解释滚动状态更新行为。
        self._validate_rolling_state(committed_column_ids, airborne_residual_states, vehicle_states, uav_states)  # 中文说明。
        self._rolling_epoch += 1  # 将滚动决策时刻增加一。
        self._completed_checkpoint_ids = _ordered_unique((*self._completed_checkpoint_ids, *completed_checkpoint_ids))  # 合并并去重已完成检查点标识。
        self._locked_column_ids = _ordered_unique(locked_column_ids)  # 保存当前滚动时刻的锁定服务列标识。
        self._committed_column_ids = _ordered_unique((*locked_column_ids, *committed_column_ids))  # 中文说明。
        self._airborne_residual_states = tuple(airborne_residual_states)  # 中文说明。
        if vehicle_states:  # 中文说明。
            self._vehicle_states = tuple(vehicle_states)  # 中文说明。
        if uav_states:  # 中文说明。
            self._uav_states = tuple(uav_states)  # 中文说明。
        if weather_state is not None:
            self._apply_measured_weather(weather_state)
        self._previous_selected_column_ids = _ordered_unique(previous_selected_column_ids)  # 保存上一滚动方案选择的服务列标识。
        self._previous_selected_road_leg_ids = _ordered_unique(previous_selected_road_leg_ids)  # 保存上一滚动方案选择的道路弧标识。
        self._previous_selected_stop_ids = _ordered_unique(previous_selected_stop_ids)  # 保存上一滚动方案选择的停靠点标识。
        return self.snapshot()  # 返回状态推进后的不可变环境快照。

    def _apply_measured_weather(self, measured: WeatherState) -> None:
        if self._energy_parameters is None:
            raise ValueError("live weather recertification requires energy model parameters")
        self._weather_states = tuple(
            WeatherState(
                weather.object_id,
                weather.name,
                weather.period_index,
                measured.wind_vector,
                measured.temperature,
            )
            for weather in self._initial_weather_states
        )
        weather_by_id = {weather.object_id: weather for weather in self._weather_states}
        primitive_by_id = {
            relation.object_id: relation
            for relation in self._system.relations()
            if isinstance(relation, FlightPrimitive)
        }
        uav_by_id = {
            element.object_id: element
            for element in self._system.elements()
            if isinstance(element, UnmannedAerialVehicle)
        }
        recertified = []
        pack_energy_by_uav = {state.uav_id: state.replacement_pack_energy for state in self._uav_states}
        for column in self._base_service_columns:
            primitives = tuple(primitive_by_id[item] for item in column.primitive_ids)
            live_column = column.recertify_measured_weather(
                weather_by_id[column.weather_period_id],
                primitives,
                replace(uav_by_id[column.uav_id], battery_state_of_charge=pack_energy_by_uav.get(
                    column.uav_id, uav_by_id[column.uav_id].battery_state_of_charge)),
                self._energy_parameters,
            )
            if live_column.admitted:
                recertified.append(live_column)
        self._service_columns = tuple(recertified)

    def _validate_rolling_state(self, committed_column_ids: tuple[str, ...], airborne_residual_states: tuple[AirborneResidualState, ...], vehicle_states: tuple[RollingVehicleState, ...], uav_states: tuple[RollingUAVState, ...]) -> None:  # 中文说明。
        """中文说明。"""
        vehicle_ids = {item.object_id for item in self._system.elements() if isinstance(item, GroundVehicle)}  # 中文说明。
        uav_ids = {item.object_id for item in self._system.elements() if isinstance(item, UnmannedAerialVehicle)}  # 中文说明。
        node_ids = {item.object_id for item in self._system.elements() if isinstance(item, RoadNode)}  # 中文说明。
        column_ids = {item.object_id for item in self._base_service_columns}  # 中文说明。
        for state in vehicle_states:  # 中文说明。
            if state.vehicle_id not in vehicle_ids:  # 中文说明。
                raise ValueError(f"unknown vehicle: {state.vehicle_id}")  # 中文说明。
            if state.current_road_node_id not in node_ids:  # 中文说明。
                raise ValueError(f"unknown road node: {state.current_road_node_id}")  # 中文说明。
        for state in uav_states:  # 中文说明。
            if state.uav_id not in uav_ids:  # 中文说明。
                raise ValueError(f"unknown UAV: {state.uav_id}")  # 中文说明。
        for column_id in (*committed_column_ids, *(state.column_id for state in airborne_residual_states)):  # 中文说明。
            if column_id not in column_ids:  # 中文说明。
                raise ValueError(f"unknown service column: {column_id}")  # 中文说明。
        for name, identifiers in (("vehicle", tuple(state.vehicle_id for state in vehicle_states)), ("UAV", tuple(state.uav_id for state in uav_states)), ("airborne column", tuple(state.column_id for state in airborne_residual_states))):  # 中文说明。
            if len(identifiers) != len(set(identifiers)):  # 中文说明。
                raise ValueError(f"duplicate {name} rolling state")  # 中文说明。

    def snapshot(self) -> SceneSnapshot:  # 实现环境快照接口。
        """按照稳定类型顺序生成全部 Agent 共享的不可变场景快照。"""  # 解释快照构造行为。
        weather_by_id = {weather.object_id: weather for weather in self._weather_states}
        elements = tuple(
            weather_by_id.get(element.object_id, element)
            if isinstance(element, WeatherState)
            else element
            for element in self._system.elements()
        )  # 读取完整场景元素集合。
        relations = self._system.relations()  # 读取完整场景关系集合。
        vehicles = tuple(element for element in elements if isinstance(element, GroundVehicle))  # 提取全部地面车辆元素。
        uavs = tuple(element for element in elements if isinstance(element, UnmannedAerialVehicle))  # 提取全部车载无人机元素。
        road_nodes = tuple(element for element in elements if isinstance(element, RoadNode))  # 提取全部道路节点元素。
        stop_sites = tuple(element for element in elements if isinstance(element, StopSite))  # 提取全部候选停靠点元素。
        tasks = tuple(element for element in elements if isinstance(element, InspectionTask))  # 提取全部构件级巡检任务元素。
        weather_states = tuple(element for element in elements if isinstance(element, WeatherState))  # 提取全部动态气象状态元素。
        road_legs = tuple(relation for relation in relations if isinstance(relation, RoadLeg))  # 提取全部道路弧关系。
        flight_primitives = tuple(relation for relation in relations if isinstance(relation, FlightPrimitive) and relation.certified)  # 提取全部认证飞行原语关系。
        start_node = next((node.object_id for node in road_nodes if node.role == "depot_start"), road_nodes[0].object_id if road_nodes else "")  # 中文说明。
        vehicle_states = self._vehicle_states or tuple(RollingVehicleState(vehicle.object_id, start_node) for vehicle in vehicles)  # 中文说明。
        uav_states = self._uav_states or tuple(RollingUAVState(uav.object_id, True, uav.battery_state_of_charge, uav.battery_state_of_charge, 0.0) for uav in uavs)  # 中文说明。
        return SceneSnapshot(self._system.object_id, self._system.name, self._rolling_epoch, self._time_horizon, elements, relations, vehicles, uavs, road_nodes, road_legs, self._road_options, stop_sites, self._stop_options, tasks, self._observation_assessments, flight_primitives, self._service_columns, weather_states, self._completed_checkpoint_ids, self._locked_column_ids, self._previous_selected_column_ids, self._previous_selected_road_leg_ids, self._previous_selected_stop_ids, self._rejection_records, self._committed_column_ids, self._airborne_residual_states, vehicle_states, uav_states, self._state_default_diagnostics, self._rolling_window_seconds)  # 返回包含车辆级停靠认证、M2 观测证据和两个时间控制的完整不可变环境快照。

    def feasibility_report(self) -> FeasibilityReport:  # 定义调用 Agent 前的基础可行性报告方法。
        """根据准入服务列数量返回环境就绪状态和阻断原因。"""  # 解释预求解可行性检查边界。
        if not self._service_columns:  # 判断环境是否缺少准入服务列。
            return FeasibilityReport(False, "no_admitted_service_columns", 0, len(self._rejection_records))  # 返回缺少准入服务列的阻断报告。
        return FeasibilityReport(True, "ready", len(self._service_columns), len(self._rejection_records))  # 返回环境可以交给 Agent 的就绪报告。

    def validate_result(self, result: ResultLike) -> ResultValidation:  # 定义不依赖具体算法的结果复核方法。
        """使用当前环境快照复核 Agent 选择的服务列基础合法性。"""  # 解释结果复核接口边界。
        return validate_selected_columns(self.snapshot(), result)  # 委托独立复核函数检查准入范围和重复选择。
