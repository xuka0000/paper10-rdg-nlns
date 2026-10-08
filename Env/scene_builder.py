"""提供只负责登记、连接和验证 Mask 对象的场景构建器。"""  # 说明本文件的环境职责。

from dataclasses import dataclass, replace  # 导入数据类工具以保存不可变构建结果。
import math  # 导入正无穷以兼容缺少第五部分时间上界的旧配置。
from pathlib import Path  # 导入路径类型以读取正式天气场景发布表。

from Env.benchmark.nrel_hrrr_weather import load_weather_scenario_set  # 导入 NREL-HRRR 正式气象场景集加载器。
from Env.data_loader import ProjectData, parse_time_controls  # 导入经过严格检查的 Data 文档和双时间控制解析器。
from Env.snapshot import RejectionRecord, RollingUAVState, RollingVehicleState, VehicleRoadOption, VehicleStopOption  # 导入物理对象拒绝记录以及车辆道路和停靠认证选项。
from Mask.base import SceneElement, SceneRelation, SceneSystem, ValidationIssue  # 导入场景对象和验证问题类型。
from Mask.flight_relation import FlightCorridor, FlightDynamicLimits, FlightPrimitive, TrajectoryObjectiveWeights  # 导入地形航路认证、二阶质点飞行原语和式 34 权重。
from Mask.geometry import Polyline3D, RotationMatrix3D, Vector3  # 导入三维向量、旋转矩阵和折线以构建空间几何。
from Mask.infrastructure import InspectionCheckpoint, RoadNode, StopSite, TowerComponent, TransmissionTower  # 导入道路、停靠、杆塔、构件和检查点元素。
from Mask.inspection import InspectionTask, ObservationPose  # 导入构件巡检任务和候选观测位姿元素。
from Mask.mobility import GroundVehicle, UnmannedAerialVehicle  # 导入地面车辆和车载无人机元素。
from Mask.observation_relation import ObservationAdmissibility, ObservationAssessment, ObservationLimits  # 导入候选观测位姿综合准入关系和评估证据。
from Mask.observation_selection import ObservationNodeAssignment, ObservationSelectionPattern, enumerate_observation_selection_patterns  # 导入服务列内嵌位姿选择、轨迹节点关联和外层枚举器。
from Mask.road_relation import RoadLeg  # 导入地形感知道路弧关系。
from Mask.service_column import LaunchRecoveryPair, ServiceColumn  # 导入起飞回收配对和服务列关系。
from Mask.terrain import CylinderObstacle, ObstacleField, TerrainSurface  # 导入地形、显式障碍和障碍场元素。
from Mask.trajectory_optimization import TrajectoryDecision, TrajectoryOptimizationConfig, TrajectoryOptimizationRequest, TrajectoryOptimizer, TrajectoryPrimitiveSpec  # 导入连续转录请求和依赖注入协议。
from Mask.weather import EnergyModelParameters, WeatherScenarioSet, WeatherState  # 导入气象状态、场景集和概率能耗参数。


def _vector3(values: object, value_path: str) -> Vector3:  # 定义三维数值列表转换函数。
    """将长度为三的数值列表转换为统一三维向量。"""  # 解释三维向量转换规则。
    if not isinstance(values, list) or len(values) != 3:  # 判断输入值是否属于长度为三的列表。
        raise ValueError(f"三维向量配置必须包含三个分量: {value_path}")  # 抛出包含数据路径的三维向量异常。
    return Vector3(float(values[0]), float(values[1]), float(values[2]))  # 按固定坐标顺序构造三维向量。


def _rotation_matrix3d(values: object, value_path: str) -> RotationMatrix3D:  # 定义三维旋转矩阵配置转换函数。
    """将三行三列数值列表转换为经过群性质校验的旋转矩阵。"""  # 解释旋转矩阵转换规则。
    if not isinstance(values, list) or len(values) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in values):  # 判断输入是否属于三行三列数组。
        raise ValueError(f"三维旋转矩阵配置必须包含三行三列: {value_path}")  # 抛出包含数据路径的矩阵形状异常。
    rows = tuple(tuple(float(value) for value in row) for row in values)  # 将嵌套列表转换为不可变浮点行记录。
    return RotationMatrix3D(rows)  # 构造并返回经过正交性和行列式校验的旋转矩阵。


def _flight_endpoint_position(system: SceneSystem, endpoint_id: str, required_clearance: float) -> Vector3:  # 定义飞行关系端点位置解析函数。
    """返回位姿绝对位置或停靠点上方满足净空的起飞回收位置。"""  # 解释飞行端点位置边界。
    endpoint = system.get_element(endpoint_id)  # 从统一场景标识空间读取飞行端点对象。
    if isinstance(endpoint, ObservationPose):  # 判断当前飞行端点是否属于候选观测位姿。
        return endpoint.position  # 返回候选观测位姿绝对三维位置。
    if isinstance(endpoint, StopSite):  # 判断当前飞行端点是否属于地面停靠点。
        return Vector3(endpoint.position.x, endpoint.position.y, endpoint.position.z + required_clearance)  # 返回停靠点上方满足规定净空的起飞回收位置。
    raise TypeError(f"飞行端点必须属于停靠点或观测位姿: {endpoint_id}")  # 拒绝不能映射为空间位置的场景对象。


def _trajectory_config(physical_config: dict[str, object]) -> TrajectoryOptimizationConfig:  # 定义物理配置到连续求解配置的转换函数。
    """读取存在时的 SLSQP 和独立审计参数并提供稳定默认值。"""  # 解释旧配置兼容边界。
    raw_record = physical_config.get("trajectory_optimizer", {})  # 读取连续轨迹求解配置记录。
    if not isinstance(raw_record, dict):  # 判断连续求解配置是否属于对象。
        raise ValueError("trajectory_optimizer 配置必须为对象")  # 拒绝不能读取字段的配置值。
    return TrajectoryOptimizationConfig(float(raw_record.get("equality_tolerance", 1.0e-6)), float(raw_record.get("inequality_tolerance", 1.0e-6)), int(raw_record.get("dense_samples_per_interval", 7)), int(raw_record.get("maximum_iterations", 500)), float(raw_record.get("function_tolerance", 1.0e-9)))  # 返回经过统一验证的连续求解配置。


def _trajectory_request_from_primitives(candidate_id: str, primitives: tuple[FlightPrimitive, ...], observation_pattern: ObservationSelectionPattern, terrain: TerrainSurface, obstacle_field: ObstacleField, uav: UnmannedAerialVehicle, weather: WeatherState, energy_parameters: EnergyModelParameters, dynamic_limits: FlightDynamicLimits, objective_weights: TrajectoryObjectiveWeights, config: TrajectoryOptimizationConfig) -> TrajectoryOptimizationRequest:  # 定义固定认证原语到连续转录请求的转换函数。
    """将地形认证基线状态拼接为服务级 SLSQP 可行初值。"""  # 解释初值来源和服务级连续性。
    if len(primitives) != len(observation_pattern.assignments) + 1 or not primitives:  # 判断原语数量是否连接起飞、全部观测节点和回收端点。
        raise ValueError("连续轨迹初值原语数量与观测模式不一致")  # 拒绝不能形成服务级连续链的输入。
    positions: list[Vector3] = []  # 创建服务级节点位置初值。
    velocities: list[Vector3] = []  # 创建服务级节点速度初值。
    accelerations: list[Vector3] = []  # 创建服务级区间加速度初值。
    durations: list[float] = []  # 创建服务级区间时间初值。
    primitive_specs: list[TrajectoryPrimitiveSpec] = []  # 创建区间到下游原语的稳定切片映射。
    boundary_node_indices: list[int] = []  # 创建相邻原语共享观测节点索引。
    interval_cursor = 0  # 初始化服务级区间游标。
    for primitive_index, primitive in enumerate(primitives):  # 逐一拼接按任务顺序排列的固定认证原语。
        intervals = primitive.trajectory_intervals()  # 读取由式 22 和式 23 展开的可行初值区间。
        if not primitive.certified or not intervals:  # 判断当前初值原语是否缺少可用连续状态。
            raise ValueError(f"连续轨迹初值原语未认证: {primitive.object_id}")  # 拒绝不可供 SLSQP 初始化的原语。
        if not positions:  # 判断当前原语是否为服务级首条原语。
            positions.append(intervals[0].start_position)  # 保存服务级起飞节点位置。
            velocities.append(intervals[0].start_velocity)  # 保存服务级起飞节点速度。
        elif positions[-1].distance_to(intervals[0].start_position) > config.equality_tolerance or (velocities[-1] - intervals[0].start_velocity).norm() > config.equality_tolerance:  # 判断相邻原语是否在审计容差内共享同一位置速度状态。
            raise ValueError("连续轨迹初值原语边界状态不连续")  # 拒绝不能形成统一服务级状态的原语集合。
        primitive_specs.append(TrajectoryPrimitiveSpec(primitive.object_id, primitive.name, primitive.source_id, primitive.target_id, primitive.corridor_id, interval_cursor, len(intervals), primitive.minimum_clearance_margin))  # 登记当前原语区间切片。
        for interval in intervals:  # 逐一拼接当前原语显式区间状态。
            accelerations.append(interval.acceleration)  # 保存当前区间实际加速度。
            durations.append(interval.duration)  # 保存当前区间持续时间。
            positions.append(interval.end_position)  # 保存当前区间终止位置作为下一节点。
            velocities.append(interval.end_velocity)  # 保存当前区间终止速度作为下一节点。
            interval_cursor += 1  # 推进服务级区间游标。
        if primitive_index < len(primitives) - 1:  # 判断当前原语终点是否属于观测节点而非最终回收点。
            boundary_node_indices.append(interval_cursor)  # 记录当前观测节点在服务级状态中的索引。
    fixed_node_positions = [(0, positions[0])]  # 使用起飞位置初始化固定节点集合。
    fixed_node_positions.extend((node_index, assignment.node_position) for node_index, assignment in zip(boundary_node_indices, observation_pattern.assignments))  # 将式 19 选中位姿绑定到对应观测节点。
    fixed_node_positions.append((len(positions) - 1, positions[-1]))  # 固定最终回收节点位置。
    service_duration = sum(assignment.service_duration for assignment in observation_pattern.assignments)  # 汇总式 36 稳定观测服务时间。
    initial_decision = TrajectoryDecision(tuple(positions), tuple(velocities), tuple(accelerations), tuple(durations))  # 构造完整服务级连续初值。
    return TrajectoryOptimizationRequest(f"trajectory-{candidate_id}", tuple(primitive_specs), tuple(fixed_node_positions), tuple(boundary_node_indices), terrain, obstacle_field, uav, weather, energy_parameters, dynamic_limits, service_duration, objective_weights, initial_decision, config)  # 返回内层连续优化请求。


class SceneValidationError(ValueError):  # 定义严格场景闭合验证异常。
    """表示场景元素或关系存在未解决的引用和物理验证问题。"""  # 解释异常触发条件。

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:  # 定义场景验证异常初始化方法。
        """保存全部验证问题并在异常文本中列出稳定问题代码。"""  # 解释异常信息组成。
        self.issues = issues  # 保存不可变场景验证问题集合。
        issue_codes = ",".join(issue.code for issue in issues)  # 汇总全部场景验证问题代码。
        super().__init__(f"场景验证失败: {issue_codes}")  # 初始化包含全部问题代码的数值异常。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存场景构建结果。
class SceneBuildResult:  # 定义完整场景系统和拒绝审计记录的组合结果。
    """绑定通过引用验证的场景系统和未进入场景的拒绝记录。"""  # 解释场景构建结果内容。

    system: SceneSystem  # 保存完成登记和验证的场景系统。
    rejection_records: tuple[RejectionRecord, ...]  # 保存场景构建期间产生的拒绝记录。
    observation_assessments: tuple[ObservationAssessment, ...] = ()  # 保存全部候选位姿的式 10 至式 15 评估证据。
    service_columns: tuple[ServiceColumn, ...] = ()  # 保存通过物理和电池准入的服务列集合。
    road_options: tuple[VehicleRoadOption, ...] = ()  # 保存按车辆评估的道路物理系数集合。
    stop_options: tuple[VehicleStopOption, ...] = ()  # 保存按车辆评估的停靠点准入结果集合。
    time_horizon: float = 0.0  # 保存当前场景调度时间上界。
    rolling_window_seconds: float = 0.0  # 保存滚动重优化与提交前缀窗口。
    vehicle_states: tuple[RollingVehicleState, ...] = ()  # 保存车辆滚动位置状态。
    uav_states: tuple[RollingUAVState, ...] = ()  # 保存无人机滚动资源状态。
    state_default_diagnostics: tuple[str, ...] = ()  # 保存旧数据默认值诊断。
    energy_parameters: EnergyModelParameters | None = None


class SceneBuilder:  # 定义 Mask 对象登记、连接和系统验证的环境构建器。
    """按照元素优先和关系随后顺序构建场景并保留拒绝证据。"""  # 解释场景构建器职责。

    def __init__(self, scene_id: str, scene_name: str) -> None:  # 定义场景构建器初始化方法。
        """创建空场景系统和空拒绝记录列表。"""  # 解释构建器初始状态。
        self._system = SceneSystem(scene_id, scene_name)  # 创建用于登记 Mask 对象的场景系统。
        self._rejection_records: list[RejectionRecord] = []  # 创建场景构建拒绝记录列表。

    @classmethod  # 声明从结构化 Data 装配完整场景的替代构造方法。
    def from_project_data(cls, project_data: ProjectData, trajectory_optimizer: TrajectoryOptimizer | None = None, weather_scenario_path: Path | None = None, weather_split: str | None = None, *, corridor_factory=None, primitive_factory=None, column_factory=ServiceColumn, dynamic_limits_factory=FlightDynamicLimits) -> SceneBuildResult:  # 定义完整 Data 场景装配方法并允许注入内层连续求解器和正式天气发布。
        """按元素、关系、物理认证和服务列顺序构建完整 P0 场景。"""  # 解释完整场景装配顺序。
        scene = project_data.scene  # 读取经过严格结构检查的场景记录。
        config = project_data.config  # 读取经过严格结构检查的物理和算法配置。
        builder = cls(str(scene["scene_id"]), str(scene["scene_name"]))  # 使用场景稳定标识和名称创建空构建器。
        terrain_record = scene["terrain"]  # 读取规则高程网格地形记录。
        terrain = TerrainSurface(str(terrain_record["id"]), str(terrain_record["name"]), tuple(float(value) for value in terrain_record["x_coordinates_m"]), tuple(float(value) for value in terrain_record["y_coordinates_m"]), tuple(tuple(float(value) for value in row) for row in terrain_record["heights_m"]))  # 构造三维地形基础物理场。
        builder.add_element(terrain)  # 将地形物理场登记到统一场景系统。
        obstacle_field_record = scene["obstacle_field"]  # 读取显式障碍场配置记录。
        obstacles = tuple(CylinderObstacle(str(record["id"]), str(record["name"]), float(record["center_x_m"]), float(record["center_y_m"]), float(record["radius_m"]), float(record["minimum_elevation_m"]), float(record["maximum_elevation_m"])) for record in obstacle_field_record["obstacles"])  # 构造全部有限圆柱显式障碍。
        for obstacle in obstacles:  # 逐一登记显式三维障碍元素。
            builder.add_element(obstacle)  # 将当前显式障碍加入统一场景系统。
        obstacle_field = ObstacleField(str(obstacle_field_record["id"]), str(obstacle_field_record["name"]), obstacles)  # 构造供净空查询使用的显式障碍场。
        builder.add_element(obstacle_field)  # 将显式障碍场登记到统一场景系统。
        road_node_by_id: dict[str, RoadNode] = {}  # 创建道路节点稳定标识索引。
        for record in scene["road_nodes"]:  # 逐一构造带地形绝对高程的道路节点。
            x_value = float(record["x_m"])  # 读取当前道路节点横向坐标。
            y_value = float(record["y_m"])  # 读取当前道路节点纵向坐标。
            position = Vector3(x_value, y_value, terrain.height_at(x_value, y_value))  # 将道路节点水平坐标映射到三维地形表面。
            node = RoadNode(str(record["id"]), str(record["name"]), position, str(record["role"]))  # 构造当前三维道路节点。
            road_node_by_id[node.object_id] = node  # 将当前道路节点加入稳定标识索引。
            builder.add_element(node)  # 将当前道路节点登记到统一场景系统。
        stop_by_id: dict[str, StopSite] = {}  # 创建候选停靠点稳定标识索引。
        for record in scene["stop_sites"]:  # 逐一构造连接道路节点的候选停靠点。
            road_node_id = str(record["road_node_id"])  # 读取当前停靠点连接的道路节点标识。
            if road_node_id not in road_node_by_id:  # 判断当前停靠点道路节点引用是否缺失。
                raise ValueError(f"停靠点引用未知道路节点: {road_node_id}")  # 抛出明确的停靠点道路引用异常。
            node_position = road_node_by_id[road_node_id].position  # 读取当前停靠点连接道路节点的三维位置。
            stop = StopSite(str(record["id"]), str(record["name"]), road_node_id, node_position, float(record["footprint_radius_m"]))  # 构造只保存物理范围的当前候选停靠点。
            stop_by_id[stop.object_id] = stop  # 将当前停靠点加入稳定标识索引。
            builder.add_element(stop)  # 将当前停靠点登记到统一场景系统。
        vehicle_by_id: dict[str, GroundVehicle] = {}  # 创建地面车辆稳定标识索引。
        vehicle_states: list[RollingVehicleState] = []  # 创建车辆滚动状态集合。
        state_default_diagnostics: list[str] = []  # 创建默认值诊断集合。
        default_start_node_id = next(str(record["id"]) for record in scene["road_nodes"] if record["role"] == "depot_start")  # 读取旧场景默认起点。
        for record in scene["vehicles"]:  # 逐一构造地面车辆物理元素。
            vehicle = GroundVehicle(str(record["id"]), str(record["name"]), float(record["maximum_road_grade_ratio"]), float(record["road_speed_mps"]), float(record["road_slope_sensitivity"]), float(record["maximum_stop_grade_ratio"]), float(record["minimum_stop_area_m2"]))  # 构造当前地面车辆。
            vehicle_by_id[vehicle.object_id] = vehicle  # 将当前车辆加入稳定标识索引。
            builder.add_element(vehicle)  # 将当前车辆登记到统一场景系统。
            current_node_id = str(record.get("current_road_node_id", default_start_node_id))  # 读取车辆当前道路节点。
            vehicle_states.append(RollingVehicleState(vehicle.object_id, current_node_id))  # 登记车辆滚动状态。
            if "current_road_node_id" not in record:  # 判断是否使用旧场景默认位置。
                state_default_diagnostics.append("vehicle.current_road_node_id=depot_start")  # 记录车辆默认位置来源。
        uav_by_id: dict[str, UnmannedAerialVehicle] = {}  # 创建车载无人机稳定标识索引。
        uav_states: list[RollingUAVState] = []  # 创建无人机滚动状态集合。
        for record in scene["uavs"]:  # 逐一构造车载无人机物理元素。
            uav = UnmannedAerialVehicle(str(record["id"]), str(record["name"]), str(record["carrier_vehicle_id"]), float(record["maximum_speed_mps"]), float(record["maximum_thrust_acceleration_mps2"]), float(record["minimum_clearance_m"]), float(record["hover_power_w"]), float(record["mass_kg"]), float(record["battery_state_of_charge_j"]), float(record["battery_reserve_ratio"]))  # 构造当前车载无人机。
            uav_by_id[uav.object_id] = uav  # 将当前无人机加入稳定标识索引。
            builder.add_element(uav)  # 将当前无人机登记到统一场景系统。
            uav_states.append(RollingUAVState(uav.object_id, bool(record.get("available", True)), uav.battery_state_of_charge, float(record.get("replacement_pack_energy_j", uav.battery_state_of_charge)), float(record.get("turnaround_time_s", 0.0))))  # 登记无人机资源状态。
            if not {"available", "replacement_pack_energy_j", "turnaround_time_s"}.issubset(record):  # 判断是否缺少新资源字段。
                state_default_diagnostics.append("uav.resource_state=legacy_defaults")  # 记录无人机旧数据默认值。
        tower_by_id: dict[str, TransmissionTower] = {}  # 创建输电杆塔稳定标识索引。
        for record in scene["towers"]:  # 逐一构造地形基座上的输电杆塔。
            x_value = float(record["x_m"])  # 读取当前杆塔基座横向坐标。
            y_value = float(record["y_m"])  # 读取当前杆塔基座纵向坐标。
            base_position = Vector3(x_value, y_value, terrain.height_at(x_value, y_value))  # 将杆塔水平位置映射到地形基座高程。
            rotation = _rotation_matrix3d(record["rotation_matrix"], f"towers.{record['id']}.rotation_matrix")  # 读取并校验杆塔完整三维旋转矩阵。
            tower = TransmissionTower(str(record["id"]), str(record["name"]), base_position, rotation)  # 构造当前输电杆塔元素。
            tower_by_id[tower.object_id] = tower  # 将当前杆塔加入稳定标识索引。
            builder.add_element(tower)  # 将当前杆塔登记到统一场景系统。
        component_by_id: dict[str, TowerComponent] = {}  # 创建杆塔构件稳定标识索引。
        for record in scene["components"]:  # 逐一构造杆塔局部坐标系中的构件。
            component = TowerComponent(str(record["id"]), str(record["name"]), str(record["tower_id"]), _vector3(record["local_origin_m"], f"components.{record['id']}.local_origin_m"), str(record["component_type"]))  # 构造当前杆塔构件元素。
            component_by_id[component.object_id] = component  # 将当前构件加入稳定标识索引。
            builder.add_element(component)  # 将当前构件登记到统一场景系统。
        checkpoint_by_id: dict[str, InspectionCheckpoint] = {}  # 创建构件检查点稳定标识索引。
        for record in scene["checkpoints"]:  # 逐一构造构件局部坐标系中的检查点。
            checkpoint = InspectionCheckpoint(str(record["id"]), str(record["name"]), str(record["component_id"]), _vector3(record["local_position_m"], f"checkpoints.{record['id']}.local_position_m"), _vector3(record["surface_normal"], f"checkpoints.{record['id']}.surface_normal"))  # 构造当前构件检查点元素。
            checkpoint_by_id[checkpoint.object_id] = checkpoint  # 将当前检查点加入稳定标识索引。
            builder.add_element(checkpoint)  # 将当前检查点登记到统一场景系统。
        pose_by_id: dict[str, ObservationPose] = {}  # 创建候选观测位姿稳定标识索引。
        for record in scene["observation_poses"]:  # 逐一构造相对于检查点定义的候选观测位姿。
            checkpoint_id = str(record["checkpoint_id"])  # 读取当前候选位姿服务的检查点标识。
            if checkpoint_id not in checkpoint_by_id:  # 判断当前候选位姿检查点引用是否缺失。
                raise ValueError(f"候选观测位姿引用未知检查点: {checkpoint_id}")  # 抛出明确的位姿检查点引用异常。
            checkpoint = checkpoint_by_id[checkpoint_id]  # 读取候选位姿对应的检查点。
            component = component_by_id[checkpoint.component_id]  # 读取检查点所属的杆塔构件。
            tower = tower_by_id[component.tower_id]  # 读取定义局部观测坐标系的杆塔。
            checkpoint_position = checkpoint.global_position(builder._system)  # 计算当前检查点的全局三维位置。
            local_offset = _vector3(record["offset_from_checkpoint_m"], f"observation_poses.{record['id']}.offset_from_checkpoint_m")  # 读取杆塔局部坐标系中的观测偏移。
            local_camera_axis = _vector3(record["camera_axis"], f"observation_poses.{record['id']}.camera_axis")  # 读取杆塔局部坐标系中的相机光轴。
            pose_position = checkpoint_position + tower.rotate_local_vector(local_offset)  # 将观测偏移随杆塔姿态旋转到全局坐标系。
            pose = ObservationPose(str(record["id"]), str(record["name"]), checkpoint_id, pose_position, tower.rotate_local_vector(local_camera_axis))  # 使用同一杆塔姿态构造全局相机光轴。
            pose_by_id[pose.object_id] = pose  # 将当前候选位姿加入稳定标识索引。
            builder.add_element(pose)  # 将当前候选位姿登记到统一场景系统。
        task_by_id: dict[str, InspectionTask] = {}  # 创建构件巡检任务稳定标识索引。
        for record in scene["tasks"]:  # 逐一构造构件检查点巡检任务。
            task = InspectionTask(str(record["id"]), str(record["name"]), str(record["checkpoint_id"]), float(record["priority"]), float(record["service_duration_s"]))  # 构造当前构件级巡检任务。
            task_by_id[task.object_id] = task  # 将当前巡检任务加入稳定标识索引。
            builder.add_element(task)  # 将当前巡检任务登记到统一场景系统。
        weather_by_id: dict[str, WeatherState] = {}  # 创建动态气象状态稳定标识索引。
        for record in scene["weather_states"]:  # 逐一构造滚动时域气象状态。
            weather = WeatherState(str(record["id"]), str(record["name"]), int(record["period_index"]), _vector3(record["wind_vector_mps"], f"weather_states.{record['id']}.wind_vector_mps"), float(record["temperature_c"]))  # 构造当前动态气象状态。
            weather_by_id[weather.object_id] = weather  # 将当前气象状态加入稳定标识索引。
            builder.add_element(weather)  # 将当前气象状态登记到统一场景系统。
        if (weather_scenario_path is None) != (weather_split is None):  # 判断天气发布路径和数据划分是否成对提供。
            raise ValueError("weather scenario path and split must be provided together")  # 拒绝无法确定场景概率的数据入口。
        weather_scenarios_by_period: dict[int, WeatherScenarioSet] = {}  # 创建滚动时段到完整气象场景集的索引。
        if weather_scenario_path is not None and weather_split is not None:  # 判断当前环境是否启用正式数据场景认证。
            for weather in weather_by_id.values():  # 逐一为场景中声明的滚动时段加载三实现气象集。
                if weather.period_index in weather_scenarios_by_period:  # 判断同一时段是否已经建立场景集。
                    continue  # 复用同一滚动时段的概率实现集合。
                scenario_set = load_weather_scenario_set(weather_scenario_path, weather_split, weather.period_index, f"{scene['scene_id']}:period-{weather.period_index}")  # 从同一正式发布表构造当前时段气象集。
                weather_scenarios_by_period[weather.period_index] = scenario_set  # 保存当前滚动时段的完整场景集。
                for scenario in scenario_set.scenarios:  # 逐一登记服务列将引用的气象实现状态。
                    builder.add_element(scenario.state)  # 将场景气象状态加入统一场景标识空间。
        physical_config = config["physical"]  # 读取 Agent 无关的物理计算配置。
        stop_spacing = float(physical_config["stop_sample_spacing_m"])  # 读取停靠点地形坡度采样间距。
        stop_assessment_by_id = {stop.object_id: stop.assess_terrain(terrain, stop_spacing) for stop in stop_by_id.values()}  # 对每个停靠点只计算一次地形坡比和几何面积。
        stop_options = tuple(VehicleStopOption.from_assessment(vehicle, stop.object_id, stop_assessment_by_id[stop.object_id]) for vehicle in vehicle_by_id.values() for stop in stop_by_id.values())  # 生成全部车辆停靠组合的准入参数。
        accessible_stop_pairs = {(option.vehicle_id, option.stop_site_id) for option in stop_options if option.accessible}  # 汇总允许进入调度模型的车辆停靠组合。
        for option in stop_options:  # 逐一保存未通过车辆能力认证的停靠组合证据。
            if not option.accessible:  # 判断当前车辆停靠组合是否被拒绝。
                pair_id = f"{option.vehicle_id}:{option.stop_site_id}"  # 构造不会在多车辆场景中混淆的组合标识。
                builder.record_rejection(RejectionRecord(pair_id, "vehicle_stop_option", "vehicle_stop_capability_limit", ("maximum_stop_grade_ratio", "minimum_stop_area_m2", "footprint_radius_m", "stop_sample_spacing_m"), "Data/scenarios/p0_minimal/scene.json"))  # 记录车辆停靠坡比或面积能力拒绝证据。
        road_options: list[VehicleRoadOption] = []  # 创建按车辆认证的道路物理系数列表。
        road_spacing = float(physical_config["road_sample_spacing_m"])  # 读取道路地形剖面最大采样间距。
        for record in scene["road_legs"]:  # 逐一构造并评估有向道路弧。
            points = tuple(Vector3(float(point[0]), float(point[1]), 0.0) for point in record["horizontal_polyline_m"])  # 将道路水平折线坐标转换为三维零高程点。
            road_leg = RoadLeg(str(record["id"]), str(record["name"]), str(record["source_node_id"]), str(record["target_node_id"]), Polyline3D(points), float(record["surface_factor"]))  # 构造当前有向道路弧关系。
            builder.add_relation(road_leg)  # 将当前道路弧登记到统一场景系统。
            for vehicle in vehicle_by_id.values():  # 逐一评估当前道路弧对每辆车辆的通行性。
                assessment = road_leg.assess(terrain, vehicle, road_spacing)  # 计算三维长度、最大坡度和地形修正时间。
                option = VehicleRoadOption(vehicle.object_id, road_leg.object_id, road_leg.source_id, road_leg.target_id, assessment.travel_time, assessment.length_3d, assessment.accessible)  # 构造供 Agent 消费的道路认证系数。
                road_options.append(option)  # 将当前车辆道路认证系数加入列表。
                if not assessment.accessible:  # 判断当前车辆是否无法通过该道路弧。
                    builder.record_rejection(RejectionRecord(road_leg.object_id, "road_leg", "vehicle_grade_limit", ("maximum_road_grade_ratio", "road_sample_spacing_m"), "Data/scenarios/p0_minimal/scene.json"))  # 记录道路坡比拒绝证据。
        observation_config = physical_config["observation_limits"]  # 读取候选观测位姿统一准入阈值。
        observation_limits = ObservationLimits(float(observation_config["minimum_distance_m"]), float(observation_config["maximum_distance_m"]), float(observation_config["maximum_incidence_angle_rad"]), float(observation_config["maximum_pointing_error_rad"]), float(observation_config["minimum_terrain_clearance_m"]), float(observation_config["minimum_obstacle_clearance_m"]), float(observation_config["line_of_sight_spacing_m"]))  # 构造观测距离、角度、净空和视线阈值。
        admitted_pose_ids: set[str] = set()  # 创建通过综合准入评估的候选观测位姿集合。
        observation_assessments: list[ObservationAssessment] = []  # 创建全部候选观测位姿的有序评估证据列表。
        assessment_by_pose_id: dict[str, ObservationAssessment] = {}  # 创建供服务列选择模式复用的位姿评估索引。
        for pose in pose_by_id.values():  # 逐一构造和评估检查点到候选位姿的关系。
            relation = ObservationAdmissibility(f"observation-{pose.object_id}", f"{pose.name}准入关系", pose.checkpoint_id, pose.object_id, observation_limits)  # 构造当前候选位姿综合准入关系。
            builder.add_relation(relation)  # 将当前观测准入关系登记到统一场景系统。
            assessment = relation.assess(builder._system, terrain, obstacle_field)  # 计算当前候选位姿几何、净空和视线条件。
            observation_assessments.append(assessment)  # 将当前候选完整评估加入 Env 可追溯证据。
            assessment_by_pose_id[pose.object_id] = assessment  # 按候选位姿标识登记评估以供服务列选择构造。
            if assessment.admissible:  # 判断当前候选观测位姿是否通过全部准入条件。
                admitted_pose_ids.add(pose.object_id)  # 将当前候选位姿加入准入集合。
            else:  # 处理未通过至少一项准入条件的候选位姿。
                builder.record_rejection(RejectionRecord(pose.object_id, "observation_pose", "observation_admissibility_failed", assessment.failed_conditions, "Data/scenarios/p0_minimal/scene.json"))  # 记录候选观测位姿失败条件。
        energy_record = physical_config["energy_model"]  # 读取服务列概率能耗参数记录。
        energy_parameters = EnergyModelParameters(float(energy_record["motion_coefficient"]), float(energy_record["acceleration_coefficient"]), float(energy_record["climb_coefficient"]), float(energy_record["launch_recovery_energy_j"]), float(energy_record["uncertainty_base_j"]), float(energy_record["uncertainty_wind_coefficient"]), float(energy_record["uncertainty_temperature_coefficient"]), float(energy_record["reference_temperature_c"]), float(energy_record["violation_probability"]))  # 构造服务列能耗和不确定性模型参数。
        clearance_spacing = float(physical_config["flight_clearance_sample_spacing_m"])  # 读取飞行航路净空认证最大采样间距。
        minimum_segment_time = float(physical_config.get("minimum_segment_time_s", 0.0))  # 读取第五部分航段时间下界并为旧配置保留零下界。
        maximum_segment_time = float(physical_config.get("maximum_segment_time_s", math.inf))  # 读取第五部分航段时间上界并为旧配置保留正无穷上界。
        maximum_dynamic_subsegments = int(physical_config.get("maximum_dynamic_subsegments_per_leg", 12))  # 读取单个几何航路段允许的有界动力学离散数量。
        maximum_observation_speed = float(physical_config.get("maximum_observation_speed_mps", 0.0))  # 读取第五部分稳定观测节点速度上界。
        objective_record = physical_config.get("trajectory_objective", {"dynamic_time_weight": 1.0, "expected_energy_weight": 0.0})  # 读取式 34 连续轨迹目标权重记录。
        if not isinstance(objective_record, dict):  # 判断式 34 权重记录是否属于对象。
            raise ValueError("trajectory_objective 配置必须为对象")  # 拒绝不能读取权重字段的配置。
        trajectory_objective_weights = TrajectoryObjectiveWeights(float(objective_record["dynamic_time_weight"]), float(objective_record["expected_energy_weight"]))  # 构造式 34 时间能耗目标权重。
        trajectory_optimization_config = _trajectory_config(physical_config)  # 构造连续求解和独立审计配置。
        expanded_candidates: list[dict[str, object]] = []  # 创建将显式候选和外层枚举模板统一展开后的候选列表。
        for candidate in project_data.service_candidates["candidates"]:  # 逐一处理 Data 中的原始候选或枚举模板。
            if "task_pose_sequence" in candidate:  # 判断当前记录是否已经显式给出任务位姿序列。
                expanded_candidates.append(candidate)  # 保留显式候选作为确定性回归输入。
                continue  # 处理下一条原始候选。
            template_id = str(candidate["id"])  # 读取外层枚举模板稳定标识。
            task_ids = tuple(str(task_id) for task_id in candidate["task_ids"])  # 读取模板声明的任务集合。
            assignments_by_task: dict[str, tuple[ObservationNodeAssignment, ...]] = {}  # 创建按任务组织的全部准入位姿分配映射。
            for task_id in task_ids:  # 逐一收集当前任务对应的准入候选位姿。
                if task_id not in task_by_id:  # 判断模板是否引用未知任务。
                    raise ValueError(f"服务列枚举模板包含未知任务引用: {template_id}:{task_id}")  # 抛出可追踪任务引用异常。
                task = task_by_id[task_id]  # 读取当前巡检任务。
                assignments = tuple(ObservationNodeAssignment.from_assessment(1, task, pose, assessment_by_pose_id[pose.object_id]) for pose in pose_by_id.values() if pose.checkpoint_id == task.checkpoint_id and pose.object_id in admitted_pose_ids)  # 从 M2 综合准入结果创建当前任务全部可选位姿分配。
                assignments_by_task[task_id] = assignments  # 保存当前任务准入位姿选择集合。
            try:  # 将没有准入位姿等领域失败转换为候选拒绝证据。
                patterns = enumerate_observation_selection_patterns(template_id, task_ids, assignments_by_task, bool(candidate["enumerate_task_orders"]))  # 枚举任务顺序和位姿笛卡尔积。
            except ValueError as error:  # 捕获不能形成任何式 16 至式 20 模式的模板。
                builder.record_rejection(RejectionRecord(template_id, "service_column_template", str(error), ("task_ids", "enumerate_task_orders"), "Data/scenarios/p0_minimal/service_candidates.json"))  # 保存外层枚举模板拒绝证据。
                continue  # 跳过不能展开的模板。
            for pattern in patterns:  # 逐一将稳定模式转换为现有显式候选装配协议。
                expanded_candidates.append({"id": pattern.pattern_id, "name": f"{candidate['name']}[{'+'.join(pattern.pose_ids)}]", "launch_stop_id": candidate["launch_stop_id"], "recovery_stop_id": candidate["recovery_stop_id"], "uav_id": candidate["uav_id"], "weather_state_id": candidate["weather_state_id"], "task_pose_sequence": [{"task_id": assignment.task_id, "pose_id": assignment.pose_id} for assignment in pattern.assignments]})  # 保存一个可独立连续求解的显式离散组合。
        admitted_columns: list[ServiceColumn] = []  # 创建通过全部物理和电池认证的服务列列表。
        for candidate in expanded_candidates:  # 逐一处理显式候选和外层枚举生成的离散组合。
            candidate_id = str(candidate["id"])  # 读取当前服务列候选稳定标识。
            launch_stop_id = str(candidate["launch_stop_id"])  # 读取当前候选起飞停靠点标识。
            recovery_stop_id = str(candidate["recovery_stop_id"])  # 读取当前候选回收停靠点标识。
            uav_id = str(candidate["uav_id"])  # 读取当前候选执行无人机标识。
            weather_id = str(candidate["weather_state_id"])  # 读取当前候选气象状态标识。
            if launch_stop_id not in stop_by_id or recovery_stop_id not in stop_by_id or uav_id not in uav_by_id or weather_id not in weather_by_id:  # 判断当前候选基础引用是否存在。
                raise ValueError(f"服务列候选包含未知基础引用: {candidate_id}")  # 抛出包含候选标识的引用异常。
            # Candidate missions start after the modeled battery replacement.
            # Keep measured battery state unchanged on the registered UAV object.
            pack_energy = next(state.replacement_pack_energy for state in uav_states if state.uav_id == uav_id)
            uav = replace(uav_by_id[uav_id], battery_state_of_charge=pack_energy)
            pair = LaunchRecoveryPair(f"pair-{candidate_id}", f"{candidate['name']}起飞回收配对", launch_stop_id, recovery_stop_id, uav_id)  # 构造当前候选起飞回收配对。
            builder.add_relation(pair)  # 将当前起飞回收配对登记到统一场景系统。
            observation_assignments: list[ObservationNodeAssignment] = []  # 创建当前候选有序观测节点分配。
            candidate_reasons: list[str] = []  # 创建当前候选预认证拒绝原因列表。
            for node_index, sequence_item in enumerate(candidate["task_pose_sequence"], start=1):  # 逐一解析当前候选任务位姿项并分配正观测节点索引。
                task_id = str(sequence_item["task_id"])  # 读取当前序列项巡检任务标识。
                pose_id = str(sequence_item["pose_id"])  # 读取当前序列项候选观测位姿标识。
                if task_id not in task_by_id or pose_id not in pose_by_id:  # 判断当前任务或位姿引用是否缺失。
                    raise ValueError(f"服务列候选包含未知任务位姿引用: {candidate_id}")  # 抛出包含候选标识的任务位姿引用异常。
                if pose_id not in admitted_pose_ids:  # 判断当前候选是否使用未通过准入的观测位姿。
                    candidate_reasons.append("unadmitted_observation_pose")  # 记录未准入观测位姿拒绝原因。
                else:  # 处理通过式 10 至式 15 准入评估的候选位姿。
                    assignment = ObservationNodeAssignment.from_assessment(node_index, task_by_id[task_id], pose_by_id[pose_id], assessment_by_pose_id[pose_id])  # 将任务、位姿、节点状态和准入证据闭合为式 16 至式 20 选中项。
                    observation_assignments.append(assignment)  # 将当前选中项加入服务列有序观测模式。
            carrier_vehicle_id = uav.carrier_vehicle_id  # 读取当前服务列对应的车辆级停靠能力主体。
            if (carrier_vehicle_id, launch_stop_id) not in accessible_stop_pairs or (carrier_vehicle_id, recovery_stop_id) not in accessible_stop_pairs:  # 判断起飞和回收组合是否均对搭载车辆可用。
                candidate_reasons.append("unavailable_vehicle_stop_pair")  # 记录车辆停靠组合不可用拒绝原因。
            if candidate_reasons:  # 判断当前候选是否已经违反观测位姿或车辆停靠前置准入。
                builder.record_rejection(RejectionRecord(candidate_id, "service_column", ";".join(dict.fromkeys(candidate_reasons)), ("task_pose_sequence", "launch_stop_id", "recovery_stop_id"), "Data/scenarios/p0_minimal/service_candidates.json"))  # 记录未进入航路认证的服务列候选拒绝证据。
                continue  # 跳过不能形成完整观测选择模式的候选。
            observation_pattern = ObservationSelectionPattern(tuple(observation_assignments))  # 构造式 16 至式 20 的不可变列内嵌选择模式。
            endpoint_ids = (launch_stop_id, *observation_pattern.pose_ids, recovery_stop_id)  # 构造连接起飞点、选中位姿序列和回收点的端点标识。
            dynamic_limits = dynamic_limits_factory(uav.maximum_speed, uav.maximum_thrust_acceleration, 9.81, minimum_segment_time, maximum_segment_time, maximum_observation_speed, maximum_dynamic_subsegments)  # 构造包含式 21、式 24、式 25 和式 31 参数的二阶质点动力学限制。
            primitives: list[FlightPrimitive] = []  # 创建当前服务列有序飞行原语列表。
            for segment_index, (source_id, target_id) in enumerate(zip(endpoint_ids[:-1], endpoint_ids[1:])):  # 逐一认证相邻服务端点之间的飞行关系。
                start_position = _flight_endpoint_position(builder._system, source_id, uav.minimum_clearance)  # 计算当前航段起点三维位置。
                end_position = _flight_endpoint_position(builder._system, target_id, uav.minimum_clearance)  # 计算当前航段终点三维位置。
                corridor_id = f"corridor-{candidate_id}-{segment_index}"  # 构造当前候选航段的稳定航路标识。
                corridor = (corridor_factory or FlightCorridor.certify)(corridor_id, f"{candidate['name']}航路{segment_index}", source_id, target_id, start_position, end_position, terrain, obstacle_field, uav.minimum_clearance, clearance_spacing)  # 执行地形和显式障碍净空认证。
                primitive = (primitive_factory or FlightPrimitive.from_corridor)(f"primitive-{candidate_id}-{segment_index}", f"{candidate['name']}原语{segment_index}", corridor, dynamic_limits)  # 将航路转换为二阶质点最短时间飞行原语。
                primitives.append(primitive)  # 将当前飞行原语加入有序原语列表。
                if corridor.certified and primitive.certified:  # 判断当前航路和动力学原语是否均通过第五部分认证。
                    builder.add_relation(corridor)  # 将通过认证的飞行航路登记到统一场景系统。
                    if trajectory_optimizer is None:  # 判断当前构建是否属于未注入连续求解器的兼容基线路径。
                        builder.add_relation(primitive)  # 仅在兼容路径登记固定认证飞行原语。
                elif corridor.certified:  # 处理空间航路可行但航段时间界不可行的飞行原语。
                    candidate_reasons.append(primitive.rejection_reason)  # 将动力学原语稳定拒绝原因加入候选拒绝列表。
                    builder.add_relation(corridor)  # 保留已经通过净空认证的空间航路证据。
                    builder.record_rejection(RejectionRecord(primitive.object_id, "flight_primitive", primitive.rejection_reason, ("minimum_segment_time_s", "maximum_segment_time_s"), "Data/configs/p0_minimal.json"))  # 记录第五部分航段时间认证拒绝证据。
                else:  # 处理未通过净空认证的飞行航路和原语。
                    candidate_reasons.append(corridor.rejection_reason)  # 将航路稳定拒绝原因加入候选拒绝列表。
                    builder.record_rejection(RejectionRecord(corridor.object_id, "flight_corridor", corridor.rejection_reason, ("flight_clearance_sample_spacing_m", "minimum_clearance_m"), "Data/scenarios/p0_minimal/service_candidates.json"))  # 记录未认证飞行航路证据。
                    builder.record_rejection(RejectionRecord(primitive.object_id, "flight_primitive", primitive.rejection_reason, ("corridor_id",), "Data/scenarios/p0_minimal/service_candidates.json"))  # 记录未认证飞行原语证据。
            if candidate_reasons:  # 判断当前候选是否已经存在位姿、停靠或航路拒绝原因。
                builder.record_rejection(RejectionRecord(candidate_id, "service_column", ";".join(dict.fromkeys(candidate_reasons)), ("task_pose_sequence", "launch_stop_id", "recovery_stop_id"), "Data/scenarios/p0_minimal/service_candidates.json"))  # 记录未进入组装阶段的服务列候选拒绝证据。
                continue  # 跳过当前未通过前置物理认证的服务列候选。
            service_primitives = tuple(primitives)  # 默认使用固定认证原语作为兼容基线或连续求解初值。
            certification_scenarios = weather_scenarios_by_period.get(weather_by_id[weather_id].period_index)  # 读取与候选名义时段对应的正式气象场景集。
            trajectory_weather = certification_scenarios.scenarios[0].state if certification_scenarios is not None else weather_by_id[weather_id]  # 使连续轨迹名义状态来自同一数据发布而非独立天气代理。
            if trajectory_optimizer is not None:  # 判断当前构建是否要求执行论文第九部分内层连续优化。
                trajectory_request = _trajectory_request_from_primitives(candidate_id, service_primitives, observation_pattern, terrain, obstacle_field, uav, trajectory_weather, energy_parameters, dynamic_limits, trajectory_objective_weights, trajectory_optimization_config)  # 从认证初值构造式 21 至式 40 连续问题。
                trajectory_result = trajectory_optimizer.solve(trajectory_request)  # 调用注入的算法层连续求解器。
                if not trajectory_result.success:  # 判断连续求解或独立审计是否失败。
                    reason = f"{trajectory_result.status}:{trajectory_result.message}"  # 组合稳定状态和求解器消息形成拒绝原因。
                    builder.record_rejection(RejectionRecord(candidate_id, "continuous_trajectory", reason, ("trajectory_optimizer", "maximum_equality_residual", "minimum_inequality_margin"), "Data/configs/p0_minimal.json"))  # 保存内层连续求解失败证据。
                    continue  # 失败关闭且不生成服务列。
                service_primitives = trajectory_result.primitives  # 使用经过独立审计的优化原语替换固定初值。
                for primitive in service_primitives:  # 逐一登记正式连续优化原语关系。
                    builder.add_relation(primitive)  # 将优化状态、目标和残差证据加入统一场景系统。
            column = column_factory.assemble_scenarios(candidate_id, str(candidate["name"]), pair, certification_scenarios, observation_pattern, service_primitives, uav, energy_parameters) if certification_scenarios is not None else column_factory.assemble(candidate_id, str(candidate["name"]), pair, weather_by_id[weather_id], observation_pattern, service_primitives, uav, energy_parameters)  # 正式路径逐场景认证，开发路径保留显式单状态兼容入口。
            if column.admitted:  # 判断当前服务列是否通过电池概率安全准入。
                admitted_columns.append(column)  # 将当前服务列加入准入列表。
                builder.add_relation(column)  # 将当前准入服务列登记到统一场景系统。
            else:  # 处理超过概率电池安全限制的服务列。
                builder.record_rejection(RejectionRecord(column.object_id, "service_column", ";".join(column.rejection_reasons), ("energy_quantile", "battery_state_of_charge_j", "battery_reserve_ratio"), "Data/scenarios/p0_minimal/service_candidates.json"))  # 记录服务列概率能量拒绝证据。
        base_result = builder.finalize(strict=True)  # 验证全部已登记元素和准入关系的引用闭合性。
        mission_horizon, rolling_window = parse_time_controls(config)  # 解析正式任务总时域与滚动更新窗口，同时兼容旧配置。
        return SceneBuildResult(system=base_result.system, rejection_records=base_result.rejection_records, observation_assessments=tuple(observation_assessments), service_columns=tuple(admitted_columns), road_options=tuple(road_options), stop_options=stop_options, time_horizon=mission_horizon, rolling_window_seconds=rolling_window, vehicle_states=tuple(vehicle_states), uav_states=tuple(uav_states), state_default_diagnostics=tuple(dict.fromkeys(state_default_diagnostics)), energy_parameters=energy_parameters)  # 返回完整场景、观测证据、准入列和两个时间控制。

    def add_element(self, element: SceneElement) -> None:  # 定义单个场景元素登记方法。
        """将一个 Mask 场景元素登记到共享标识空间。"""  # 解释元素登记行为。
        self._system.add_element(element)  # 复用场景系统的元素登记和重复检查。

    def add_relation(self, relation: SceneRelation) -> None:  # 定义单个场景关系登记方法。
        """将一个 Mask 场景关系登记到共享标识空间。"""  # 解释关系登记行为。
        self._system.add_relation(relation)  # 复用场景系统的关系登记和重复检查。

    def record_rejection(self, rejection: RejectionRecord) -> None:  # 定义物理对象拒绝记录方法。
        """保存未进入准入集合的对象及其 Data 来源和拒绝原因。"""  # 解释拒绝记录行为。
        self._rejection_records.append(rejection)  # 将拒绝记录按发生顺序加入审计列表。

    def finalize(self, strict: bool = True) -> SceneBuildResult:  # 定义场景构建完成方法。
        """验证全部登记对象并在严格模式下拒绝任何未闭合问题。"""  # 解释场景完成行为。
        issues = self._system.validate()  # 汇总全部元素和关系的场景验证问题。
        if strict and issues:  # 判断严格模式下是否存在未解决验证问题。
            raise SceneValidationError(issues)  # 抛出保存全部问题的场景验证异常。
        return SceneBuildResult(self._system, tuple(self._rejection_records))  # 返回场景系统和不可变拒绝记录集合。
