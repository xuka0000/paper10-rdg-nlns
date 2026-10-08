"""定义式 21 至式 40 的连续轨迹转录对象、公式计算和独立审计。"""  # 说明本文件只承担算法无关的数学模型职责。

import math  # 导入数学函数以验证连续状态和汇总约束证据。
from dataclasses import dataclass  # 导入数据类工具以保存不可变转录对象。
from typing import Protocol  # 导入协议类型以允许 Env 注入下层求解器而不依赖 Agent。

import numpy as np  # 导入数组库以编解码 SciPy 一维决策向量。
from scipy.optimize import Bounds  # 导入连续变量边界对象供算法层直接消费。

from Mask.flight_relation import FlightDynamicLimits, FlightPrimitive, TrajectoryObjectiveWeights  # 导入动力学限制、优化原语结果和式 34 权重。
from Mask.geometry import Vector3  # 导入三维向量以表达位置、速度和加速度。
from Mask.mobility import UnmannedAerialVehicle  # 导入无人机速度、推力、净空、功率和电池能力。
from Mask.terrain import ObstacleField, TerrainSurface  # 导入地形和显式障碍场以计算式 26 和式 27。
from Mask.weather import EnergyModelParameters, WeatherState  # 导入式 35 至式 38 的气象和能耗参数。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑记录保存全部连续决策状态。
class TrajectoryDecision:  # 定义式 21 至式 33 的服务列级连续变量集合。
    """保存节点位置速度、区间实际加速度和区间持续时间。"""  # 解释决策变量职责。

    positions: tuple[Vector3, ...]  # 保存 K 加一组轨迹节点位置。
    velocities: tuple[Vector3, ...]  # 保存 K 加一组轨迹节点速度。
    accelerations: tuple[Vector3, ...]  # 保存 K 组分段恒定实际加速度。
    durations: tuple[float, ...]  # 保存 K 组正区间持续时间。

    def __post_init__(self) -> None:  # 定义连续决策构造后的结构验证。
        """验证状态长度、区间数量和持续时间有限性。"""  # 解释决策变量输入约束。
        interval_count = len(self.durations)  # 读取连续轨迹区间数量。
        if interval_count < 1:  # 判断轨迹是否缺少运动区间。
            raise ValueError("连续轨迹至少需要一个运动区间")  # 拒绝不能连接起飞回收端点的空轨迹。
        if len(self.positions) != interval_count + 1 or len(self.velocities) != interval_count + 1 or len(self.accelerations) != interval_count:  # 判断各状态序列长度是否符合转录结构。
            raise ValueError("连续轨迹位置、速度、加速度和时间长度不一致")  # 拒绝不能逐区间应用式 22 和式 23 的输入。
        if any(not math.isfinite(duration) or duration <= 0.0 for duration in self.durations):  # 判断全部持续时间是否为正有限数值。
            raise ValueError("连续轨迹区间时间必须为正有限数值")  # 拒绝无物理意义的区间时间。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑记录保存连续求解和审计参数。
class TrajectoryOptimizationConfig:  # 定义 SLSQP 停止参数和独立复核容差。
    """保存等式、不等式、密集采样、迭代次数和目标停止容差。"""  # 解释配置职责。

    equality_tolerance: float  # 保存最大允许等式残差。
    inequality_tolerance: float  # 保存最大允许负不等式余量。
    dense_samples_per_interval: int  # 保存每个区间独立净空复核采样数。
    max_iterations: int  # 保存连续求解最大迭代次数。
    function_tolerance: float  # 保存连续目标停止容差。

    def __post_init__(self) -> None:  # 定义连续求解配置构造后验证。
        """验证容差为正有限数且计数参数为正整数。"""  # 解释配置输入约束。
        tolerances = (self.equality_tolerance, self.inequality_tolerance, self.function_tolerance)  # 汇总必须为正有限数的求解容差。
        if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):  # 判断容差是否有效。
            raise ValueError("连续轨迹求解和审计容差必须为正有限数值")  # 拒绝无效数值停止条件。
        if not isinstance(self.dense_samples_per_interval, int) or self.dense_samples_per_interval < 2:  # 判断密集采样数量是否足以检查区间内部。
            raise ValueError("每个连续轨迹区间至少需要两个密集复核样本")  # 拒绝只检查单个端点的配置。
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:  # 判断最大迭代次数是否为正整数。
            raise ValueError("连续轨迹最大迭代次数必须为正整数")  # 拒绝不能执行求解的配置。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑记录描述服务级决策中的单条原语切片。
class TrajectoryPrimitiveSpec:  # 定义优化区间到下游 FlightPrimitive 的稳定映射。
    """保存原语引用、区间起点和区间数量。"""  # 解释切片映射职责。

    object_id: str  # 保存优化原语稳定标识。
    name: str  # 保存优化原语可读名称。
    source_id: str  # 保存原语起点对象标识。
    target_id: str  # 保存原语终点对象标识。
    corridor_id: str  # 保存原语来源航路标识。
    first_interval_index: int  # 保存原语在服务级决策中的首区间索引。
    interval_count: int  # 保存原语包含的连续区间数量。
    minimum_clearance_margin: float  # 保存来源走廊初值的净空余量。

    def __post_init__(self) -> None:  # 定义原语切片构造后的引用和范围验证。
        """验证稳定标识、非负首索引和正区间数量。"""  # 解释切片输入约束。
        if any(not identifier.strip() for identifier in (self.object_id, self.name, self.source_id, self.target_id, self.corridor_id)):  # 判断原语标识或名称是否为空白。
            raise ValueError("连续轨迹原语映射标识不能为空")  # 拒绝不可追踪的原语切片。
        if self.first_interval_index < 0 or self.interval_count < 1:  # 判断原语区间范围是否有效。
            raise ValueError("连续轨迹原语区间范围无效")  # 拒绝越界或空原语切片。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑记录绑定连续模型全部输入。
class TrajectoryOptimizationRequest:  # 定义一条离散服务列组合的内层连续优化问题。
    """保存原语映射、固定状态、物理场、参数、初值和复核配置。"""  # 解释内层请求职责。

    object_id: str  # 保存连续优化请求稳定标识。
    primitive_specs: tuple[TrajectoryPrimitiveSpec, ...]  # 保存服务级区间到飞行原语的切片映射。
    fixed_node_positions: tuple[tuple[int, Vector3], ...]  # 保存起飞、观测和回收节点固定位置。
    observation_node_indices: tuple[int, ...]  # 保存承担稳定观测任务的节点索引。
    terrain: TerrainSurface  # 保存式 26 使用的唯一地形场。
    obstacle_field: ObstacleField  # 保存式 27 使用的显式障碍场。
    uav: UnmannedAerialVehicle  # 保存速度、推力、净空、功率和电池能力。
    weather: WeatherState  # 保存当前离散组合使用的气象时段。
    energy_parameters: EnergyModelParameters  # 保存式 35 至式 38 能耗参数。
    dynamic_limits: FlightDynamicLimits  # 保存式 21、式 24、式 25 和式 31 动力学限制。
    service_duration: float  # 保存全部稳定观测服务时间。
    objective_weights: TrajectoryObjectiveWeights  # 保存式 34 时间和期望能耗权重。
    initial_decision: TrajectoryDecision  # 保存由认证走廊和确定性原语生成的可行初值。
    config: TrajectoryOptimizationConfig  # 保存求解停止参数和独立复核容差。

    def __post_init__(self) -> None:  # 定义连续优化请求构造后的闭合验证。
        """验证标识、原语切片、固定节点、观测节点和服务时间。"""  # 解释请求输入约束。
        if not self.object_id.strip() or not self.primitive_specs:  # 判断请求是否缺少稳定标识或原语映射。
            raise ValueError("连续轨迹优化请求必须具有标识和原语映射")  # 拒绝不能形成服务列的请求。
        if not math.isfinite(self.service_duration) or self.service_duration < 0.0:  # 判断服务时间是否为非负有限数。
            raise ValueError("连续轨迹服务时间必须为非负有限数值")  # 拒绝无法计算式 36 的服务时间。
        node_count = len(self.initial_decision.positions)  # 读取初值节点数量。
        fixed_indices = tuple(index for index, _ in self.fixed_node_positions)  # 提取固定位置节点索引。
        if len(fixed_indices) != len(set(fixed_indices)) or any(index < 0 or index >= node_count for index in fixed_indices):  # 判断固定位置节点是否重复或越界。
            raise ValueError("连续轨迹固定位置节点索引无效")  # 拒绝不闭合的固定状态映射。
        if any(index < 0 or index >= node_count for index in self.observation_node_indices):  # 判断观测节点索引是否越界。
            raise ValueError("连续轨迹观测节点索引无效")  # 拒绝无法应用式 31 的请求。
        interval_count = len(self.initial_decision.durations)  # 读取服务级初值区间数量。
        covered_intervals: list[int] = []  # 创建由全部原语切片覆盖的区间索引列表。
        for spec in self.primitive_specs:  # 逐一展开原语区间切片。
            covered_intervals.extend(range(spec.first_interval_index, spec.first_interval_index + spec.interval_count))  # 将当前原语区间范围加入覆盖列表。
        if tuple(covered_intervals) != tuple(range(interval_count)):  # 判断原语切片是否按顺序无遗漏无重叠覆盖全部区间。
            raise ValueError("连续轨迹原语映射必须完整覆盖全部区间")  # 拒绝不能重构下游飞行原语的请求。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑记录保存独立约束复核结果。
class TrajectoryConstraintAudit:  # 定义不信任求解器成功标志的连续轨迹审计证据。
    """记录准入状态、稳定状态、消息和三类最坏约束证据。"""  # 解释审计输出职责。

    admitted: bool  # 保存连续轨迹是否通过独立复核。
    status: str  # 保存 optimized、solver_failed 或 audit_failed 稳定状态。
    message: str  # 保存求解器或审计消息。
    maximum_equality_residual: float  # 保存全部等式约束最大绝对残差。
    minimum_inequality_margin: float  # 保存全部节点和区间不等式最小余量。
    minimum_dense_clearance_margin: float  # 保存区间密集采样最小净空余量。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑记录保存算法层连续求解结果。
class TrajectoryOptimizationResult:  # 定义 Env 可消费的统一内层求解结果。
    """绑定求解状态、连续决策、目标、能耗、审计和优化原语。"""  # 解释求解结果职责。

    success: bool  # 保存是否同时通过求解器和独立审计。
    status: str  # 保存稳定求解状态。
    message: str  # 保存求解器消息。
    solver_method: str  # 保存实际连续求解方法名称。
    decision: TrajectoryDecision | None  # 保存存在时的连续状态解。
    objective_value: float | None  # 保存存在时的式 34 目标值。
    expected_energy: float | None  # 保存存在时的式 36 期望能耗。
    audit: TrajectoryConstraintAudit  # 保存独立约束复核证据。
    primitives: tuple[FlightPrimitive, ...]  # 保存仅在准入时生成的优化飞行原语。


class TrajectoryOptimizer(Protocol):  # 定义 Env 可以注入的连续轨迹求解器协议。
    """要求算法层使用统一请求并返回可审计结果。"""  # 解释依赖反转边界。

    def solve(self, request: TrajectoryOptimizationRequest) -> TrajectoryOptimizationResult:  # 声明统一内层求解入口。
        """求解一个离散服务列组合的连续轨迹问题。"""  # 解释协议方法职责。
        ...  # 由具体算法层实现连续轨迹求解协议。


def encode_decision_vector(decision: TrajectoryDecision) -> np.ndarray:  # 定义不可变连续状态到 SciPy 一维向量的编码函数。
    """按位置、速度、加速度和时间顺序返回一维浮点决策向量。"""  # 解释稳定编码顺序。
    values = [component for vector in decision.positions for component in (vector.x, vector.y, vector.z)]  # 编码全部节点位置分量。
    values.extend(component for vector in decision.velocities for component in (vector.x, vector.y, vector.z))  # 编码全部节点速度分量。
    values.extend(component for vector in decision.accelerations for component in (vector.x, vector.y, vector.z))  # 编码全部区间加速度分量。
    values.extend(decision.durations)  # 编码全部区间持续时间。
    return np.asarray(values, dtype=float)  # 返回独立的一维浮点数组。


def decode_decision_vector(request: TrajectoryOptimizationRequest, values: np.ndarray) -> TrajectoryDecision:  # 定义 SciPy 一维向量到不可变连续状态的解码函数。
    """按照请求初值网格解码位置、速度、加速度和时间。"""  # 解释解码尺寸来源。
    flattened = np.asarray(values, dtype=float).reshape(-1)  # 将任意形状输入转换为一维浮点数组。
    interval_count = len(request.initial_decision.durations)  # 读取请求固定的区间数量。
    node_count = interval_count + 1  # 计算节点数量。
    expected_length = node_count * 6 + interval_count * 4  # 计算三维位置速度、三维加速度和时间的总变量数量。
    if len(flattened) != expected_length or not np.isfinite(flattened).all():  # 判断输入长度和有限性是否满足请求网格。
        raise ValueError("连续轨迹决策向量长度或数值无效")  # 拒绝不能形成完整状态的数值解。
    cursor = 0  # 初始化顺序解码游标。

    def read_vectors(count: int) -> tuple[Vector3, ...]:  # 定义从当前游标读取三维向量序列的局部函数。
        """从当前游标读取指定数量的三维向量并推进游标。"""  # 解释局部解码函数的状态更新职责。
        nonlocal cursor  # 声明更新外层解码游标。
        vectors = tuple(Vector3(float(flattened[cursor + 3 * index]), float(flattened[cursor + 3 * index + 1]), float(flattened[cursor + 3 * index + 2])) for index in range(count))  # 按连续三个分量重建向量。
        cursor += 3 * count  # 将游标推进到下一类变量起点。
        return vectors  # 返回不可变向量序列。

    positions = read_vectors(node_count)  # 解码全部节点位置。
    velocities = read_vectors(node_count)  # 解码全部节点速度。
    accelerations = read_vectors(interval_count)  # 解码全部区间实际加速度。
    durations = tuple(float(value) for value in flattened[cursor:])  # 解码剩余全部区间时间。
    return TrajectoryDecision(positions, velocities, accelerations, durations)  # 返回经过结构验证的连续状态。


def decision_bounds(request: TrajectoryOptimizationRequest) -> Bounds:  # 定义连续决策向量的分量边界构造函数。
    """返回地形定义域、速度、加速度和式 21 时间组成的 SciPy 边界。"""  # 解释边界与非线性约束的互补关系。
    interval_count = len(request.initial_decision.durations)  # 读取区间数量。
    node_count = interval_count + 1  # 计算节点数量。
    x_min, x_max = request.terrain.x_coordinates[0], request.terrain.x_coordinates[-1]  # 读取地形横向定义域。
    y_min, y_max = request.terrain.y_coordinates[0], request.terrain.y_coordinates[-1]  # 读取地形纵向定义域。
    position_lower = [component for _ in range(node_count) for component in (x_min, y_min, -math.inf)]  # 构造全部位置分量下界。
    position_upper = [component for _ in range(node_count) for component in (x_max, y_max, math.inf)]  # 构造全部位置分量上界。
    speed_limit = request.dynamic_limits.maximum_speed  # 读取速度分量保守边界。
    velocity_lower = [-speed_limit] * (3 * node_count)  # 构造全部速度分量下界。
    velocity_upper = [speed_limit] * (3 * node_count)  # 构造全部速度分量上界。
    acceleration_component_limit = request.dynamic_limits.maximum_thrust_acceleration + request.dynamic_limits.gravity  # 构造覆盖推力球的实际加速度分量边界。
    acceleration_lower = [-acceleration_component_limit] * (3 * interval_count)  # 构造全部加速度分量下界。
    acceleration_upper = [acceleration_component_limit] * (3 * interval_count)  # 构造全部加速度分量上界。
    duration_lower = [max(request.dynamic_limits.minimum_segment_time, 1.0e-9)] * interval_count  # 构造式 21 时间下界。
    duration_upper = [request.dynamic_limits.maximum_segment_time] * interval_count  # 构造式 21 时间上界。
    return Bounds(np.asarray((*position_lower, *velocity_lower, *acceleration_lower, *duration_lower), dtype=float), np.asarray((*position_upper, *velocity_upper, *acceleration_upper, *duration_upper), dtype=float))  # 返回与稳定编码顺序一致的变量边界。


def trajectory_equality_residuals(request: TrajectoryOptimizationRequest, decision: TrajectoryDecision) -> np.ndarray:  # 定义式 22、式 23 和固定状态等式残差函数。
    """返回位置速度递推、固定节点和零速端点的有符号残差。"""  # 解释等式残差零值语义。
    residuals: list[float] = []  # 创建稳定顺序的等式残差列表。
    for interval_index, duration in enumerate(decision.durations):  # 逐一处理分段恒加速度区间。
        start_position = decision.positions[interval_index]  # 读取当前区间起始位置。
        end_position = decision.positions[interval_index + 1]  # 读取当前区间终止位置。
        start_velocity = decision.velocities[interval_index]  # 读取当前区间起始速度。
        end_velocity = decision.velocities[interval_index + 1]  # 读取当前区间终止速度。
        acceleration = decision.accelerations[interval_index]  # 读取当前区间实际加速度。
        expected_position = start_position + start_velocity.scaled(duration) + acceleration.scaled(0.5 * duration**2)  # 按式 22 递推终止位置。
        expected_velocity = start_velocity + acceleration.scaled(duration)  # 按式 23 递推终止速度。
        position_difference = end_position - expected_position  # 计算位置递推残差向量。
        velocity_difference = end_velocity - expected_velocity  # 计算速度递推残差向量。
        residuals.extend((position_difference.x, position_difference.y, position_difference.z, velocity_difference.x, velocity_difference.y, velocity_difference.z))  # 按三维分量保存递推残差。
    for node_index, fixed_position in request.fixed_node_positions:  # 逐一应用起飞、观测和回收固定位置。
        difference = decision.positions[node_index] - fixed_position  # 计算当前固定节点位置残差。
        residuals.extend((difference.x, difference.y, difference.z))  # 保存固定位置三维残差。
    for endpoint_velocity in (decision.velocities[0], decision.velocities[-1]):  # 逐一起飞和回收零速端点。
        residuals.extend((endpoint_velocity.x, endpoint_velocity.y, endpoint_velocity.z))  # 保存式 30 零速状态残差。
    return np.asarray(residuals, dtype=float)  # 返回供求解器和独立审计复用的等式残差数组。


def _position_clearance_margins(request: TrajectoryOptimizationRequest, position: Vector3) -> tuple[float, ...]:  # 定义单个空间位置的定义域、地形和障碍余量函数。
    """返回水平定义域和式 26、式 27 的非负可行余量。"""  # 解释位置余量语义。
    x_min, x_max = request.terrain.x_coordinates[0], request.terrain.x_coordinates[-1]  # 读取地形横向定义域。
    y_min, y_max = request.terrain.y_coordinates[0], request.terrain.y_coordinates[-1]  # 读取地形纵向定义域。
    domain_margins = (position.x - x_min, x_max - position.x, position.y - y_min, y_max - position.y)  # 计算四个水平边界余量。
    if not request.terrain.contains_xy(position.x, position.y):  # 判断当前位置是否越出地形定义域。
        return (*domain_margins, -1.0e6, -1.0e6)  # 使用有限负哨兵标记不能查询高程和障碍净空的位置。
    terrain_margin = position.z - request.terrain.height_at(position.x, position.y) - request.uav.minimum_clearance  # 计算式 26 地形净空余量。
    obstacle_clearance = request.obstacle_field.signed_clearance(position)  # 查询当前位置到最近显式障碍的有符号净空。
    obstacle_margin = 1.0e6 if math.isinf(obstacle_clearance) else obstacle_clearance - request.uav.minimum_clearance  # 计算式 27 余量并用有限正值表达空障碍场。
    return (*domain_margins, terrain_margin, obstacle_margin)  # 返回全部位置可行余量。


def trajectory_inequality_margins(request: TrajectoryOptimizationRequest, decision: TrajectoryDecision) -> np.ndarray:  # 定义式 21、式 24 至式 27 和式 31 余量函数。
    """返回全部以非负表示可行的连续轨迹不等式余量。"""  # 解释求解器不等式方向。
    margins: list[float] = []  # 创建稳定顺序的不等式余量列表。
    limits = request.dynamic_limits  # 读取动力学限制。
    gravity_vector = limits.gravity_vector  # 读取竖直向下重力向量。
    for duration in decision.durations:  # 逐一应用式 21 时间上下界。
        margins.append(duration - limits.minimum_segment_time)  # 保存时间下界余量。
        margins.append(limits.maximum_segment_time - duration)  # 保存时间上界余量。
    for velocity in decision.velocities:  # 逐一应用式 24 三维速度球。
        margins.append(limits.maximum_speed**2 - velocity.norm() ** 2)  # 保存平方形式速度余量。
    for acceleration in decision.accelerations:  # 逐一应用式 25 推力比加速度球。
        thrust_acceleration = acceleration - gravity_vector  # 计算实际加速度相对于重力的推力比加速度。
        margins.append(limits.maximum_thrust_acceleration**2 - thrust_acceleration.norm() ** 2)  # 保存平方形式推力余量。
    for position in decision.positions:  # 逐一应用节点地形和显式障碍净空。
        margins.extend(_position_clearance_margins(request, position))  # 保存当前位置定义域和净空余量。
    for node_index in request.observation_node_indices:  # 逐一应用式 31 稳定观测速度限制。
        margins.append(limits.maximum_observation_speed**2 - decision.velocities[node_index].norm() ** 2)  # 保存观测节点平方速度余量。
    return np.asarray(margins, dtype=float)  # 返回供 SLSQP 和独立审计共同使用的余量数组。


def trajectory_expected_energy(request: TrajectoryOptimizationRequest, decision: TrajectoryDecision) -> float:  # 定义式 35 和式 36 的连续状态能耗函数。
    """从每个区间起点状态、实际加速度和正爬升速度计算期望能耗。"""  # 解释离散积分规则。
    parameters = request.energy_parameters  # 读取能耗模型参数。
    expected_energy = parameters.launch_recovery_energy + request.uav.hover_power * request.service_duration  # 初始化起飞回收和稳定观测能耗。
    gravity_vector = request.dynamic_limits.gravity_vector  # 读取与动力学一致的重力向量。
    for interval_index, duration in enumerate(decision.durations):  # 逐一积分分段恒加速度区间功率。
        velocity = decision.velocities[interval_index]  # 读取式 35 当前节点速度。
        acceleration = decision.accelerations[interval_index]  # 读取式 35 当前区间实际加速度。
        relative_air_velocity = velocity - request.weather.wind_vector  # 计算相对于当前气象风速的速度。
        thrust_acceleration = acceleration - gravity_vector  # 计算式 35 推力比加速度。
        vertical_speed = (decision.positions[interval_index + 1].z - decision.positions[interval_index].z) / duration  # 计算当前区间平均竖向速度。
        power = request.uav.hover_power + parameters.motion_coefficient * relative_air_velocity.norm() ** 2 + parameters.acceleration_coefficient * thrust_acceleration.norm() ** 2 + parameters.climb_coefficient * max(vertical_speed, 0.0)  # 按式 35 计算调度层区间功率。
        expected_energy += power * duration  # 按式 36 累加当前区间功率时间积分。
    return expected_energy  # 返回服务列期望能耗。


def trajectory_objective(request: TrajectoryOptimizationRequest, decision: TrajectoryDecision) -> float:  # 定义式 34 连续轨迹目标函数。
    """返回动力学飞行时间与式 36 期望能耗的加权和。"""  # 解释局部值函数语义。
    flight_duration = sum(decision.durations)  # 按式 32 汇总全部航段时间。
    expected_energy = trajectory_expected_energy(request, decision)  # 计算式 36 期望能耗。
    return request.objective_weights.dynamic_time_weight * flight_duration + request.objective_weights.expected_energy_weight * expected_energy  # 按式 34 返回连续目标值。


def _dense_clearance_margin(request: TrajectoryOptimizationRequest, decision: TrajectoryDecision) -> float:  # 定义区间内部密集净空独立复核函数。
    """使用解析恒加速度状态在每个区间内部重新计算最小净空余量。"""  # 解释密集检查独立性。
    minimum_margin = math.inf  # 初始化全部区间密集位置的最小余量。
    sample_count = request.config.dense_samples_per_interval  # 读取每区间密集采样数量。
    for interval_index, duration in enumerate(decision.durations):  # 逐一处理连续轨迹区间。
        start_position = decision.positions[interval_index]  # 读取区间起始位置。
        start_velocity = decision.velocities[interval_index]  # 读取区间起始速度。
        acceleration = decision.accelerations[interval_index]  # 读取区间恒定实际加速度。
        for sample_index in range(sample_count + 1):  # 采样包含两个端点的区间时刻。
            sample_time = duration * sample_index / sample_count  # 计算当前相对区间时刻。
            position = start_position + start_velocity.scaled(sample_time) + acceleration.scaled(0.5 * sample_time**2)  # 按式 22 解析计算区间内部位置。
            position_margins = _position_clearance_margins(request, position)  # 重算当前密集位置定义域和两类净空余量。
            minimum_margin = min(minimum_margin, *position_margins)  # 更新全部密集位置最小余量。
    return minimum_margin  # 返回密集复核最坏净空余量。


def audit_trajectory_decision(request: TrajectoryOptimizationRequest, decision: TrajectoryDecision, solver_success: bool, message: str) -> TrajectoryConstraintAudit:  # 定义不信任求解器成功标志的独立审计入口。
    """重算全部连续约束并仅在求解成功和三类证据通过时准入。"""  # 解释失败关闭语义。
    equality_residuals = trajectory_equality_residuals(request, decision)  # 独立重算全部等式约束残差。
    inequality_margins = trajectory_inequality_margins(request, decision)  # 独立重算全部节点和区间不等式余量。
    maximum_equality_residual = float(np.max(np.abs(equality_residuals)))  # 汇总最大绝对等式残差。
    minimum_inequality_margin = float(np.min(inequality_margins))  # 汇总最小不等式余量。
    minimum_dense_clearance_margin = float(_dense_clearance_margin(request, decision))  # 汇总区间内部密集净空最小余量。
    constraints_passed = maximum_equality_residual <= request.config.equality_tolerance and minimum_inequality_margin >= -request.config.inequality_tolerance and minimum_dense_clearance_margin >= -request.config.inequality_tolerance  # 判断全部独立复核证据是否位于容差内。
    admitted = bool(solver_success and constraints_passed)  # 仅在求解器和独立复核同时成功时准入。
    status = "optimized" if admitted else ("audit_failed" if solver_success else "solver_failed")  # 生成稳定连续轨迹状态。
    return TrajectoryConstraintAudit(admitted, status, message, maximum_equality_residual, minimum_inequality_margin, minimum_dense_clearance_margin)  # 返回完整不可变审计证据。
