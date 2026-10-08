"""定义地形认证航路和不涉及下层飞控的二阶质点飞行原语。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以计算动力学根和航段时间。
from dataclasses import dataclass  # 导入数据类工具以保存不可变飞行对象。

from Mask.base import SceneObject, SceneRelation, SceneSystem, ValidationIssue  # 导入场景关系和验证基础类型。
from Mask.geometry import Polyline3D, Vector3  # 导入三维折线和向量以表达航路与状态。
from Mask.terrain import ObstacleField, TerrainSurface  # 导入地形和显式障碍场以认证空间净空。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义飞行动力学限制。
class FlightDynamicLimits:  # 定义二阶质点飞行时间计算所需的物理上限。
    """保存速度、推力、重力、航段时间和稳定观测速度限制。"""  # 解释动力学限制内容。

    maximum_speed: float  # 保存无人机最大三维速度。
    maximum_thrust_acceleration: float  # 保存无人机最大推力比加速度。
    gravity: float = 9.81  # 保存竖直向下的重力加速度大小。
    minimum_segment_time: float = 0.0  # 保存相邻轨迹节点之间允许的最小航段时间。
    maximum_segment_time: float = math.inf  # 保存相邻轨迹节点之间允许的最大航段时间。
    maximum_observation_speed: float = 0.0  # 保存稳定观测节点允许的最大三维速度。
    maximum_subsegments_per_leg: int = 12  # 保存单个几何航路段允许的最大动力学离散段数。

    def __post_init__(self) -> None:  # 定义动力学限制初始化后的参数验证方法。
        """验证速度、推力和重力参数为正有限数值且推力能够悬停。"""  # 解释动力学限制输入约束。
        numeric_values = (self.maximum_speed, self.maximum_thrust_acceleration, self.gravity, self.minimum_segment_time, self.maximum_observation_speed)  # 汇总必须为有限数值的动力学限制参数。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部动力学限制参数是否有限。
            raise ValueError("飞行动力学限制必须为有限数值")  # 抛出明确的动力学有限性异常。
        if self.maximum_speed <= 0.0:  # 判断最大速度是否为正数。
            raise ValueError("飞行最大速度必须为正数")  # 抛出明确的最大速度异常。
        if self.gravity <= 0.0:  # 判断重力加速度是否为正数。
            raise ValueError("重力加速度必须为正数")  # 抛出明确的重力参数异常。
        if self.maximum_thrust_acceleration <= self.gravity:  # 判断最大推力比加速度是否足以抵消重力。
            raise ValueError("最大推力比加速度必须大于重力加速度")  # 抛出明确的推力能力异常。
        if self.minimum_segment_time < 0.0:  # 判断航段时间下界是否为非负数。
            raise ValueError("航段时间下界不能为负数")  # 抛出明确的航段时间下界异常。
        if math.isnan(self.maximum_segment_time) or self.maximum_segment_time <= 0.0:  # 判断航段时间上界是否为正数或正无穷。
            raise ValueError("航段时间上界必须为正数")  # 抛出明确的航段时间上界异常。
        if self.maximum_segment_time < self.minimum_segment_time:  # 判断航段时间上下界顺序是否正确。
            raise ValueError("航段时间上界不能小于下界")  # 抛出明确的航段时间区间异常。
        if self.maximum_observation_speed < 0.0 or self.maximum_observation_speed > self.maximum_speed:  # 判断稳定观测速度上界是否位于物理速度范围内。
            raise ValueError("稳定观测速度上界必须位于零和最大飞行速度之间")  # 抛出明确的稳定观测速度异常。
        if isinstance(self.maximum_subsegments_per_leg, bool) or not isinstance(self.maximum_subsegments_per_leg, int) or self.maximum_subsegments_per_leg <= 0:  # 判断动力学离散段数上界是否为正整数。
            raise ValueError("单航路段最大动力学离散数必须为正整数")  # 拒绝无界或无效的离散配置。

    @property  # 将重力向量暴露为只读属性。
    def gravity_vector(self) -> Vector3:  # 定义重力向量读取方法。
        """返回采用竖直向上为正方向时的重力向量。"""  # 解释重力向量坐标约定。
        return Vector3(0.0, 0.0, -self.gravity)  # 返回竖直向下的重力加速度向量。

    def directional_accelerations(self, direction: Vector3) -> tuple[float, float]:  # 定义给定航向的加速和制动能力计算方法。
        """由推力球约束计算沿航向加速和反航向制动的最大实际加速度。"""  # 解释方向加速度计算含义。
        direction_unit = direction.normalized()  # 将给定航向单位化。
        gravity_projection = direction_unit.dot(self.gravity_vector)  # 计算重力在航向上的投影。
        discriminant = gravity_projection**2 + self.maximum_thrust_acceleration**2 - self.gravity**2  # 计算推力球与航向直线交点判别式。
        if discriminant <= 0.0:  # 判断给定方向是否缺少正实际加速度能力。
            raise ValueError("推力上限不足以沿给定方向形成正加速度")  # 抛出明确的方向动力学不可行异常。
        root = math.sqrt(discriminant)  # 计算方向推力约束的公共平方根项。
        acceleration = gravity_projection + root  # 计算沿航向的最大正实际加速度。
        braking = -gravity_projection + root  # 计算沿反航向的最大制动加速度大小。
        if acceleration <= 0.0 or braking <= 0.0:  # 判断加速和制动能力是否均为正数。
            raise ValueError("给定方向缺少有效加速或制动能力")  # 抛出明确的方向动力学异常。
        return acceleration, braking  # 返回沿航向加速能力和反航向制动能力。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义局部轨迹目标权重。
class TrajectoryObjectiveWeights:  # 定义式 34 时间项和期望能耗项的符号权重。
    """保存固定认证轨迹目标评估使用的非负时间与能耗权重。"""

    dynamic_time_weight: float  # 保存局部轨迹动力学飞行时间权重。
    expected_energy_weight: float  # 保存局部轨迹期望能耗权重。

    def __post_init__(self) -> None:  # 定义局部轨迹目标权重初始化后验证方法。
        """验证两个目标权重为有限非负数且不能同时为零。"""
        weights = (self.dynamic_time_weight, self.expected_energy_weight)  # 汇总式 34 使用的两个目标权重。
        if not all(math.isfinite(value) and value >= 0.0 for value in weights):  # 判断两个权重是否均为有限非负数。
            raise ValueError("局部轨迹目标权重必须为有限非负数")  # 抛出明确的轨迹目标权重异常。
        if all(value == 0.0 for value in weights):  # 判断两个目标权重是否同时为零。
            raise ValueError("局部轨迹目标权重不能同时为零")  # 拒绝没有任何优化含义的零目标。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存一个运动阶段。
class MotionPhase:  # 定义加速、巡航或制动阶段的状态摘要。
    """记录阶段类型、时间、距离、速度端点和实际加速度向量。"""  # 解释运动阶段内容。

    phase_type: str  # 保存加速、巡航或制动阶段类型。
    duration: float  # 保存当前运动阶段持续时间。
    distance: float  # 保存当前运动阶段沿航向移动距离。
    start_speed: float  # 保存当前运动阶段起始标量速度。
    end_speed: float  # 保存当前运动阶段结束标量速度。
    direction: Vector3  # 保存当前运动阶段的单位航向。
    acceleration: Vector3  # 保存当前运动阶段的实际三维加速度。

    def __post_init__(self) -> None:  # 定义运动阶段初始化后的参数验证方法。
        """验证阶段类型、时间、距离和速度边界。"""  # 解释运动阶段输入约束。
        if self.phase_type not in {"accelerate", "cruise", "decelerate"}:  # 判断运动阶段类型是否受支持。
            raise ValueError("飞行运动阶段类型无效")  # 抛出明确的运动阶段类型异常。
        numeric_values = (self.duration, self.distance, self.start_speed, self.end_speed)  # 汇总运动阶段标量字段。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部运动阶段标量是否有限。
            raise ValueError("飞行运动阶段参数必须为有限数值")  # 抛出明确的阶段有限性异常。
        if self.duration < 0.0 or self.distance < 0.0 or self.start_speed < 0.0 or self.end_speed < 0.0:  # 判断时间、距离和速度是否为非负数。
            raise ValueError("飞行运动阶段参数不能为负数")  # 抛出明确的阶段非负约束异常。

    @property  # 将平均速度向量暴露为只读属性。
    def average_velocity(self) -> Vector3:  # 定义阶段平均速度向量读取方法。
        """使用阶段端点标量速度平均值构造平均三维速度。"""  # 解释平均速度近似方法。
        average_speed = 0.5 * (self.start_speed + self.end_speed)  # 计算阶段端点标量速度平均值。
        return self.direction.scaled(average_speed)  # 返回沿单位航向的平均三维速度。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义一个分段恒加速度轨迹区间。
class TrajectoryInterval:  # 定义式 22 和式 23 对应的显式状态递推区间。
    """保存区间起止时刻、位置、速度和恒定实际加速度。"""

    phase_type: str  # 保存当前区间对应的运动阶段类型。
    start_time: float  # 保存当前区间相对飞行原语起点的开始时刻。
    duration: float  # 保存当前区间持续时间。
    start_position: Vector3  # 保存当前区间起始三维位置。
    end_position: Vector3  # 保存由式 22 递推得到的终止三维位置。
    start_velocity: Vector3  # 保存当前区间起始三维速度。
    end_velocity: Vector3  # 保存由式 23 递推得到的终止三维速度。
    acceleration: Vector3  # 保存当前区间恒定实际三维加速度。

    def __post_init__(self) -> None:  # 定义轨迹区间初始化后的结构验证方法。
        """验证阶段类型、开始时刻和持续时间具有物理意义。"""
        if self.phase_type not in {"accelerate", "cruise", "decelerate"}:  # 判断轨迹区间阶段类型是否受支持。
            raise ValueError("轨迹区间阶段类型无效")  # 抛出明确的轨迹区间类型异常。
        if not math.isfinite(self.start_time) or self.start_time < 0.0:  # 判断轨迹区间开始时刻是否为非负有限数值。
            raise ValueError("轨迹区间开始时刻必须为非负有限数值")  # 抛出明确的轨迹区间开始时刻异常。
        if not math.isfinite(self.duration) or self.duration <= 0.0:  # 判断轨迹区间持续时间是否为正有限数值。
            raise ValueError("轨迹区间持续时间必须为正有限数值")  # 抛出明确的轨迹区间持续时间异常。

    @property  # 将区间终止时刻暴露为只读派生量。
    def end_time(self) -> float:  # 定义轨迹区间终止时刻读取方法。
        """返回开始时刻与持续时间之和。"""
        return self.start_time + self.duration  # 返回当前轨迹区间终止时刻。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存一个飞行航段剖面。
class FlightSegmentProfile:  # 定义两个连续航路折点之间的最短时间状态剖面。
    """记录航段几何、峰值速度和有序运动阶段。"""  # 解释航段剖面内容。

    start: Vector3  # 保存航段起点三维位置。
    end: Vector3  # 保存航段终点三维位置。
    direction: Vector3  # 保存航段单位航向。
    length: float  # 保存航段三维长度。
    peak_speed: float  # 保存航段达到的峰值标量速度。
    phases: tuple[MotionPhase, ...]  # 保存航段有序运动阶段集合。

    @property  # 将航段持续时间暴露为只读属性。
    def duration(self) -> float:  # 定义航段持续时间读取方法。
        """返回航段全部运动阶段持续时间之和。"""  # 解释航段时间汇总方法。
        return sum(phase.duration for phase in self.phases)  # 汇总并返回全部运动阶段时间。

    def trajectory_intervals(self, start_time: float = 0.0) -> tuple[TrajectoryInterval, ...]:  # 定义航段显式状态区间展开方法。
        """从给定起始时刻按式 22 和式 23 展开全部分段恒加速度区间。"""
        if not math.isfinite(start_time) or start_time < 0.0:  # 判断调用方给定起始时刻是否为非负有限数值。
            raise ValueError("航段起始时刻必须为非负有限数值")  # 抛出明确的航段起始时刻异常。
        intervals: list[TrajectoryInterval] = []  # 创建按时间顺序保存的显式轨迹区间列表。
        current_time = start_time  # 使用调用方给定值初始化当前轨迹时刻。
        current_position = self.start  # 使用航段起点初始化当前三维位置。
        for phase in self.phases:  # 按运动顺序展开当前航段全部阶段。
            if phase.duration <= 1.0e-15:  # 判断当前阶段是否仅为浮点舍入产生的零时长阶段。
                continue  # 跳过不能形成真实轨迹区间的零时长阶段。
            start_velocity = phase.direction.scaled(phase.start_speed)  # 从阶段标量速度和单位航向构造起始三维速度。
            end_velocity = start_velocity + phase.acceleration.scaled(phase.duration)  # 按式 23 递推区间终止三维速度。
            end_position = current_position + start_velocity.scaled(phase.duration) + phase.acceleration.scaled(0.5 * phase.duration**2)  # 按式 22 递推区间终止三维位置。
            interval = TrajectoryInterval(phase.phase_type, current_time, phase.duration, current_position, end_position, start_velocity, end_velocity, phase.acceleration)  # 创建当前阶段对应的完整状态区间。
            intervals.append(interval)  # 将当前显式状态区间加入有序结果。
            current_time = interval.end_time  # 将当前时刻推进到区间终点。
            current_position = interval.end_position  # 将当前位置推进到区间终点。
        return tuple(intervals)  # 返回不可变的航段显式状态区间集合。


def minimum_time_segment(start: Vector3, end: Vector3, limits: FlightDynamicLimits) -> FlightSegmentProfile:  # 定义零端点速度航段最短时间计算函数。
    """在速度和方向推力约束下构造三角形或梯形速度剖面。"""  # 解释航段时间计算边界。
    displacement = end - start  # 计算航段起点到终点的三维位移。
    length = displacement.norm()  # 计算航段三维长度。
    if length <= 0.0:  # 判断航段是否具有正长度。
        raise ValueError("飞行航段起点和终点不能重合")  # 抛出明确的零长度航段异常。
    direction = displacement.normalized()  # 计算航段单位航向。
    acceleration_limit, braking_limit = limits.directional_accelerations(direction)  # 计算航向加速和制动能力。
    unconstrained_peak_speed = math.sqrt(2.0 * length / (1.0 / acceleration_limit + 1.0 / braking_limit))  # 计算无最大速度约束时的三角形峰值速度。
    if unconstrained_peak_speed <= limits.maximum_speed:  # 判断航段是否无需恒速巡航即可完成。
        peak_speed = unconstrained_peak_speed  # 使用三角形速度剖面的自然峰值速度。
        acceleration_duration = peak_speed / acceleration_limit  # 计算加速阶段持续时间。
        braking_duration = peak_speed / braking_limit  # 计算制动阶段持续时间。
        acceleration_distance = 0.5 * peak_speed**2 / acceleration_limit  # 计算加速阶段移动距离。
        braking_distance = 0.5 * peak_speed**2 / braking_limit  # 计算制动阶段移动距离。
        phases = (  # 构造三角形速度剖面的两个运动阶段。
            MotionPhase("accelerate", acceleration_duration, acceleration_distance, 0.0, peak_speed, direction, direction.scaled(acceleration_limit)),  # 构造沿航向加速阶段。
            MotionPhase("decelerate", braking_duration, braking_distance, peak_speed, 0.0, direction, direction.scaled(-braking_limit)),  # 构造沿反航向制动阶段。
        )  # 结束三角形速度剖面阶段集合。
    else:  # 处理航段达到最大速度并包含巡航阶段的情况。
        peak_speed = limits.maximum_speed  # 将航段峰值速度限制为给定最大速度。
        acceleration_duration = peak_speed / acceleration_limit  # 计算达到最大速度所需加速时间。
        braking_duration = peak_speed / braking_limit  # 计算从最大速度制动到零所需时间。
        acceleration_distance = 0.5 * peak_speed**2 / acceleration_limit  # 计算达到最大速度所需加速距离。
        braking_distance = 0.5 * peak_speed**2 / braking_limit  # 计算从最大速度制动到零所需距离。
        cruise_distance = max(0.0, length - acceleration_distance - braking_distance)  # 计算速度受限后的剩余巡航距离。
        cruise_duration = cruise_distance / peak_speed  # 计算恒速巡航阶段持续时间。
        phases = (  # 构造梯形速度剖面的三个运动阶段。
            MotionPhase("accelerate", acceleration_duration, acceleration_distance, 0.0, peak_speed, direction, direction.scaled(acceleration_limit)),  # 构造沿航向加速阶段。
            MotionPhase("cruise", cruise_duration, cruise_distance, peak_speed, peak_speed, direction, Vector3(0.0, 0.0, 0.0)),  # 构造零实际加速度巡航阶段。
            MotionPhase("decelerate", braking_duration, braking_distance, peak_speed, 0.0, direction, direction.scaled(-braking_limit)),  # 构造沿反航向制动阶段。
        )  # 结束梯形速度剖面阶段集合。
    natural_duration = sum(phase.duration for phase in phases)  # 计算速度和推力约束下的物理最短航段时间。
    if natural_duration > limits.maximum_segment_time + 1.0e-12:  # 判断物理最短时间是否超过式 21 航段时间上界。
        raise ValueError("物理最短飞行时间超过航段时间上界")  # 拒绝不能在给定时间上界内完成的航段。
    if natural_duration < limits.minimum_segment_time:  # 判断最短时间是否短于式 21 航段时间下界。
        time_scale = limits.minimum_segment_time / natural_duration  # 计算保持空间路径不变所需的时间伸缩比例。
        phases = tuple(  # 通过时间伸缩降低速度和实际加速度并保持每阶段移动距离。
            MotionPhase(
                phase.phase_type,
                phase.duration * time_scale,
                phase.distance,
                phase.start_speed / time_scale,
                phase.end_speed / time_scale,
                phase.direction,
                phase.acceleration.scaled(1.0 / time_scale**2),
            )
            for phase in phases
        )
        peak_speed /= time_scale  # 按相同时间比例降低航段峰值速度。
    return FlightSegmentProfile(start, end, direction, length, peak_speed, phases)  # 返回不可变航段最短时间剖面。


def _bounded_dynamic_segments(
    start: Vector3,
    end: Vector3,
    limits: FlightDynamicLimits,
) -> tuple[tuple[Vector3, ...], tuple[FlightSegmentProfile, ...]]:
    """Return the smallest uniform partition satisfying the segment-time bound."""
    displacement = end - start
    for subsegment_count in range(1, limits.maximum_subsegments_per_leg + 1):
        points = tuple(
            start + displacement.scaled(index / subsegment_count)
            for index in range(subsegment_count + 1)
        )
        try:
            profiles = tuple(
                minimum_time_segment(left, right, limits)
                for left, right in zip(points[:-1], points[1:])
            )
        except ValueError as error:
            if "航段时间上界" not in str(error):
                raise
            continue
        return points, profiles
    raise ValueError("物理最短飞行时间超过航段时间上界")


def _clearance_margins(path: Polyline3D, terrain: TerrainSurface, obstacle_field: ObstacleField, required_clearance: float, sample_spacing: float) -> tuple[float, float]:  # 定义航路地形和显式障碍净空余量计算函数。
    """返回采样航路相对于规定净空的最小地形余量和显式障碍余量。"""  # 解释内部净空计算语义。
    terrain_margin = math.inf  # 初始化最小地形净空余量。
    obstacle_margin = math.inf  # 初始化最小显式障碍净空余量。
    for sample in path.sample(sample_spacing):  # 逐一检查航路采样点。
        sample_terrain_margin = sample.z - terrain.height_at(sample.x, sample.y) - required_clearance  # 计算采样点相对于规定值的地形净空余量。
        sample_obstacle_margin = obstacle_field.signed_clearance(sample) - required_clearance  # 计算采样点相对于规定值的显式障碍净空余量。
        terrain_margin = min(terrain_margin, sample_terrain_margin)  # 更新航路最小地形净空余量。
        obstacle_margin = min(obstacle_margin, sample_obstacle_margin)  # 更新航路最小显式障碍净空余量。
    return terrain_margin, obstacle_margin  # 返回地形和显式障碍两类最小净空余量。


def _remove_consecutive_duplicates(points: tuple[Vector3, ...]) -> tuple[Vector3, ...]:  # 定义连续重复航路点删除函数。
    """保留输入顺序并删除仅与前一个点相同的航路点。"""  # 解释航路点清理规则。
    cleaned: list[Vector3] = []  # 创建清理后的航路点列表。
    for point in points:  # 逐一处理输入航路点。
        if not cleaned or cleaned[-1] != point:  # 判断当前点是否不同于已保留的最后一个点。
            cleaned.append(point)  # 将非连续重复点加入清理结果。
    return tuple(cleaned)  # 返回不可变的清理后航路点集合。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义地形认证航路。
class FlightCorridor(SceneRelation):  # 定义两个场景位置对象之间的三维飞行关系。
    """保存最终空间折线、净空余量、构造方式和认证状态。"""  # 解释飞行航路职责。

    path: Polyline3D  # 保存认证或拒绝时评估的三维航路折线。
    required_clearance: float  # 保存航路规定的最小地形和障碍净空。
    sample_spacing: float  # 保存净空认证最大采样间隔。
    minimum_clearance_margin: float  # 保存扣除规定净空后的最小剩余余量。
    certified: bool  # 保存航路是否通过物理净空认证。
    construction: str  # 保存直达或抬升巡航下降等航路构造方式。
    rejection_reason: str = ""  # 保存航路未通过认证时的稳定拒绝原因。

    def __post_init__(self) -> None:  # 定义飞行航路初始化后的参数验证方法。
        """验证关系端点、净空、采样间隔和状态字段。"""  # 解释飞行航路输入约束。
        SceneRelation.__post_init__(self)  # 显式复用场景关系的标识和端点验证。
        if self.required_clearance < 0.0:  # 判断规定净空是否为非负数。
            raise ValueError("飞行航路规定净空不能为负数")  # 抛出明确的航路净空异常。
        if self.sample_spacing <= 0.0:  # 判断净空采样间隔是否为正数。
            raise ValueError("飞行航路采样间隔必须为正数")  # 抛出明确的航路采样异常。
        if not self.construction.strip():  # 判断航路构造方式是否为空白。
            raise ValueError("飞行航路构造方式不能为空")  # 抛出明确的航路构造方式异常。
        if not self.certified and not self.rejection_reason.strip():  # 判断拒绝航路是否缺少拒绝原因。
            raise ValueError("未认证飞行航路必须记录拒绝原因")  # 抛出明确的航路拒绝记录异常。

    @classmethod  # 声明航路认证替代构造方法。
    def certify(cls, object_id: str, name: str, source_id: str, target_id: str, start: Vector3, end: Vector3, terrain: TerrainSurface, obstacle_field: ObstacleField, required_clearance: float, sample_spacing: float) -> "FlightCorridor":  # 定义直达和抬升航路认证方法。
        """优先认证直达航路并在仅受地形阻挡时尝试保守抬升航路。"""  # 解释航路认证策略和边界。
        direct_path = Polyline3D((start, end))  # 构建起点到终点的直达三维航路。
        terrain_margin, obstacle_margin = _clearance_margins(direct_path, terrain, obstacle_field, required_clearance, sample_spacing)  # 计算直达航路两类净空余量。
        if terrain_margin >= 0.0 and obstacle_margin >= 0.0:  # 判断直达航路是否同时满足地形和显式障碍净空。
            return cls(object_id, name, source_id, target_id, direct_path, required_clearance, sample_spacing, min(terrain_margin, obstacle_margin), True, "direct", "")  # 返回通过认证的直达航路。
        if obstacle_margin < 0.0:  # 判断直达航路是否受到显式三维障碍限制。
            return cls(object_id, name, source_id, target_id, direct_path, required_clearance, sample_spacing, min(terrain_margin, obstacle_margin), False, "rejected", "explicit_obstacle_requires_reroute")  # 返回需要未来一般三维绕障算法的拒绝结果。
        horizontal_path = Polyline3D((Vector3(start.x, start.y, 0.0), Vector3(end.x, end.y, 0.0)))  # 构建仅用于地形查询的水平投影航路。
        terrain_samples = horizontal_path.sample(sample_spacing)  # 按净空认证间隔采样水平投影航路。
        safe_elevation = max(terrain.height_at(point.x, point.y) for point in terrain_samples) + required_clearance  # 计算覆盖沿线最高地形的保守巡航高程。
        safe_elevation = max(safe_elevation, start.z, end.z)  # 确保保守巡航高程不低于两个航路端点。
        lifted_points = _remove_consecutive_duplicates((start, Vector3(start.x, start.y, safe_elevation), Vector3(end.x, end.y, safe_elevation), end))  # 构造并清理抬升巡航下降航路点。
        lifted_path = Polyline3D(lifted_points)  # 创建抬升巡航下降三维折线。
        lifted_terrain_margin, lifted_obstacle_margin = _clearance_margins(lifted_path, terrain, obstacle_field, required_clearance, sample_spacing)  # 计算抬升航路两类净空余量。
        if lifted_terrain_margin >= -1.0e-12 and lifted_obstacle_margin >= 0.0:  # 判断抬升航路是否在数值容差内满足全部净空。
            minimum_margin = min(max(0.0, lifted_terrain_margin), lifted_obstacle_margin)  # 截断微小负浮点误差并计算最小剩余净空。
            return cls(object_id, name, source_id, target_id, lifted_path, required_clearance, sample_spacing, minimum_margin, True, "climb_cruise_descent", "")  # 返回通过认证的抬升巡航下降航路。
        return cls(object_id, name, source_id, target_id, lifted_path, required_clearance, sample_spacing, min(lifted_terrain_margin, lifted_obstacle_margin), False, "rejected", "terrain_clearance_unresolved")  # 返回抬升后仍不满足净空的拒绝结果。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景关系验证接口。
        """检查航路源和目标对象是否存在并报告未认证状态。"""  # 解释飞行航路场景验证规则。
        issues: list[ValidationIssue] = []  # 创建飞行航路验证问题列表。
        if not system.contains(self.source_id):  # 判断航路源对象是否缺失。
            issues.append(ValidationIssue("missing_flight_source", self.object_id, self.source_id))  # 记录航路源对象缺失问题。
        if not system.contains(self.target_id):  # 判断航路目标对象是否缺失。
            issues.append(ValidationIssue("missing_flight_target", self.object_id, self.target_id))  # 记录航路目标对象缺失问题。
        if not self.certified:  # 判断航路是否未通过物理认证。
            issues.append(ValidationIssue("uncertified_flight_corridor", self.object_id, "", self.rejection_reason))  # 记录航路未认证问题及拒绝原因。
        return tuple(issues)  # 返回不可变飞行航路验证问题集合。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义飞行原语。
class FlightPrimitive(SceneRelation):  # 定义调度层可消费的二阶质点飞行关系。
    """保存认证航路、分段速度剖面、总时间和净空证据。"""  # 解释飞行原语职责。

    corridor_id: str  # 保存生成该原语的飞行航路标识。
    path: Polyline3D  # 保存飞行原语使用的三维空间折线。
    segments: tuple[FlightSegmentProfile, ...]  # 保存每个连续航段的最短时间状态剖面。
    duration: float  # 保存全部航段持续时间之和。
    minimum_clearance_margin: float  # 保存来源航路最小剩余净空。
    certified: bool  # 保存飞行原语是否可以进入服务列。
    rejection_reason: str = ""  # 保存飞行原语不可用时的稳定拒绝原因。
    optimized_intervals: tuple[TrajectoryInterval, ...] = ()  # 保存连续求解器返回的显式分段恒加速度区间。
    optimization_objective: float | None = None  # 保存式 34 服务级连续优化目标值。
    maximum_equality_residual: float | None = None  # 保存连续轨迹独立审计最大等式残差。
    minimum_inequality_margin: float | None = None  # 保存连续轨迹独立审计最小不等式余量。

    def __post_init__(self) -> None:  # 定义飞行原语初始化后的参数验证方法。
        """验证关系端点、航路引用、持续时间和认证状态。"""  # 解释飞行原语输入约束。
        SceneRelation.__post_init__(self)  # 显式复用场景关系的标识和端点验证。
        if not self.corridor_id.strip():  # 判断来源飞行航路标识是否为空白。
            raise ValueError("飞行原语航路标识不能为空")  # 抛出明确的航路引用异常。
        if self.certified and (not math.isfinite(self.duration) or self.duration <= 0.0):  # 判断认证原语持续时间是否为正有限数值。
            raise ValueError("认证飞行原语持续时间必须为正有限数值")  # 抛出明确的原语持续时间异常。
        if not self.certified and not self.rejection_reason.strip():  # 判断未认证原语是否缺少拒绝原因。
            raise ValueError("未认证飞行原语必须记录拒绝原因")  # 抛出明确的原语拒绝记录异常。
        if self.optimized_intervals:  # 判断当前原语是否由连续求解器显式生成。
            if self.segments:  # 判断优化原语是否同时保留旧最短时间分段。
                raise ValueError("优化飞行原语不能同时保存固定分段剖面")  # 拒绝两套状态来源并存造成证据冲突。
            interval_duration = sum(interval.duration for interval in self.optimized_intervals)  # 汇总显式优化区间持续时间。
            if not math.isclose(interval_duration, self.duration, rel_tol=0.0, abs_tol=1.0e-8):  # 判断原语持续时间是否与显式区间一致。
                raise ValueError("优化飞行原语持续时间与显式区间不一致")  # 拒绝下游调度时间和连续状态冲突。
            evidence_values = (self.optimization_objective, self.maximum_equality_residual, self.minimum_inequality_margin)  # 汇总优化原语必须保存的求解和审计证据。
            if any(value is None or not math.isfinite(value) for value in evidence_values):  # 判断优化证据是否缺失或非有限。
                raise ValueError("优化飞行原语必须保存有限目标和约束审计证据")  # 拒绝不可追踪的连续求解结果。
        elif any(value is not None for value in (self.optimization_objective, self.maximum_equality_residual, self.minimum_inequality_margin)):  # 判断固定原语是否错误携带部分优化证据。
            raise ValueError("固定飞行原语不能保存不完整的优化证据")  # 拒绝错误标记固定轨迹为优化结果。

    @property  # 将连续优化状态暴露为只读属性。
    def optimized(self) -> bool:  # 定义飞行原语是否来自连续求解器的读取方法。
        """返回当前原语是否保存经过审计的显式优化区间。"""  # 解释优化状态判定来源。
        return bool(self.optimized_intervals)  # 仅以完整显式区间作为优化证据。

    @classmethod  # 声明飞行原语替代构造方法。
    def from_corridor(cls, object_id: str, name: str, corridor: FlightCorridor, limits: FlightDynamicLimits) -> "FlightPrimitive":  # 定义从认证航路生成飞行原语的方法。
        """为航路的每个连续空间段生成零端点速度最短时间剖面。"""  # 解释原语生成方法和保守边界。
        if not corridor.certified:  # 判断来源飞行航路是否未通过认证。
            return cls(object_id, name, corridor.source_id, corridor.target_id, corridor.object_id, corridor.path, (), math.inf, corridor.minimum_clearance_margin, False, corridor.rejection_reason)  # 返回继承航路拒绝原因的未认证原语。
        primitive_points = [corridor.path.points[0]]  # 使用认证航路起点初始化动力学离散路径。
        segment_profiles: list[FlightSegmentProfile] = []  # 创建有界动力学航段剖面列表。
        try:  # 将航段时间界不可行转换为可审计的原语拒绝状态。
            for start, end in zip(corridor.path.points[:-1], corridor.path.points[1:]):  # 逐一处理认证航路的连续几何段。
                points, profiles = _bounded_dynamic_segments(start, end, limits)  # 寻找满足时间上界的最小有界离散。
                primitive_points.extend(points[1:])  # 追加当前几何段的非重复动力学节点。
                segment_profiles.extend(profiles)  # 追加当前几何段的动力学剖面。
        except ValueError as error:  # 捕获航段时间上界小于物理最短时间的领域异常。
            if "航段时间上界" not in str(error):  # 判断当前异常是否不属于式 21 时间上界不可行。
                raise  # 继续传播与时间认证无关的编程或数据异常。
            return cls(object_id, name, corridor.source_id, corridor.target_id, corridor.object_id, corridor.path, (), math.inf, corridor.minimum_clearance_margin, False, "segment_time_above_maximum")  # 返回具有稳定时间拒绝原因的未认证原语。
        segments = tuple(segment_profiles)  # 固化全部有界动力学航段剖面。
        path = Polyline3D(tuple(primitive_points))  # 将动力学离散节点保存为原语路径。
        duration = sum(segment.duration for segment in segments)  # 汇总全部航段持续时间。
        return cls(object_id, name, corridor.source_id, corridor.target_id, corridor.object_id, path, segments, duration, corridor.minimum_clearance_margin, True, "")  # 返回通过动力学计算的飞行原语。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景关系验证接口。
        """检查原语端点和来源航路是否存在并报告未认证状态。"""  # 解释飞行原语场景验证规则。
        issues: list[ValidationIssue] = []  # 创建飞行原语验证问题列表。
        if not system.contains(self.source_id):  # 判断飞行原语源对象是否缺失。
            issues.append(ValidationIssue("missing_flight_source", self.object_id, self.source_id))  # 记录飞行原语源对象缺失问题。
        if not system.contains(self.target_id):  # 判断飞行原语目标对象是否缺失。
            issues.append(ValidationIssue("missing_flight_target", self.object_id, self.target_id))  # 记录飞行原语目标对象缺失问题。
        if not system.contains(self.corridor_id):  # 判断飞行原语来源航路是否缺失。
            issues.append(ValidationIssue("missing_flight_corridor", self.object_id, self.corridor_id))  # 记录来源飞行航路缺失问题。
        if not self.certified:  # 判断飞行原语是否未通过认证。
            issues.append(ValidationIssue("uncertified_flight_primitive", self.object_id, "", self.rejection_reason))  # 记录飞行原语未认证问题及拒绝原因。
        return tuple(issues)  # 返回不可变飞行原语验证问题集合。

    def phases(self) -> tuple[MotionPhase, ...]:  # 定义全部运动阶段读取方法。
        """按照航路顺序展平并返回全部航段运动阶段。"""  # 解释阶段集合输出顺序。
        if self.optimized_intervals:  # 判断当前原语是否使用显式连续优化状态。
            phases: list[MotionPhase] = []  # 创建兼容可视化和阶段摘要的优化运动阶段列表。
            for interval in self.optimized_intervals:  # 逐一转换显式分段恒加速度区间。
                displacement = interval.end_position - interval.start_position  # 计算当前区间三维位移。
                average_velocity = interval.start_velocity + interval.end_velocity  # 构造两倍平均速度向量以确定运动方向。
                if average_velocity.norm() > 1.0e-12:  # 判断当前区间是否具有可用平均速度方向。
                    direction = average_velocity.normalized()  # 使用平均速度方向表达当前阶段。
                elif displacement.norm() > 1.0e-12:  # 判断零平均速度区间是否仍具有非零位移。
                    direction = displacement.normalized()  # 使用空间位移方向表达当前阶段。
                else:  # 处理数值退化的零位移区间。
                    direction = Vector3(1.0, 0.0, 0.0)  # 使用稳定单位轴避免摘要阶段出现零方向。
                phases.append(MotionPhase(interval.phase_type, interval.duration, displacement.norm(), interval.start_velocity.norm(), interval.end_velocity.norm(), direction, interval.acceleration))  # 创建不替代正式状态的兼容阶段摘要。
            return tuple(phases)  # 返回优化区间对应的阶段摘要。
        return tuple(phase for segment in self.segments for phase in segment.phases)  # 展平并返回全部航段运动阶段。

    def trajectory_intervals(self) -> tuple[TrajectoryInterval, ...]:  # 定义飞行原语显式状态区间展开方法。
        """按照航路顺序展开全部航段的分段恒加速度状态区间。"""
        if not self.certified:  # 判断当前飞行原语是否未通过物理认证。
            return ()  # 未认证原语不生成可供调度消费的轨迹状态。
        if self.optimized_intervals:  # 判断当前原语是否保存连续求解器显式状态。
            return self.optimized_intervals  # 直接返回经过独立审计的优化区间。
        intervals: list[TrajectoryInterval] = []  # 创建飞行原语全部显式轨迹区间列表。
        current_time = 0.0  # 将飞行原语起飞时刻定义为相对零时刻。
        for segment in self.segments:  # 按认证航路顺序遍历全部几何航段。
            segment_intervals = segment.trajectory_intervals(current_time)  # 从当前累计时刻展开当前航段状态区间。
            intervals.extend(segment_intervals)  # 将当前航段状态区间追加到飞行原语结果。
            if segment_intervals:  # 判断当前航段是否生成至少一个正时长区间。
                current_time = segment_intervals[-1].end_time  # 将累计时刻推进到当前航段终点。
        return tuple(intervals)  # 返回不可变的飞行原语显式状态区间集合。

    @property  # 将位置剖面暴露为全局符号注册表声明的只读接口。
    def position_profile(self) -> tuple[tuple[float, Vector3], ...]:  # 定义飞行原语位置状态剖面读取方法。
        """返回按相对时刻排列的轨迹位置状态。"""
        intervals = self.trajectory_intervals()  # 展开当前飞行原语全部轨迹区间。
        if not intervals:  # 判断当前飞行原语是否缺少可用轨迹区间。
            return ()  # 返回空位置剖面以表达未认证状态。
        return ((intervals[0].start_time, intervals[0].start_position), *((interval.end_time, interval.end_position) for interval in intervals))  # 返回起飞状态和全部区间终点位置。

    @property  # 将速度剖面暴露为全局符号注册表声明的只读接口。
    def velocity_profile(self) -> tuple[tuple[float, Vector3], ...]:  # 定义飞行原语速度状态剖面读取方法。
        """返回按相对时刻排列的轨迹速度状态。"""
        intervals = self.trajectory_intervals()  # 展开当前飞行原语全部轨迹区间。
        if not intervals:  # 判断当前飞行原语是否缺少可用轨迹区间。
            return ()  # 返回空速度剖面以表达未认证状态。
        return ((intervals[0].start_time, intervals[0].start_velocity), *((interval.end_time, interval.end_velocity) for interval in intervals))  # 返回起飞状态和全部区间终点速度。

    @property  # 将加速度剖面暴露为全局符号注册表声明的只读接口。
    def acceleration_profile(self) -> tuple[tuple[float, Vector3], ...]:  # 定义飞行原语加速度状态剖面读取方法。
        """返回每个分段恒加速度区间起点时刻和实际加速度。"""
        return tuple((interval.start_time, interval.acceleration) for interval in self.trajectory_intervals())  # 返回与显式轨迹区间一一对应的实际加速度记录。

    @property  # 将累计正向爬升量暴露为只读派生量。
    def positive_climb(self) -> float:  # 定义式 33 累计正向爬升量读取方法。
        """返回认证航路相邻折点正高程增量之和。"""
        if not self.certified:  # 判断当前飞行原语是否未通过物理认证。
            return math.inf  # 使用正无穷表达未认证原语缺少可用爬升量。
        return sum(max(0.0, end.z - start.z) for start, end in zip(self.path.points[:-1], self.path.points[1:]))  # 按式 33 汇总全部正向高程增量。



def evaluate_fixed_trajectory_value(primitives: tuple[FlightPrimitive, ...], expected_energy: float, weights: TrajectoryObjectiveWeights) -> float:  # 定义式 34 服务列级固定认证轨迹目标评估函数。
    """返回有序飞行原语集合的时间能耗加权值，但不执行连续轨迹优化。"""
    if not primitives:  # 判断服务列轨迹是否缺少飞行原语。
        raise ValueError("固定轨迹目标评估至少需要一条飞行原语")  # 拒绝缺少起飞至回收运动的空轨迹。
    if not all(primitive.certified for primitive in primitives):  # 判断服务列轨迹是否包含未认证飞行原语。
        raise ValueError("未认证飞行原语不能进入固定轨迹目标评估")  # 拒绝为物理不可行轨迹生成有限目标值。
    if not math.isfinite(expected_energy) or expected_energy < 0.0:  # 判断第六部分提供的期望能耗是否为有限非负数。
        raise ValueError("轨迹期望能耗必须为有限非负数")  # 抛出明确的期望能耗输入异常。
    total_duration = sum(primitive.duration for primitive in primitives)  # 汇总起飞至观测和观测至回收全部原语时间。
    return weights.dynamic_time_weight * total_duration + weights.expected_energy_weight * expected_energy  # 按式 34 计算完整固定认证轨迹的加权目标值。
