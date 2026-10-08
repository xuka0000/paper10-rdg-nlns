"""定义动态气象状态、能耗参数和不重复计量重力的概率能耗计算。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以验证参数并汇总不确定性。
from dataclasses import dataclass  # 导入数据类工具以保存不可变气象和能耗对象。

from scipy.stats import norm  # 导入正态分布分位函数以计算概率安全能耗。

from Mask.base import SceneElement, SceneObject, SceneSystem, ValidationIssue  # 导入场景元素和验证基础类型。
from Mask.flight_relation import FlightPrimitive, MotionPhase, TrajectoryInterval  # 导入飞行原语、运动阶段和优化轨迹区间以积分能耗。
from Mask.geometry import Vector3  # 导入三维向量以表达风速和相对速度。
from Mask.mobility import UnmannedAerialVehicle  # 导入无人机以读取质量和悬停功率。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义气象时段状态。
class WeatherState(SceneElement):  # 定义影响航次能耗的动态气象元素。
    """保存气象时段序号、三维风速向量和环境温度。"""  # 解释气象状态职责。

    period_index: int  # 保存滚动时域内的气象时段序号。
    wind_vector: Vector3  # 保存全局坐标系中的三维风速向量。
    temperature: float  # 保存摄氏环境温度。

    def __post_init__(self) -> None:  # 定义气象状态初始化后的参数验证方法。
        """验证场景对象字段、时段序号和有限温度。"""  # 解释气象状态输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if self.period_index < 0:  # 判断气象时段序号是否为非负整数。
            raise ValueError("气象时段序号不能为负数")  # 抛出明确的气象时段异常。
        if not math.isfinite(self.temperature):  # 判断环境温度是否为有限数值。
            raise ValueError("环境温度必须为有限数值")  # 抛出明确的环境温度异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """返回空结果以表示气象状态已经在构造阶段验证。"""  # 解释气象状态验证结果。
        return ()  # 返回无验证问题结果。


@dataclass(frozen=True, slots=True)  # 使用不可变数据类绑定一个气象实现及其条件概率。
class WeatherScenario:  # 定义同一滚动时刻下的一个不确定气象实现。
    """保存气象状态及其在当前场景集内的严格正概率。"""  # 解释气象实现职责。

    state: WeatherState  # 保存进入物理能耗模型的风矢量和环境温度。
    probability: float  # 保存当前气象实现在场景集内的条件概率。
    scenario_id: str | None = None  # 保存跨滚动时段稳定的同一不确定实现标识。

    def __post_init__(self) -> None:  # 定义气象实现初始化后的概率验证。
        """拒绝非有限、零值和负值场景概率。"""  # 解释单场景概率约束。
        if not math.isfinite(self.probability) or self.probability <= 0.0:  # 判断概率是否为严格正有限数。
            raise ValueError("weather scenario probability must be finite and positive")  # 拒绝不能采样的气象实现。
        if self.scenario_id is not None and not self.scenario_id.strip():  # 判断显式场景标识是否为空白。
            raise ValueError("weather scenario identifier cannot be empty")  # 拒绝不能跨时段对齐的显式身份。


@dataclass(frozen=True, slots=True)  # 使用不可变数据类保存一个时刻的完整气象场景集。
class WeatherScenarioSet:  # 定义服务列认证和奖励采样共享的气象场景集。
    """验证有序场景标识和概率并计算场景能耗总体矩。"""  # 解释场景集职责。

    epoch_id: str  # 保存滚动时刻或认证时段稳定标识。
    scenarios: tuple[WeatherScenario, ...]  # 保存有序气象实现集合。

    def __post_init__(self) -> None:  # 定义气象场景集初始化后的闭合验证。
        """验证非空集合、唯一标识、共同周期和归一化概率。"""  # 解释场景集输入约束。
        if not self.epoch_id.strip():  # 判断滚动时刻标识是否为空。
            raise ValueError("weather scenario epoch identifier cannot be empty")  # 拒绝无身份的场景集。
        if not self.scenarios:  # 判断场景集是否为空。
            raise ValueError("weather scenario set cannot be empty")  # 拒绝不能认证能耗的空集合。
        identifiers = self.scenario_ids  # 提取有序场景标识以检查唯一性。
        if len(set(identifiers)) != len(identifiers):  # 判断场景标识是否重复。
            raise ValueError("weather scenario identifiers must be unique")  # 拒绝重复概率质量。
        period_indices = {scenario.state.period_index for scenario in self.scenarios}  # 汇总全部气象实现的滚动时段序号。
        if len(period_indices) != 1:  # 判断同一集合是否混入多个滚动时段。
            raise ValueError("weather scenarios in one set must share a period index")  # 拒绝把时间序列误作不确定实现。
        probability_sum = math.fsum(self.probabilities)  # 使用高精度求和检查概率质量。
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-9):  # 判断场景概率是否归一化。
            raise ValueError(f"weather scenario probabilities must sum to one: {probability_sum}")  # 拒绝概率漂移。

    @property  # 将场景标识序列暴露为只读派生字段。
    def scenario_ids(self) -> tuple[str, ...]:  # 定义场景标识读取方法。
        """返回场景集中有序气象状态标识。"""  # 解释场景身份来源。
        return tuple(scenario.scenario_id or scenario.state.object_id for scenario in self.scenarios)  # 优先返回跨时段稳定标识。

    @property  # 将场景概率序列暴露为只读派生字段。
    def probabilities(self) -> tuple[float, ...]:  # 定义场景概率读取方法。
        """返回与场景标识顺序一致的概率序列。"""  # 解释概率顺序约束。
        return tuple(float(scenario.probability) for scenario in self.scenarios)  # 返回不可变概率序列。

    def aggregate_energies(
        self,
        energies: tuple[float, ...],
        *,
        conditional_standard_deviations: tuple[float, ...] | None = None,
        violation_probability: float,
    ) -> "ProbabilisticEnergy":  # 定义场景能耗矩聚合方法。
        """按全方差公式聚合场景均值、条件方差和保护分位。"""  # 解释总体矩计算边界。
        if len(energies) != len(self.scenarios):  # 判断逐场景能耗是否完整。
            raise ValueError("weather scenario energies must match the complete scenario set")  # 拒绝缺失场景能耗。
        numeric = tuple(float(value) for value in energies)  # 规范化逐场景能耗数值。
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):  # 判断能耗是否为非负有限值。
            raise ValueError("weather scenario energies must be finite and nonnegative")  # 拒绝无效物理能耗。
        conditional = (
            tuple(0.0 for _ in self.scenarios)
            if conditional_standard_deviations is None
            else tuple(float(value) for value in conditional_standard_deviations)
        )  # 未显式提供条件不确定性时保留旧的离散场景总体矩语义。
        if len(conditional) != len(self.scenarios):  # 判断每个气象实现是否都有条件标准差。
            raise ValueError(
                "weather scenario conditional standard deviations must match the complete scenario set"
            )  # 拒绝温度或风速不确定性追踪缺项。
        if any(not math.isfinite(value) or value < 0.0 for value in conditional):  # 判断条件标准差是否有效。
            raise ValueError(
                "weather scenario conditional standard deviations must be finite and nonnegative"
            )  # 拒绝无效的场景内不确定性。
        mean = math.fsum(probability * energy for probability, energy in zip(self.probabilities, numeric))  # 计算式中概率加权均值。
        variance = math.fsum(
            probability * (standard_deviation**2 + (energy - mean) ** 2)
            for probability, energy, standard_deviation in zip(
                self.probabilities, numeric, conditional
            )
        )  # 按全方差公式合并场景内不确定性与场景间能耗差异。
        return ProbabilisticEnergy(mean, math.sqrt(max(variance, 0.0)), violation_probability)  # 返回统一保护能耗对象。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义能耗模型参数。
class EnergyModelParameters:  # 定义飞行、服务和不确定性能耗计算系数。
    """保存运动、加速度、服务、一次性能耗和标准差参数。"""  # 解释能耗参数职责。

    motion_coefficient: float  # 保存相对风速平方能耗系数。
    acceleration_coefficient: float  # 保存实际加速度平方能耗系数。
    climb_coefficient: float  # 保存正竖向速度功率系数。
    launch_recovery_energy: float  # 保存每个服务列一次性起飞回收能耗。
    uncertainty_base: float  # 保存能耗标准差基础项。
    uncertainty_wind_coefficient: float  # 保存风速不确定性系数。
    uncertainty_temperature_coefficient: float  # 保存温度偏差不确定性系数。
    reference_temperature: float  # 保存温度不确定性参考温度。
    violation_probability: float  # 保存能量安全允许失效概率。

    def __post_init__(self) -> None:  # 定义能耗参数初始化后的参数验证方法。
        """验证全部能耗系数的有限性、非负性和概率范围。"""  # 解释能耗参数输入约束。
        numeric_values = (self.motion_coefficient, self.acceleration_coefficient, self.climb_coefficient, self.launch_recovery_energy, self.uncertainty_base, self.uncertainty_wind_coefficient, self.uncertainty_temperature_coefficient, self.reference_temperature, self.violation_probability)  # 汇总全部能耗模型参数。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部能耗模型参数是否有限。
            raise ValueError("能耗模型参数必须为有限数值")  # 抛出明确的能耗参数有限性异常。
        nonnegative_values = (self.motion_coefficient, self.acceleration_coefficient, self.climb_coefficient, self.launch_recovery_energy, self.uncertainty_base, self.uncertainty_wind_coefficient, self.uncertainty_temperature_coefficient)  # 汇总应为非负数的能耗参数。
        if any(value < 0.0 for value in nonnegative_values):  # 判断非负能耗参数是否违反范围。
            raise ValueError("能耗系数和不确定性系数不能为负数")  # 抛出明确的能耗系数范围异常。
        if not 0.0 < self.violation_probability < 0.5:  # 判断允许失效概率是否位于安全分位有效范围。
            raise ValueError("能量安全失效概率必须位于零和二分之一之间")  # 抛出明确的能量安全概率异常。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存单个运动阶段能耗。
class PhaseEnergy:  # 定义运动阶段四类能耗分项。
    """记录悬停、相对风速、实际加速度和正爬升机械能耗。"""  # 解释阶段能耗内容。

    hover: float  # 保存阶段悬停基准能耗。
    motion: float  # 保存阶段相对风速运动能耗。
    acceleration: float  # 保存阶段实际加速度能耗。
    climb: float  # 保存阶段正爬升机械能耗。

    @property  # 将阶段总能耗暴露为只读属性。
    def total(self) -> float:  # 定义阶段总能耗读取方法。
        """返回四类阶段能耗分项之和。"""  # 解释阶段能耗汇总方法。
        return self.hover + self.motion + self.acceleration + self.climb  # 汇总并返回阶段总能耗。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存服务列能耗分解。
class EnergyBreakdown:  # 定义飞行、观测和起飞回收能耗分项。
    """记录可重构期望能耗的全部物理和服务分项。"""  # 解释能耗分解内容。

    hover: float  # 保存全部飞行阶段悬停基准能耗。
    motion: float  # 保存全部飞行阶段相对风速运动能耗。
    acceleration: float  # 保存全部飞行阶段实际加速度能耗。
    climb: float  # 保存全部飞行阶段正爬升机械能耗。
    service: float  # 保存全部检查点稳定观测服务能耗。
    launch_recovery: float  # 保存每个服务列一次性起飞回收能耗。

    @property  # 将期望总能耗暴露为只读属性。
    def total(self) -> float:  # 定义期望总能耗读取方法。
        """返回全部飞行和服务能耗分项之和。"""  # 解释期望能耗汇总方法。
        return self.hover + self.motion + self.acceleration + self.climb + self.service + self.launch_recovery  # 汇总并返回期望总能耗。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义概率能耗。
class ProbabilisticEnergy:  # 定义期望值、标准差和允许失效概率组成的能耗对象。
    """使用正态近似分位数计算服务列概率安全能耗。"""  # 解释概率能耗职责。

    mean: float  # 保存服务列期望能耗。
    standard_deviation: float  # 保存服务列能耗标准差。
    violation_probability: float  # 保存能量安全允许失效概率。

    def __post_init__(self) -> None:  # 定义概率能耗初始化后的参数验证方法。
        """验证均值、标准差和允许失效概率。"""  # 解释概率能耗输入约束。
        numeric_values = (self.mean, self.standard_deviation, self.violation_probability)  # 汇总概率能耗参数。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部概率能耗参数是否有限。
            raise ValueError("概率能耗参数必须为有限数值")  # 抛出明确的概率能耗有限性异常。
        if self.mean < 0.0 or self.standard_deviation < 0.0:  # 判断能耗均值和标准差是否为非负数。
            raise ValueError("能耗均值和标准差不能为负数")  # 抛出明确的概率能耗范围异常。
        if not 0.0 < self.violation_probability < 0.5:  # 判断允许失效概率是否位于安全分位有效范围。
            raise ValueError("能量安全失效概率必须位于零和二分之一之间")  # 抛出明确的失效概率异常。

    @property  # 将概率安全分位能耗暴露为只读属性。
    def quantile(self) -> float:  # 定义分位能耗读取方法。
        """返回均值加安全分位系数乘标准差的能耗。"""  # 解释分位能耗计算方法。
        quantile_factor = float(norm.ppf(1.0 - self.violation_probability))  # 计算给定失效概率对应的标准正态分位系数。
        return self.mean + quantile_factor * self.standard_deviation  # 返回概率安全分位能耗。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存服务能耗估计结果。
class ServiceEnergyEstimate:  # 定义能耗分解和概率安全结果的组合对象。
    """将可重构能耗分项与概率能耗参数绑定到同一结果。"""  # 解释服务能耗估计内容。

    breakdown: EnergyBreakdown  # 保存期望能耗的全部分项。
    probabilistic: ProbabilisticEnergy  # 保存期望值、标准差和分位能耗。


def calculate_phase_energy(phase: MotionPhase, weather: WeatherState, uav: UnmannedAerialVehicle, parameters: EnergyModelParameters) -> PhaseEnergy:  # 定义单个运动阶段能耗计算函数。
    """根据实际速度、风速、实际加速度和正爬升速度计算阶段能耗。"""  # 解释阶段能耗计算边界。
    relative_air_velocity = phase.average_velocity - weather.wind_vector  # 计算运动阶段平均速度相对于风速的空气速度。
    hover_energy = uav.hover_power * phase.duration  # 计算阶段悬停基准能耗。
    motion_energy = parameters.motion_coefficient * relative_air_velocity.norm() ** 2 * phase.duration  # 计算相对风速平方运动能耗。
    thrust_acceleration = phase.acceleration - Vector3(0.0, 0.0, -9.81)  # 计算式 35 中实际加速度相对于重力的推力比加速度。
    acceleration_energy = parameters.acceleration_coefficient * thrust_acceleration.norm() ** 2 * phase.duration  # 使用推力比加速度计算阶段加速度能耗。
    positive_vertical_speed = max(phase.average_velocity.z, 0.0)  # 提取阶段平均速度中的正竖向分量。
    climb_energy = parameters.climb_coefficient * positive_vertical_speed * phase.duration  # 按式 35 显式正爬升系数计算阶段能耗。
    return PhaseEnergy(hover_energy, motion_energy, acceleration_energy, climb_energy)  # 返回不可变阶段能耗分项。


def calculate_trajectory_interval_energy(interval: TrajectoryInterval, weather: WeatherState, uav: UnmannedAerialVehicle, parameters: EnergyModelParameters) -> PhaseEnergy:  # 定义优化轨迹区间的式 35 能耗计算函数。
    """根据节点速度、实际加速度和区间正爬升速度严格计算式 35 能耗。"""  # 解释优化状态计算边界。
    relative_air_velocity = interval.start_velocity - weather.wind_vector  # 使用式 35 当前轨迹节点速度计算相对空气速度。
    hover_energy = uav.hover_power * interval.duration  # 计算区间悬停基准能耗。
    motion_energy = parameters.motion_coefficient * relative_air_velocity.norm() ** 2 * interval.duration  # 计算相对风速平方运动能耗。
    thrust_acceleration = interval.acceleration - Vector3(0.0, 0.0, -9.81)  # 计算实际加速度相对于重力的推力比加速度。
    acceleration_energy = parameters.acceleration_coefficient * thrust_acceleration.norm() ** 2 * interval.duration  # 计算推力比加速度能耗。
    positive_vertical_speed = max((interval.end_position.z - interval.start_position.z) / interval.duration, 0.0)  # 计算式 35 区间正竖向速度。
    climb_energy = parameters.climb_coefficient * positive_vertical_speed * interval.duration  # 计算显式正爬升能耗。
    return PhaseEnergy(hover_energy, motion_energy, acceleration_energy, climb_energy)  # 返回优化区间四类能耗分项。


def calculate_service_energy(primitives: tuple[FlightPrimitive, ...], weather: WeatherState, uav: UnmannedAerialVehicle, parameters: EnergyModelParameters, service_duration: float) -> ServiceEnergyEstimate:  # 定义服务列飞行和观测总能耗计算函数。
    """汇总全部认证飞行阶段并加入一次性服务能耗和气象不确定性。"""  # 解释服务列能耗计算方法。
    if service_duration < 0.0 or not math.isfinite(service_duration):  # 判断总观测服务时间是否为非负有限数值。
        raise ValueError("总观测服务时间必须为非负有限数值")  # 抛出明确的服务时间异常。
    if any(not primitive.certified for primitive in primitives):  # 判断输入飞行原语是否全部通过认证。
        raise ValueError("概率能耗只能由认证飞行原语计算")  # 拒绝使用未认证原语形成能耗证据。
    hover_energy = 0.0  # 初始化全部飞行阶段悬停基准能耗。
    motion_energy = 0.0  # 初始化全部飞行阶段相对风速运动能耗。
    acceleration_energy = 0.0  # 初始化全部飞行阶段实际加速度能耗。
    climb_energy = 0.0  # 初始化全部飞行阶段正爬升机械能耗。
    flight_duration = 0.0  # 初始化全部飞行原语持续时间。
    for primitive in primitives:  # 逐一处理服务列包含的飞行原语。
        flight_duration += primitive.duration  # 累加当前飞行原语持续时间。
        energy_items = primitive.trajectory_intervals() if primitive.optimized else primitive.phases()  # 优化原语使用正式节点状态，固定原语保留阶段积分基线。
        for energy_item in energy_items:  # 逐一处理当前飞行原语的正式区间或基线阶段。
            phase_energy = calculate_trajectory_interval_energy(energy_item, weather, uav, parameters) if primitive.optimized else calculate_phase_energy(energy_item, weather, uav, parameters)  # 按原语证据层级计算能耗分项。
            hover_energy += phase_energy.hover  # 累加当前阶段悬停基准能耗。
            motion_energy += phase_energy.motion  # 累加当前阶段相对风速运动能耗。
            acceleration_energy += phase_energy.acceleration  # 累加当前阶段实际加速度能耗。
            climb_energy += phase_energy.climb  # 累加当前阶段正爬升机械能耗。
    service_energy = uav.hover_power * service_duration  # 按式 36 使用无人机悬停功率计算稳定观测服务能耗。
    breakdown = EnergyBreakdown(hover_energy, motion_energy, acceleration_energy, climb_energy, service_energy, parameters.launch_recovery_energy)  # 构造可重构的期望能耗分项。
    wind_uncertainty = parameters.uncertainty_wind_coefficient * weather.wind_vector.norm()  # 计算平均风速引起的标准差贡献。
    temperature_uncertainty = parameters.uncertainty_temperature_coefficient * (weather.temperature - parameters.reference_temperature)  # 计算平均温度偏差引起的标准差贡献。
    standard_deviation = math.sqrt(parameters.uncertainty_base**2 + wind_uncertainty**2 + temperature_uncertainty**2)  # 按式 37 使用平方和开根号汇总能耗标准差。
    probabilistic = ProbabilisticEnergy(breakdown.total, standard_deviation, parameters.violation_probability)  # 构造服务列概率能耗对象。
    return ServiceEnergyEstimate(breakdown, probabilistic)  # 返回能耗分解和概率能耗组合结果。
