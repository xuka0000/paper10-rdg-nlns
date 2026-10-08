"""定义起飞回收配对和连接物理模型与调度模型的服务列关系。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以验证服务列系数有限性。
from dataclasses import dataclass, replace  # 导入数据类工具以保存和更新不可变服务列对象。

from Mask.base import SceneRelation, SceneSystem, ValidationIssue  # 导入场景关系和验证基础类型。
from Mask.flight_relation import FlightPrimitive  # 导入飞行原语以组装服务列时间和能耗。
from Mask.infrastructure import StopSite  # 导入停靠点类型以验证起飞回收端点。
from Mask.mobility import UnmannedAerialVehicle  # 导入无人机以执行电池准入判断。
from Mask.observation_selection import ObservationSelectionPattern  # 导入列内嵌观测选择模式以建立覆盖和节点状态。
from Mask.weather import EnergyModelParameters, WeatherScenarioSet, WeatherState, calculate_service_energy  # 导入气象场景集和服务能耗计算接口。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义起飞回收配对。
class LaunchRecoveryPair(SceneRelation):  # 定义一个无人机从起飞停靠点到回收停靠点的关系。
    """保存起飞停靠点、回收停靠点和执行无人机标识。"""  # 解释起飞回收配对职责。

    uav_id: str  # 保存执行该起飞回收配对的无人机标识。

    def __post_init__(self) -> None:  # 定义起飞回收配对初始化后的参数验证方法。
        """验证关系端点和非空无人机标识。"""  # 解释起飞回收配对输入约束。
        SceneRelation.__post_init__(self)  # 显式复用场景关系的标识和端点验证。
        if not self.uav_id.strip():  # 判断执行无人机标识是否为空白。
            raise ValueError("起飞回收配对无人机标识不能为空")  # 抛出明确的无人机引用异常。

    @property  # 将起飞停靠点标识暴露为只读属性。
    def launch_stop_id(self) -> str:  # 定义起飞停靠点标识读取方法。
        """返回场景关系源端点对应的起飞停靠点标识。"""  # 解释起飞端点映射。
        return self.source_id  # 返回关系源对象标识。

    @property  # 将回收停靠点标识暴露为只读属性。
    def recovery_stop_id(self) -> str:  # 定义回收停靠点标识读取方法。
        """返回场景关系目标端点对应的回收停靠点标识。"""  # 解释回收端点映射。
        return self.target_id  # 返回关系目标对象标识。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景关系验证接口。
        """检查起飞点、回收点和无人机是否存在且类型正确。"""  # 解释起飞回收配对场景验证规则。
        issues: list[ValidationIssue] = []  # 创建起飞回收配对验证问题列表。
        for role, stop_id in (("launch", self.launch_stop_id), ("recovery", self.recovery_stop_id)):  # 逐一检查起飞和回收停靠点。
            if not system.contains(stop_id):  # 判断当前停靠点是否缺失。
                issues.append(ValidationIssue(f"missing_{role}_stop", self.object_id, stop_id))  # 记录当前停靠点缺失问题。
                continue  # 跳过缺失停靠点的类型检查。
            if not isinstance(system.get_element(stop_id), StopSite):  # 判断当前端点对象是否属于停靠点。
                issues.append(ValidationIssue(f"invalid_{role}_stop_type", self.object_id, stop_id))  # 记录当前停靠点类型错误问题。
        if not system.contains(self.uav_id):  # 判断执行无人机是否缺失。
            issues.append(ValidationIssue("missing_uav", self.object_id, self.uav_id))  # 记录执行无人机缺失问题。
        elif not isinstance(system.get_element(self.uav_id), UnmannedAerialVehicle):  # 判断执行对象是否属于无人机。
            issues.append(ValidationIssue("invalid_uav_type", self.object_id, self.uav_id))  # 记录执行无人机类型错误问题。
        return tuple(issues)  # 返回不可变起飞回收配对验证问题集合。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义服务列。
class ServiceColumn(SceneRelation):  # 定义连接下层物理模型和上层调度模型的关系对象。
    """保存任务顺序、飞行原语、时间偏移、概率能耗和准入状态。"""  # 解释服务列职责。

    launch_recovery_pair_id: str  # 保存服务列使用的起飞回收配对标识。
    uav_id: str  # 保存服务列执行无人机标识。
    weather_period_id: str  # 保存服务列使用的气象时段标识。
    observation_pattern: ObservationSelectionPattern  # 保存式 16 至式 20 的有序观测节点选择模式。
    primitive_ids: tuple[str, ...]  # 保存连接起飞点、位姿和回收点的飞行原语标识。
    start_offsets: tuple[tuple[str, float], ...]  # 保存检查点开始时间相对起飞时刻的偏移。
    completion_offsets: tuple[tuple[str, float], ...]  # 保存检查点完成时间相对起飞时刻的偏移。
    dynamic_duration: float  # 保存服务列全部飞行和服务持续时间。
    energy_mean: float  # 保存服务列期望能耗。
    energy_standard_deviation: float  # 保存服务列能耗标准差。
    energy_quantile: float  # 保存服务列概率安全分位能耗。
    minimum_clearance_margin: float  # 保存全部飞行原语的最小剩余净空。
    admitted: bool  # 保存服务列是否可以进入上层调度模型。
    rejection_reasons: tuple[str, ...]  # 保存服务列未准入时的全部稳定拒绝原因。
    trajectory_status: str = "fixed_baseline"  # 保存连续轨迹优化状态或旧基线标识。
    trajectory_objective: float | None = None  # 保存式 34 连续轨迹目标值。
    maximum_equality_residual: float | None = None  # 保存连续轨迹最大等式残差。
    minimum_inequality_margin: float | None = None  # 保存连续轨迹最小不等式余量。
    weather_scenario_ids: tuple[str, ...] = ()  # 保存认证使用的有序气象场景标识。
    weather_scenario_probabilities: tuple[float, ...] = ()  # 保存与场景标识对应的条件概率。
    weather_scenario_energies: tuple[float, ...] = ()  # 保存每个气象场景下的物理能耗。
    weather_certification_mode: str = "legacy_single_state"  # 区分正式场景集认证与开发兼容入口。
    weather_scenario_standard_deviations: tuple[float, ...] = ()  # 保存每个气象实现内由基础、风速和温度项形成的条件标准差。

    def __post_init__(self) -> None:  # 定义服务列初始化后的参数验证方法。
        """验证关系端点、标识序列、时间能耗系数和准入记录。"""  # 解释服务列输入约束。
        SceneRelation.__post_init__(self)  # 显式复用场景关系的标识和端点验证。
        identifiers = (self.launch_recovery_pair_id, self.uav_id, self.weather_period_id, *self.task_ids, *self.pose_ids, *self.primitive_ids, *self.covered_checkpoint_ids)  # 汇总服务列使用的全部稳定标识。
        if any(not identifier.strip() for identifier in identifiers):  # 判断服务列是否包含空白稳定标识。
            raise ValueError("服务列对象标识和引用标识不能为空")  # 抛出明确的服务列标识异常。
        if len(self.primitive_ids) != len(self.task_ids) + 1:  # 判断飞行原语数量是否连接全部任务位姿和两个停靠端点。
            raise ValueError("服务列飞行原语数量必须比任务数量多一")  # 抛出明确的服务列原语数量异常。
        numeric_values = (self.dynamic_duration, self.energy_mean, self.energy_standard_deviation, self.energy_quantile, self.minimum_clearance_margin)  # 汇总服务列数值系数。
        if self.admitted and not all(math.isfinite(value) for value in numeric_values):  # 判断准入服务列数值系数是否全部有限。
            raise ValueError("准入服务列的时间、能耗和净空系数必须为有限数值")  # 抛出明确的服务列有限性异常。
        if self.admitted and self.rejection_reasons:  # 判断准入服务列是否错误保留拒绝原因。
            raise ValueError("准入服务列不能包含拒绝原因")  # 抛出明确的服务列状态冲突异常。
        if not self.admitted and not self.rejection_reasons:  # 判断拒绝服务列是否缺少拒绝原因。
            raise ValueError("拒绝服务列必须记录至少一个拒绝原因")  # 抛出明确的服务列拒绝记录异常。
        if self.trajectory_status not in {"fixed_baseline", "optimized"}:  # 判断服务列轨迹状态是否属于受支持的证据层级。
            raise ValueError("服务列轨迹状态无效")  # 拒绝不能解释的轨迹来源。
        trajectory_evidence = (self.trajectory_objective, self.maximum_equality_residual, self.minimum_inequality_margin)  # 汇总连续优化证据字段。
        if self.trajectory_status == "optimized" and any(value is None or not math.isfinite(value) for value in trajectory_evidence):  # 判断优化服务列是否缺少有限目标或约束证据。
            raise ValueError("优化服务列必须保存有限轨迹目标和约束证据")  # 拒绝不可追踪的连续优化服务列。
        if self.trajectory_status == "fixed_baseline" and any(value is not None for value in trajectory_evidence):  # 判断固定基线是否错误携带部分优化证据。
            raise ValueError("固定基线服务列不能保存优化轨迹证据")  # 拒绝混淆固定评价和连续优化。
        scenario_lengths = (len(self.weather_scenario_ids), len(self.weather_scenario_probabilities), len(self.weather_scenario_energies))  # 汇总气象认证追踪字段长度。
        if self.weather_certification_mode not in {"legacy_single_state", "scenario_set", "live_measured_state"}:  # 判断气象认证来源是否可解释。
            raise ValueError("服务列气象认证模式无效")  # 拒绝未知认证路径。
        has_scenario_trace = any(scenario_lengths)  # 判断服务列是否携带新增逐场景认证追踪。
        if len(set(scenario_lengths)) != 1 or (self.weather_certification_mode == "scenario_set" and not has_scenario_trace):  # 判断正式逐场景认证追踪是否完整非空。
            raise ValueError("服务列必须保存完整且非空的气象场景认证追踪")  # 拒绝缺失概率或能耗的正式服务列。
        if has_scenario_trace:  # 对新组装列验证场景身份和概率闭合。
            if len(set(self.weather_scenario_ids)) != len(self.weather_scenario_ids):  # 判断气象场景标识是否重复。
                raise ValueError("服务列气象场景标识必须唯一")  # 拒绝重复场景概率质量。
            probability_sum = math.fsum(self.weather_scenario_probabilities)  # 汇总服务列认证概率。
            if any(not math.isfinite(value) or value <= 0.0 for value in self.weather_scenario_probabilities) or not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-9):  # 判断场景概率是否严格正且归一化。
                raise ValueError("服务列气象场景概率必须为严格正有限数且总和为一")  # 拒绝不可采样概率追踪。
        if self.weather_certification_mode == "scenario_set" and self.admitted:  # 对正式准入列复算全部场景总体矩。
            if any(not math.isfinite(value) or value < 0.0 for value in self.weather_scenario_energies):  # 判断逐场景能耗是否有效。
                raise ValueError("准入服务列逐场景能耗必须为非负有限数")  # 拒绝无效场景能耗证据。
            if len(self.weather_scenario_standard_deviations) != len(self.weather_scenario_ids):  # 判断温度和风速条件不确定性是否逐场景闭合。
                raise ValueError("正式服务列必须保存完整的逐场景条件标准差")  # 拒绝在正式保护能耗中丢失温度不确定性。
            if any(not math.isfinite(value) or value < 0.0 for value in self.weather_scenario_standard_deviations):  # 判断逐场景条件标准差是否有效。
                raise ValueError("准入服务列逐场景条件标准差必须为非负有限数")  # 拒绝不可复算的场景内不确定性。
            scenario_mean = math.fsum(probability * energy for probability, energy in zip(self.weather_scenario_probabilities, self.weather_scenario_energies))  # 复算概率加权均值。
            scenario_variance = math.fsum(
                probability * (standard_deviation**2 + (energy - scenario_mean) ** 2)
                for probability, energy, standard_deviation in zip(
                    self.weather_scenario_probabilities,
                    self.weather_scenario_energies,
                    self.weather_scenario_standard_deviations,
                )
            )  # 按全方差公式复算场景内与场景间不确定性。
            if not math.isclose(self.energy_mean, scenario_mean, rel_tol=1.0e-9, abs_tol=1.0e-7):  # 判断保存均值是否与场景追踪一致。
                raise ValueError("服务列期望能耗与逐场景认证追踪不一致")  # 拒绝均值证据漂移。
            if not math.isclose(self.energy_standard_deviation, math.sqrt(max(scenario_variance, 0.0)), rel_tol=1.0e-9, abs_tol=1.0e-7):  # 判断保存标准差是否与场景追踪一致。
                raise ValueError("服务列能耗标准差与逐场景认证追踪不一致")  # 拒绝方差证据漂移。

    @property  # 将起飞停靠点标识暴露为只读属性。
    def launch_stop_id(self) -> str:  # 定义服务列起飞停靠点标识读取方法。
        """返回场景关系源端点对应的起飞停靠点标识。"""  # 解释服务列起飞端点映射。
        return self.source_id  # 返回关系源对象标识。

    @property  # 将回收停靠点标识暴露为只读属性。
    def recovery_stop_id(self) -> str:  # 定义服务列回收停靠点标识读取方法。
        """返回场景关系目标端点对应的回收停靠点标识。"""  # 解释服务列回收端点映射。
        return self.target_id  # 返回关系目标对象标识。

    @property  # 将任务标识序列暴露为选择模式的只读派生参数。
    def task_ids(self) -> tuple[str, ...]:  # 定义服务列有序任务标识读取方法。
        """返回观测选择模式中的有序巡检任务标识。"""  # 解释任务标识来源。
        return self.observation_pattern.task_ids  # 从唯一选择模式派生任务序列。

    @property  # 将位姿标识序列暴露为选择模式的只读派生参数。
    def pose_ids(self) -> tuple[str, ...]:  # 定义服务列有序位姿标识读取方法。
        """返回观测选择模式中的有序候选位姿标识。"""  # 解释位姿标识来源。
        return self.observation_pattern.pose_ids  # 从唯一选择模式派生位姿序列。

    @property  # 将检查点覆盖序列暴露为选择模式的只读派生参数。
    def covered_checkpoint_ids(self) -> tuple[str, ...]:  # 定义服务列覆盖检查点读取方法。
        """返回由式 17 观测选择结果派生的检查点覆盖序列。"""  # 解释覆盖参数来源。
        return self.observation_pattern.covered_checkpoint_ids  # 从唯一选择模式派生覆盖集合。

    @classmethod  # 声明服务列组装替代构造方法。
    def assemble(cls, object_id: str, name: str, pair: LaunchRecoveryPair, weather: WeatherState, observation_pattern: ObservationSelectionPattern, primitives: tuple[FlightPrimitive, ...], uav: UnmannedAerialVehicle, parameters: EnergyModelParameters) -> "ServiceColumn":  # 定义从物理对象组装服务列的方法。
        """按照任务顺序汇总时间偏移、概率能耗、净空和电池准入状态。"""  # 解释服务列组装方法。
        assignments = observation_pattern.assignments  # 读取已经通过式 16 至式 20 约束的有序观测节点。
        if len(primitives) != len(assignments) + 1:  # 判断飞行原语数量是否连接全部任务位姿和两个停靠点。
            raise ValueError("服务列飞行原语数量必须比任务数量多一")  # 抛出明确的飞行原语数量异常。
        if pair.uav_id != uav.object_id:  # 判断起飞回收配对和输入无人机是否一致。
            raise ValueError("起飞回收配对与服务列无人机不一致")  # 抛出明确的无人机映射异常。
        cumulative_time = 0.0  # 初始化相对于起飞时刻的累计时间。
        start_offsets: list[tuple[str, float]] = []  # 创建检查点开始时间偏移列表。
        completion_offsets: list[tuple[str, float]] = []  # 创建检查点完成时间偏移列表。
        for sequence_index, assignment in enumerate(assignments):  # 按服务顺序处理每个选中观测节点。
            cumulative_time += primitives[sequence_index].duration  # 累加到达当前任务位姿的飞行时间。
            start_offsets.append((assignment.checkpoint_id, cumulative_time))  # 记录当前检查点开始时间偏移。
            cumulative_time += assignment.service_duration  # 累加当前检查点稳定观测服务时间。
            completion_offsets.append((assignment.checkpoint_id, cumulative_time))  # 记录当前检查点完成时间偏移。
        cumulative_time += primitives[-1].duration  # 累加最后观测位姿到回收停靠点的飞行时间。
        all_primitives_certified = all(primitive.certified for primitive in primitives)  # 判断全部飞行原语是否通过物理认证。
        rejection_reasons: list[str] = []  # 创建服务列拒绝原因列表。
        if not all_primitives_certified:  # 判断服务列是否包含未认证飞行原语。
            rejection_reasons.append("uncertified_flight_primitive")  # 记录未认证飞行原语拒绝原因。
            energy_mean = math.inf  # 使用无穷值标记未认证服务列期望能耗不可用。
            energy_standard_deviation = math.inf  # 使用无穷值标记未认证服务列能耗标准差不可用。
            energy_quantile = math.inf  # 使用无穷值标记未认证服务列分位能耗不可用。
        else:  # 处理全部飞行原语均通过认证的服务列。
            total_service_duration = sum(assignment.service_duration for assignment in assignments)  # 汇总全部检查点稳定观测服务时间。
            energy_estimate = calculate_service_energy(primitives, weather, uav, parameters, total_service_duration)  # 计算服务列能耗分解和概率能耗。
            energy_mean = energy_estimate.probabilistic.mean  # 读取服务列期望能耗。
            energy_standard_deviation = energy_estimate.probabilistic.standard_deviation  # 读取服务列能耗标准差。
            energy_quantile = energy_estimate.probabilistic.quantile  # 读取服务列概率安全分位能耗。
            if energy_quantile > uav.available_battery:  # 判断服务列分位能耗是否超过单航次可用电池。
                rejection_reasons.append("insufficient_battery")  # 记录电池能量不足拒绝原因。
        minimum_clearance_margin = min(primitive.minimum_clearance_margin for primitive in primitives)  # 计算全部飞行原语的最小剩余净空。
        admitted = not rejection_reasons  # 仅在不存在任何拒绝原因时准入服务列。
        primitive_ids = tuple(primitive.object_id for primitive in primitives)  # 提取服务列有序飞行原语标识。
        all_primitives_optimized = all(primitive.optimized for primitive in primitives)  # 判断当前服务列是否全部由连续求解器生成。
        if all_primitives_optimized:  # 处理具有完整式 34 求解证据的服务列。
            objective_values = tuple(float(primitive.optimization_objective) for primitive in primitives if primitive.optimization_objective is not None)  # 读取每条原语保存的服务级目标值。
            if len(objective_values) != len(primitives) or any(not math.isclose(value, objective_values[0], rel_tol=0.0, abs_tol=1.0e-8) for value in objective_values[1:]):  # 判断服务级目标证据是否在全部原语间一致。
                raise ValueError("优化服务列原语的式 34 目标证据不一致")  # 拒绝混合不同连续求解结果的原语集合。
            trajectory_status = "optimized"  # 标记正式连续优化服务列。
            trajectory_objective = objective_values[0]  # 保存共同的服务级式 34 目标值。
            maximum_equality_residual = max(float(primitive.maximum_equality_residual) for primitive in primitives if primitive.maximum_equality_residual is not None)  # 汇总最坏等式残差。
            minimum_inequality_margin = min(float(primitive.minimum_inequality_margin) for primitive in primitives if primitive.minimum_inequality_margin is not None)  # 汇总最坏不等式余量。
        else:  # 处理未注入连续求解器的兼容基线路径。
            trajectory_status = "fixed_baseline"  # 明确标记固定轨迹基线而非连续优化。
            trajectory_objective = None  # 固定基线不保存式 34 优化目标证据。
            maximum_equality_residual = None  # 固定基线不保存连续等式审计证据。
            minimum_inequality_margin = None  # 固定基线不保存连续不等式审计证据。
        return cls(object_id, name, pair.launch_stop_id, pair.recovery_stop_id, pair.object_id, uav.object_id, weather.object_id, observation_pattern, primitive_ids, tuple(start_offsets), tuple(completion_offsets), cumulative_time, energy_mean, energy_standard_deviation, energy_quantile, minimum_clearance_margin, admitted, tuple(rejection_reasons), trajectory_status, trajectory_objective, maximum_equality_residual, minimum_inequality_margin, (weather.object_id,), (1.0,), (energy_mean,), "legacy_single_state")  # 返回带单状态兼容追踪的不可变服务列关系。

    @classmethod  # 声明正式气象场景集服务列组装方法。
    def assemble_scenarios(cls, object_id: str, name: str, pair: LaunchRecoveryPair, weather_scenarios: WeatherScenarioSet, observation_pattern: ObservationSelectionPattern, primitives: tuple[FlightPrimitive, ...], uav: UnmannedAerialVehicle, parameters: EnergyModelParameters) -> "ServiceColumn":  # 定义逐场景认证服务列入口。
        """逐场景计算物理能耗并由概率总体矩执行电池准入。"""  # 解释正式场景认证方法。
        representative = weather_scenarios.scenarios[0].state  # 使用首个状态复用时间、轨迹和关系组装逻辑。
        base = cls.assemble(object_id, name, pair, representative, observation_pattern, primitives, uav, parameters)  # 构造共享的时间、轨迹和引用字段。
        if not all(primitive.certified for primitive in primitives):  # 判断物理原语是否允许计算场景能耗。
            scenario_energies = tuple(math.inf for _ in weather_scenarios.scenarios)  # 为未认证列保存完整但不可用的场景位置。
            aggregate_mean = math.inf  # 标记未认证列均值不可用。
            aggregate_standard_deviation = math.inf  # 标记未认证列标准差不可用。
            aggregate_quantile = math.inf  # 标记未认证列保护能耗不可用。
        else:  # 处理通过轨迹认证的服务列。
            total_service_duration = sum(assignment.service_duration for assignment in observation_pattern.assignments)  # 汇总稳定观测服务时间。
            scenario_estimates = tuple(calculate_service_energy(primitives, scenario.state, uav, parameters, total_service_duration) for scenario in weather_scenarios.scenarios)  # 对每个气象实现重算物理能耗及其条件不确定性。
            scenario_energies = tuple(estimate.breakdown.total for estimate in scenario_estimates)  # 保存逐场景条件均值供奖励与认证共同追踪。
            scenario_standard_deviations = tuple(estimate.probabilistic.standard_deviation for estimate in scenario_estimates)  # 保留温度和风速对每个实现内标准差的贡献。
            aggregate = weather_scenarios.aggregate_energies(scenario_energies, conditional_standard_deviations=scenario_standard_deviations, violation_probability=parameters.violation_probability)  # 按全方差公式计算正式保护能耗。
            aggregate_mean = aggregate.mean  # 保存概率加权期望能耗。
            aggregate_standard_deviation = aggregate.standard_deviation  # 保存场景总体标准差。
            aggregate_quantile = aggregate.quantile  # 保存正态矩匹配保护能耗。
        if not all(primitive.certified for primitive in primitives):  # 为未认证列补齐不可用的条件标准差位置。
            scenario_standard_deviations = tuple(math.inf for _ in weather_scenarios.scenarios)  # 保持正式追踪数组严格对齐。
        rejection_reasons = [reason for reason in base.rejection_reasons if reason != "insufficient_battery"]  # 删除单状态兼容入口产生的电池判定。
        if aggregate_quantile > uav.available_battery and "insufficient_battery" not in rejection_reasons:  # 使用场景保护能耗重新执行电池准入。
            rejection_reasons.append("insufficient_battery")  # 记录场景集保护能耗超限。
        return replace(base, energy_mean=aggregate_mean, energy_standard_deviation=aggregate_standard_deviation, energy_quantile=aggregate_quantile, admitted=not rejection_reasons, rejection_reasons=tuple(rejection_reasons), weather_scenario_ids=weather_scenarios.scenario_ids, weather_scenario_probabilities=weather_scenarios.probabilities, weather_scenario_energies=scenario_energies, weather_certification_mode="scenario_set", weather_scenario_standard_deviations=scenario_standard_deviations)  # 返回包含条件不确定性的完整场景追踪服务列。

    def recertify_measured_weather(
        self,
        weather: WeatherState,
        primitives: tuple[FlightPrimitive, ...],
        uav: UnmannedAerialVehicle,
        parameters: EnergyModelParameters,
    ) -> "ServiceColumn":
        """Recompute one existing trajectory under the current measured weather."""
        if tuple(primitive.object_id for primitive in primitives) != self.primitive_ids:
            raise ValueError(f"service column primitive order changed: {self.object_id}")
        if uav.object_id != self.uav_id:
            raise ValueError(f"service column UAV changed: {self.object_id}")
        service_duration = math.fsum(
            assignment.service_duration
            for assignment in self.observation_pattern.assignments
        )
        estimate = calculate_service_energy(
            primitives,
            weather,
            uav,
            parameters,
            service_duration,
        )
        rejection_reasons = [
            reason for reason in self.rejection_reasons
            if reason != "insufficient_battery"
        ]
        if estimate.probabilistic.quantile > uav.available_battery:
            rejection_reasons.append("insufficient_battery")
        return replace(
            self,
            energy_mean=estimate.probabilistic.mean,
            energy_standard_deviation=estimate.probabilistic.standard_deviation,
            energy_quantile=estimate.probabilistic.quantile,
            admitted=not rejection_reasons,
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            weather_scenario_ids=(weather.object_id,),
            weather_scenario_probabilities=(1.0,),
            weather_scenario_energies=(estimate.breakdown.total,),
            weather_certification_mode="live_measured_state",
            weather_scenario_standard_deviations=(
                estimate.probabilistic.standard_deviation,
            ),
        )

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景关系验证接口。
        """检查服务列引用对象是否存在并报告未准入状态。"""  # 解释服务列场景验证规则。
        issues: list[ValidationIssue] = []  # 创建服务列验证问题列表。
        referenced_ids = (self.launch_stop_id, self.recovery_stop_id, self.launch_recovery_pair_id, self.uav_id, self.weather_period_id, *self.task_ids, *self.pose_ids, *self.primitive_ids)  # 汇总服务列全部引用对象标识。
        for reference_id in referenced_ids:  # 逐一检查服务列引用对象。
            if not system.contains(reference_id):  # 判断当前引用对象是否缺失。
                issues.append(ValidationIssue("missing_service_column_reference", self.object_id, reference_id))  # 记录服务列引用对象缺失问题。
        if not self.admitted:  # 判断服务列是否未通过准入。
            issues.append(ValidationIssue("unadmitted_service_column", self.object_id, "", ",".join(self.rejection_reasons)))  # 记录服务列未准入问题及全部拒绝原因。
        return tuple(issues)  # 返回不可变服务列验证问题集合。
