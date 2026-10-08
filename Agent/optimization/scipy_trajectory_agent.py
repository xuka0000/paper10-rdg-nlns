"""使用 SciPy SLSQP 求解式 21 至式 40 的服务列连续轨迹问题。"""  # 说明本文件承担内层连续算法职责。

import json  # 导入 JSON 解析器以读取连续求解配置。
import math  # 导入数学函数以检查求解器返回数值和分类运动阶段。
import warnings  # 导入告警控制以收敛已知 SLSQP 边界裁剪诊断。
from pathlib import Path  # 导入路径工具以定位物理算法配置文件。

import numpy as np  # 导入数组库以检查 SciPy 数值解。
from scipy.optimize import minimize  # 导入论文第九部分声明的连续非线性求解入口。

from Mask.flight_relation import FlightPrimitive, MotionPhase, TrajectoryInterval  # 导入下游飞行原语和显式连续状态对象。
from Mask.geometry import Polyline3D, Vector3  # 导入三维路径和值对象以重构优化原语。
from Mask.trajectory_optimization import (  # 导入算法无关的连续模型、审计和结果协议。
    TrajectoryConstraintAudit,
    TrajectoryDecision,
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationRequest,
    TrajectoryOptimizationResult,
    _position_clearance_margins,
    audit_trajectory_decision,
    decision_bounds,
    decode_decision_vector,
    encode_decision_vector,
    trajectory_equality_residuals,
    trajectory_expected_energy,
    trajectory_inequality_margins,
    trajectory_objective,
)


def _phase_type(start_velocity: Vector3, end_velocity: Vector3) -> str:  # 定义显式状态到兼容运动阶段类型的分类函数。
    """根据区间端点速度范数返回加速、巡航或制动类型。"""  # 解释分类只用于阶段摘要。
    start_speed = start_velocity.norm()  # 计算区间起始速度大小。
    end_speed = end_velocity.norm()  # 计算区间终止速度大小。
    if end_speed > start_speed + 1.0e-9:  # 判断区间是否净加速。
        return "accelerate"  # 返回加速阶段类型。
    if end_speed + 1.0e-9 < start_speed:  # 判断区间是否净制动。
        return "decelerate"  # 返回制动阶段类型。
    return "cruise"  # 将端点速度大小不变的区间归为巡航阶段。


def _remove_consecutive_duplicate_positions(points: tuple[Vector3, ...]) -> tuple[Vector3, ...]:  # 定义优化路径连续重复节点清理函数。
    """保留输入顺序并删除仅与前一个节点相同的位置。"""  # 解释路径清理边界。
    cleaned: list[Vector3] = []  # 创建清理后的路径节点列表。
    for point in points:  # 逐一处理优化节点位置。
        if not cleaned or point != cleaned[-1]:  # 判断当前节点是否不同于最后保留节点。
            cleaned.append(point)  # 保存非连续重复位置。
    return tuple(cleaned)  # 返回不可变清理结果。


def _solver_inequality_row_mask(
    request: TrajectoryOptimizationRequest,
) -> np.ndarray:
    """Exclude position margins already fixed by equality constraints."""
    interval_count = len(request.initial_decision.durations)
    node_count = interval_count + 1
    row_count = len(
        trajectory_inequality_margins(request, request.initial_decision)
    )
    mask = np.ones(row_count, dtype=bool)
    position_row_offset = 2 * interval_count + node_count + interval_count
    for node_index, _ in request.fixed_node_positions:
        first_row = position_row_offset + 6 * node_index
        mask[first_row : first_row + 6] = False
    return mask


def _trajectory_objective_gradient(
    request: TrajectoryOptimizationRequest,
    decision: TrajectoryDecision,
) -> np.ndarray:
    """Return the exact gradient of the time-energy objective."""
    interval_count = len(decision.durations)
    node_count = interval_count + 1
    velocity_offset = 3 * node_count
    acceleration_offset = 6 * node_count
    duration_offset = acceleration_offset + 3 * interval_count
    gradient = np.zeros(duration_offset + interval_count, dtype=float)
    parameters = request.energy_parameters
    time_weight = request.objective_weights.dynamic_time_weight
    energy_weight = request.objective_weights.expected_energy_weight
    gravity = request.dynamic_limits.gravity_vector
    wind = request.weather.wind_vector
    for index, duration in enumerate(decision.durations):
        velocity = decision.velocities[index]
        acceleration = decision.accelerations[index]
        relative_velocity = velocity - wind
        thrust_acceleration = acceleration - gravity
        base_power = (
            request.uav.hover_power
            + parameters.motion_coefficient * relative_velocity.norm() ** 2
            + parameters.acceleration_coefficient
            * thrust_acceleration.norm()
            ** 2
        )
        gradient[duration_offset + index] = (
            time_weight + energy_weight * base_power
        )
        velocity_scale = (
            2.0 * energy_weight * parameters.motion_coefficient * duration
        )
        gradient[
            velocity_offset + 3 * index : velocity_offset + 3 * index + 3
        ] = velocity_scale * np.asarray(
            (relative_velocity.x, relative_velocity.y, relative_velocity.z)
        )
        acceleration_scale = (
            2.0
            * energy_weight
            * parameters.acceleration_coefficient
            * duration
        )
        gradient[
            acceleration_offset
            + 3 * index : acceleration_offset
            + 3 * index
            + 3
        ] = acceleration_scale * np.asarray(
            (
                thrust_acceleration.x,
                thrust_acceleration.y,
                thrust_acceleration.z,
            )
        )
        height_change = (
            decision.positions[index + 1].z - decision.positions[index].z
        )
        if height_change > 0.0:
            climb_gradient = energy_weight * parameters.climb_coefficient
            gradient[3 * index + 2] -= climb_gradient
            gradient[3 * (index + 1) + 2] += climb_gradient
    return gradient


def _trajectory_equality_jacobian(
    request: TrajectoryOptimizationRequest,
    decision: TrajectoryDecision,
) -> np.ndarray:
    """Return the exact Jacobian of dynamics and fixed-state equalities."""
    interval_count = len(decision.durations)
    node_count = interval_count + 1
    velocity_offset = 3 * node_count
    acceleration_offset = 6 * node_count
    duration_offset = acceleration_offset + 3 * interval_count
    row_count = 6 * interval_count + 3 * len(request.fixed_node_positions) + 6
    jacobian = np.zeros(
        (row_count, duration_offset + interval_count), dtype=float
    )
    for index, duration in enumerate(decision.durations):
        velocity = decision.velocities[index]
        acceleration = decision.accelerations[index]
        velocity_values = (velocity.x, velocity.y, velocity.z)
        acceleration_values = (
            acceleration.x,
            acceleration.y,
            acceleration.z,
        )
        for component in range(3):
            position_row = 6 * index + component
            velocity_row = position_row + 3
            start_position_column = 3 * index + component
            end_position_column = 3 * (index + 1) + component
            start_velocity_column = velocity_offset + 3 * index + component
            end_velocity_column = (
                velocity_offset + 3 * (index + 1) + component
            )
            acceleration_column = (
                acceleration_offset + 3 * index + component
            )
            duration_column = duration_offset + index
            jacobian[position_row, start_position_column] = -1.0
            jacobian[position_row, end_position_column] = 1.0
            jacobian[position_row, start_velocity_column] = -duration
            jacobian[position_row, acceleration_column] = -0.5 * duration**2
            jacobian[position_row, duration_column] = -(
                velocity_values[component]
                + acceleration_values[component] * duration
            )
            jacobian[velocity_row, start_velocity_column] = -1.0
            jacobian[velocity_row, end_velocity_column] = 1.0
            jacobian[velocity_row, acceleration_column] = -duration
            jacobian[velocity_row, duration_column] = -acceleration_values[
                component
            ]
    row = 6 * interval_count
    for node_index, _ in request.fixed_node_positions:
        for component in range(3):
            jacobian[row + component, 3 * node_index + component] = 1.0
        row += 3
    for node_index in (0, node_count - 1):
        for component in range(3):
            jacobian[
                row + component,
                velocity_offset + 3 * node_index + component,
            ] = 1.0
        row += 3
    return jacobian


def _position_margin_jacobian(
    request: TrajectoryOptimizationRequest,
    position: Vector3,
) -> np.ndarray:
    """Differentiate the terrain and obstacle oracle locally in 3-D."""
    values = (position.x, position.y, position.z)
    jacobian = np.zeros((6, 3), dtype=float)
    for component, value in enumerate(values):
        step = math.sqrt(np.finfo(float).eps) * max(1.0, abs(value))
        lower = list(values)
        upper = list(values)
        lower[component] -= step
        upper[component] += step
        lower_margins = _position_clearance_margins(
            request, Vector3(*lower)
        )
        upper_margins = _position_clearance_margins(
            request, Vector3(*upper)
        )
        jacobian[:, component] = (
            np.asarray(upper_margins) - np.asarray(lower_margins)
        ) / (2.0 * step)
    return jacobian


def _trajectory_inequality_jacobian(
    request: TrajectoryOptimizationRequest,
    decision: TrajectoryDecision,
) -> np.ndarray:
    """Return a sparse-by-construction Jacobian for all inequality margins."""
    interval_count = len(decision.durations)
    node_count = interval_count + 1
    velocity_offset = 3 * node_count
    acceleration_offset = 6 * node_count
    duration_offset = acceleration_offset + 3 * interval_count
    row_count = len(trajectory_inequality_margins(request, decision))
    jacobian = np.zeros(
        (row_count, duration_offset + interval_count), dtype=float
    )
    for index in range(interval_count):
        duration_column = duration_offset + index
        jacobian[2 * index, duration_column] = 1.0
        jacobian[2 * index + 1, duration_column] = -1.0
    speed_row_offset = 2 * interval_count
    for index, velocity in enumerate(decision.velocities):
        jacobian[
            speed_row_offset + index,
            velocity_offset + 3 * index : velocity_offset + 3 * index + 3,
        ] = -2.0 * np.asarray((velocity.x, velocity.y, velocity.z))
    thrust_row_offset = speed_row_offset + node_count
    gravity = request.dynamic_limits.gravity_vector
    for index, acceleration in enumerate(decision.accelerations):
        thrust = acceleration - gravity
        jacobian[
            thrust_row_offset + index,
            acceleration_offset
            + 3 * index : acceleration_offset
            + 3 * index
            + 3,
        ] = -2.0 * np.asarray((thrust.x, thrust.y, thrust.z))
    position_row_offset = thrust_row_offset + interval_count
    for index, position in enumerate(decision.positions):
        jacobian[
            position_row_offset
            + 6 * index : position_row_offset
            + 6 * index
            + 6,
            3 * index : 3 * index + 3,
        ] = _position_margin_jacobian(request, position)
    observation_row_offset = position_row_offset + 6 * node_count
    for row_index, node_index in enumerate(request.observation_node_indices):
        velocity = decision.velocities[node_index]
        jacobian[
            observation_row_offset + row_index,
            velocity_offset
            + 3 * node_index : velocity_offset
            + 3 * node_index
            + 3,
        ] = -2.0 * np.asarray((velocity.x, velocity.y, velocity.z))
    return jacobian


def _is_time_scalable_out_and_back(
    request: TrajectoryOptimizationRequest,
) -> bool:
    """Recognize the controlled single-task return-flight structure."""
    if len(request.primitive_specs) != 2:
        return False
    first, second = request.primitive_specs
    boundary = first.first_interval_index + first.interval_count
    if second.first_interval_index != boundary:
        return False
    if request.observation_node_indices != (boundary,):
        return False
    if first.interval_count < 1 or second.interval_count < 1:
        return False
    decision = request.initial_decision
    tolerance = request.config.equality_tolerance
    if decision.positions[0].distance_to(decision.positions[-1]) > tolerance:
        return False
    boundary_indices = (0, boundary, len(decision.positions) - 1)
    return all(
        decision.velocities[index].norm() <= tolerance
        for index in boundary_indices
    )


def _time_scale_interval_groups(
    request: TrajectoryOptimizationRequest,
) -> tuple[tuple[int, int], ...]:
    """Partition each primitive at zero-velocity motion-segment boundaries."""
    groups: list[tuple[int, int]] = []
    tolerance = request.config.equality_tolerance
    velocities = request.initial_decision.velocities
    for spec in request.primitive_specs:
        first = spec.first_interval_index
        stop = first + spec.interval_count
        group_first = first
        for node_index in range(first + 1, stop + 1):
            if velocities[node_index].norm() <= tolerance:
                groups.append((group_first, node_index))
                group_first = node_index
        if group_first != stop:
            raise ValueError("time-scaled primitive does not end at zero velocity")
    return tuple(groups)


def _time_scale_group_bounds(
    request: TrajectoryOptimizationRequest,
    groups: tuple[tuple[int, int], ...],
) -> tuple[tuple[float, float], ...]:
    """Derive scale bounds that enforce every interval duration bound."""
    bounds = []
    durations = request.initial_decision.durations
    for first, stop in groups:
        lower = max(
            1.0,
            *(
                request.dynamic_limits.minimum_segment_time / durations[index]
                for index in range(first, stop)
            ),
        )
        upper = min(
            request.dynamic_limits.maximum_segment_time / durations[index]
            for index in range(first, stop)
        )
        if lower > upper + request.config.equality_tolerance:
            raise ValueError(
                f"time-scaled motion segment has incompatible duration bounds: "
                f"{first}:{stop}"
            )
        bounds.append((lower, max(lower, upper)))
    return tuple(bounds)


def _time_scaled_decision(
    request: TrajectoryOptimizationRequest,
    scales: np.ndarray,
    groups: tuple[tuple[int, int], ...] | None = None,
) -> TrajectoryDecision:
    """Reconstruct states after independently scaling zero-velocity segments."""
    initial = request.initial_decision
    velocities = list(initial.velocities)
    accelerations = list(initial.accelerations)
    durations = list(initial.durations)
    active_groups = groups or _time_scale_interval_groups(request)
    if len(scales) != len(active_groups):
        raise ValueError("time-scale vector does not match motion-segment groups")
    for scale, (first, stop) in zip(scales, active_groups):
        value = float(scale)
        for node_index in range(first, stop + 1):
            velocities[node_index] = initial.velocities[node_index].scaled(
                1.0 / value
            )
        for interval_index in range(first, stop):
            accelerations[interval_index] = initial.accelerations[
                interval_index
            ].scaled(1.0 / value**2)
            durations[interval_index] = initial.durations[interval_index] * value
    return TrajectoryDecision(
        initial.positions,
        tuple(velocities),
        tuple(accelerations),
        tuple(durations),
    )


def _time_scale_dynamic_margins(
    request: TrajectoryOptimizationRequest,
    decision: TrajectoryDecision,
) -> np.ndarray:
    """Keep only margins that can change on the time-scaled path."""
    interval_count = len(decision.durations)
    node_count = interval_count + 1
    margins = trajectory_inequality_margins(request, decision)
    position_start = 2 * interval_count + node_count + interval_count
    position_stop = position_start + 6 * node_count
    return np.concatenate((margins[:position_start], margins[position_stop:]))


class ScipyTrajectoryAgent:  # 定义基于 SLSQP 的服务列连续轨迹 Agent。
    """调用 SciPy minimize 并在独立审计通过后生成优化飞行原语。"""  # 解释算法职责和失败关闭语义。

    def __init__(self, config: TrajectoryOptimizationConfig) -> None:  # 定义连续轨迹 Agent 初始化方法。
        """保存经过数据类验证的求解和审计配置。"""  # 解释初始化输入。
        self._config = config  # 保存不可变连续求解配置。

    @classmethod  # 声明从项目配置创建连续轨迹 Agent 的命名构造器。
    def from_config(cls, config_path: Path) -> "ScipyTrajectoryAgent":  # 定义 JSON 配置读取入口。
        """从 physical.trajectory_optimizer 读取 SLSQP 和审计参数。"""  # 解释配置来源。
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))  # 解析兼容 BOM 的项目配置 JSON。
        record = data.get("physical", {}).get("trajectory_optimizer", {})  # 读取存在时的连续求解参数记录。
        config = TrajectoryOptimizationConfig(  # 构造经过统一验证的连续求解配置。
            equality_tolerance=float(record.get("equality_tolerance", 1.0e-6)),  # 读取最大等式残差容差。
            inequality_tolerance=float(record.get("inequality_tolerance", 1.0e-6)),  # 读取最小不等式余量容差。
            dense_samples_per_interval=int(record.get("dense_samples_per_interval", 7)),  # 读取每区间密集净空复核样本数。
            max_iterations=int(record.get("maximum_iterations", 500)),  # 读取 SLSQP 最大迭代次数。
            function_tolerance=float(record.get("function_tolerance", 1.0e-9)),  # 读取目标停止容差。
        )  # 结束连续求解配置构造。
        return cls(config)  # 返回正式连续轨迹 Agent。

    def _build_primitives(self, request: TrajectoryOptimizationRequest, decision: TrajectoryDecision, audit: TrajectoryConstraintAudit, objective_value: float) -> tuple[FlightPrimitive, ...]:  # 定义连续状态到下游飞行原语的重构方法。
        """按照原语切片生成带显式优化区间和审计证据的认证原语。"""  # 解释成功结果重构边界。
        primitives: list[FlightPrimitive] = []  # 创建按服务顺序保存的优化原语列表。
        for spec in request.primitive_specs:  # 逐一处理服务级连续决策中的原语切片。
            start_index = spec.first_interval_index  # 读取当前原语首区间索引。
            stop_index = start_index + spec.interval_count  # 计算当前原语终止节点索引。
            intervals: list[TrajectoryInterval] = []  # 创建当前原语局部相对时刻区间列表。
            current_time = 0.0  # 将每条原语起点定义为局部零时刻。
            for interval_index in range(start_index, stop_index):  # 逐一重构当前原语包含的连续区间。
                duration = decision.durations[interval_index]  # 读取当前区间持续时间。
                start_velocity = decision.velocities[interval_index]  # 读取当前区间起始速度。
                end_velocity = decision.velocities[interval_index + 1]  # 读取当前区间终止速度。
                interval = TrajectoryInterval(_phase_type(start_velocity, end_velocity), current_time, duration, decision.positions[interval_index], decision.positions[interval_index + 1], start_velocity, end_velocity, decision.accelerations[interval_index])  # 创建显式分段恒加速度区间。
                intervals.append(interval)  # 保存当前优化区间。
                current_time = interval.end_time  # 推进原语局部时刻。
            path_points = _remove_consecutive_duplicate_positions(tuple(decision.positions[start_index : stop_index + 1]))  # 清理用于几何输出的连续重复节点。
            if len(path_points) < 2:  # 判断当前原语是否缺少正空间路径。
                raise ValueError(f"优化飞行原语缺少正路径长度: {spec.object_id}")  # 拒绝不能表达飞行动作的优化切片。
            primitive = FlightPrimitive(  # 构造下游服务列可消费的优化原语。
                object_id=spec.object_id,  # 保存原语稳定标识。
                name=spec.name,  # 保存原语可读名称。
                source_id=spec.source_id,  # 保存原语起点对象标识。
                target_id=spec.target_id,  # 保存原语终点对象标识。
                corridor_id=spec.corridor_id,  # 保存来源走廊标识。
                path=Polyline3D(path_points),  # 保存优化节点形成的三维路径。
                segments=(),  # 明确优化原语不复用固定最短时间分段。
                duration=current_time,  # 保存显式区间持续时间之和。
                minimum_clearance_margin=min(spec.minimum_clearance_margin, audit.minimum_dense_clearance_margin),  # 保存初值走廊与优化密集复核中的最小净空证据。
                certified=True,  # 标记当前原语已经通过连续独立审计。
                rejection_reason="",  # 认证原语没有拒绝原因。
                optimized_intervals=tuple(intervals),  # 保存正式连续状态区间。
                optimization_objective=objective_value,  # 保存式 34 服务级目标值。
                maximum_equality_residual=audit.maximum_equality_residual,  # 保存最大等式残差。
                minimum_inequality_margin=audit.minimum_inequality_margin,  # 保存最小不等式余量。
            )  # 结束优化原语构造。
            primitives.append(primitive)  # 将当前原语加入有序结果。
        return tuple(primitives)  # 返回完整优化原语集合。

    def _solve_time_scaled_out_and_back(
        self,
        request: TrajectoryOptimizationRequest,
    ) -> TrajectoryOptimizationResult:
        initial = request.initial_decision
        initial_audit = audit_trajectory_decision(
            request,
            initial,
            solver_success=True,
            message="certified initial path",
        )
        initial_objective = trajectory_objective(request, initial)
        initial_energy = trajectory_expected_energy(request, initial)
        static_path_valid = (
            initial_audit.maximum_equality_residual
            <= request.config.equality_tolerance
            and initial_audit.minimum_dense_clearance_margin
            >= -request.config.inequality_tolerance
        )
        if not static_path_valid:
            failed_audit = audit_trajectory_decision(
                request,
                initial,
                solver_success=False,
                message="certified path fails continuous clearance audit",
            )
            return TrajectoryOptimizationResult(
                False,
                "infeasible",
                failed_audit.message,
                "SLSQP",
                initial,
                initial_objective,
                initial_energy,
                failed_audit,
                (),
            )
        groups = _time_scale_interval_groups(request)
        try:
            bounds = _time_scale_group_bounds(request, groups)
        except ValueError as error:
            failed_audit = audit_trajectory_decision(
                request,
                initial,
                solver_success=False,
                message=str(error),
            )
            return TrajectoryOptimizationResult(
                False,
                "infeasible",
                str(error),
                "SLSQP",
                initial,
                initial_objective,
                initial_energy,
                failed_audit,
                (),
            )
        initial_scales = np.asarray([lower for lower, _ in bounds], dtype=float)
        raw_result = minimize(
            fun=lambda scales: trajectory_objective(
                request,
                _time_scaled_decision(request, scales, groups),
            ),
            x0=initial_scales,
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {
                    "type": "ineq",
                    "fun": lambda scales: _time_scale_dynamic_margins(
                        request,
                        _time_scaled_decision(request, scales, groups),
                    ),
                },
            ),
            options={
                "maxiter": self._config.max_iterations,
                "ftol": self._config.function_tolerance,
                "disp": False,
            },
        )
        if not bool(raw_result.success):
            interior_start = np.asarray(
                [min(lower + 0.05, upper) for lower, upper in bounds],
                dtype=float,
            )
            if np.any(interior_start > initial_scales + 1.0e-12):
                raw_result = minimize(
                    fun=lambda scales: trajectory_objective(
                        request,
                        _time_scaled_decision(request, scales, groups),
                    ),
                    x0=interior_start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=(
                        {
                            "type": "ineq",
                            "fun": lambda scales: _time_scale_dynamic_margins(
                                request,
                                _time_scaled_decision(request, scales, groups),
                            ),
                        },
                    ),
                    options={
                        "maxiter": self._config.max_iterations,
                        "ftol": self._config.function_tolerance,
                        "disp": False,
                    },
                )
        message = str(raw_result.message)
        if raw_result.x is None or not np.isfinite(
            np.asarray(raw_result.x, dtype=float)
        ).all():
            audit = TrajectoryConstraintAudit(
                False,
                "solver_failed",
                message,
                math.inf,
                -math.inf,
                -math.inf,
            )
            return TrajectoryOptimizationResult(
                False,
                "numerical_failure",
                message,
                "SLSQP",
                None,
                None,
                None,
                audit,
                (),
            )
        decision = _time_scaled_decision(
            request,
            np.asarray(raw_result.x, dtype=float),
            groups,
        )
        audit = audit_trajectory_decision(
            request,
            decision,
            bool(raw_result.success),
            message,
        )
        objective_value = trajectory_objective(request, decision)
        expected_energy = trajectory_expected_energy(request, decision)
        if raw_result.success and audit.admitted:
            primitives = self._build_primitives(
                request,
                decision,
                audit,
                objective_value,
            )
            return TrajectoryOptimizationResult(
                True,
                "optimized",
                message,
                "SLSQP",
                decision,
                objective_value,
                expected_energy,
                audit,
                primitives,
            )
        status = (
            "audit_failed"
            if raw_result.success
            else "iteration_limit"
            if int(getattr(raw_result, "status", -1)) == 9
            else "infeasible"
            if int(getattr(raw_result, "status", -1)) in {2, 4}
            else "numerical_failure"
        )
        return TrajectoryOptimizationResult(
            False,
            status,
            message,
            "SLSQP",
            decision,
            objective_value,
            expected_energy,
            audit,
            (),
        )

    def solve(self, request: TrajectoryOptimizationRequest) -> TrajectoryOptimizationResult:  # 定义式 34 连续非线性优化入口。
        """使用 SLSQP 求解连续状态并在独立复核后失败关闭。"""  # 解释求解流程。
        if _is_time_scalable_out_and_back(request):
            return self._solve_time_scaled_out_and_back(request)
        initial_vector = encode_decision_vector(request.initial_decision)  # 将认证初值编码为 SciPy 一维变量。
        inequality_mask = _solver_inequality_row_mask(request)

        def decode(values: np.ndarray) -> TrajectoryDecision:  # 定义当前请求的局部决策解码函数。
            """使用当前连续优化请求解码一维决策向量。"""  # 解释局部函数闭包的请求来源。
            return decode_decision_vector(request, values)  # 委托模型层稳定解码接口。

        constraints = (
            {
                "type": "eq",
                "fun": lambda values: trajectory_equality_residuals(
                    request, decode(values)
                ),
                "jac": lambda values: _trajectory_equality_jacobian(
                    request, decode(values)
                ),
            },
            {
                "type": "ineq",
                "fun": lambda values: trajectory_inequality_margins(
                    request, decode(values)
                )[inequality_mask],
                "jac": lambda values: _trajectory_inequality_jacobian(
                    request, decode(values)
                )[inequality_mask],
            },
        )
        with warnings.catch_warnings():  # 将已知的 SLSQP 临时越界裁剪限制在求解器内部。
            warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds", category=RuntimeWarning)  # 忽略随后仍由边界和独立审计控制的临时试探点告警。
            raw_result = minimize(  # 调用论文第九部分声明的 SciPy 连续优化器。
                fun=lambda values: trajectory_objective(request, decode(values)),  # 使用式 34 时间能耗值函数。
                jac=lambda values: _trajectory_objective_gradient(
                    request, decode(values)
                ),
                x0=initial_vector,  # 使用地形认证确定性原语形成可行初值。
                method="SLSQP",  # 使用支持一般等式和不等式约束的序列二次规划方法。
                bounds=decision_bounds(request),  # 应用地形定义域和变量分量边界。
                constraints=constraints,  # 应用算法无关模型层提供的连续约束。
                options={"maxiter": self._config.max_iterations, "ftol": self._config.function_tolerance, "disp": False},  # 设置迭代上限和目标停止容差。
            )  # 完成连续优化调用。
        message = str(raw_result.message)  # 保存 SciPy 返回的可读求解消息。
        decision: TrajectoryDecision | None = None  # 初始化不存在的连续状态解。
        if raw_result.x is not None and np.isfinite(np.asarray(raw_result.x, dtype=float)).all():  # 判断 SciPy 是否返回有限数值向量。
            try:  # 捕获边界数值不能形成正时间状态的解码异常。
                decision = decode(np.asarray(raw_result.x, dtype=float))  # 解码存在的有限连续状态。
            except ValueError:  # 处理不能形成有效连续状态的数据。
                decision = None  # 保持失败关闭状态。
        if decision is None:  # 判断求解器是否没有可审计数值状态。
            audit = TrajectoryConstraintAudit(False, "solver_failed", message, math.inf, -math.inf, -math.inf)  # 构造缺少数值解的失败审计记录。
            return TrajectoryOptimizationResult(False, "numerical_failure", message, "SLSQP", None, None, None, audit, ())  # 返回不含原语的数值失败结果。
        audit = audit_trajectory_decision(request, decision, bool(raw_result.success), message)  # 不信任成功标志并重算全部连续约束。
        objective_value = trajectory_objective(request, decision)  # 从解码状态独立重算式 34 目标值。
        expected_energy = trajectory_expected_energy(request, decision)  # 从解码状态独立重算式 36 期望能耗。
        if raw_result.success and audit.admitted:  # 判断求解器和独立复核是否同时成功。
            try:  # 捕获不能重构正路径原语的退化连续解。
                primitives = self._build_primitives(request, decision, audit, objective_value)  # 生成供服务列和 M5 消费的优化原语。
            except ValueError as error:  # 处理退化路径或原语证据异常。
                failed_audit = TrajectoryConstraintAudit(False, "audit_failed", str(error), audit.maximum_equality_residual, audit.minimum_inequality_margin, audit.minimum_dense_clearance_margin)  # 将重构失败纳入独立审计证据。
                return TrajectoryOptimizationResult(False, "audit_failed", str(error), "SLSQP", decision, objective_value, expected_energy, failed_audit, ())  # 失败关闭且不返回原语。
            return TrajectoryOptimizationResult(True, "optimized", message, "SLSQP", decision, objective_value, expected_energy, audit, primitives)  # 返回完整连续优化成功结果。
        if raw_result.success:  # 判断求解器声称成功但独立审计未通过。
            status = "audit_failed"  # 使用独立审计失败状态覆盖求解器成功标志。
        elif int(getattr(raw_result, "status", -1)) == 9:  # 判断 SciPy 是否达到迭代上限。
            status = "iteration_limit"  # 记录稳定迭代上限状态。
        elif int(getattr(raw_result, "status", -1)) in {2, 4}:  # 判断 SciPy 是否报告约束不兼容或不等式不可行。
            status = "infeasible"  # 记录稳定不可行状态。
        else:  # 处理其他线搜索、奇异矩阵或数值失败。
            status = "numerical_failure"  # 记录稳定数值失败状态。
        return TrajectoryOptimizationResult(False, status, message, "SLSQP", decision, objective_value, expected_energy, audit, ())  # 返回具有数值诊断但不含准入原语的失败结果。
