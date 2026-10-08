"""使用 scipy.optimize.milp 实现滚动车辆无人机协同调度 Agent。"""  # 说明本文件的算法职责。

import math  # 导入数学函数以验证 Agent 配置和处理无穷边界。
import time  # 导入计时工具以记录 Agent 求解时间。
from dataclasses import dataclass, replace  # 导入数据类工具以保存配置并安全覆盖评估时间上限。
from pathlib import Path  # 导入路径工具以定位 Data 中的 Agent 配置文件。
from types import SimpleNamespace  # 中文说明。
from typing import Any  # 导入通用类型以兼容统一环境快照协议。

import numpy as np  # 导入数组库以读取 SciPy 数值解。
from scipy.optimize import milp  # 导入 SciPy 混合整数线性规划求解器。

from Agent.base_agent import BaseAgent  # 导入统一 Agent 抽象接口。
from Agent.optimization.linear_model import BuiltLinearModel, LinearModelBuilder, VariableReference  # 导入命名稀疏线性模型构建工具。
from Agent.results import AgentResult  # 导入统一 Agent 结果类型。
from Env.data_loader import CONFIG_ALLOWED_KEYS, CONFIG_COMMON_REQUIRED_KEYS, parse_time_controls, read_json_object, require_record_keys  # 导入严格 JSON、双时域和嵌套字段检查接口。
from Env.snapshot import SceneSnapshot  # 导入正式环境快照类型以声明求解接口。


def _effective_column(snapshot: Any, column: Any) -> Any:  # 中文说明。
    """Use aligned snapshot semantics with a legacy test-snapshot fallback. 中文说明。"""
    if hasattr(snapshot, "column_state"):  # 中文说明。
        return snapshot.column_state(column.object_id)  # 中文说明。
    return SimpleNamespace(  # 中文说明。
        is_committed=column.object_id in getattr(snapshot, "locked_column_ids", ()),
        is_airborne=False,
        effective_checkpoint_ids=column.covered_checkpoint_ids,
        effective_completion_offsets=column.completion_offsets,
        effective_duration=column.dynamic_duration,
        effective_protected_energy=column.energy_quantile,
        usable_battery_energy=math.inf,
        requires_launch_stop=True,
    )


def _rolling_uav_state(snapshot: Any, uav: Any) -> Any:  # 中文说明。
    """中文说明。"""
    if hasattr(snapshot, "uav_state"):  # 中文说明。
        return snapshot.uav_state(uav.object_id)  # 中文说明。
    return SimpleNamespace(available=True, turnaround_time=0.0)  # 中文说明。


CONSTRAINT_FAMILIES = (  # 定义论文公式和代码共同使用的稳定 M5 约束族。
    "vehicle_start_flow",  # 车辆起点仓库出流约束族。
    "vehicle_end_flow",  # 车辆终点仓库入流约束族。
    "vehicle_stop_flow",  # 车辆中间道路节点流守恒约束族。
    "road_time",  # 车辆道路时间传递约束族。
    "column_launch_stop",  # 服务列起飞停靠点联结约束族。
    "column_recovery_stop",  # 服务列回收停靠点联结约束族。
    "checkpoint_coverage",  # 检查点覆盖或漏检约束族。
    "column_duration",  # 服务列动力学持续时间约束族。
    "launch_after_arrival",  # 车辆到达起飞点后起飞约束族。
    "departure_after_launch",  # 无人机起飞后车辆离开起飞点约束族。
    "recovery_after_vehicle",  # 车辆到达回收点后回收约束族。
    "departure_after_recovery",  # 无人机回收后车辆离站约束族。
    "uav_nonoverlap",  # 同一无人机服务列不重叠约束族。
    "checkpoint_completion_lower",  # 检查点完成时间下界联结约束族。
    "checkpoint_completion_upper",  # 检查点完成时间上界联结约束族。
    "makespan",  # 最大完成时间约束族。
    "rolling_lock",  # 滚动服务列锁定约束族。
    "plan_change",  # 相对上一滚动方案变更线性化约束族。
)  # 结束稳定 M5 约束族声明。

OBJECTIVE_COMPONENTS = ("completion", "makespan", "vehicle_time", "energy", "missed", "waiting", "plan_change")  # 定义七类可重构线性目标项。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义 SciPy MILP Agent 配置。
class ScipyMilpConfig:  # 定义时序常数、目标权重和求解器停止参数。
    """保存 M5 线性化和 SciPy MILP 求解所需的全部算法参数。"""  # 解释 SciPy Agent 配置内容。

    time_big_m: float  # 保存时序联结使用的充分大常数。
    weight_completion: float  # 保存风险加权完成时间系数。
    weight_makespan: float  # 保存最大完成时间系数。
    weight_vehicle_time: float  # 保存车辆道路行驶时间系数。
    weight_energy: float  # 保存概率安全能耗系数。
    weight_missed: float  # 保存检查点漏检惩罚系数。
    weight_waiting: float  # 保存回收等待时间系数。
    weight_plan_change: float  # 保存滚动方案变更惩罚系数。
    time_limit: float  # 保存 SciPy MILP 求解时间上限。
    mip_relative_gap: float  # 保存允许的相对整数间隙。

    def __post_init__(self) -> None:  # 定义 SciPy Agent 配置初始化后的参数验证方法。
        """验证充分大常数、目标权重、时间上限和整数间隙。"""  # 解释 SciPy Agent 配置输入约束。
        numeric_values = (self.time_big_m, self.weight_completion, self.weight_makespan, self.weight_vehicle_time, self.weight_energy, self.weight_missed, self.weight_waiting, self.weight_plan_change, self.time_limit, self.mip_relative_gap)  # 汇总全部 SciPy Agent 配置参数。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部 SciPy Agent 配置参数是否有限。
            raise ValueError("SciPy MILP Agent 配置必须为有限数值")  # 抛出明确的 Agent 配置有限性异常。
        if self.time_big_m <= 0.0 or self.time_limit <= 0.0:  # 判断时序常数和求解时间上限是否为正数。
            raise ValueError("时序充分大常数和求解时间上限必须为正数")  # 抛出明确的时序和时间上限异常。
        objective_weights = (self.weight_completion, self.weight_makespan, self.weight_vehicle_time, self.weight_energy, self.weight_missed, self.weight_waiting, self.weight_plan_change)  # 汇总全部线性目标权重。
        if any(weight < 0.0 for weight in objective_weights):  # 判断线性目标权重是否为非负数。
            raise ValueError("SciPy MILP 目标权重不能为负数")  # 抛出明确的目标权重异常。
        if not 0.0 <= self.mip_relative_gap < 1.0:  # 判断相对整数间隙是否位于有效范围。
            raise ValueError("相对整数间隙必须位于零到一之间")  # 抛出明确的整数间隙异常。


def classify_milp_result(result: object) -> tuple[bool, str, bool]:
    """Separate finite incumbent availability from a proof of optimality."""
    vector = getattr(result, "x", None)
    objective = getattr(result, "fun", None)
    try:
        values = np.asarray(vector, dtype=float)
        finite_candidate = (
            vector is not None
            and objective is not None
            and values.size > 0
            and bool(np.isfinite(values).all())
            and math.isfinite(float(objective))
        )
    except (TypeError, ValueError):
        finite_candidate = False
    if not finite_candidate:
        return False, "no_solution", False
    try:
        status = int(getattr(result, "status", -1))
    except (TypeError, ValueError):
        status = -1
    if bool(getattr(result, "success", False)) and status == 0:
        return True, "optimal", True
    if status == 1:
        return True, "time_limit_incumbent", False
    return True, "other_incumbent", False


@dataclass(slots=True)  # 使用紧凑数据类保存 M5 模型及其变量映射。
class SchedulingModelArtifacts:  # 定义求解、测试和结果重构共同使用的模型产物。
    """绑定数值线性模型、变量族映射和目标分项系数。"""  # 解释 M5 模型产物内容。

    model: BuiltLinearModel  # 保存可以直接传给 SciPy MILP 的数值模型。
    road_use: dict[tuple[str, str], VariableReference]  # 保存车辆道路弧变量映射。
    stop_visit: dict[tuple[str, str], VariableReference]  # 保存车辆停靠点变量映射。
    column_select: dict[tuple[str, str], VariableReference]  # 保存车辆服务列变量映射。
    checkpoint_completion: dict[str, VariableReference]  # 保存检查点完成时间变量映射。
    objective_records: dict[str, tuple[tuple[int, float], ...]]  # 保存各目标项使用的变量索引和加权系数。

    @property  # 将目标项名称暴露为只读属性。
    def objective_terms(self) -> tuple[str, ...]:  # 定义目标项名称读取方法。
        """按照固定七类目标顺序返回目标分项名称。"""  # 解释目标项输出顺序。
        return tuple(self.objective_records)  # 返回目标分项映射的稳定键顺序。


class ScipyMilpAgent(BaseAgent):  # 定义基于 SciPy MILP 的小规模精确调度 Agent。
    """从 Env 认证系数构建 M5 并返回可由 Env 独立复核的调度结果。"""  # 解释 SciPy MILP Agent 职责。

    def __init__(self, config: ScipyMilpConfig) -> None:  # 定义 SciPy MILP Agent 初始化方法。
        """保存经过数值范围验证的算法配置。"""  # 解释 Agent 初始状态。
        self._config = config  # 保存不可变 SciPy MILP Agent 配置。

    def with_time_limit(self, time_limit_seconds: float) -> "ScipyMilpAgent":
        """Return an independent solver that differs only in its time budget."""
        return type(self)(
            replace(self._config, time_limit=float(time_limit_seconds))
        )

    @classmethod  # 声明从 Data 算法配置构建 Agent 的替代构造方法。
    def from_config(cls, config_path: Path, time_limit_override: float | None = None) -> "ScipyMilpAgent":  # 定义 SciPy Agent 配置读取和可选评估时间上限入口。
        """读取 Agent 专属配置字段并构造经过数值范围检查的算法实例。"""  # 解释配置读取职责边界。
        config_data = read_json_object(config_path, CONFIG_COMMON_REQUIRED_KEYS, CONFIG_ALLOWED_KEYS)  # 严格读取旧单时域或正式双时域配置。
        parse_time_controls(config_data)  # 与环境构造共享时间控制校验，拒绝混合或不完整形式。
        agent_keys = {"time_big_m_seconds", "weight_completion", "weight_makespan", "weight_vehicle_time", "weight_energy", "weight_missed", "weight_waiting", "weight_plan_change", "time_limit_seconds", "mip_relative_gap"}  # 定义 SciPy Agent 允许的全部配置字段。
        agent_data = require_record_keys(config_data["agent"], agent_keys, agent_keys, "agent")  # 检查 Agent 配置字段完整性和未知字段。
        scipy_config = ScipyMilpConfig(float(agent_data["time_big_m_seconds"]), float(agent_data["weight_completion"]), float(agent_data["weight_makespan"]), float(agent_data["weight_vehicle_time"]), float(agent_data["weight_energy"]), float(agent_data["weight_missed"]), float(agent_data["weight_waiting"]), float(agent_data["weight_plan_change"]), float(agent_data["time_limit_seconds"]), float(agent_data["mip_relative_gap"]))  # 构造经过统一数值验证的 SciPy MILP 配置。
        if time_limit_override is not None:  # 判断评估清单是否显式覆盖求解时间上限。
            scipy_config = replace(scipy_config, time_limit=float(time_limit_override))  # 通过数据类验证路径构造覆盖后的配置。
        return cls(scipy_config)  # 返回仅承担算法职责的 SciPy MILP Agent。

    def build_model(self, snapshot: Any) -> SchedulingModelArtifacts:  # 定义 M5 命名稀疏线性模型构建方法。
        """根据不可变环境快照建立完整车辆无人机滚动调度 MILP。"""  # 解释 M5 模型构建输入输出。
        builder = LinearModelBuilder()  # 创建空命名稀疏线性模型构建器。
        objective_records: dict[str, list[tuple[int, float]]] = {component: [] for component in OBJECTIVE_COMPONENTS}  # 创建七类目标项变量系数记录。
        vehicles = tuple(snapshot.vehicles)  # 读取环境快照中的地面车辆集合。
        uavs = tuple(snapshot.uavs)  # 读取环境快照中的车载无人机集合。
        road_nodes = tuple(snapshot.road_nodes)  # 读取环境快照中的道路节点集合。
        road_options = tuple(option for option in snapshot.road_options if option.accessible)  # 读取通过车辆认证的道路选项集合。
        stop_sites = tuple(snapshot.stop_sites)  # 读取环境快照中的候选停靠点集合。
        stop_options = tuple(option for option in snapshot.stop_options if option.accessible)  # 读取通过车辆能力认证的停靠组合集合。
        remaining_tasks = tuple(task for task in snapshot.tasks if task.checkpoint_id not in set(snapshot.completed_checkpoint_ids))  # 删除已经完成检查点对应的巡检任务。
        service_columns = tuple(column for column in snapshot.service_columns if column.admitted)  # 读取环境快照中的准入服务列集合。
        uav_by_id = {uav.object_id: uav for uav in uavs}  # 按稳定标识索引车载无人机。
        effective_by_id = {column.object_id: _effective_column(snapshot, column) for column in service_columns}  # 中文说明。
        uav_state_by_id = {uav.object_id: _rolling_uav_state(snapshot, uav) for uav in uavs}  # 中文说明。
        stop_by_id = {stop.object_id: stop for stop in stop_sites}  # 按稳定标识索引候选停靠点。
        start_nodes = tuple(node for node in road_nodes if node.role == "depot_start")  # 提取起点仓库道路节点。
        end_nodes = tuple(node for node in road_nodes if node.role == "depot_end")  # 提取终点仓库道路节点。
        if len(start_nodes) != 1 or len(end_nodes) != 1:  # 判断环境是否具有唯一的起点和终点仓库副本。
            raise ValueError("M5 要求唯一的起点仓库节点和终点仓库节点")  # 抛出明确的仓库节点结构异常。
        start_node_id = start_nodes[0].object_id  # 保存唯一的起点仓库节点标识。
        end_node_id = end_nodes[0].object_id  # 保存唯一的终点仓库节点标识。
        vehicle_start_node_ids = {  # 中文说明。
            vehicle.object_id: snapshot.vehicle_state(vehicle.object_id).current_road_node_id
            if hasattr(snapshot, "vehicle_state") else start_node_id
            for vehicle in vehicles
        }
        road_use: dict[tuple[str, str], VariableReference] = {}  # 创建车辆道路弧变量映射。
        stop_visit: dict[tuple[str, str], VariableReference] = {}  # 创建车辆停靠点变量映射。
        column_select: dict[tuple[str, str], VariableReference] = {}  # 创建车辆服务列变量映射。
        checkpoint_missed: dict[str, VariableReference] = {}  # 创建检查点漏检变量映射。
        vehicle_arrival: dict[tuple[str, str], VariableReference] = {}  # 创建车辆道路节点到达时间变量映射。
        vehicle_departure: dict[tuple[str, str], VariableReference] = {}  # 创建车辆道路节点离开时间变量映射。
        column_launch: dict[tuple[str, str], VariableReference] = {}  # 创建服务列起飞时间变量映射。
        column_recovery: dict[tuple[str, str], VariableReference] = {}  # 创建服务列回收时间变量映射。
        recovery_wait: dict[tuple[str, str], VariableReference] = {}  # 创建服务列回收等待时间变量映射。
        checkpoint_completion: dict[str, VariableReference] = {}  # 创建检查点完成时间变量映射。
        column_precedence: dict[tuple[str, str, str], VariableReference] = {}  # 创建同一无人机服务列先后关系变量映射。
        plan_change: dict[tuple[str, str, str], VariableReference] = {}  # 创建道路、停靠和服务列方案变更变量映射。
        for option in road_options:  # 逐一创建通过认证的车辆道路弧变量。
            key = (option.vehicle_id, option.road_leg_id)  # 构造车辆道路弧变量稳定键。
            coefficient = self._config.weight_vehicle_time * option.travel_time  # 计算道路行驶时间加权目标系数。
            variable = builder.add_binary(f"road_use[{option.vehicle_id},{option.road_leg_id}]", coefficient)  # 创建车辆道路弧选择变量。
            road_use[key] = variable  # 记录车辆道路弧变量引用。
            objective_records["vehicle_time"].append((variable.index, coefficient))  # 记录车辆道路时间目标分项。
        for option in stop_options:  # 逐一创建通过认证的车辆停靠点访问变量。
            if option.stop_site_id not in stop_by_id:  # 判断车辆停靠选项是否引用未知物理停靠点。
                raise ValueError(f"车辆停靠选项引用未知停靠点: {option.vehicle_id},{option.stop_site_id}")  # 拒绝无法联结物理停靠点的选项。
            option_key = (option.vehicle_id, option.stop_site_id)  # 构造车辆停靠变量稳定键。
            if option_key in stop_visit:  # 判断环境是否错误地提供重复车辆停靠选项。
                raise ValueError(f"车辆停靠选项重复: {option.vehicle_id},{option.stop_site_id}")  # 拒绝会产生重复变量名称的选项。
            stop_visit[option_key] = builder.add_binary(f"stop_visit[{option.vehicle_id},{option.stop_site_id}]")  # 创建并记录通过认证的车辆停靠点访问变量。
        for vehicle in vehicles:  # 逐一处理地面车辆。
            for node in road_nodes:  # 逐一创建车辆道路节点时序变量。
                vehicle_arrival[(vehicle.object_id, node.object_id)] = builder.add_continuous(f"vehicle_arrival[{vehicle.object_id},{node.object_id}]", 0.0, snapshot.time_horizon)  # 创建并记录车辆到达时间变量。
                vehicle_departure[(vehicle.object_id, node.object_id)] = builder.add_continuous(f"vehicle_departure[{vehicle.object_id},{node.object_id}]", 0.0, snapshot.time_horizon)  # 创建并记录车辆离开时间变量。
        for column in service_columns:  # 逐一创建与执行无人机所属车辆兼容的服务列变量。
            if column.uav_id not in uav_by_id:  # 判断服务列执行无人机是否存在于环境快照。
                raise ValueError(f"服务列引用未知无人机: {column.object_id}")  # 抛出包含服务列标识的无人机引用异常。
            vehicle_id = uav_by_id[column.uav_id].carrier_vehicle_id  # 读取执行无人机所属地面车辆标识。
            key = (vehicle_id, column.object_id)  # 构造车辆服务列变量稳定键。
            effective = effective_by_id[column.object_id]  # 中文说明。
            uav_state = uav_state_by_id[column.uav_id]  # 中文说明。
            energy_coefficient = self._config.weight_energy * effective.effective_protected_energy  # 计算服务列概率能耗加权目标系数。
            can_select = uav_state.available or getattr(uav_state,'available_after_seconds',0.)>0 or effective.is_committed
            column_variable = builder.add_binary(f"column_select[{vehicle_id},{column.object_id}]", energy_coefficient, upper=1.0 if can_select else 0.0)  # 创建车辆服务列选择变量。
            column_select[key] = column_variable  # 记录车辆服务列变量引用。
            objective_records["energy"].append((column_variable.index, energy_coefficient))  # 记录概率安全能耗目标分项。
            column_launch[key] = builder.add_continuous(f"column_launch[{vehicle_id},{column.object_id}]", 0.0, 0.0 if effective.is_airborne else snapshot.time_horizon)  # 创建并记录服务列起飞时间变量。
            column_recovery[key] = builder.add_continuous(f"column_recovery[{vehicle_id},{column.object_id}]", 0.0, snapshot.time_horizon)  # 创建并记录服务列回收时间变量。
            waiting_coefficient = self._config.weight_waiting + self._config.weight_energy * getattr(uav_by_id[column.uav_id], "hover_power", 0.0)  # 中文说明。
            waiting_variable = builder.add_continuous(f"recovery_wait[{vehicle_id},{column.object_id}]", 0.0, snapshot.time_horizon, waiting_coefficient)  # 创建服务列回收等待时间变量。
            recovery_wait[key] = waiting_variable  # 记录服务列回收等待时间变量引用。
            objective_records["waiting"].append((waiting_variable.index, self._config.weight_waiting))  # 记录回收等待时间目标分项。
            objective_records["energy"].append((waiting_variable.index, self._config.weight_energy * getattr(uav_by_id[column.uav_id], "hover_power", 0.0)))  # 中文说明。
        for task in remaining_tasks:  # 逐一创建剩余检查点任务变量。
            checkpoint_id = task.checkpoint_id  # 读取当前任务对应的检查点标识。
            if checkpoint_id in checkpoint_completion:  # 判断多个任务是否错误地复用同一检查点。
                raise ValueError(f"一个检查点只能对应一个剩余任务: {checkpoint_id}")  # 抛出明确的检查点任务重复异常。
            completion_coefficient = self._config.weight_completion * task.priority  # 计算风险加权完成时间目标系数。
            completion_variable = builder.add_continuous(f"checkpoint_completion[{checkpoint_id}]", 0.0, snapshot.time_horizon, completion_coefficient)  # 创建检查点完成时间变量。
            checkpoint_completion[checkpoint_id] = completion_variable  # 记录检查点完成时间变量引用。
            objective_records["completion"].append((completion_variable.index, completion_coefficient))  # 记录风险加权完成时间目标分项。
            missed_coefficient = self._config.weight_missed * task.priority  # 计算检查点漏检加权惩罚系数。
            missed_variable = builder.add_binary(f"checkpoint_missed[{checkpoint_id}]", missed_coefficient)  # 创建检查点漏检变量。
            checkpoint_missed[checkpoint_id] = missed_variable  # 记录检查点漏检变量引用。
            objective_records["missed"].append((missed_variable.index, missed_coefficient))  # 记录检查点漏检目标分项。
        makespan = builder.add_continuous("makespan", 0.0, snapshot.time_horizon, self._config.weight_makespan)  # 创建最大完成时间变量。
        objective_records["makespan"].append((makespan.index, self._config.weight_makespan))  # 记录最大完成时间目标分项。
        locked_column_sequence = tuple(
            dict.fromkeys(
                (
                    *snapshot.locked_column_ids,
                    *getattr(snapshot, "committed_column_ids", ()),
                    *(
                        state.column_id
                        for state in getattr(snapshot, "airborne_residual_states", ())
                    ),
                )
            )
        )
        locked_column_ids = set(locked_column_sequence)
        for uav in uavs:  # 逐一处理车载无人机服务列先后关系。
            uav_columns = sorted((column for column in service_columns if column.uav_id == uav.object_id), key=lambda column: column.object_id)  # 按稳定标识排序当前无人机准入服务列。
            for first_index, first_column in enumerate(uav_columns):  # 逐一选择服务列对中的第一条服务列。
                for second_column in uav_columns[first_index + 1 :]:  # 逐一选择稳定顺序中后续服务列。
                    if (
                        first_column.object_id in locked_column_ids
                        and second_column.object_id in locked_column_ids
                    ):
                        continue
                    key = (uav.object_id, first_column.object_id, second_column.object_id)  # 构造无人机服务列先后变量稳定键。
                    column_precedence[key] = builder.add_binary(f"column_precedence[{uav.object_id},{first_column.object_id},{second_column.object_id}]")  # 创建并记录服务列先后关系变量。
        self._add_vehicle_constraints(builder, vehicles, road_nodes, road_options, road_use, stop_sites, stop_visit, vehicle_arrival, vehicle_departure, vehicle_start_node_ids, end_node_id)  # 添加车辆流、停靠联结和道路时序约束。
        self._add_column_constraints(builder, service_columns, effective_by_id, uav_by_id, stop_by_id, column_select, stop_visit, vehicle_arrival, vehicle_departure, column_launch, column_recovery, recovery_wait)  # 添加服务列停靠和起飞回收同步约束。
        for column in service_columns:
            delay=getattr(uav_state_by_id[column.uav_id],'available_after_seconds',0.)
            if delay>0 and not effective_by_id[column.object_id].is_airborne:
                vehicle_id=uav_by_id[column.uav_id].carrier_vehicle_id
                key=(vehicle_id,column.object_id)
                builder.add_constraint(f'uav_ready_time[{column.object_id}]',
                    ((column_launch[key],1.),(column_select[key],-self._config.time_big_m)),
                    delay-self._config.time_big_m,math.inf)
        self._add_coverage_constraints(builder, remaining_tasks, service_columns, effective_by_id, uav_by_id, column_select, checkpoint_missed, checkpoint_completion, column_launch, makespan)  # 添加检查点覆盖、完成时间和最大完成时间约束。
        self._add_nonoverlap_constraints(builder, service_columns, uav_by_id, uav_state_by_id, column_select, column_launch, column_recovery, column_precedence)  # 添加同一无人机服务列不重叠约束。
        self._add_locked_nonoverlap_chains(
            builder,
            service_columns,
            uav_by_id,
            uav_state_by_id,
            stop_by_id,
            effective_by_id,
            column_launch,
            column_recovery,
            locked_column_sequence,
        )
        self._add_rolling_constraints(builder, snapshot, road_use, stop_visit, column_select, plan_change, objective_records)  # 添加滚动锁定和三类方案变更约束。
        built_model = builder.build()  # 将命名变量和约束转换为 SciPy 稀疏数值模型。
        immutable_objective_records = {name: tuple(records) for name, records in objective_records.items()}  # 将目标分项变量系数转换为不可变元组。
        return SchedulingModelArtifacts(built_model, road_use, stop_visit, column_select, checkpoint_completion, immutable_objective_records)  # 返回 M5 数值模型和结果重构映射。

    def _add_vehicle_constraints(self, builder: LinearModelBuilder, vehicles: tuple[Any, ...], road_nodes: tuple[Any, ...], road_options: tuple[Any, ...], road_use: dict[tuple[str, str], VariableReference], stop_sites: tuple[Any, ...], stop_visit: dict[tuple[str, str], VariableReference], vehicle_arrival: dict[tuple[str, str], VariableReference], vehicle_departure: dict[tuple[str, str], VariableReference], vehicle_start_node_ids: dict[str, str], end_node_id: str) -> None:  # 定义车辆流和道路时序约束添加方法。
        """添加仓库流、节点流守恒、停靠访问联结和道路时间传递约束。"""  # 解释车辆约束范围。
        big_m = self._config.time_big_m  # 读取时序线性化充分大常数。
        for vehicle in vehicles:  # 逐一处理地面车辆路径约束。
            vehicle_id = vehicle.object_id  # 保存当前地面车辆标识。
            start_node_id = vehicle_start_node_ids[vehicle_id]  # 中文说明。
            vehicle_options = tuple(option for option in road_options if option.vehicle_id == vehicle_id)  # 提取当前车辆通过认证的道路选项。
            outgoing_start = tuple((road_use[(vehicle_id, option.road_leg_id)], 1.0) for option in vehicle_options if option.source_node_id == start_node_id)  # 构造起点仓库出流项。
            incoming_start = tuple((road_use[(vehicle_id, option.road_leg_id)], 1.0) for option in vehicle_options if option.target_node_id == start_node_id)  # 构造起点仓库入流项。
            incoming_end = tuple((road_use[(vehicle_id, option.road_leg_id)], 1.0) for option in vehicle_options if option.target_node_id == end_node_id)  # 构造终点仓库入流项。
            outgoing_end = tuple((road_use[(vehicle_id, option.road_leg_id)], 1.0) for option in vehicle_options if option.source_node_id == end_node_id)  # 构造终点仓库出流项。
            builder.add_constraint(f"vehicle_start_flow[{vehicle_id}]", outgoing_start, 1.0, 1.0)  # 要求车辆恰好从起点仓库驶出一次。
            builder.add_constraint(f"vehicle_start_in_zero[{vehicle_id}]", incoming_start, 0.0, 0.0)  # 禁止车辆返回起点仓库副本。
            builder.add_constraint(f"vehicle_end_flow[{vehicle_id}]", incoming_end, 1.0, 1.0)  # 要求车辆恰好进入终点仓库一次。
            builder.add_constraint(f"vehicle_end_out_zero[{vehicle_id}]", outgoing_end, 0.0, 0.0)  # 禁止车辆离开终点仓库副本。
            builder.add_constraint(f"vehicle_start_time[{vehicle_id},{start_node_id}]", ((vehicle_arrival[(vehicle_id, start_node_id)], 1.0),), 0.0, 0.0)  # 将车辆当前起点到达时间固定为零。
            for node in road_nodes:  # 逐一处理道路节点流守恒和停留时间。
                node_id = node.object_id  # 保存当前道路节点标识。
                builder.add_constraint(f"vehicle_dwell_nonnegative[{vehicle_id},{node_id}]", ((vehicle_departure[(vehicle_id, node_id)], 1.0), (vehicle_arrival[(vehicle_id, node_id)], -1.0)), 0.0, math.inf)  # 要求车辆离开时间不早于到达时间。
                if node_id in {start_node_id, end_node_id}:  # 判断当前节点是否属于仓库副本。
                    continue  # 仓库副本使用专门的起终点流约束。
                incoming_terms = tuple((road_use[(vehicle_id, option.road_leg_id)], 1.0) for option in vehicle_options if option.target_node_id == node_id)  # 构造当前节点车辆入流项。
                outgoing_terms = tuple((road_use[(vehicle_id, option.road_leg_id)], -1.0) for option in vehicle_options if option.source_node_id == node_id)  # 构造当前节点车辆负出流项。
                builder.add_constraint(f"vehicle_stop_flow[{vehicle_id},{node_id}]", (*incoming_terms, *outgoing_terms), 0.0, 0.0)  # 要求中间道路节点入流等于出流。
            for stop in stop_sites:  # 逐一联结车辆停靠点访问变量和道路流。
                stop_key = (vehicle_id, stop.object_id)  # 构造当前车辆停靠组合键。
                if stop_key not in stop_visit:  # 判断当前物理停靠点是否未通过该车辆能力认证。
                    continue  # 允许车辆通过道路节点但不创建停靠访问决策。
                visit = stop_visit[stop_key]  # 读取当前通过认证的车辆停靠点访问变量。
                incoming_terms = tuple((road_use[(vehicle_id, option.road_leg_id)], 1.0) for option in vehicle_options if option.target_node_id == stop.road_node_id)  # 构造停靠点道路节点入流项。
                outgoing_terms = tuple((road_use[(vehicle_id, option.road_leg_id)], 1.0) for option in vehicle_options if option.source_node_id == stop.road_node_id)  # 构造停靠点道路节点出流项。
                if stop.road_node_id == start_node_id:  # 中文说明。
                    incoming_terms = outgoing_terms  # 中文说明。
                elif stop.road_node_id == end_node_id:
                    outgoing_terms = incoming_terms
                builder.add_constraint(f"stop_visit_in_link[{vehicle_id},{stop.object_id}]", (*incoming_terms, (visit, -1.0)), 0.0, 0.0)  # 要求停靠点访问变量等于道路节点入流。
                builder.add_constraint(f"stop_visit_out_link[{vehicle_id},{stop.object_id}]", (*outgoing_terms, (visit, -1.0)), 0.0, 0.0)  # 要求停靠点访问变量等于道路节点出流。
            for option in vehicle_options:  # 逐一添加当前车辆道路时间传递约束。
                road_variable = road_use[(vehicle_id, option.road_leg_id)]  # 读取当前车辆道路弧选择变量。
                terms = ((vehicle_arrival[(vehicle_id, option.target_node_id)], 1.0), (vehicle_departure[(vehicle_id, option.source_node_id)], -1.0), (road_variable, -big_m))  # 构造道路时间传递线性表达式。
                builder.add_constraint(f"road_time[{vehicle_id},{option.road_leg_id}]", terms, option.travel_time - big_m, math.inf)  # 仅在道路弧被选择时要求到达时间覆盖行驶时间。

    def _add_column_constraints(self, builder: LinearModelBuilder, service_columns: tuple[Any, ...], effective_by_id: dict[str, Any], uav_by_id: dict[str, Any], stop_by_id: dict[str, Any], column_select: dict[tuple[str, str], VariableReference], stop_visit: dict[tuple[str, str], VariableReference], vehicle_arrival: dict[tuple[str, str], VariableReference], vehicle_departure: dict[tuple[str, str], VariableReference], column_launch: dict[tuple[str, str], VariableReference], column_recovery: dict[tuple[str, str], VariableReference], recovery_wait: dict[tuple[str, str], VariableReference]) -> None:  # 定义服务列停靠和同步约束添加方法。
        """添加服务列选择、车辆到达、动力学时间、回收和离站联结约束。"""  # 解释服务列同步约束范围。
        big_m = self._config.time_big_m  # 读取时序线性化充分大常数。
        for column in service_columns:  # 逐一处理准入服务列同步约束。
            vehicle_id = uav_by_id[column.uav_id].carrier_vehicle_id  # 读取服务列执行无人机所属车辆标识。
            key = (vehicle_id, column.object_id)  # 构造车辆服务列变量稳定键。
            selected = column_select[key]  # 读取服务列选择变量。
            launch = column_launch[key]  # 读取服务列起飞时间变量。
            recovery = column_recovery[key]  # 读取服务列回收时间变量。
            waiting = recovery_wait[key]  # 读取服务列回收等待时间变量。
            effective = effective_by_id[column.object_id]  # 中文说明。
            if column.launch_stop_id not in stop_by_id or column.recovery_stop_id not in stop_by_id:  # 判断服务列起飞和回收停靠点是否存在于环境快照。
                raise ValueError(f"服务列引用未知停靠点: {column.object_id}")  # 抛出包含服务列标识的停靠点引用异常。
            if (vehicle_id, column.launch_stop_id) not in stop_visit or (vehicle_id, column.recovery_stop_id) not in stop_visit:  # 判断准入服务列是否引用未通过车辆能力认证的停靠组合。
                raise ValueError(f"服务列引用未准入车辆停靠组合: {column.object_id}")  # 拒绝 Env 与 Agent 准入集合不一致的服务列。
            launch_stop = stop_by_id[column.launch_stop_id]  # 读取服务列起飞停靠点对象。
            recovery_stop = stop_by_id[column.recovery_stop_id]  # 读取服务列回收停靠点对象。
            if effective.requires_launch_stop:  # 中文说明。
                builder.add_constraint(f"column_launch_stop[{vehicle_id},{column.object_id}]", ((selected, 1.0), (stop_visit[(vehicle_id, column.launch_stop_id)], -1.0)), -math.inf, 0.0)  # 要求选择服务列时车辆访问起飞停靠点。
            builder.add_constraint(f"column_recovery_stop[{vehicle_id},{column.object_id}]", ((selected, 1.0), (stop_visit[(vehicle_id, column.recovery_stop_id)], -1.0)), -math.inf, 0.0)  # 要求选择服务列时车辆访问回收停靠点。
            builder.add_constraint(f"column_launch_activation[{vehicle_id},{column.object_id}]", ((launch, 1.0), (selected, -big_m)), -math.inf, 0.0)  # 将未选择服务列的起飞时间压缩为零。
            builder.add_constraint(f"column_recovery_activation[{vehicle_id},{column.object_id}]", ((recovery, 1.0), (selected, -big_m)), -math.inf, 0.0)  # 将未选择服务列的回收时间压缩为零。
            if effective.requires_launch_stop:  # 中文说明。
                builder.add_constraint(f"launch_after_arrival[{vehicle_id},{column.object_id}]", ((launch, 1.0), (vehicle_arrival[(vehicle_id, launch_stop.road_node_id)], -1.0), (selected, -big_m)), -big_m, math.inf)  # 仅在选择服务列时要求车辆到达后起飞。
                builder.add_constraint(f"departure_after_launch[{vehicle_id},{column.object_id}]", ((vehicle_departure[(vehicle_id, launch_stop.road_node_id)], 1.0), (launch, -1.0), (selected, -big_m)), -big_m, math.inf)  # 仅在选择服务列时要求车辆在无人机起飞后离开起飞停靠点。
            builder.add_constraint(f"column_duration[{vehicle_id},{column.object_id}]", ((recovery, 1.0), (launch, -1.0), (selected, -big_m)), effective.effective_duration - big_m, math.inf)  # 仅在选择服务列时要求回收时间覆盖动力学航次持续时间。
            builder.add_constraint(f"recovery_after_vehicle[{vehicle_id},{column.object_id}]", ((recovery, 1.0), (vehicle_arrival[(vehicle_id, recovery_stop.road_node_id)], -1.0), (selected, -big_m)), -big_m, math.inf)  # 仅在选择服务列时要求车辆到达后完成回收。
            builder.add_constraint(f"departure_after_recovery[{vehicle_id},{column.object_id}]", ((vehicle_departure[(vehicle_id, recovery_stop.road_node_id)], 1.0), (recovery, -1.0), (selected, -big_m)), -big_m, math.inf)  # 仅在选择服务列时要求车辆回收无人机后离站。
            builder.add_constraint(f"recovery_wait_definition[{vehicle_id},{column.object_id}]", ((waiting, 1.0), (recovery, -1.0), (launch, 1.0), (selected, -big_m)), -effective.effective_duration - big_m, math.inf)  # 定义选择服务列后的非负回收等待时间。
            builder.add_constraint(f"recovery_wait_activation[{vehicle_id},{column.object_id}]", ((waiting, 1.0), (selected, -big_m)), -math.inf, 0.0)  # 将未选择服务列的回收等待时间压缩为零。
            hover_power = getattr(uav_by_id[column.uav_id], "hover_power", 0.0)  # 中文说明。
            if math.isfinite(effective.usable_battery_energy):  # 中文说明。
                builder.add_constraint(f"column_battery[{vehicle_id},{column.object_id}]", ((waiting, hover_power), (selected, effective.effective_protected_energy - effective.usable_battery_energy)), -math.inf, 0.0)  # 中文说明。

    def _add_coverage_constraints(self, builder: LinearModelBuilder, tasks: tuple[Any, ...], service_columns: tuple[Any, ...], effective_by_id: dict[str, Any], uav_by_id: dict[str, Any], column_select: dict[tuple[str, str], VariableReference], checkpoint_missed: dict[str, VariableReference], checkpoint_completion: dict[str, VariableReference], column_launch: dict[tuple[str, str], VariableReference], makespan: VariableReference) -> None:  # 定义检查点覆盖和完成时间约束添加方法。
        """添加覆盖或漏检、完成时间双边联结和最大完成时间约束。"""  # 解释覆盖和完成时间约束范围。
        big_m = self._config.time_big_m  # 读取时序线性化充分大常数。
        for task in tasks:  # 逐一处理剩余检查点任务。
            checkpoint_id = task.checkpoint_id  # 读取当前任务对应的检查点标识。
            covering_columns = tuple(column for column in service_columns if checkpoint_id in effective_by_id[column.object_id].effective_checkpoint_ids)  # 提取覆盖当前检查点的准入服务列。
            coverage_terms = []  # 创建检查点覆盖约束线性项列表。
            for column in covering_columns:  # 逐一处理覆盖当前检查点的服务列。
                vehicle_id = uav_by_id[column.uav_id].carrier_vehicle_id  # 读取当前服务列所属车辆标识。
                coverage_terms.append((column_select[(vehicle_id, column.object_id)], 1.0))  # 将当前服务列选择变量加入覆盖约束。
            coverage_terms.append((checkpoint_missed[checkpoint_id], 1.0))  # 将检查点漏检变量加入覆盖约束。
            builder.add_constraint(f"checkpoint_coverage[{checkpoint_id}]", tuple(coverage_terms), 1.0, 1.0)  # 要求每个检查点恰好被一个服务列覆盖或标记漏检。
            completion = checkpoint_completion[checkpoint_id]  # 读取当前检查点完成时间变量。
            builder.add_constraint(f"checkpoint_completion_activation[{checkpoint_id}]", ((completion, 1.0), (checkpoint_missed[checkpoint_id], big_m)), -math.inf, big_m)  # 将漏检检查点完成时间压缩为零。
            builder.add_constraint(f"makespan[{checkpoint_id}]", ((makespan, 1.0), (completion, -1.0)), 0.0, math.inf)  # 要求最大完成时间不小于当前检查点完成时间。
            for column in covering_columns:  # 逐一添加覆盖当前检查点的完成时间联结约束。
                vehicle_id = uav_by_id[column.uav_id].carrier_vehicle_id  # 读取当前服务列所属车辆标识。
                key = (vehicle_id, column.object_id)  # 构造车辆服务列变量稳定键。
                completion_offset = dict(effective_by_id[column.object_id].effective_completion_offsets)[checkpoint_id]  # 读取当前检查点在服务列中的完成时间偏移。
                selected = column_select[key]  # 读取当前服务列选择变量。
                launch = column_launch[key]  # 读取当前服务列起飞时间变量。
                builder.add_constraint(f"checkpoint_completion_lower[{checkpoint_id},{column.object_id}]", ((completion, 1.0), (launch, -1.0), (selected, -big_m)), completion_offset - big_m, math.inf)  # 选择服务列时锁定检查点完成时间下界。
                builder.add_constraint(f"checkpoint_completion_upper[{checkpoint_id},{column.object_id}]", ((completion, 1.0), (launch, -1.0), (selected, big_m)), -math.inf, completion_offset + big_m)  # 选择服务列时锁定检查点完成时间上界。

    def _add_nonoverlap_constraints(self, builder: LinearModelBuilder, service_columns: tuple[Any, ...], uav_by_id: dict[str, Any], uav_state_by_id: dict[str, Any], column_select: dict[tuple[str, str], VariableReference], column_launch: dict[tuple[str, str], VariableReference], column_recovery: dict[tuple[str, str], VariableReference], column_precedence: dict[tuple[str, str, str], VariableReference]) -> None:  # 定义同一无人机服务列不重叠约束添加方法。
        """使用一对方向约束和先后二元变量排除同时选择服务列的时间重叠。"""  # 解释无人机不重叠线性化方法。
        big_m = self._config.time_big_m  # 读取时序线性化充分大常数。
        for (uav_id, first_column_id, second_column_id), precedence in column_precedence.items():  # 逐一处理同一无人机的无序服务列对。
            vehicle_id = uav_by_id[uav_id].carrier_vehicle_id  # 读取当前无人机所属车辆标识。
            first_key = (vehicle_id, first_column_id)  # 构造第一条车辆服务列稳定键。
            second_key = (vehicle_id, second_column_id)  # 构造第二条车辆服务列稳定键。
            first_selected = column_select[first_key]  # 读取第一条服务列选择变量。
            second_selected = column_select[second_key]  # 读取第二条服务列选择变量。
            turnaround = uav_state_by_id[uav_id].turnaround_time  # 中文说明。
            forward_terms = ((column_launch[second_key], 1.0), (column_recovery[first_key], -1.0), (precedence, -big_m), (first_selected, -big_m), (second_selected, -big_m))  # 构造第一条服务列先于第二条的条件表达式。
            backward_terms = ((column_launch[first_key], 1.0), (column_recovery[second_key], -1.0), (precedence, big_m), (first_selected, -big_m), (second_selected, -big_m))  # 构造第二条服务列先于第一条的条件表达式。
            builder.add_constraint(f"uav_nonoverlap_forward[{uav_id},{first_column_id},{second_column_id}]", forward_terms, -3.0 * big_m + turnaround, math.inf)  # 在先后变量为一且两列被选时执行正向不重叠约束。
            builder.add_constraint(f"uav_nonoverlap_backward[{uav_id},{first_column_id},{second_column_id}]", backward_terms, -2.0 * big_m + turnaround, math.inf)  # 在先后变量为零且两列被选时执行反向不重叠约束。

    def _add_locked_nonoverlap_chains(
        self,
        builder: LinearModelBuilder,
        service_columns: tuple[Any, ...],
        uav_by_id: dict[str, Any],
        uav_state_by_id: dict[str, Any],
        stop_by_id: dict[str, Any],
        effective_by_id: dict[str, Any],
        column_launch: dict[tuple[str, str], VariableReference],
        column_recovery: dict[tuple[str, str], VariableReference],
        locked_column_sequence: tuple[str, ...],
    ) -> None:
        """Order already-fixed columns with a linear deterministic chain."""
        sequence_rank = {
            column_id: index
            for index, column_id in enumerate(locked_column_sequence)
        }
        for uav_id, uav_state in uav_state_by_id.items():
            locked_columns = tuple(
                sorted(
                    (
                        column
                        for column in service_columns
                        if column.uav_id == uav_id
                        and column.object_id in sequence_rank
                    ),
                    key=lambda column: (
                        0 if effective_by_id[column.object_id].is_airborne else 1,
                        sequence_rank[column.object_id],
                    ),
                )
            )
            vehicle_id = uav_by_id[uav_id].carrier_vehicle_id
            for first, second in zip(locked_columns[:-1], locked_columns[1:]):
                builder.add_constraint(
                    f"uav_locked_sequence[{uav_id},{first.object_id},{second.object_id}]",
                    (
                        (column_launch[(vehicle_id, second.object_id)], 1.0),
                        (column_recovery[(vehicle_id, first.object_id)], -1.0),
                    ),
                    uav_state.turnaround_time,
                    math.inf,
                )

    def _add_rolling_constraints(self, builder: LinearModelBuilder, snapshot: Any, road_use: dict[tuple[str, str], VariableReference], stop_visit: dict[tuple[str, str], VariableReference], column_select: dict[tuple[str, str], VariableReference], plan_change: dict[tuple[str, str, str], VariableReference], objective_records: dict[str, list[tuple[int, float]]]) -> None:  # 定义滚动锁定和方案变更约束添加方法。
        """固定锁定服务列并线性化道路、停靠和服务列相对上一方案的绝对变化。"""  # 解释滚动计划约束范围。
        selected_column_by_id = {column_id: variable for (_, column_id), variable in column_select.items()}  # 按服务列标识索引当前选择变量。
        airborne_ids = tuple(state.column_id for state in getattr(snapshot, "airborne_residual_states", ()))  # 中文说明。
        all_locked_ids = (*snapshot.locked_column_ids, *getattr(snapshot, "committed_column_ids", ()), *airborne_ids)  # 中文说明。
        locked_ids = tuple(dict.fromkeys(all_locked_ids))  # 中文说明。
        for locked_column_id in locked_ids:  # 逐一处理滚动计划锁定服务列。
            if locked_column_id not in selected_column_by_id:  # 判断锁定服务列是否存在于当前准入集合。
                raise ValueError(f"锁定服务列不在当前准入集合: {locked_column_id}")  # 抛出包含锁定服务列标识的滚动状态异常。
            variable = selected_column_by_id[locked_column_id]  # 读取锁定服务列选择变量。
            vehicle_id = next(vehicle_id for (vehicle_id, column_id), candidate in column_select.items() if column_id == locked_column_id and candidate == variable)  # 查找锁定服务列所属车辆标识。
            builder.add_constraint(f"rolling_lock[{vehicle_id},{locked_column_id}]", ((variable, 1.0),), 1.0, 1.0)  # 将锁定服务列选择变量固定为一。
        previous_roads = set(getattr(snapshot, "previous_selected_road_leg_ids", ()))  # 读取上一滚动方案道路弧标识集合。
        previous_stops = set(getattr(snapshot, "previous_selected_stop_ids", ()))  # 读取上一滚动方案停靠点标识集合。
        previous_columns = set(snapshot.previous_selected_column_ids)  # 读取上一滚动方案服务列标识集合。
        variable_groups = (  # 构造三类当前方案变量及其上一方案集合。
            ("road", road_use, previous_roads),  # 登记车辆道路弧方案变化组。
            ("stop", stop_visit, previous_stops),  # 登记车辆停靠点方案变化组。
            ("column", column_select, previous_columns),  # 登记车辆服务列方案变化组。
        )  # 结束三类方案变化组集合。
        for group_name, variables, previous_ids in variable_groups:  # 逐一处理道路、停靠和服务列方案变化。
            for (vehicle_id, object_id), variable in variables.items():  # 逐一处理当前方案中的二元决策变量。
                coefficient = self._config.weight_plan_change  # 读取滚动方案变更目标系数。
                change = builder.add_continuous(f"plan_change[{group_name},{vehicle_id},{object_id}]", 0.0, 1.0, coefficient)  # 创建当前二元变量的绝对变化辅助变量。
                plan_change[(group_name, vehicle_id, object_id)] = change  # 记录当前方案变化变量引用。
                objective_records["plan_change"].append((change.index, coefficient))  # 记录滚动方案变更目标分项。
                previous_value = 1.0 if object_id in previous_ids else 0.0  # 将上一方案对象选择状态转换为零一常数。
                builder.add_constraint(f"plan_change_positive[{group_name},{vehicle_id},{object_id}]", ((change, 1.0), (variable, -1.0)), -previous_value, math.inf)  # 约束变化变量不小于当前值减上一值。
                builder.add_constraint(f"plan_change_negative[{group_name},{vehicle_id},{object_id}]", ((change, 1.0), (variable, 1.0)), previous_value, math.inf)  # 约束变化变量不小于上一值减当前值。

    def solve(self, snapshot: SceneSnapshot) -> AgentResult:  # 实现统一 Agent 求解接口。
        return self.solve_with_deadline(snapshot, deadline=None)

    def solve_with_deadline(self, snapshot: SceneSnapshot, *, deadline) -> AgentResult:
        """构建 M5、调用 SciPy MILP 并重构可由 Env 独立复核的结果。"""  # 解释 SciPy MILP 求解行为。
        start_time = time.perf_counter()  # 记录 Agent 求解开始时间。
        artifacts = self.build_model(snapshot)  # 根据环境快照构建完整 M5 数值模型。
        remaining = min(self._config.time_limit, deadline.remaining_seconds()) if deadline is not None else self._config.time_limit
        if remaining <= 0.:
            return AgentResult(False, 1, 'runtime_deadline_after_model_build', None, (), (), (), (), (), (),
                               (('solver_termination', 'runtime_deadline'),), time.perf_counter()-start_time)
        options = {"time_limit": remaining, "mip_rel_gap": self._config.mip_relative_gap, "presolve": True}  # 构造 SciPy MILP 求解停止和预处理参数。
        result = milp(c=artifacts.model.objective, integrality=artifacts.model.integrality, bounds=artifacts.model.bounds, constraints=artifacts.model.linear_constraint, options=options)  # 调用 SciPy 混合整数线性规划求解器。
        runtime_seconds = time.perf_counter() - start_time  # 计算 Agent 构建和求解总运行时间。
        success, solver_termination, solver_optimal = classify_milp_result(result)  # 区分有限候选与最优性证明。
        named_values = artifacts.model.variable_names  # 读取稳定变量名称顺序。
        if result.x is None:  # 判断 SciPy 是否没有返回数值解向量。
            value_map: dict[str, float] = {}  # 使用空映射表示不存在可重构变量值。
        else:  # 处理 SciPy 返回数值解向量的情况。
            value_map = {name: float(result.x[index]) for index, name in enumerate(named_values)}  # 按稳定变量名称重构数值解。
        selected_roads = tuple(road_id for (_, road_id), variable in artifacts.road_use.items() if value_map.get(variable.name, 0.0) > 0.5)  # 提取数值大于二分之一的道路弧选择。
        selected_stops = tuple(stop_id for (_, stop_id), variable in artifacts.stop_visit.items() if value_map.get(variable.name, 0.0) > 0.5)  # 提取数值大于二分之一的停靠点选择。
        selected_columns = tuple(column_id for (_, column_id), variable in artifacts.column_select.items() if value_map.get(variable.name, 0.0) > 0.5)  # 提取数值大于二分之一的服务列选择。
        completion_times = tuple((checkpoint_id, value_map.get(variable.name, 0.0)) for checkpoint_id, variable in artifacts.checkpoint_completion.items())  # 重构全部剩余检查点完成时间。
        objective_decomposition = tuple((component, sum(value_map.get(named_values[index], 0.0) * coefficient for index, coefficient in records)) for component, records in artifacts.objective_records.items())  # 按七类目标项重构加权目标分解。
        diagnostics = (  # 构造稳定求解诊断字段。
            ("variable_count", str(len(artifacts.model.variable_names))),  # 记录 M5 变量数量。
            ("constraint_count", str(len(artifacts.model.constraint_names))),  # 记录 M5 约束数量。
            ("solver_status", str(int(result.status))),
            ("solver_termination", solver_termination),
            ("solver_optimal", "true" if solver_optimal else "false"),
            ("solver_message", str(result.message)),
            ("mip_gap", str(getattr(result, "mip_gap", "unavailable"))),  # 记录 SciPy 返回的相对整数间隙。
            ("mip_node_count", str(getattr(result, "mip_node_count", "unavailable"))),  # 记录 SciPy 分支定界节点数量。
        )  # 结束稳定求解诊断字段集合。
        objective_value = float(result.fun) if result.fun is not None else None  # 读取存在时的 SciPy 目标函数值。
        variable_values = tuple((name, value_map[name]) for name in named_values if name in value_map)  # 按稳定列顺序保存全部命名变量值。
        return AgentResult(success, int(result.status), str(result.message), objective_value, variable_values, objective_decomposition, selected_roads, selected_stops, selected_columns, completion_times, diagnostics, runtime_seconds)  # 返回统一可追踪 Agent 结果。
