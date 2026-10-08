"""定义不同 Agent 共享的可追踪决策结果对象。"""  # 说明本文件的算法职责。

from dataclasses import dataclass  # 导入数据类工具以保存不可变算法结果。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义 Agent 结果。
class AgentResult:  # 定义 Env 和外部运行器共同消费的算法输出。
    """保存求解状态、命名变量、选择对象、目标分解和诊断信息。"""  # 解释 Agent 结果内容。

    success: bool  # 保存算法是否返回可用决策。
    status: int  # 保存具体求解器或算法的稳定状态码。
    message: str  # 保存具体求解器或算法的可读状态说明。
    objective_value: float | None  # 保存可用决策对应的目标函数值。
    variable_values: tuple[tuple[str, float], ...]  # 保存按稳定变量名称排列的数值结果。
    objective_decomposition: tuple[tuple[str, float], ...]  # 保存可重构的目标函数分项。
    selected_road_leg_ids: tuple[str, ...]  # 保存结果选择的道路弧标识。
    selected_stop_ids: tuple[str, ...]  # 保存结果选择的停靠点标识。
    selected_column_ids: tuple[str, ...]  # 保存结果选择的服务列标识。
    checkpoint_completion_times: tuple[tuple[str, float], ...]  # 保存检查点完成时间结果。
    diagnostics: tuple[tuple[str, str], ...]  # 保存模型规模、间隙和验证信息等诊断字段。
    runtime_seconds: float  # 保存算法总运行时间。
