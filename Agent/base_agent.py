"""定义所有优化、启发式、强化学习和图神经网络 Agent 的统一接口。"""  # 说明本文件的算法职责。

from abc import ABC, abstractmethod  # 导入抽象基类工具以约束算法接口。

from Agent.results import AgentResult  # 导入统一 Agent 结果类型。
from Env.snapshot import SceneSnapshot  # 导入不可变环境快照类型。


class BaseAgent(ABC):  # 定义全部决策和求解算法共享的抽象基类。
    """要求算法读取环境快照并返回统一结构的决策结果。"""  # 解释 Agent 抽象边界。

    @abstractmethod  # 声明具体 Agent 必须实现求解或决策方法。
    def solve(self, snapshot: SceneSnapshot) -> AgentResult:  # 定义统一 Agent 求解接口。
        """根据不可变环境快照生成可由 Env 独立复核的结果。"""  # 解释 Agent 接口输入输出。
        raise NotImplementedError  # 阻止直接调用未实现的抽象 Agent 方法。
