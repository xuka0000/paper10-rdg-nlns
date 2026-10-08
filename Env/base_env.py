"""定义所有优化、启发式和学习算法共享的环境抽象接口。"""  # 说明本文件的环境职责。

from abc import ABC, abstractmethod  # 导入抽象基类工具以约束环境接口。

from Env.snapshot import SceneSnapshot  # 导入不可变场景快照类型。


class BaseEnvironment(ABC):  # 定义全部巡检环境实现共享的抽象基类。
    """要求环境能够重置状态并生成版本化只读快照。"""  # 解释环境抽象边界。

    @abstractmethod  # 声明具体环境必须实现重置方法。
    def reset(self) -> SceneSnapshot:  # 定义环境重置接口。
        """将环境恢复到初始滚动状态并返回新快照。"""  # 解释重置接口输出。
        raise NotImplementedError  # 阻止直接调用未实现的抽象重置方法。

    @abstractmethod  # 声明具体环境必须实现快照方法。
    def snapshot(self) -> SceneSnapshot:  # 定义环境快照接口。
        """返回当前环境状态的不可变版本化快照。"""  # 解释快照接口输出。
        raise NotImplementedError  # 阻止直接调用未实现的抽象快照方法。
