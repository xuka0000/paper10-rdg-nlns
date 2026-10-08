"""Replay records for post-repair RDG-DAC transitions. 中文说明。"""

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from collections import deque
# 导入映射协议以验证 checkpoint 中的 Replay 状态对象。
from collections.abc import Mapping
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from dataclasses import dataclass
# 导入数学函数以拒绝非有限收益样本。
import math
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import random

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.graph_neural_network.heterogeneous_service_graph import HeterogeneousServiceGraph


@dataclass(frozen=True, slots=True)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class SearchTransition:
    """说明当前算法或测试步骤。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    state_graph: HeterogeneousServiceGraph
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    destroy_action: int
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    repair_action: int
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    reward_samples: tuple[float, ...]
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    next_state_graph: HeterogeneousServiceGraph
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    done: bool

    # 在转移进入 Replay 之前验证动作和经验收益分布。
    def __post_init__(self) -> None:
        """拒绝非法动作索引、空收益分布和非有限样本。"""
        # 要求两个动作均为非负整数且布尔值不能冒充动作。
        if type(self.destroy_action) is not int or type(self.repair_action) is not int or self.destroy_action < 0 or self.repair_action < 0:
            # 抛出稳定动作索引错误供 checkpoint 和训练入口共同识别。
            raise ValueError("搜索动作索引必须为非负整数")
        # 要求收益分布使用不可变元组且至少包含一个有限值。
        if not isinstance(self.reward_samples, tuple) or not self.reward_samples or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in self.reward_samples):
            # 抛出明确的经验收益样本完整性错误。
            raise ValueError("搜索转移收益样本必须为非空有限数值元组")
        # 要求终止标记保持布尔语义。
        if not isinstance(self.done, bool):
            # 拒绝数值或字符串终止标记进入训练状态。
            raise ValueError("搜索转移终止标记必须为布尔值")

    @property
    # 说明以样本均值提供日志兼容指标而不替代分布训练目标。
    def mean_reward(self) -> float:
        """返回当前动作 Monte Carlo 收益样本的算术均值。"""
        # 拒绝无法形成经验分布的空样本记录。
        if not self.reward_samples:
            # 抛出明确的回放数据完整性异常。
            raise ValueError("Search transition requires reward samples")
        # 计算仅用于汇总报告的样本均值。
        return sum(self.reward_samples) / len(self.reward_samples)


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class ReplayBuffer:
    """Bounded deterministic replay storage. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __init__(self, capacity: int, seed: int = 0) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if capacity <= 0:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Replay capacity must be positive")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._records: deque[SearchTransition] = deque(maxlen=capacity)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._rng = random.Random(seed)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def append(self, transition: SearchTransition) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._records.append(transition)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def sample(self, batch_size: int) -> tuple[SearchTransition, ...]:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if batch_size <= 0 or batch_size > len(self._records):
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Replay batch size is outside the available range")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return tuple(self._rng.sample(tuple(self._records), batch_size))

    # 导出 Replay 记录和私有采样器状态供完整 checkpoint 使用。
    def state_dict(self) -> dict[str, object]:
        """返回容量、记录和下一次采样所需的 RNG 状态。"""
        # 读取构造时固定的 deque 最大容量。
        capacity = self._records.maxlen
        # 确认有界 Replay 不会意外失去容量约束。
        if capacity is None:
            # 拒绝无法可靠恢复的无界内部状态。
            raise RuntimeError("Replay 内部容量缺失")
        # 返回不暴露可变 deque 的不可变记录快照。
        return {"capacity": capacity, "records": tuple(self._records), "rng_state": self._rng.getstate()}

    # 从完整 checkpoint 恢复 Replay 记录和下一次采样位置。
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """验证全部载荷后原子地替换 Replay 内容和 RNG 状态。"""
        # 定义 Replay checkpoint 必须且只能包含的字段。
        expected_keys = {"capacity", "records", "rng_state"}
        # 要求载荷属于映射并且字段集合完全一致。
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            # 拒绝缺失或未知字段以防语义漂移。
            raise ValueError("Replay checkpoint 字段不完整或包含未知字段")
        # 读取当前 Replay 构造容量。
        current_capacity = self._records.maxlen
        # 要求保存容量为普通整数且与当前对象一致。
        if type(state["capacity"]) is not int or state["capacity"] != current_capacity:
            # 拒绝可能改变淘汰历史的容量漂移。
            raise ValueError("Replay 容量与 checkpoint 不一致")
        # 读取并检查保存记录容器。
        raw_records = state["records"]
        # 要求记录使用不可变元组保存。
        if not isinstance(raw_records, tuple):
            # 拒绝无法证明顺序稳定的记录载荷。
            raise ValueError("Replay checkpoint 记录必须为元组")
        # 将已收窄记录保存为局部变量以便完整验证。
        records = tuple(raw_records)
        # 要求记录数不超过当前固定容量。
        if current_capacity is None or len(records) > current_capacity:
            # 拒绝会在恢复过程中静默淘汰历史的载荷。
            raise ValueError("Checkpoint 中的 Replay 记录超过当前容量")
        # 要求每条保存记录都属于正式搜索转移类型。
        if any(not isinstance(item, SearchTransition) for item in records):
            # 拒绝其他对象混入训练回放。
            raise ValueError("Replay checkpoint 包含无效转移记录")
        # 使用临时随机数生成器预验证 Python RNG 状态。
        candidate_rng = random.Random()
        # 捕获损坏或版本不兼容的 RNG 状态。
        try:
            # 在不影响当前 Replay 的前提下加载候选状态。
            candidate_rng.setstate(state["rng_state"])
        # 将底层类型错误统一转换为领域错误。
        except (TypeError, ValueError) as error:
            # 保留原异常作为调试原因并拒绝半恢复。
            raise ValueError("Replay checkpoint RNG 状态无效") from error
        # 全部验证通过后清空当前 Replay 记录。
        self._records.clear()
        # 按保存顺序恢复全部记录。
        self._records.extend(records)
        # 最后恢复私有采样器状态以保持下一批样本一致。
        self._rng.setstate(candidate_rng.getstate())

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __len__(self) -> int:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return len(self._records)
