"""定义服务列内部的观测位姿选择与轨迹节点关联对象。"""  # 说明本文件对应式 16 至式 20 的建模职责。

import math  # 导入数学函数以验证服务持续时间有限性。
from dataclasses import dataclass, replace  # 导入数据类工具以保存不可变选择对象并重建轨迹节点索引。
from itertools import permutations, product  # 导入排列和笛卡尔积以枚举任务顺序及候选位姿组合。
from typing import Mapping  # 导入只读映射协议以接收按任务组织的准入分配。

from Mask.geometry import Vector3  # 导入三维向量以保存轨迹节点位置和相机光轴。
from Mask.inspection import InspectionTask, ObservationPose  # 导入巡检任务和候选位姿以构造选中节点。
from Mask.observation_relation import ObservationAssessment  # 导入候选位姿准入证据以执行式 16。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义一个被选中的观测节点。
class ObservationNodeAssignment:  # 定义候选服务列内隐式取值为一的位姿选择。
    """绑定轨迹节点、任务、检查点、准入位姿及其位置和单位相机光轴。"""  # 解释节点分配职责。

    node_index: int  # 保存服务列内部观测轨迹节点索引。
    task_id: str  # 保存该节点执行的巡检任务标识。
    checkpoint_id: str  # 保存该节点覆盖的检查点标识。
    pose_id: str  # 保存该节点选中的候选观测位姿标识。
    node_position: Vector3  # 保存式 19 联结后的轨迹节点位置。
    camera_axis: Vector3  # 保存式 20 联结后的实际单位相机光轴。
    service_duration: float  # 保存该节点稳定观测服务持续时间。
    admissibility_conditions: tuple[tuple[str, bool], ...]  # 保存式 15 及其计算域前置条件证据。
    selection_value: int = 1  # 保存列内嵌的式 16 选择值，候选列只存储取值为一的项。

    def __post_init__(self) -> None:  # 定义选中观测节点构造后的闭合验证。
        """验证索引、引用、单位光轴、服务时间、准入证据和隐式选择值。"""  # 解释节点分配输入约束。
        if not isinstance(self.node_index, int) or self.node_index < 1:  # 判断观测节点索引是否为正整数。
            raise ValueError("观测轨迹节点索引必须为正整数")  # 拒绝与起飞端点或无效节点混淆的索引。
        if any(not identifier.strip() for identifier in (self.task_id, self.checkpoint_id, self.pose_id)):  # 判断任务、检查点和位姿标识是否为空白。
            raise ValueError("观测节点任务、检查点和位姿标识不能为空")  # 拒绝不可追溯的节点分配。
        if not math.isfinite(self.service_duration) or self.service_duration < 0.0:  # 判断稳定观测服务时间是否为非负有限值。
            raise ValueError("观测节点服务时间必须为非负有限数值")  # 拒绝无法汇总服务列时长的节点分配。
        if not self.admissibility_conditions or not all(passed for _, passed in self.admissibility_conditions):  # 判断候选位姿是否通过全部准入与计算域条件。
            raise ValueError("未准入候选观测位姿不能形成轨迹节点分配")  # 强制执行式 16 的准入上界。
        if self.selection_value != 1:  # 判断列内存储项是否表示实际选中的位姿。
            raise ValueError("观测节点分配的内嵌选择值必须等于一")  # 拒绝把自由二元变量混入候选服务列参数。
        object.__setattr__(self, "camera_axis", self.camera_axis.normalized())  # 统一保存式 20 使用的单位相机光轴。

    @classmethod  # 声明从任务、位姿和准入评估创建节点分配的命名构造器。
    def from_assessment(cls, node_index: int, task: InspectionTask, pose: ObservationPose, assessment: ObservationAssessment) -> "ObservationNodeAssignment":  # 定义经过式 15 认证的节点分配构造路径。
        """仅在任务、位姿和准入证据引用一致且位姿准入时创建选中节点。"""  # 解释命名构造器边界。
        if task.checkpoint_id != pose.checkpoint_id:  # 判断任务和位姿是否服务同一检查点。
            raise ValueError("观测节点任务与候选位姿的检查点不一致")  # 拒绝跨检查点错误关联。
        if assessment.checkpoint_id != task.checkpoint_id or assessment.pose_id != pose.object_id:  # 判断评估来源是否与当前任务位姿一致。
            raise ValueError("观测准入评估与任务位姿引用不一致")  # 拒绝复用其他候选的准入证据。
        if not assessment.admissible:  # 判断当前候选位姿是否通过全部式 15 条件。
            raise ValueError("未准入候选观测位姿不能形成轨迹节点分配")  # 在候选列构造边界强制式 16。
        return cls(node_index, task.object_id, task.checkpoint_id, pose.object_id, pose.position, pose.camera_axis, task.service_duration, assessment.conditions)  # 将式 19 和式 20 的状态直接绑定到选中位姿。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义一条服务列的观测选择模式。
class ObservationSelectionPattern:  # 定义式 16 至式 20 在候选服务列中的完整实现。
    """保存有序选中节点并派生任务、位姿、覆盖、节点位置和相机光轴参数。"""  # 解释选择模式职责。

    assignments: tuple[ObservationNodeAssignment, ...]  # 保存按执行顺序排列的观测节点分配。
    pattern_id: str = ""  # 保存外层枚举生成的稳定离散模式标识，显式单模式可留空。

    def __post_init__(self) -> None:  # 定义选择模式构造后的唯一性验证。
        """验证非空、严格节点顺序、单节点单任务和单检查点单姿态。"""  # 解释选择模式输入约束。
        if not self.assignments:  # 判断候选服务列是否包含观测任务。
            raise ValueError("观测选择模式至少需要一个节点分配")  # 拒绝没有巡检任务的空服务列模式。
        if self.pattern_id and not self.pattern_id.strip():  # 判断非空模式标识是否仅由空白字符组成。
            raise ValueError("观测选择模式标识不能仅包含空白字符")  # 拒绝不可追踪的外层模式标识。
        node_indices = tuple(assignment.node_index for assignment in self.assignments)  # 提取全部观测轨迹节点索引。
        if len(node_indices) != len(set(node_indices)):  # 判断一个轨迹节点是否承担多个任务。
            raise ValueError("观测选择模式不能重复使用轨迹节点")  # 强制执行式 18。
        if tuple(sorted(node_indices)) != node_indices:  # 判断节点分配是否按轨迹顺序保存。
            raise ValueError("观测选择模式必须按轨迹节点递增排序")  # 保证服务时间和飞行原语汇总顺序确定。
        checkpoint_ids = self.covered_checkpoint_ids  # 提取全部被选中位姿覆盖的检查点。
        if len(checkpoint_ids) != len(set(checkpoint_ids)):  # 判断同一检查点是否选择多个位姿。
            raise ValueError("观测选择模式不能为同一检查点选择多个位姿")  # 保证式 17 的覆盖参数为二元值。
        task_ids = self.task_ids  # 提取全部任务标识。
        pose_ids = self.pose_ids  # 提取全部位姿标识。
        if len(task_ids) != len(set(task_ids)) or len(pose_ids) != len(set(pose_ids)):  # 判断任务或候选位姿是否被重复使用。
            raise ValueError("观测选择模式不能重复使用任务或候选位姿")  # 拒绝重复服务记录。

    @property  # 将有序任务标识暴露为只读派生参数。
    def task_ids(self) -> tuple[str, ...]:  # 定义任务标识序列读取方法。
        """返回按轨迹节点顺序排列的巡检任务标识。"""  # 解释任务顺序语义。
        return tuple(assignment.task_id for assignment in self.assignments)  # 从节点分配派生任务顺序。

    @property  # 将有序位姿标识暴露为只读派生参数。
    def pose_ids(self) -> tuple[str, ...]:  # 定义位姿标识序列读取方法。
        """返回按轨迹节点顺序排列的选中候选位姿标识。"""  # 解释位姿顺序语义。
        return tuple(assignment.pose_id for assignment in self.assignments)  # 从节点分配派生位姿顺序。

    @property  # 将覆盖检查点暴露为只读派生参数。
    def covered_checkpoint_ids(self) -> tuple[str, ...]:  # 定义覆盖检查点序列读取方法。
        """返回按轨迹节点顺序排列的覆盖检查点标识。"""  # 解释覆盖顺序语义。
        return tuple(assignment.checkpoint_id for assignment in self.assignments)  # 从选中位姿派生服务列覆盖。

    @property  # 将式 17 覆盖参数暴露为只读派生结果。
    def checkpoint_coverage(self) -> tuple[tuple[str, int], ...]:  # 定义二元覆盖参数读取方法。
        """返回服务列中每个被选中检查点的二元覆盖值一。"""  # 解释式 17 的列内参数化语义。
        return tuple((checkpoint_id, 1) for checkpoint_id in self.covered_checkpoint_ids)  # 将唯一选中检查点映射为覆盖值一。

    @property  # 将式 19 节点位置暴露为只读派生结果。
    def node_positions(self) -> tuple[tuple[int, Vector3], ...]:  # 定义节点位置映射读取方法。
        """返回轨迹节点索引及其选中候选位姿位置。"""  # 解释式 19 映射语义。
        return tuple((assignment.node_index, assignment.node_position) for assignment in self.assignments)  # 从分配直接派生节点位置。

    @property  # 将式 20 相机光轴暴露为只读派生结果。
    def camera_axes(self) -> tuple[tuple[int, Vector3], ...]:  # 定义节点相机光轴映射读取方法。
        """返回轨迹节点索引及其选中候选位姿单位光轴。"""  # 解释式 20 映射语义。
        return tuple((assignment.node_index, assignment.camera_axis) for assignment in self.assignments)  # 从分配直接派生节点相机光轴。


def enumerate_observation_selection_patterns(template_id: str, task_ids: tuple[str, ...], assignments_by_task: Mapping[str, tuple[ObservationNodeAssignment, ...]], enumerate_task_orders: bool) -> tuple[ObservationSelectionPattern, ...]:  # 定义式 16 至式 20 的外层有限枚举器。
    """返回全部任务顺序和准入位姿组合对应的稳定选择模式。"""  # 解释外层枚举结果。
    if not template_id.strip():  # 判断候选模板标识是否为空白。
        raise ValueError("观测选择模板标识不能为空")  # 拒绝不能派生稳定模式标识的模板。
    if not task_ids or any(not task_id.strip() for task_id in task_ids):  # 判断任务集合是否为空或包含空白标识。
        raise ValueError("观测选择枚举至少需要一个有效任务标识")  # 拒绝不能形成服务列的任务集合。
    if len(task_ids) != len(set(task_ids)):  # 判断候选模板是否重复声明任务。
        raise ValueError("观测选择枚举不能包含重复任务")  # 拒绝重复服务同一任务的离散模板。
    normalized_choices: dict[str, tuple[ObservationNodeAssignment, ...]] = {}  # 创建按任务保存的稳定准入位姿选择映射。
    for task_id in task_ids:  # 逐一验证模板声明的任务位姿集合。
        choices = tuple(sorted(assignments_by_task.get(task_id, ()), key=lambda assignment: assignment.pose_id))  # 按位姿标识稳定排序当前任务选择。
        if not choices:  # 判断当前任务是否没有任何准入位姿。
            raise ValueError(f"任务缺少准入位姿: {task_id}")  # 拒绝不能满足式 16 的离散模板。
        if any(assignment.task_id != task_id for assignment in choices):  # 判断映射键与节点分配任务引用是否一致。
            raise ValueError(f"准入位姿分配与任务标识不一致: {task_id}")  # 拒绝跨任务错误组合。
        normalized_choices[task_id] = choices  # 保存经过验证和排序的当前任务选择。
    task_orders = tuple(permutations(task_ids)) if enumerate_task_orders else (task_ids,)  # 根据模板设置生成全部任务排列或保留输入顺序。
    patterns: list[ObservationSelectionPattern] = []  # 创建有序外层离散模式结果。
    for task_order in task_orders:  # 逐一处理一个任务执行顺序。
        ordered_choice_sets = tuple(normalized_choices[task_id] for task_id in task_order)  # 按当前任务顺序读取位姿选择集合。
        for selected_choices in product(*ordered_choice_sets):  # 枚举当前任务顺序下的位姿笛卡尔积。
            assignments = tuple(replace(assignment, node_index=node_index) for node_index, assignment in enumerate(selected_choices, start=1))  # 按当前执行顺序重新建立连续观测节点索引。
            pattern_id = "::".join((template_id, "+".join(task_order), "+".join(assignment.pose_id for assignment in assignments)))  # 由模板、任务顺序和位姿顺序派生稳定模式标识。
            patterns.append(ObservationSelectionPattern(assignments, pattern_id))  # 创建并保存经过唯一性复核的离散模式。
    return tuple(patterns)  # 返回保持稳定枚举顺序的不可变模式集合。
