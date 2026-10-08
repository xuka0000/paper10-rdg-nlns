"""定义候选观测位姿和构件级巡检任务元素。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以验证巡检参数有限性。
from dataclasses import dataclass  # 导入数据类工具以保存不可变巡检对象。

from Mask.base import SceneElement, SceneObject, SceneSystem, ValidationIssue  # 导入场景元素和验证基础类型。
from Mask.geometry import Vector3  # 导入三维向量以表达位姿位置和相机方向。
from Mask.infrastructure import InspectionCheckpoint  # 导入检查点类型以验证巡检引用。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义候选观测位姿。
class ObservationPose(SceneElement):  # 定义检查点对应的相机位置和光轴元素。
    """保存所属检查点、无人机位置和相机光轴方向。"""  # 解释候选观测位姿职责。

    checkpoint_id: str  # 保存候选位姿服务的检查点标识。
    position: Vector3  # 保存候选位姿的全局三维位置。
    camera_axis: Vector3  # 保存从相机出发的光轴方向向量。

    def __post_init__(self) -> None:  # 定义观测位姿初始化后的参数验证方法。
        """验证检查点引用和非零相机光轴。"""  # 解释候选观测位姿输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if not self.checkpoint_id.strip():  # 判断候选位姿检查点标识是否为空白。
            raise ValueError("观测位姿检查点标识不能为空")  # 抛出明确的检查点引用异常。
        if self.camera_axis.norm() <= 0.0:  # 判断相机光轴是否为零向量。
            raise ValueError("观测位姿相机光轴不能为零向量")  # 抛出明确的相机光轴异常。
        object.__setattr__(self, "camera_axis", self.camera_axis.normalized())  # 将候选相机光轴统一保存为式 12 和式 20 使用的单位向量。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """检查候选位姿引用的检查点是否存在且类型正确。"""  # 解释候选观测位姿场景验证规则。
        if not system.contains(self.checkpoint_id):  # 判断候选位姿引用的检查点是否缺失。
            return (ValidationIssue("missing_checkpoint", self.object_id, self.checkpoint_id),)  # 返回检查点引用缺失问题。
        referenced = system.get_element(self.checkpoint_id)  # 读取候选位姿引用的场景元素。
        if not isinstance(referenced, InspectionCheckpoint):  # 判断引用对象是否属于检查点。
            return (ValidationIssue("invalid_checkpoint_type", self.object_id, self.checkpoint_id),)  # 返回检查点类型错误问题。
        return ()  # 返回无验证问题结果。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义构件级巡检任务。
class InspectionTask(SceneElement):  # 定义具有优先级和服务时间的检查任务元素。
    """保存所属检查点、任务优先级和观测服务持续时间。"""  # 解释巡检任务职责。

    checkpoint_id: str  # 保存任务对应的检查点标识。
    priority: float  # 保存任务风险或价值优先级权重。
    service_duration: float  # 保存稳定观测所需服务持续时间。

    def __post_init__(self) -> None:  # 定义巡检任务初始化后的参数验证方法。
        """验证检查点引用、优先级和服务时间。"""  # 解释巡检任务输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if not self.checkpoint_id.strip():  # 判断巡检任务检查点标识是否为空白。
            raise ValueError("巡检任务检查点标识不能为空")  # 抛出明确的检查点引用异常。
        if not math.isfinite(self.priority) or self.priority <= 0.0:  # 判断任务优先级是否为正有限数值。
            raise ValueError("巡检任务优先级必须为正有限数值")  # 抛出明确的任务优先级异常。
        if not math.isfinite(self.service_duration) or self.service_duration < 0.0:  # 判断任务服务时间是否为非负有限数值。
            raise ValueError("巡检任务服务时间必须为非负有限数值")  # 抛出明确的任务服务时间异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """检查巡检任务引用的检查点是否存在且类型正确。"""  # 解释巡检任务场景验证规则。
        if not system.contains(self.checkpoint_id):  # 判断巡检任务引用的检查点是否缺失。
            return (ValidationIssue("missing_checkpoint", self.object_id, self.checkpoint_id),)  # 返回检查点引用缺失问题。
        referenced = system.get_element(self.checkpoint_id)  # 读取巡检任务引用的场景元素。
        if not isinstance(referenced, InspectionCheckpoint):  # 判断引用对象是否属于检查点。
            return (ValidationIssue("invalid_checkpoint_type", self.object_id, self.checkpoint_id),)  # 返回检查点类型错误问题。
        return ()  # 返回无验证问题结果。
