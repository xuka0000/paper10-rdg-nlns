"""定义道路节点、停靠点、杆塔、构件和检查点等基础设施元素。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以计算面积和杆塔坐标旋转。
from dataclasses import dataclass  # 导入数据类工具以保存不可变基础设施对象。

from Mask.base import SceneElement, SceneObject, SceneSystem, ValidationIssue  # 导入场景元素和验证基础类型。
from Mask.geometry import RotationMatrix3D, Vector3  # 导入三维向量和旋转矩阵以表达基础设施位置与朝向。
from Mask.terrain import TerrainSurface  # 导入地形对象以评估停靠点可用性。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义道路节点。
class RoadNode(SceneElement):  # 定义地面道路网络中的三维节点元素。
    """保存道路节点三维位置以及可选的节点角色。"""  # 解释道路节点职责。

    position: Vector3  # 保存道路节点的三维位置。
    role: str = "ordinary"  # 保存普通、起点仓库或终点仓库等节点角色。

    def __post_init__(self) -> None:  # 定义道路节点初始化后的参数验证方法。
        """验证场景对象字段和非空节点角色。"""  # 解释道路节点输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if not self.role.strip():  # 判断道路节点角色是否为空白。
            raise ValueError("道路节点角色不能为空")  # 抛出明确的节点角色异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """返回空结果以表示道路节点结构已经在构造阶段验证。"""  # 解释道路节点验证结果。
        return ()  # 返回无验证问题结果。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存停靠点评估结果。
class StopAssessment:  # 定义候选停靠点的地形可用性结果。
    """记录不依赖具体车辆能力的最大局部坡比和几何面积。"""  # 解释停靠点评估内容。

    maximum_local_grade_ratio: float  # 保存停靠范围内最大局部坡比。
    footprint_area: float  # 保存圆形停靠范围的几何面积。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义候选停靠点。
class StopSite(SceneElement):  # 定义车辆停车和无人机起飞回收接口元素。
    """保存道路连接、中心位置和圆形物理停靠范围。"""  # 解释停靠点职责。

    road_node_id: str  # 保存停靠点连接的道路节点标识。
    position: Vector3  # 保存停靠点中心三维位置。
    footprint_radius: float  # 保存圆形停靠范围半径。

    def __post_init__(self) -> None:  # 定义停靠点初始化后的参数验证方法。
        """验证道路引用和停靠范围半径。"""  # 解释停靠点输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if not self.road_node_id.strip():  # 判断道路节点引用是否为空白。
            raise ValueError("停靠点道路节点标识不能为空")  # 抛出明确的道路节点引用异常。
        if self.footprint_radius <= 0.0:  # 判断停靠范围半径是否为正数。
            raise ValueError("停靠点范围半径必须为正数")  # 抛出明确的停靠范围半径异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """检查停靠点引用的道路节点是否存在且类型正确。"""  # 解释停靠点场景验证规则。
        if not system.contains(self.road_node_id):  # 判断停靠点引用的道路节点是否缺失。
            return (ValidationIssue("missing_road_node", self.object_id, self.road_node_id),)  # 返回道路节点缺失问题。
        referenced = system.get_element(self.road_node_id)  # 读取停靠点引用的场景元素。
        if not isinstance(referenced, RoadNode):  # 判断引用对象是否属于道路节点。
            return (ValidationIssue("invalid_road_node_type", self.object_id, self.road_node_id),)  # 返回道路节点类型错误问题。
        return ()  # 返回无验证问题结果。

    def assess_terrain(self, terrain: TerrainSurface, sample_spacing: float) -> StopAssessment:  # 定义停靠点地形和几何评估方法。
        """返回圆形范围内最大坡比和独立的几何面积。"""  # 解释停靠点评估方法。
        maximum_grade_ratio = terrain.maximum_grade_in_circle(self.position.x, self.position.y, self.footprint_radius, sample_spacing)  # 计算停靠范围最大局部坡比。
        footprint_area = math.pi * self.footprint_radius**2  # 计算不依赖车辆阈值和坡比合格状态的圆形面积。
        return StopAssessment(maximum_grade_ratio, footprint_area)  # 返回不可变停靠点物理评估结果。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义输电杆塔。
class TransmissionTower(SceneElement):  # 定义具有地形基座和完整三维朝向的杆塔元素。
    """保存杆塔基座全局位置和局部坐标系三维旋转矩阵。"""  # 解释杆塔坐标职责。

    base_position: Vector3  # 保存杆塔基座全局三维位置。
    rotation: RotationMatrix3D  # 保存杆塔局部坐标系到全局坐标系的右手正交旋转。

    def __post_init__(self) -> None:  # 定义杆塔初始化后的参数验证方法。
        """复用场景对象验证并依赖旋转值对象保证朝向合法。"""  # 解释杆塔输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """返回空结果以表示杆塔结构已经在构造阶段验证。"""  # 解释杆塔验证结果。
        return ()  # 返回无验证问题结果。

    def rotate_local_vector(self, local_vector: Vector3) -> Vector3:  # 定义杆塔局部向量到全局方向的旋转方法。
        """按照杆塔完整三维朝向旋转局部向量。"""  # 解释杆塔旋转方法。
        return self.rotation.apply(local_vector)  # 复用统一旋转值对象返回全局向量。

    def global_from_local(self, local_position: Vector3) -> Vector3:  # 定义杆塔局部位置到全局位置的转换方法。
        """将局部位置旋转后叠加杆塔基座全局位置。"""  # 解释杆塔坐标转换方法。
        return self.base_position + self.rotate_local_vector(local_position)  # 返回局部位置对应的全局三维位置。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义杆塔构件。
class TowerComponent(SceneElement):  # 定义隶属于一基杆塔的具体检查构件。
    """保存所属杆塔、局部原点和构件类型。"""  # 解释杆塔构件职责。

    tower_id: str  # 保存所属杆塔稳定标识。
    local_origin: Vector3  # 保存构件原点相对于杆塔基座的局部位置。
    component_type: str  # 保存绝缘子、横担或连接件等构件类型。

    def __post_init__(self) -> None:  # 定义构件初始化后的参数验证方法。
        """验证场景对象字段、杆塔引用和构件类型。"""  # 解释构件输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if not self.tower_id.strip():  # 判断所属杆塔标识是否为空白。
            raise ValueError("构件所属杆塔标识不能为空")  # 抛出明确的杆塔引用异常。
        if not self.component_type.strip():  # 判断构件类型是否为空白。
            raise ValueError("构件类型不能为空")  # 抛出明确的构件类型异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """检查构件引用的杆塔是否存在且类型正确。"""  # 解释构件场景验证规则。
        if not system.contains(self.tower_id):  # 判断构件引用的杆塔是否缺失。
            return (ValidationIssue("missing_tower", self.object_id, self.tower_id),)  # 返回杆塔引用缺失问题。
        referenced = system.get_element(self.tower_id)  # 读取构件引用的场景元素。
        if not isinstance(referenced, TransmissionTower):  # 判断引用对象是否属于输电杆塔。
            return (ValidationIssue("invalid_tower_type", self.object_id, self.tower_id),)  # 返回杆塔类型错误问题。
        return ()  # 返回无验证问题结果。

    def global_origin(self, system: SceneSystem) -> Vector3:  # 定义构件全局原点计算方法。
        """根据所属杆塔基座和偏航角返回构件全局原点。"""  # 解释构件坐标计算方法。
        tower = system.get_element(self.tower_id)  # 读取构件所属杆塔对象。
        if not isinstance(tower, TransmissionTower):  # 判断引用对象是否属于输电杆塔。
            raise TypeError("构件所属对象不是输电杆塔")  # 抛出明确的杆塔类型异常。
        return tower.global_from_local(self.local_origin)  # 返回构件局部原点对应的全局位置。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义构件检查点。
class InspectionCheckpoint(SceneElement):  # 定义构件上的具体三维视觉检查点。
    """保存所属构件、构件局部位置和局部表面法向量。"""  # 解释检查点职责。

    component_id: str  # 保存所属杆塔构件稳定标识。
    local_position: Vector3  # 保存检查点相对于构件原点的局部位置。
    surface_normal: Vector3  # 保存检查点表面外法向量。

    def __post_init__(self) -> None:  # 定义检查点初始化后的参数验证方法。
        """验证场景对象字段、构件引用和非零表面法向量。"""  # 解释检查点输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if not self.component_id.strip():  # 判断所属构件标识是否为空白。
            raise ValueError("检查点所属构件标识不能为空")  # 抛出明确的构件引用异常。
        if self.surface_normal.norm() <= 0.0:  # 判断检查点表面法向量是否为零向量。
            raise ValueError("检查点表面法向量不能为零向量")  # 抛出明确的表面法向量异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """检查检查点引用的构件是否存在且类型正确。"""  # 解释检查点场景验证规则。
        if not system.contains(self.component_id):  # 判断检查点引用的构件是否缺失。
            return (ValidationIssue("missing_component", self.object_id, self.component_id),)  # 返回构件引用缺失问题。
        referenced = system.get_element(self.component_id)  # 读取检查点引用的场景元素。
        if not isinstance(referenced, TowerComponent):  # 判断引用对象是否属于杆塔构件。
            return (ValidationIssue("invalid_component_type", self.object_id, self.component_id),)  # 返回构件类型错误问题。
        return ()  # 返回无验证问题结果。

    def global_position(self, system: SceneSystem) -> Vector3:  # 定义检查点全局位置计算方法。
        """叠加构件和检查点局部位移并通过杆塔坐标系转换。"""  # 解释检查点坐标计算方法。
        component = system.get_element(self.component_id)  # 读取检查点所属构件对象。
        if not isinstance(component, TowerComponent):  # 判断引用对象是否属于杆塔构件。
            raise TypeError("检查点所属对象不是杆塔构件")  # 抛出明确的构件类型异常。
        tower = system.get_element(component.tower_id)  # 读取构件所属杆塔对象。
        if not isinstance(tower, TransmissionTower):  # 判断构件引用对象是否属于输电杆塔。
            raise TypeError("检查点关联构件的所属对象不是输电杆塔")  # 抛出明确的杆塔类型异常。
        combined_local = component.local_origin + self.local_position  # 叠加构件原点和检查点局部位置。
        return tower.global_from_local(combined_local)  # 返回组合局部位置对应的全局三维位置。

    def global_surface_normal(self, system: SceneSystem) -> Vector3:  # 定义检查点全局表面法向量计算方法。
        """按照所属杆塔偏航角旋转并单位化局部表面法向量。"""  # 解释表面法向量转换方法。
        component = system.get_element(self.component_id)  # 读取检查点所属构件对象。
        if not isinstance(component, TowerComponent):  # 判断引用对象是否属于杆塔构件。
            raise TypeError("检查点所属对象不是杆塔构件")  # 抛出明确的构件类型异常。
        tower = system.get_element(component.tower_id)  # 读取构件所属杆塔对象。
        if not isinstance(tower, TransmissionTower):  # 判断构件引用对象是否属于输电杆塔。
            raise TypeError("检查点关联构件的所属对象不是输电杆塔")  # 抛出明确的杆塔类型异常。
        return tower.rotate_local_vector(self.surface_normal).normalized()  # 返回旋转并单位化后的全局表面法向量。
