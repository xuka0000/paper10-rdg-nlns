"""定义道路弧关系及其三维坡度、通行性和行驶时间计算。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以计算坡度修正和无穷时间。
from dataclasses import dataclass  # 导入数据类工具以保存不可变道路对象。

from Mask.base import SceneObject, SceneRelation, SceneSystem, ValidationIssue  # 导入场景关系和验证基础类型。
from Mask.geometry import Polyline3D, Vector3  # 导入三维折线和向量以表达道路几何。
from Mask.infrastructure import RoadNode  # 导入道路节点以验证道路端点类型。
from Mask.mobility import GroundVehicle  # 导入地面车辆以读取道路通行能力。
from Mask.terrain import TerrainSurface  # 导入地形物理场以生成道路高程剖面。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存道路评估结果。
class RoadAssessment:  # 定义一条道路弧针对一辆车辆的物理评估结果。
    """记录三维长度、最大坡度、通行标识和地形修正行驶时间。"""  # 解释道路评估内容。

    length_3d: float  # 保存道路高程剖面的三维曲线长度。
    maximum_absolute_grade_ratio: float  # 保存道路剖面中的最大绝对坡比。
    accessible: bool  # 保存车辆是否能够通行该道路弧。
    travel_time: float  # 保存地形和路面修正后的车辆行驶时间。
    terrain_profile: tuple[Vector3, ...]  # 保存道路采样点对应的三维地形剖面。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义道路弧关系。
class RoadLeg(SceneRelation):  # 定义两个道路节点之间的有向道路关系。
    """保存道路水平折线和路面速度修正系数。"""  # 解释道路弧职责。

    horizontal_polyline: Polyline3D  # 保存道路中心线的水平投影折线。
    surface_factor: float  # 保存路面条件对应的有效速度比例。

    def __post_init__(self) -> None:  # 定义道路弧初始化后的参数验证方法。
        """验证关系端点字段和路面速度修正系数。"""  # 解释道路弧输入约束。
        SceneRelation.__post_init__(self)  # 显式复用场景关系的端点验证。
        if not math.isfinite(self.surface_factor) or not 0.0 < self.surface_factor <= 1.0:  # 判断路面速度修正系数是否位于有效范围。
            raise ValueError("道路路面修正系数必须位于零到一之间")  # 抛出明确的路面系数异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景关系验证接口。
        """检查道路弧起点和终点是否存在且均属于道路节点。"""  # 解释道路弧场景验证规则。
        issues: list[ValidationIssue] = []  # 创建道路端点验证问题列表。
        for endpoint_role, endpoint_id in (("source", self.source_id), ("target", self.target_id)):  # 逐一检查道路弧两个端点。
            if not system.contains(endpoint_id):  # 判断当前道路端点是否缺失。
                issues.append(ValidationIssue(f"missing_{endpoint_role}_road_node", self.object_id, endpoint_id))  # 记录道路端点缺失问题。
                continue  # 跳过缺失端点的类型检查。
            referenced = system.get_element(endpoint_id)  # 读取当前道路端点对象。
            if not isinstance(referenced, RoadNode):  # 判断当前端点对象是否属于道路节点。
                issues.append(ValidationIssue(f"invalid_{endpoint_role}_road_node_type", self.object_id, endpoint_id))  # 记录道路端点类型错误问题。
        return tuple(issues)  # 返回不可变的道路端点验证问题集合。

    def assess(self, terrain: TerrainSurface, vehicle: GroundVehicle, max_spacing: float) -> RoadAssessment:  # 定义道路弧地形感知物理评估方法。
        """沿道路水平折线采样地形并计算坡度、通行性和时间。"""  # 解释道路弧评估方法。
        horizontal_samples = self.horizontal_polyline.sample(max_spacing)  # 按最大间距采样道路水平折线。
        terrain_profile = tuple(Vector3(point.x, point.y, terrain.height_at(point.x, point.y)) for point in horizontal_samples)  # 将水平采样点映射到地形高程剖面。
        length_3d = 0.0  # 初始化道路剖面三维长度。
        maximum_grade_ratio = 0.0  # 初始化道路剖面最大绝对坡比。
        for start, end in zip(terrain_profile[:-1], terrain_profile[1:]):  # 逐一评估相邻道路剖面采样点。
            horizontal_distance = math.hypot(end.x - start.x, end.y - start.y)  # 计算当前剖面段水平投影距离。
            vertical_difference = end.z - start.z  # 计算当前剖面段高程差。
            segment_length = start.distance_to(end)  # 计算当前剖面段三维长度。
            length_3d += segment_length  # 将当前剖面段长度加入道路总长度。
            if horizontal_distance <= 0.0 and abs(vertical_difference) > 0.0:  # 判断当前剖面段是否为非物理竖直道路。
                segment_grade_ratio = math.inf  # 使用无穷坡比标记竖直道路段。
            elif horizontal_distance <= 0.0:  # 判断当前剖面段是否没有水平和竖向变化。
                segment_grade_ratio = 0.0  # 将零变化道路段坡比设为零。
            else:  # 处理具有正水平距离的常规道路段。
                segment_grade_ratio = abs(vertical_difference) / horizontal_distance  # 计算当前道路段绝对坡比。
            maximum_grade_ratio = max(maximum_grade_ratio, segment_grade_ratio)  # 更新道路剖面最大绝对坡比。
        accessible = maximum_grade_ratio <= vehicle.maximum_road_grade_ratio  # 判断车辆坡比能力是否覆盖道路最大坡比。
        if accessible:  # 判断道路弧是否可以用于车辆行驶。
            slope_factor = math.exp(-vehicle.road_slope_sensitivity * maximum_grade_ratio)  # 计算最大坡比对应的速度衰减系数。
            effective_speed = vehicle.road_speed * self.surface_factor * slope_factor  # 计算地形和路面共同修正后的有效速度。
            travel_time = length_3d / effective_speed  # 计算车辆通过道路弧的行驶时间。
        else:  # 处理车辆无法通行道路弧的情况。
            travel_time = math.inf  # 使用正无穷标记不可通行道路时间。
        return RoadAssessment(length_3d, maximum_grade_ratio, accessible, travel_time, terrain_profile)  # 返回不可变道路弧评估结果。
