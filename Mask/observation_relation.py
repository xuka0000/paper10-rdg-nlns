"""定义构件检查点与候选观测位姿之间的综合准入关系。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以计算角度和验证阈值。
from dataclasses import dataclass  # 导入数据类工具以保存不可变准入对象。

from Mask.base import SceneObject, SceneRelation, SceneSystem, ValidationIssue  # 导入场景关系和验证基础类型。
from Mask.geometry import Polyline3D, Vector3  # 导入三维折线和向量以计算视线几何。
from Mask.infrastructure import InspectionCheckpoint  # 导入检查点类型以读取目标位置和表面法向量。
from Mask.inspection import ObservationPose  # 导入候选观测位姿类型以读取相机状态。
from Mask.terrain import ObstacleField, TerrainSurface  # 导入地形和显式障碍场以计算净空。


def angle_between(first: Vector3, second: Vector3) -> float:  # 定义两个非零三维向量之间的夹角计算函数。
    """返回零到圆周率之间的数值稳定欧氏夹角。"""  # 解释夹角函数输出范围。
    first_unit = first.normalized()  # 将第一个向量单位化。
    second_unit = second.normalized()  # 将第二个向量单位化。
    cosine_value = max(-1.0, min(1.0, first_unit.dot(second_unit)))  # 将浮点点积截断到反余弦定义域。
    return math.acos(cosine_value)  # 返回两个单位向量的反余弦夹角。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义观测准入阈值。
class ObservationLimits:  # 定义距离、角度、净空和视线采样限制。
    """保存候选观测位姿准入所需的全部物理阈值。"""  # 解释观测阈值职责。

    minimum_distance: float  # 保存最小观测距离。
    maximum_distance: float  # 保存最大观测距离。
    maximum_incidence_angle: float  # 保存最大构件表面入射角。
    maximum_pointing_error: float  # 保存最大相机光轴指向误差。
    minimum_terrain_clearance: float  # 保存候选位姿最小离地高度。
    minimum_obstacle_clearance: float  # 保存候选位姿最小显式障碍净空。
    line_of_sight_spacing: float  # 保存视线认证最大采样间隔。

    def __post_init__(self) -> None:  # 定义观测阈值初始化后的参数验证方法。
        """验证距离顺序、角度范围、净空阈值和采样间距。"""  # 解释观测阈值输入约束。
        numeric_values = (self.minimum_distance, self.maximum_distance, self.maximum_incidence_angle, self.maximum_pointing_error, self.minimum_terrain_clearance, self.minimum_obstacle_clearance, self.line_of_sight_spacing)  # 汇总全部观测阈值。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部观测阈值是否有限。
            raise ValueError("观测位姿准入阈值必须为有限数值")  # 抛出明确的观测阈值有限性异常。
        if self.minimum_distance <= 0.0 or self.maximum_distance < self.minimum_distance:  # 判断观测距离上下限是否有效。
            raise ValueError("观测距离上下限设置无效")  # 抛出明确的观测距离异常。
        if not 0.0 <= self.maximum_incidence_angle <= math.pi:  # 判断最大入射角是否位于有效范围。
            raise ValueError("最大观测入射角必须位于零到圆周率之间")  # 抛出明确的入射角异常。
        if not 0.0 <= self.maximum_pointing_error <= math.pi:  # 判断最大指向误差是否位于有效范围。
            raise ValueError("最大相机指向误差必须位于零到圆周率之间")  # 抛出明确的指向误差异常。
        if self.minimum_terrain_clearance < 0.0 or self.minimum_obstacle_clearance < 0.0:  # 判断两个最小净空阈值是否为非负数。
            raise ValueError("观测位姿最小净空不能为负数")  # 抛出明确的观测净空异常。
        if self.line_of_sight_spacing <= 0.0:  # 判断视线采样间距是否为正数。
            raise ValueError("视线采样间距必须为正数")  # 抛出明确的视线采样异常。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存观测准入结果。
class ObservationAssessment:  # 定义候选观测位姿的完整可解释评估结果。
    """记录几何量、净空量、视线状态和每项准入条件。"""  # 解释观测评估内容。

    checkpoint_id: str  # 保存评估来源检查点标识以支持式 10 至式 15 的追溯。
    pose_id: str  # 保存被评估候选观测位姿标识。
    view_direction: Vector3 | None  # 保存式 10 的检查点到相机单位方向，退化距离时为空。
    distance: float  # 保存候选位姿到检查点的观测距离。
    incidence_angle: float  # 保存构件表面法向与相机位置方向的夹角。
    pointing_error: float  # 保存相机光轴与目标方向的夹角。
    agl_height: float  # 保存候选位姿相对地形高度。
    obstacle_clearance: float  # 保存候选位姿到最近显式障碍的净空。
    line_of_sight_clear: bool  # 保存候选位姿到检查点的视线是否无阻挡。
    conditions: tuple[tuple[str, bool], ...]  # 保存各项准入条件名称和判断结果。

    @property  # 将综合准入状态暴露为只读属性。
    def admissible(self) -> bool:  # 定义综合准入状态读取方法。
        """仅在全部观测条件均通过时返回真。"""  # 解释综合准入逻辑。
        return all(condition_passed for _, condition_passed in self.conditions)  # 返回全部条件的逻辑与结果。

    @property  # 将失败条件暴露为只读属性。
    def failed_conditions(self) -> tuple[str, ...]:  # 定义失败条件名称读取方法。
        """按照固定评估顺序返回全部未通过条件名称。"""  # 解释失败条件输出语义。
        return tuple(condition_name for condition_name, condition_passed in self.conditions if not condition_passed)  # 返回全部失败条件名称。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义观测准入关系。
class ObservationAdmissibility(SceneRelation):  # 定义从检查点到候选观测位姿的物理关系。
    """使用统一阈值计算候选位姿几何、净空和视线准入结果。"""  # 解释观测准入关系职责。

    limits: ObservationLimits  # 保存该关系使用的观测准入阈值。

    def __post_init__(self) -> None:  # 定义观测准入关系初始化后的验证方法。
        """复用场景关系端点验证。"""  # 解释观测准入关系输入约束。
        SceneRelation.__post_init__(self)  # 显式复用场景关系的标识和端点验证。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景关系验证接口。
        """检查关系源属于检查点且关系目标属于候选观测位姿。"""  # 解释观测准入关系场景验证规则。
        issues: list[ValidationIssue] = []  # 创建关系端点验证问题列表。
        if not system.contains(self.source_id):  # 判断关系源检查点是否缺失。
            issues.append(ValidationIssue("missing_checkpoint", self.object_id, self.source_id))  # 记录关系源检查点缺失问题。
        elif not isinstance(system.get_element(self.source_id), InspectionCheckpoint):  # 判断关系源对象是否属于检查点。
            issues.append(ValidationIssue("invalid_checkpoint_type", self.object_id, self.source_id))  # 记录关系源检查点类型错误问题。
        if not system.contains(self.target_id):  # 判断关系目标位姿是否缺失。
            issues.append(ValidationIssue("missing_observation_pose", self.object_id, self.target_id))  # 记录关系目标位姿缺失问题。
        elif not isinstance(system.get_element(self.target_id), ObservationPose):  # 判断关系目标对象是否属于候选观测位姿。
            issues.append(ValidationIssue("invalid_observation_pose_type", self.object_id, self.target_id))  # 记录关系目标位姿类型错误问题。
        return tuple(issues)  # 返回不可变的关系端点验证问题集合。

    def assess(self, system: SceneSystem, terrain: TerrainSurface, obstacle_field: ObstacleField) -> ObservationAssessment:  # 定义候选观测位姿综合准入评估方法。
        """计算距离、入射角、指向误差、净空和视线条件。"""  # 解释观测准入评估方法。
        checkpoint = system.get_element(self.source_id)  # 读取关系源检查点对象。
        pose = system.get_element(self.target_id)  # 读取关系目标候选位姿对象。
        if not isinstance(checkpoint, InspectionCheckpoint):  # 判断关系源对象是否属于检查点。
            raise TypeError("观测准入关系源对象不是检查点")  # 抛出明确的关系源类型异常。
        if not isinstance(pose, ObservationPose):  # 判断关系目标对象是否属于候选观测位姿。
            raise TypeError("观测准入关系目标对象不是候选位姿")  # 抛出明确的关系目标类型异常。
        checkpoint_position = checkpoint.global_position(system)  # 计算检查点全局三维位置。
        surface_normal = checkpoint.global_surface_normal(system)  # 计算检查点全局表面法向量。
        checkpoint_to_camera = pose.position - checkpoint_position  # 计算检查点指向相机的三维向量。
        camera_to_checkpoint = checkpoint_position - pose.position  # 计算相机指向检查点的三维向量。
        distance = checkpoint_to_camera.norm()  # 计算候选位姿到检查点的观测距离。
        if distance <= 0.0:  # 判断候选位姿是否与检查点重合。
            view_direction = None  # 将不可定义的式 10 单位方向记录为空。
            incidence_angle = math.pi  # 使用最大角度标记无法定义的表面入射关系。
            pointing_error = math.pi  # 使用最大角度标记无法定义的相机指向关系。
        else:  # 处理候选位姿与检查点不重合的常规情况。
            view_direction = checkpoint_to_camera.normalized()  # 计算并保存式 10 的检查点到相机单位方向。
            incidence_angle = angle_between(surface_normal, view_direction)  # 计算构件表面观测入射角。
            pointing_error = angle_between(pose.camera_axis, camera_to_checkpoint)  # 计算相机光轴指向误差。
        pose_in_terrain_domain = terrain.contains_xy(pose.position.x, pose.position.y)  # 判断候选位姿水平坐标是否位于地形定义域。
        agl_height = pose.position.z - terrain.height_at(pose.position.x, pose.position.y) if pose_in_terrain_domain else -math.inf  # 在定义域内计算离地高度，域外使用负无穷拒绝标记。
        obstacle_clearance = obstacle_field.signed_clearance(pose.position)  # 计算候选位姿到最近显式障碍的净空。
        sight_samples: tuple[Vector3, ...] = ()  # 初始化退化候选不具有有效视线采样点。
        if distance > 0.0:  # 仅为具有正观测距离的候选构建三维视线。
            sight_line = Polyline3D((pose.position, checkpoint_position))  # 构建相机到检查点的三维视线折线。
            sight_samples = sight_line.sample(self.limits.line_of_sight_spacing)  # 按最大间隔采样三维视线。
        terrain_domain_valid = bool(sight_samples) and all(terrain.contains_xy(point.x, point.y) for point in sight_samples)  # 要求完整视线位于声明的地形水平定义域。
        line_of_sight_clear = terrain_domain_valid  # 只有视线可在地形定义域内完整评估时才初始化为无遮挡。
        if line_of_sight_clear:  # 判断当前视线是否具有完整地形查询定义域。
            for sight_point in sight_samples[1:-1]:  # 逐一检查不含两个端点的视线内部采样点。
                terrain_margin = sight_point.z - terrain.height_at(sight_point.x, sight_point.y)  # 计算视线采样点相对地形高程余量。
                obstacle_margin = obstacle_field.signed_clearance(sight_point)  # 计算视线采样点到显式障碍的有符号净空。
                if terrain_margin < 0.0 or obstacle_margin < 0.0:  # 判断视线采样点是否穿过地形或显式障碍。
                    line_of_sight_clear = False  # 将视线状态标记为受阻。
                    break  # 发现首个遮挡后停止继续采样。
        conditions = (  # 按固定顺序构造全部观测准入条件。
            ("nonzero_observation_distance", distance > 0.0),  # 判断式 10 至式 12 是否具有非零距离定义域。
            ("terrain_domain", pose_in_terrain_domain and terrain_domain_valid),  # 判断候选位置和完整视线是否位于地形定义域。
            ("distance", self.limits.minimum_distance <= distance <= self.limits.maximum_distance),  # 判断观测距离是否位于允许范围。
            ("incidence_angle", incidence_angle <= self.limits.maximum_incidence_angle),  # 判断构件表面入射角是否满足上限。
            ("pointing_error", pointing_error <= self.limits.maximum_pointing_error),  # 判断相机光轴指向误差是否满足上限。
            ("terrain_clearance", agl_height >= self.limits.minimum_terrain_clearance),  # 判断候选位姿离地高度是否满足下限。
            ("obstacle_clearance", obstacle_clearance >= self.limits.minimum_obstacle_clearance),  # 判断候选位姿显式障碍净空是否满足下限。
            ("line_of_sight", line_of_sight_clear),  # 判断相机到检查点的视线是否无阻挡。
        )  # 结束观测准入条件集合。
        return ObservationAssessment(self.source_id, self.target_id, view_direction, distance, incidence_angle, pointing_error, agl_height, obstacle_clearance, line_of_sight_clear, conditions)  # 返回具有来源标识和式 10 方向的完整不可变观测评估结果。
