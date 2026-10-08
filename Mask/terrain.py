"""定义规则网格地形、显式圆柱障碍和三维障碍场。"""  # 说明本文件的建模职责。

import bisect  # 导入二分定位工具以查找规则网格单元。
import math  # 导入数学函数以计算距离和坡度。
from dataclasses import dataclass  # 导入数据类工具以保存不可变地形对象。

from Mask.base import SceneElement, SceneObject, SceneSystem, ValidationIssue  # 导入场景元素和验证基础类型。
from Mask.geometry import Vector3  # 导入三维向量以表达空间查询点。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义规则网格地形。
class TerrainSurface(SceneElement):  # 定义作为基础物理场的三维地形元素。
    """保存严格递增的规则网格坐标和对应地形高程。"""  # 解释地形数据结构。

    x_coordinates: tuple[float, ...]  # 保存横向规则网格坐标。
    y_coordinates: tuple[float, ...]  # 保存纵向规则网格坐标。
    heights: tuple[tuple[float, ...], ...]  # 保存按横向索引和纵向索引排列的高程矩阵。

    def __post_init__(self) -> None:  # 定义地形初始化后的结构验证方法。
        """验证网格坐标、高程矩阵形状和全部数值的有限性。"""  # 解释地形输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if len(self.x_coordinates) < 2 or len(self.y_coordinates) < 2:  # 判断两个坐标轴是否都至少包含两个点。
            raise ValueError("地形规则网格的每个坐标轴至少需要两个点")  # 抛出明确的网格点数异常。
        if any(right <= left for left, right in zip(self.x_coordinates[:-1], self.x_coordinates[1:])):  # 判断横向坐标是否严格递增。
            raise ValueError("地形横向坐标必须严格递增")  # 抛出明确的横向坐标异常。
        if any(right <= left for left, right in zip(self.y_coordinates[:-1], self.y_coordinates[1:])):  # 判断纵向坐标是否严格递增。
            raise ValueError("地形纵向坐标必须严格递增")  # 抛出明确的纵向坐标异常。
        if len(self.heights) != len(self.x_coordinates):  # 判断高程矩阵横向维度是否匹配。
            raise ValueError("地形高程矩阵横向维度与坐标数量不一致")  # 抛出明确的矩阵横向维度异常。
        if any(len(row) != len(self.y_coordinates) for row in self.heights):  # 判断每个高程矩阵行的纵向维度是否匹配。
            raise ValueError("地形高程矩阵纵向维度与坐标数量不一致")  # 抛出明确的矩阵纵向维度异常。
        numeric_values = (*self.x_coordinates, *self.y_coordinates, *(value for row in self.heights for value in row))  # 汇总全部地形数值。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部地形数值是否有限。
            raise ValueError("地形坐标和高程必须为有限数值")  # 抛出明确的地形有限性异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """返回空结果以表示地形结构已经在构造阶段验证。"""  # 解释地形验证结果。
        return ()  # 返回无验证问题结果。

    def contains_xy(self, x_value: float, y_value: float) -> bool:  # 定义地形水平范围查询方法。
        """判断给定水平坐标是否位于闭合地形网格范围内。"""  # 解释范围查询语义。
        return self.x_coordinates[0] <= x_value <= self.x_coordinates[-1] and self.y_coordinates[0] <= y_value <= self.y_coordinates[-1]  # 返回水平坐标范围判断结果。

    def height_at(self, x_value: float, y_value: float) -> float:  # 定义任意水平位置的地形高程查询方法。
        """通过所在规则网格单元的双线性插值返回地形高程。"""  # 解释高程计算方法。
        if not self.contains_xy(x_value, y_value):  # 判断查询点是否位于地形范围内。
            raise ValueError("查询坐标超出地形范围")  # 拒绝未声明的地形外推。
        x_index = min(bisect.bisect_right(self.x_coordinates, x_value) - 1, len(self.x_coordinates) - 2)  # 定位查询点所在横向网格单元。
        y_index = min(bisect.bisect_right(self.y_coordinates, y_value) - 1, len(self.y_coordinates) - 2)  # 定位查询点所在纵向网格单元。
        x_left = self.x_coordinates[x_index]  # 取得网格单元左侧横坐标。
        x_right = self.x_coordinates[x_index + 1]  # 取得网格单元右侧横坐标。
        y_lower = self.y_coordinates[y_index]  # 取得网格单元下侧纵坐标。
        y_upper = self.y_coordinates[y_index + 1]  # 取得网格单元上侧纵坐标。
        x_fraction = (x_value - x_left) / (x_right - x_left)  # 计算查询点在横向单元内的比例。
        y_fraction = (y_value - y_lower) / (y_upper - y_lower)  # 计算查询点在纵向单元内的比例。
        height_left_lower = self.heights[x_index][y_index]  # 取得单元左下角高程。
        height_left_upper = self.heights[x_index][y_index + 1]  # 取得单元左上角高程。
        height_right_lower = self.heights[x_index + 1][y_index]  # 取得单元右下角高程。
        height_right_upper = self.heights[x_index + 1][y_index + 1]  # 取得单元右上角高程。
        height_lower = height_left_lower * (1.0 - x_fraction) + height_right_lower * x_fraction  # 计算单元下边界的线性插值高程。
        height_upper = height_left_upper * (1.0 - x_fraction) + height_right_upper * x_fraction  # 计算单元上边界的线性插值高程。
        return height_lower * (1.0 - y_fraction) + height_upper * y_fraction  # 沿纵向插值并返回双线性高程。

    def grade_at(self, x_value: float, y_value: float, spacing: float) -> float:  # 定义局部地形坡度计算方法。
        """通过范围内双向有限差分计算局部坡度范数。"""  # 解释坡度近似方法。
        if not math.isfinite(spacing) or spacing <= 0.0:  # 判断差分间距是否为正有限数值。
            raise ValueError("地形坡度差分间距必须为正有限数值")  # 抛出明确的差分间距异常。
        x_lower = max(self.x_coordinates[0], x_value - spacing)  # 计算范围内横向左侧差分坐标。
        x_upper = min(self.x_coordinates[-1], x_value + spacing)  # 计算范围内横向右侧差分坐标。
        y_lower = max(self.y_coordinates[0], y_value - spacing)  # 计算范围内纵向下侧差分坐标。
        y_upper = min(self.y_coordinates[-1], y_value + spacing)  # 计算范围内纵向上侧差分坐标。
        if x_upper <= x_lower or y_upper <= y_lower:  # 判断查询点附近是否具有有效差分范围。
            raise ValueError("地形范围不足以计算局部坡度")  # 抛出明确的局部差分范围异常。
        x_grade = (self.height_at(x_upper, y_value) - self.height_at(x_lower, y_value)) / (x_upper - x_lower)  # 计算横向高程梯度。
        y_grade = (self.height_at(x_value, y_upper) - self.height_at(x_value, y_lower)) / (y_upper - y_lower)  # 计算纵向高程梯度。
        return math.hypot(x_grade, y_grade)  # 返回水平梯度的欧氏范数。

    def maximum_grade_in_circle(self, center_x: float, center_y: float, radius: float, spacing: float) -> float:  # 定义圆形范围最大局部坡度计算方法。
        """在候选停靠圆形范围内采样并返回最大局部坡度。"""  # 解释范围坡度用途。
        if radius <= 0.0 or spacing <= 0.0:  # 判断停靠半径和采样间距是否有效。
            raise ValueError("停靠范围半径和采样间距必须为正数")  # 抛出明确的范围采样参数异常。
        sample_count = max(1, math.ceil(radius / spacing))  # 计算每个方向需要覆盖的采样层数。
        maximum_grade = 0.0  # 初始化最大局部坡度。
        for x_step in range(-sample_count, sample_count + 1):  # 遍历圆形范围横向采样索引。
            for y_step in range(-sample_count, sample_count + 1):  # 遍历圆形范围纵向采样索引。
                x_offset = x_step * spacing  # 计算当前采样点横向偏移。
                y_offset = y_step * spacing  # 计算当前采样点纵向偏移。
                if math.hypot(x_offset, y_offset) > radius + 1.0e-12:  # 判断当前采样点是否位于圆形范围外。
                    continue  # 跳过圆形范围外的网格采样点。
                sample_x = center_x + x_offset  # 计算当前采样点横坐标。
                sample_y = center_y + y_offset  # 计算当前采样点纵坐标。
                if not self.contains_xy(sample_x, sample_y):  # 判断停靠范围是否越过地形边界。
                    return math.inf  # 使用无穷坡度标记边界外停靠范围不可用。
                local_grade = self.grade_at(sample_x, sample_y, min(spacing, radius))  # 计算当前采样点局部坡度。
                maximum_grade = max(maximum_grade, local_grade)  # 更新圆形范围内最大局部坡度。
        return maximum_grade  # 返回停靠范围最大局部坡度。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义有限圆柱障碍。
class CylinderObstacle(SceneElement):  # 定义不能压缩进地形高度函数的显式三维障碍。
    """保存竖直有限圆柱的中心、半径和高程范围。"""  # 解释圆柱障碍几何。

    center_x: float  # 保存圆柱中心横坐标。
    center_y: float  # 保存圆柱中心纵坐标。
    radius: float  # 保存圆柱水平半径。
    minimum_elevation: float  # 保存圆柱底部绝对高程。
    maximum_elevation: float  # 保存圆柱顶部绝对高程。

    def __post_init__(self) -> None:  # 定义圆柱障碍初始化后的参数验证方法。
        """验证圆柱半径、高程顺序和全部数值的有限性。"""  # 解释圆柱输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        numeric_values = (self.center_x, self.center_y, self.radius, self.minimum_elevation, self.maximum_elevation)  # 汇总圆柱障碍数值字段。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部圆柱数值是否有限。
            raise ValueError("圆柱障碍参数必须为有限数值")  # 抛出明确的圆柱有限性异常。
        if self.radius <= 0.0:  # 判断圆柱半径是否为正数。
            raise ValueError("圆柱障碍半径必须为正数")  # 抛出明确的圆柱半径异常。
        if self.maximum_elevation <= self.minimum_elevation:  # 判断圆柱顶部是否高于底部。
            raise ValueError("圆柱障碍顶部高程必须高于底部高程")  # 抛出明确的圆柱高程范围异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """返回空结果以表示圆柱结构已经在构造阶段验证。"""  # 解释圆柱验证结果。
        return ()  # 返回无验证问题结果。

    def signed_clearance(self, point: Vector3) -> float:  # 定义空间点到有限圆柱的有符号净空计算方法。
        """返回圆柱外部正距离和圆柱内部负穿透深度。"""  # 解释有符号净空约定。
        radial_distance = math.hypot(point.x - self.center_x, point.y - self.center_y)  # 计算点到圆柱轴线的水平距离。
        radial_delta = radial_distance - self.radius  # 计算点相对于圆柱侧面的水平有符号距离。
        if point.z < self.minimum_elevation:  # 判断点是否位于圆柱底部以下。
            vertical_delta = self.minimum_elevation - point.z  # 计算点到底部平面的竖直距离。
        elif point.z > self.maximum_elevation:  # 判断点是否位于圆柱顶部以上。
            vertical_delta = point.z - self.maximum_elevation  # 计算点到顶部平面的竖直距离。
        else:  # 处理点位于圆柱高程范围内的情况。
            vertical_delta = 0.0  # 将高程范围内的外部竖直距离设为零。
        if radial_delta > 0.0 or vertical_delta > 0.0:  # 判断点是否位于圆柱实体外部。
            return math.hypot(max(radial_delta, 0.0), vertical_delta)  # 返回水平和竖直外部距离组成的欧氏净空。
        radial_penetration = self.radius - radial_distance  # 计算点到圆柱侧面的内部穿透余量。
        lower_penetration = point.z - self.minimum_elevation  # 计算点到圆柱底面的内部余量。
        upper_penetration = self.maximum_elevation - point.z  # 计算点到圆柱顶面的内部余量。
        return -min(radial_penetration, lower_penetration, upper_penetration)  # 返回到最近边界的负穿透深度。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义显式障碍场。
class ObstacleField(SceneElement):  # 定义一组不能由单值高度函数表示的三维障碍。
    """聚合显式三维障碍并提供最近有符号净空查询。"""  # 解释障碍场职责。

    obstacles: tuple[CylinderObstacle, ...]  # 保存显式圆柱障碍集合。

    def __post_init__(self) -> None:  # 定义障碍场初始化后的结构验证方法。
        """验证场景对象字段和障碍标识唯一性。"""  # 解释障碍场输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        obstacle_ids = [obstacle.object_id for obstacle in self.obstacles]  # 提取全部显式障碍标识。
        if len(obstacle_ids) != len(set(obstacle_ids)):  # 判断障碍场内部是否存在重复标识。
            raise ValueError("显式障碍场不能包含重复障碍标识")  # 抛出明确的障碍重复异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """返回空结果以表示障碍结构已经在构造阶段验证。"""  # 解释障碍场验证结果。
        return ()  # 返回无验证问题结果。

    def signed_clearance(self, point: Vector3) -> float:  # 定义空间点到全部显式障碍的最近净空查询方法。
        """返回空间点相对于最近显式障碍的有符号净空。"""  # 解释障碍场净空语义。
        if not self.obstacles:  # 判断障碍场是否为空。
            return math.inf  # 使用正无穷表示不存在显式障碍限制。
        return min(obstacle.signed_clearance(point) for obstacle in self.obstacles)  # 返回全部显式障碍净空的最小值。
