"""定义三维向量、三维折线和保守路径采样等基础几何能力。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以计算范数和采样段数。
from dataclasses import dataclass  # 导入数据类工具以保存不可变几何对象。

import numpy as np  # 导入数组库以向数值计算模块提供向量表示。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义三维向量。
class Vector3:  # 定义统一的三维笛卡尔向量类型。
    """保存以米或相应物理单位表示的三个笛卡尔分量。"""  # 解释向量的坐标语义。

    x: float  # 保存横向 x 分量。
    y: float  # 保存横向 y 分量。
    z: float  # 保存竖向 z 分量。

    def __post_init__(self) -> None:  # 定义向量初始化后的有限性验证方法。
        """拒绝包含无穷值或非数值的三维向量。"""  # 解释向量输入约束。
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z)):  # 判断三个分量是否全部有限。
            raise ValueError("三维向量分量必须为有限数值")  # 抛出明确的向量输入异常。

    def as_array(self) -> np.ndarray:  # 定义 NumPy 数组转换方法。
        """返回适合数值计算的浮点数组副本。"""  # 解释转换结果的数据所有权。
        return np.asarray((self.x, self.y, self.z), dtype=float)  # 构造并返回三维浮点数组。

    def __add__(self, other: "Vector3") -> "Vector3":  # 定义三维向量加法。
        """返回当前向量与另一个向量的逐分量和。"""  # 解释向量加法语义。
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)  # 构造向量加法结果。

    def __sub__(self, other: "Vector3") -> "Vector3":  # 定义三维向量减法。
        """返回当前向量减去另一个向量的逐分量差。"""  # 解释向量减法语义。
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)  # 构造向量减法结果。

    def scaled(self, factor: float) -> "Vector3":  # 定义三维向量数乘方法。
        """返回当前向量乘以给定有限标量后的新向量。"""  # 解释向量数乘语义。
        if not math.isfinite(factor):  # 判断缩放系数是否为有限数值。
            raise ValueError("向量缩放系数必须为有限数值")  # 抛出明确的缩放系数异常。
        return Vector3(self.x * factor, self.y * factor, self.z * factor)  # 构造向量数乘结果。

    def dot(self, other: "Vector3") -> float:  # 定义三维向量点积方法。
        """计算当前向量与另一个向量的欧氏点积。"""  # 解释点积的数学含义。
        return self.x * other.x + self.y * other.y + self.z * other.z  # 返回逐分量乘积之和。

    def norm(self) -> float:  # 定义三维向量欧氏范数方法。
        """计算当前向量的欧氏长度。"""  # 解释范数的数学含义。
        return math.sqrt(self.dot(self))  # 通过自身点积的平方根计算范数。

    def distance_to(self, other: "Vector3") -> float:  # 定义两个三维点之间的距离方法。
        """计算当前点到另一个点的三维欧氏距离。"""  # 解释距离的数学含义。
        return (self - other).norm()  # 通过差向量范数计算点间距离。

    def normalized(self) -> "Vector3":  # 定义三维向量单位化方法。
        """返回方向不变且范数为一的单位向量。"""  # 解释单位化结果。
        vector_norm = self.norm()  # 计算当前向量范数。
        if vector_norm <= 0.0:  # 判断当前向量是否为零向量。
            raise ValueError("零向量不能单位化")  # 抛出明确的零向量异常。
        return self.scaled(1.0 / vector_norm)  # 按范数倒数缩放并返回单位向量。

    def interpolate(self, other: "Vector3", fraction: float) -> "Vector3":  # 定义两个三维点之间的线性插值方法。
        """按照零到一之间的比例返回线段上的插值点。"""  # 解释插值参数和结果。
        if not 0.0 <= fraction <= 1.0:  # 判断插值比例是否位于闭区间内。
            raise ValueError("插值比例必须位于零到一之间")  # 抛出明确的插值范围异常。
        return self + (other - self).scaled(fraction)  # 按差向量比例构造插值点。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义三维旋转矩阵。
class RotationMatrix3D:  # 定义右手正交三维旋转值对象。
    """保存并应用行列式为正一的三维旋转矩阵。"""  # 解释矩阵物理语义。

    rows: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]  # 保存按行排列的三乘三矩阵分量。

    def __post_init__(self) -> None:  # 定义旋转矩阵构造后的群性质验证。
        """拒绝错误形状、非有限、非正交或反射矩阵。"""  # 解释矩阵输入约束。
        try:  # 捕获不能转换为浮点数组的输入。
            matrix = np.asarray(self.rows, dtype=float)  # 将不可变行记录转换为数值校验数组。
        except (TypeError, ValueError) as error:  # 处理非数值矩阵成员。
            raise ValueError("三维旋转矩阵必须包含有限数值") from error  # 转换为稳定领域异常。
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():  # 判断矩阵形状和有限性是否合格。
            raise ValueError("三维旋转矩阵必须是有限三乘三矩阵")  # 拒绝不能表达三维线性变换的输入。
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9, rtol=0.0):  # 判断矩阵是否在数值容差内保持长度和正交关系。
            raise ValueError("三维旋转矩阵必须满足正交性")  # 拒绝包含缩放或剪切的矩阵。
        if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-9):  # 判断矩阵是否属于右手旋转群而非反射变换。
            raise ValueError("三维旋转矩阵行列式必须为正一")  # 拒绝行列式为负一的反射矩阵。

    @classmethod  # 声明单位旋转的命名构造器。
    def identity(cls) -> "RotationMatrix3D":  # 定义保持三维向量不变的单位旋转。
        """返回三维单位旋转矩阵。"""  # 解释命名构造器输出。
        return cls(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))  # 构造标准三阶单位矩阵。

    def apply(self, vector: Vector3) -> Vector3:  # 定义局部向量到旋转后向量的变换。
        """返回矩阵与给定三维向量的乘积。"""  # 解释变换结果。
        result = np.asarray(self.rows, dtype=float) @ vector.as_array()  # 执行三维矩阵向量乘法。
        return Vector3(float(result[0]), float(result[1]), float(result[2]))  # 将数值结果恢复为统一三维向量。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义三维折线。
class Polyline3D:  # 定义由有序三维点组成的折线类型。
    """保存至少两个连续不同的三维点并提供长度和采样计算。"""  # 解释折线的数据约束和能力。

    points: tuple[Vector3, ...]  # 保存按路径顺序排列的三维点。

    def __post_init__(self) -> None:  # 定义折线初始化后的结构验证方法。
        """拒绝少于两个点或包含连续重复点的折线。"""  # 解释折线结构约束。
        if len(self.points) < 2:  # 判断折线是否至少包含两个点。
            raise ValueError("三维折线至少需要两个点")  # 抛出明确的点数异常。
        for start, end in zip(self.points[:-1], self.points[1:]):  # 逐一检查相邻折线点。
            if start == end:  # 判断相邻点是否完全重复。
                raise ValueError("三维折线不能包含连续重复点")  # 抛出明确的零长度线段异常。

    def length(self) -> float:  # 定义折线总长度计算方法。
        """返回全部相邻线段三维欧氏长度之和。"""  # 解释折线长度定义。
        return sum(start.distance_to(end) for start, end in zip(self.points[:-1], self.points[1:]))  # 汇总全部相邻线段长度。

    def sample(self, max_spacing: float) -> tuple[Vector3, ...]:  # 定义受最大间距约束的折线采样方法。
        """返回包含全部折点且相邻距离不超过给定上限的采样点。"""  # 解释采样结果的保守性保证。
        if not math.isfinite(max_spacing) or max_spacing <= 0.0:  # 判断最大采样间距是否为正有限数值。
            raise ValueError("最大采样间距必须为正有限数值")  # 抛出明确的采样参数异常。
        samples: list[Vector3] = [self.points[0]]  # 使用折线起点初始化采样结果。
        for start, end in zip(self.points[:-1], self.points[1:]):  # 逐一采样每条相邻线段。
            segment_length = start.distance_to(end)  # 计算当前线段三维长度。
            interval_count = max(1, math.ceil(segment_length / max_spacing))  # 计算满足最大间距的最少区间数。
            for interval_index in range(1, interval_count + 1):  # 逐一生成当前线段除起点外的采样点。
                fraction = interval_index / interval_count  # 计算当前采样点在线段上的比例。
                samples.append(start.interpolate(end, fraction))  # 将线性插值点加入采样结果。
        return tuple(samples)  # 返回不可变的折线采样点集合。
