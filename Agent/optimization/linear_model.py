"""构建具有稳定变量名和约束名的 SciPy 稀疏线性模型。"""  # 说明本文件的算法职责。

import math  # 导入数学函数以验证模型标量参数。
from dataclasses import dataclass  # 导入数据类工具以保存不可变变量和模型对象。
from typing import Iterable  # 导入可迭代类型以描述稀疏约束项。

import numpy as np  # 导入数组库以构建 SciPy 目标、边界和解向量。
from scipy.optimize import Bounds, LinearConstraint  # 导入 SciPy 变量边界和线性约束类型。
from scipy.sparse import csr_array  # 导入压缩稀疏行矩阵以保存约束系数。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义变量引用。
class VariableReference:  # 定义稳定变量名称和数值列索引的组合对象。
    """允许约束构建代码通过名称来源对象引用数值变量列。"""  # 解释变量引用职责。

    name: str  # 保存变量的稳定可读名称。
    index: int  # 保存变量在 SciPy 数组中的列索引。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义已构建线性模型。
class BuiltLinearModel:  # 定义传给 SciPy MILP 的全部数组、稀疏矩阵和名称映射。
    """绑定数值模型输入和公式到代码追踪所需的稳定名称。"""  # 解释已构建模型内容。

    objective: np.ndarray  # 保存最小化目标函数系数向量。
    integrality: np.ndarray  # 保存连续或整数变量类型标识向量。
    lower_bounds: np.ndarray  # 保存变量下界向量。
    upper_bounds: np.ndarray  # 保存变量上界向量。
    constraint_matrix: csr_array  # 保存稀疏线性约束系数矩阵。
    constraint_lower_bounds: np.ndarray  # 保存线性约束下界向量。
    constraint_upper_bounds: np.ndarray  # 保存线性约束上界向量。
    bounds: Bounds  # 保存 SciPy 变量边界对象。
    linear_constraint: LinearConstraint  # 保存 SciPy 线性约束对象。
    variable_names: tuple[str, ...]  # 保存按照列索引排列的稳定变量名称。
    constraint_names: tuple[str, ...]  # 保存按照行索引排列的稳定约束名称。


class LinearModelBuilder:  # 定义命名变量注册和稀疏约束汇总构建器。
    """以登记顺序创建变量和约束并生成 SciPy MILP 数值输入。"""  # 解释线性模型构建器职责。

    def __init__(self) -> None:  # 定义线性模型构建器初始化方法。
        """创建空变量注册表和空约束行集合。"""  # 解释构建器初始状态。
        self._variable_names: list[str] = []  # 创建稳定变量名称列表。
        self._variable_index_by_name: dict[str, int] = {}  # 创建变量名称到列索引的映射。
        self._objective: list[float] = []  # 创建目标函数系数列表。
        self._integrality: list[int] = []  # 创建变量积分类型标识列表。
        self._lower_bounds: list[float] = []  # 创建变量下界列表。
        self._upper_bounds: list[float] = []  # 创建变量上界列表。
        self._constraint_names: list[str] = []  # 创建稳定约束名称列表。
        self._constraint_name_set: set[str] = set()
        self._constraint_rows: list[dict[int, float]] = []  # 创建稀疏约束行系数映射列表。
        self._constraint_lower_bounds: list[float] = []  # 创建线性约束下界列表。
        self._constraint_upper_bounds: list[float] = []  # 创建线性约束上界列表。

    def add_binary(self, name: str, objective_coefficient: float = 0.0, upper: float = 1.0) -> VariableReference:  # 定义二元变量登记方法。
        """创建下界为零、上界为一且具有整数标识的变量。"""  # 解释二元变量数值边界。
        return self._add_variable(name, 0.0, upper, 1, objective_coefficient)  # 委托通用变量登记方法创建二元变量。

    def add_continuous(self, name: str, lower: float = 0.0, upper: float = math.inf, objective_coefficient: float = 0.0) -> VariableReference:  # 定义连续变量登记方法。
        """创建具有给定上下界和目标系数的连续变量。"""  # 解释连续变量登记参数。
        return self._add_variable(name, lower, upper, 0, objective_coefficient)  # 委托通用变量登记方法创建连续变量。

    def _add_variable(self, name: str, lower: float, upper: float, integrality: int, objective_coefficient: float) -> VariableReference:  # 定义通用变量登记方法。
        """验证名称、边界和目标系数并分配稳定列索引。"""  # 解释内部变量登记行为。
        if not name.strip():  # 判断变量名称是否为空白。
            raise ValueError("变量名称不能为空")  # 抛出明确的变量名称异常。
        if name in self._variable_index_by_name:  # 判断变量名称是否已经登记。
            raise ValueError(f"变量名称重复: {name}")  # 抛出包含冲突名称的变量重复异常。
        if math.isnan(lower) or math.isnan(upper) or lower > upper:  # 判断变量边界是否包含非数或逆序范围。
            raise ValueError(f"变量边界无效: {name}")  # 抛出包含变量名称的边界异常。
        if not math.isfinite(objective_coefficient):  # 判断目标函数系数是否为有限数值。
            raise ValueError(f"变量目标系数必须有限: {name}")  # 抛出包含变量名称的目标系数异常。
        index = len(self._variable_names)  # 使用当前变量数量分配新列索引。
        self._variable_names.append(name)  # 将变量名称加入稳定顺序列表。
        self._variable_index_by_name[name] = index  # 记录变量名称到列索引的映射。
        self._objective.append(float(objective_coefficient))  # 将目标函数系数加入数值列表。
        self._integrality.append(integrality)  # 将变量积分类型标识加入数值列表。
        self._lower_bounds.append(float(lower))  # 将变量下界加入数值列表。
        self._upper_bounds.append(float(upper))  # 将变量上界加入数值列表。
        return VariableReference(name, index)  # 返回稳定变量名称和列索引引用。

    def add_constraint(self, name: str, terms: Iterable[tuple[VariableReference, float]], lower: float = -math.inf, upper: float = math.inf) -> None:  # 定义命名稀疏线性约束登记方法。
        """合并重复变量系数并保存约束行、下界和上界。"""  # 解释约束登记行为。
        if not name.strip():  # 判断约束名称是否为空白。
            raise ValueError("约束名称不能为空")  # 抛出明确的约束名称异常。
        if name in self._constraint_name_set:  # 判断约束名称是否已经登记。
            raise ValueError(f"约束名称重复: {name}")  # 抛出包含冲突名称的约束重复异常。
        if math.isnan(lower) or math.isnan(upper) or lower > upper:  # 判断约束边界是否包含非数或逆序范围。
            raise ValueError(f"约束边界无效: {name}")  # 抛出包含约束名称的边界异常。
        row: dict[int, float] = {}  # 创建当前约束行的稀疏系数映射。
        for variable, coefficient in terms:  # 逐一处理约束中的变量系数项。
            if variable.index < 0 or variable.index >= len(self._variable_names):  # 判断变量引用列索引是否属于当前构建器。
                raise ValueError(f"约束引用未知变量: {variable.name}")  # 抛出包含变量名称的未知引用异常。
            if self._variable_names[variable.index] != variable.name:  # 判断变量名称是否与当前构建器列索引一致。
                raise ValueError(f"约束变量引用不匹配: {variable.name}")  # 抛出明确的变量引用冲突异常。
            if not math.isfinite(coefficient):  # 判断约束系数是否为有限数值。
                raise ValueError(f"约束系数必须有限: {name}")  # 抛出包含约束名称的系数异常。
            row[variable.index] = row.get(variable.index, 0.0) + float(coefficient)  # 累加同一变量在当前约束中的重复系数。
        row = {index: coefficient for index, coefficient in row.items() if abs(coefficient) > 0.0}  # 删除累加后恰好为零的稀疏系数。
        self._constraint_names.append(name)  # 将约束名称加入稳定顺序列表。
        self._constraint_name_set.add(name)
        self._constraint_rows.append(row)  # 将稀疏约束行加入数值列表。
        self._constraint_lower_bounds.append(float(lower))  # 将约束下界加入数值列表。
        self._constraint_upper_bounds.append(float(upper))  # 将约束上界加入数值列表。

    def build(self) -> BuiltLinearModel:  # 定义 SciPy 线性模型构建方法。
        """将登记数据转换为数组、压缩稀疏行矩阵和 SciPy 约束对象。"""  # 解释模型构建输出。
        variable_count = len(self._variable_names)  # 计算已登记变量数量。
        constraint_count = len(self._constraint_names)  # 计算已登记约束数量。
        row_indices: list[int] = []  # 创建非零系数对应的约束行索引列表。
        column_indices: list[int] = []  # 创建非零系数对应的变量列索引列表。
        coefficients: list[float] = []  # 创建非零约束系数列表。
        for row_index, row in enumerate(self._constraint_rows):  # 按稳定顺序遍历稀疏约束行。
            for column_index, coefficient in sorted(row.items()):  # 按变量列索引顺序遍历当前行非零系数。
                row_indices.append(row_index)  # 记录当前非零系数的约束行索引。
                column_indices.append(column_index)  # 记录当前非零系数的变量列索引。
                coefficients.append(coefficient)  # 记录当前非零约束系数。
        constraint_matrix = csr_array((coefficients, (row_indices, column_indices)), shape=(constraint_count, variable_count), dtype=float)  # 构建压缩稀疏行约束矩阵。
        objective = np.asarray(self._objective, dtype=float)  # 构建目标函数系数数组。
        integrality = np.asarray(self._integrality, dtype=int)  # 构建变量积分类型数组。
        lower_bounds = np.asarray(self._lower_bounds, dtype=float)  # 构建变量下界数组。
        upper_bounds = np.asarray(self._upper_bounds, dtype=float)  # 构建变量上界数组。
        constraint_lower_bounds = np.asarray(self._constraint_lower_bounds, dtype=float)  # 构建约束下界数组。
        constraint_upper_bounds = np.asarray(self._constraint_upper_bounds, dtype=float)  # 构建约束上界数组。
        bounds = Bounds(lower_bounds, upper_bounds)  # 构建 SciPy 变量边界对象。
        linear_constraint = LinearConstraint(constraint_matrix, constraint_lower_bounds, constraint_upper_bounds)  # 构建 SciPy 线性约束对象。
        return BuiltLinearModel(objective, integrality, lower_bounds, upper_bounds, constraint_matrix, constraint_lower_bounds, constraint_upper_bounds, bounds, linear_constraint, tuple(self._variable_names), tuple(self._constraint_names))  # 返回数值输入和稳定名称绑定的线性模型。

    def reconstruct(self, solution: np.ndarray) -> dict[str, float]:  # 定义数值解向量重构方法。
        """按照稳定变量登记顺序将数值解转换为名称到数值的映射。"""  # 解释解重构输出。
        flattened = np.asarray(solution, dtype=float).reshape(-1)  # 将输入解转换为一维浮点数组。
        if len(flattened) != len(self._variable_names):  # 判断解向量长度是否与变量数量一致。
            raise ValueError("解向量长度与变量数量不一致")  # 抛出明确的解向量长度异常。
        return {name: float(flattened[index]) for index, name in enumerate(self._variable_names)}  # 按稳定变量名称重构并返回数值映射。
