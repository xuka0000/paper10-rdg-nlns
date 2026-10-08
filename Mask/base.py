"""定义场景元素、场景关系、场景系统和验证问题的基础抽象。"""  # 说明本文件的建模职责。

from abc import ABC, abstractmethod  # 导入抽象基类工具以约束子类接口。
from dataclasses import dataclass, field  # 导入数据类工具以保存稳定对象数据。
from typing import Iterable  # 导入可迭代类型以描述对象集合接口。


class DuplicateObjectError(ValueError):  # 定义场景对象标识重复异常。
    """表示元素或关系使用了场景中已经存在的稳定标识。"""  # 解释异常触发条件。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类保存验证问题。
class ValidationIssue:  # 定义统一的场景验证问题对象。
    """记录验证问题代码、所属对象、引用对象和可读说明。"""  # 解释验证问题的数据内容。

    code: str  # 保存便于程序处理的稳定问题代码。
    object_id: str  # 保存产生验证问题的场景对象标识。
    reference_id: str = ""  # 保存缺失或冲突的引用对象标识。
    message: str = ""  # 保存面向使用者的补充说明。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义场景对象。
class SceneObject(ABC):  # 定义全部场景对象共享的抽象基类。
    """保存稳定标识和名称并要求子类提供场景验证行为。"""  # 解释基类职责。

    object_id: str  # 保存对象在整个场景中的唯一标识。
    name: str  # 保存对象的中文或英文可读名称。

    def __post_init__(self) -> None:  # 定义基础字段的初始化后验证方法。
        """拒绝空白的对象标识和对象名称。"""  # 解释基础字段验证规则。
        if not self.object_id.strip():  # 判断稳定标识是否为空白。
            raise ValueError("场景对象标识不能为空")  # 抛出明确的标识验证异常。
        if not self.name.strip():  # 判断可读名称是否为空白。
            raise ValueError("场景对象名称不能为空")  # 抛出明确的名称验证异常。

    @abstractmethod  # 声明所有具体场景对象必须实现验证方法。
    def validate(self, system: "SceneSystem") -> tuple[ValidationIssue, ...]:  # 定义对象相对于完整场景的验证接口。
        """返回当前对象相对于给定场景系统的全部验证问题。"""  # 解释验证接口的输入输出。
        raise NotImplementedError  # 阻止直接调用未实现的抽象验证方法。


class SceneElement(SceneObject, ABC):  # 定义独立物理对象和任务对象的抽象基类。
    """表示可以独立登记到场景系统中的元素对象。"""  # 解释元素基类的语义边界。


@dataclass(frozen=True, slots=True)  # 使用不可变数据类定义场景关系基础字段。
class SceneRelation(SceneObject, ABC):  # 定义两个场景对象之间关系的抽象基类。
    """保存关系源对象和目标对象并要求子类实现物理验证。"""  # 解释关系基类职责。

    source_id: str  # 保存关系源场景对象的稳定标识。
    target_id: str  # 保存关系目标场景对象的稳定标识。

    def __post_init__(self) -> None:  # 定义场景关系的初始化后验证方法。
        """验证基础对象字段和关系端点标识。"""  # 解释关系字段验证规则。
        SceneObject.__post_init__(self)  # 显式调用基类验证以兼容带 slots 的数据类继承。
        if not self.source_id.strip():  # 判断关系源标识是否为空白。
            raise ValueError("场景关系源标识不能为空")  # 抛出明确的关系源验证异常。
        if not self.target_id.strip():  # 判断关系目标标识是否为空白。
            raise ValueError("场景关系目标标识不能为空")  # 抛出明确的关系目标验证异常。


@dataclass(slots=True)  # 使用可变紧凑数据类保存场景对象注册表。
class SceneSystem:  # 定义元素和关系的统一场景聚合类。
    """在共享标识空间内登记元素和关系并提供只读查询接口。"""  # 解释场景系统职责。

    object_id: str  # 保存场景系统的稳定标识。
    name: str  # 保存场景系统的可读名称。
    _elements: dict[str, SceneElement] = field(default_factory=dict, init=False, repr=False)  # 保存按标识索引的场景元素。
    _relations: dict[str, SceneRelation] = field(default_factory=dict, init=False, repr=False)  # 保存按标识索引的场景关系。

    def __post_init__(self) -> None:  # 定义场景系统的初始化后验证方法。
        """拒绝空白的场景系统标识和名称。"""  # 解释场景系统字段验证规则。
        if not self.object_id.strip():  # 判断场景系统标识是否为空白。
            raise ValueError("场景系统标识不能为空")  # 抛出明确的系统标识验证异常。
        if not self.name.strip():  # 判断场景系统名称是否为空白。
            raise ValueError("场景系统名称不能为空")  # 抛出明确的系统名称验证异常。

    def contains(self, object_id: str) -> bool:  # 定义场景对象存在性查询方法。
        """判断给定稳定标识是否已经登记为元素或关系。"""  # 解释查询方法的返回语义。
        return object_id in self._elements or object_id in self._relations  # 返回共享标识空间中的查询结果。

    def add_element(self, element: SceneElement) -> None:  # 定义场景元素登记方法。
        """将一个场景元素加入共享标识空间。"""  # 解释元素登记行为。
        self._reject_duplicate(element.object_id)  # 在写入前检查稳定标识是否重复。
        self._elements[element.object_id] = element  # 按稳定标识保存场景元素。

    def add_relation(self, relation: SceneRelation) -> None:  # 定义场景关系登记方法。
        """将一个场景关系加入共享标识空间。"""  # 解释关系登记行为。
        self._reject_duplicate(relation.object_id)  # 在写入前检查稳定标识是否重复。
        self._relations[relation.object_id] = relation  # 按稳定标识保存场景关系。

    def add_elements(self, elements: Iterable[SceneElement]) -> None:  # 定义批量元素登记方法。
        """按照输入顺序登记一组场景元素。"""  # 解释批量登记行为。
        for element in elements:  # 逐一处理输入的场景元素。
            self.add_element(element)  # 复用单个元素登记和重复检查逻辑。

    def add_relations(self, relations: Iterable[SceneRelation]) -> None:  # 定义批量关系登记方法。
        """按照输入顺序登记一组场景关系。"""  # 解释批量登记行为。
        for relation in relations:  # 逐一处理输入的场景关系。
            self.add_relation(relation)  # 复用单个关系登记和重复检查逻辑。

    def get_element(self, object_id: str) -> SceneElement:  # 定义场景元素查询方法。
        """根据稳定标识返回已经登记的场景元素。"""  # 解释元素查询行为。
        return self._elements[object_id]  # 从元素注册表返回目标对象。

    def get_relation(self, object_id: str) -> SceneRelation:  # 定义场景关系查询方法。
        """根据稳定标识返回已经登记的场景关系。"""  # 解释关系查询行为。
        return self._relations[object_id]  # 从关系注册表返回目标对象。

    def elements(self) -> tuple[SceneElement, ...]:  # 定义全部场景元素读取方法。
        """按照登记顺序返回不可变的场景元素集合。"""  # 解释元素集合的只读边界。
        return tuple(self._elements.values())  # 将内部元素视图转换为不可变元组。

    def relations(self) -> tuple[SceneRelation, ...]:  # 定义全部场景关系读取方法。
        """按照登记顺序返回不可变的场景关系集合。"""  # 解释关系集合的只读边界。
        return tuple(self._relations.values())  # 将内部关系视图转换为不可变元组。

    def validate(self) -> tuple[ValidationIssue, ...]:  # 定义完整场景验证方法。
        """汇总全部元素和关系相对于当前场景的验证问题。"""  # 解释系统级验证行为。
        issues: list[ValidationIssue] = []  # 创建可追加的验证问题列表。
        for element in self._elements.values():  # 逐一验证全部场景元素。
            issues.extend(element.validate(self))  # 将元素验证问题加入汇总列表。
        for relation in self._relations.values():  # 逐一验证全部场景关系。
            issues.extend(relation.validate(self))  # 将关系验证问题加入汇总列表。
        return tuple(issues)  # 返回不可变的系统验证问题集合。

    def _reject_duplicate(self, object_id: str) -> None:  # 定义共享标识重复检查方法。
        """在给定标识已经存在时抛出包含该标识的异常。"""  # 解释内部检查行为。
        if self.contains(object_id):  # 判断元素或关系是否已经使用给定标识。
            raise DuplicateObjectError(f"场景对象标识重复: {object_id}")  # 抛出包含冲突标识的明确异常。
