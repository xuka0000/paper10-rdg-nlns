"""定义地面车辆和车载无人机的物理能力元素。"""  # 说明本文件的建模职责。

import math  # 导入数学函数以验证设备参数有限性。
from dataclasses import dataclass  # 导入数据类工具以保存不可变设备对象。

from Mask.base import SceneElement, SceneObject, SceneSystem, ValidationIssue  # 导入场景元素和验证基础类型。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义地面车辆。
class GroundVehicle(SceneElement):  # 定义道路通行和车载平台能力元素。
    """保存车辆道路行驶能力和起飞回收停靠能力。"""  # 解释地面车辆职责。

    maximum_road_grade_ratio: float  # 保存车辆允许的最大绝对道路坡比。
    road_speed: float  # 保存车辆在标准平路条件下的道路速度。
    road_slope_sensitivity: float  # 保存道路坡比引起速度衰减的灵敏度。
    maximum_stop_grade_ratio: float  # 保存车辆停靠允许的最大局部坡比。
    minimum_stop_area: float  # 保存车辆停车和无人机起飞回收所需最小面积。

    def __post_init__(self) -> None:  # 定义地面车辆初始化后的参数验证方法。
        """验证车辆道路和停靠能力参数。"""  # 解释地面车辆输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        numeric_values = (self.maximum_road_grade_ratio, self.road_speed, self.road_slope_sensitivity, self.maximum_stop_grade_ratio, self.minimum_stop_area)  # 汇总地面车辆数值参数。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部地面车辆参数是否有限。
            raise ValueError("地面车辆参数必须为有限数值")  # 抛出明确的车辆有限性异常。
        if self.maximum_road_grade_ratio < 0.0 or self.maximum_stop_grade_ratio < 0.0:  # 判断车辆道路和停靠坡比上限是否为非负数。
            raise ValueError("车辆最大道路和停靠坡比不能为负数")  # 抛出明确的坡比能力异常。
        if self.road_speed <= 0.0:  # 判断车辆标准道路速度是否为正数。
            raise ValueError("车辆道路速度必须为正数")  # 抛出明确的道路速度异常。
        if self.road_slope_sensitivity < 0.0:  # 判断坡比速度衰减灵敏度是否为非负数。
            raise ValueError("车辆道路坡比灵敏度不能为负数")  # 抛出明确的坡比灵敏度异常。
        if self.minimum_stop_area <= 0.0:  # 判断车辆最小停靠面积是否为正数。
            raise ValueError("车辆最小停靠面积必须为正数")  # 抛出明确的停靠面积异常。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """返回空结果以表示车辆结构已经在构造阶段验证。"""  # 解释车辆验证结果。
        return ()  # 返回无验证问题结果。


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类定义车载无人机。
class UnmannedAerialVehicle(SceneElement):  # 定义影响航次时间和能耗的无人机物理能力元素。
    """保存所属车辆、速度、推力、净空、功率、质量和电池能力。"""  # 解释无人机职责。

    carrier_vehicle_id: str  # 保存搭载该无人机的地面车辆标识。
    maximum_speed: float  # 保存无人机允许的最大三维速度。
    maximum_thrust_acceleration: float  # 保存无人机最大推力比加速度。
    minimum_clearance: float  # 保存无人机最小地形和障碍净空。
    hover_power: float  # 保存无人机悬停基准功率。
    mass: float  # 保存无人机任务质量。
    battery_state_of_charge: float  # 保存当前航次开始时的电池荷电能量。
    battery_reserve_ratio: float = 0.0  # 保存任务结束后必须保留的电池比例。

    def __post_init__(self) -> None:  # 定义无人机初始化后的参数验证方法。
        """验证无人机引用和全部物理能力参数。"""  # 解释无人机输入约束。
        SceneObject.__post_init__(self)  # 复用场景对象的标识和名称验证。
        if not self.carrier_vehicle_id.strip():  # 判断搭载车辆标识是否为空白。
            raise ValueError("无人机搭载车辆标识不能为空")  # 抛出明确的搭载车辆引用异常。
        numeric_values = (self.maximum_speed, self.maximum_thrust_acceleration, self.minimum_clearance, self.hover_power, self.mass, self.battery_state_of_charge, self.battery_reserve_ratio)  # 汇总无人机数值参数。
        if not all(math.isfinite(value) for value in numeric_values):  # 判断全部无人机参数是否有限。
            raise ValueError("无人机物理参数必须为有限数值")  # 抛出明确的无人机有限性异常。
        if self.maximum_speed <= 0.0:  # 判断无人机最大速度是否为正数。
            raise ValueError("无人机最大速度必须为正数")  # 抛出明确的无人机速度异常。
        if self.maximum_thrust_acceleration <= 9.81:  # 判断无人机推力比加速度是否足以抵消标准重力。
            raise ValueError("无人机最大推力比加速度必须大于重力加速度")  # 抛出明确的推力能力异常。
        if self.minimum_clearance < 0.0:  # 判断无人机最小净空是否为非负数。
            raise ValueError("无人机最小净空不能为负数")  # 抛出明确的无人机净空异常。
        if self.hover_power <= 0.0:  # 判断无人机悬停功率是否为正数。
            raise ValueError("无人机悬停功率必须为正数")  # 抛出明确的悬停功率异常。
        if self.mass <= 0.0:  # 判断无人机任务质量是否为正数。
            raise ValueError("无人机质量必须为正数")  # 抛出明确的无人机质量异常。
        if self.battery_state_of_charge <= 0.0:  # 判断无人机当前荷电能量是否为正数。
            raise ValueError("无人机当前荷电能量必须为正数")  # 抛出明确的电池能量异常。
        if not 0.0 <= self.battery_reserve_ratio < 1.0:  # 判断电池预留比例是否位于有效范围。
            raise ValueError("无人机电池预留比例必须位于零和一之间")  # 抛出明确的预留比例异常。

    @property  # 将式 39 可用电池能量暴露为只读派生属性。
    def available_battery(self) -> float:  # 定义当前航次可用电池能量读取方法。
        """返回扣除预留比例后的当前可用电池能量。"""  # 解释式 39 计算语义。
        return (1.0 - self.battery_reserve_ratio) * self.battery_state_of_charge  # 按式 39 计算并返回可用电池能量。

    def validate(self, system: SceneSystem) -> tuple[ValidationIssue, ...]:  # 实现场景元素验证接口。
        """检查无人机引用的搭载车辆是否存在且类型正确。"""  # 解释无人机场景验证规则。
        if not system.contains(self.carrier_vehicle_id):  # 判断无人机搭载车辆是否缺失。
            return (ValidationIssue("missing_carrier_vehicle", self.object_id, self.carrier_vehicle_id),)  # 返回搭载车辆缺失问题。
        referenced = system.get_element(self.carrier_vehicle_id)  # 读取无人机引用的场景元素。
        if not isinstance(referenced, GroundVehicle):  # 判断引用对象是否属于地面车辆。
            return (ValidationIssue("invalid_carrier_vehicle_type", self.object_id, self.carrier_vehicle_id),)  # 返回搭载车辆类型错误问题。
        return ()  # 返回无验证问题结果。
