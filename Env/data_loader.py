"""提供不包含决策逻辑的严格 JSON 配置读取和标识检查。"""  # 说明本文件的环境职责。

import json  # 导入 JSON 解析器以读取 Data 配置文件。
import math  # 导入数学函数以拒绝非有限数值配置。
from dataclasses import dataclass  # 导入数据类工具以保存不可变的三份 Data 文档。
from pathlib import Path  # 导入路径工具以定位配置文件。
from typing import Any, Iterable  # 导入通用值和可迭代类型以描述配置记录。


SCENE_REQUIRED_KEYS = {"scene_id", "scene_name", "terrain", "obstacle_field", "road_nodes", "stop_sites", "vehicles", "uavs", "road_legs", "towers", "components", "checkpoints", "observation_poses", "tasks", "weather_states"}  # 定义场景文件必须包含的顶层字段。
CONFIG_COMMON_REQUIRED_KEYS = {"physical", "agent"}  # 定义所有时间形式共享的物理和 Agent 必填字段。
CONFIG_REQUIRED_KEYS = CONFIG_COMMON_REQUIRED_KEYS | {"time_horizon_seconds"}  # 保留旧 Agent 导入常量的原有单时域语义。
CONFIG_ALLOWED_KEYS = CONFIG_COMMON_REQUIRED_KEYS | {"time_horizon_seconds", "mission_horizon_seconds", "rolling_window_seconds"}  # 允许 Env 读取旧单时域或正式双时域，但不允许其他顶层字段。
SERVICE_REQUIRED_KEYS = {"candidates"}  # 定义服务列候选文件必须包含的顶层字段。
M1_SCENE_RECORD_KEYS = {  # 定义本轮完整对齐所保护的 M1 场景记录字段。
    "stop_sites": {"id", "name", "road_node_id", "footprint_radius_m"},  # 固定候选停靠点物理字段。
    "vehicles": {"id", "name", "maximum_road_grade_ratio", "road_speed_mps", "road_slope_sensitivity", "maximum_stop_grade_ratio", "minimum_stop_area_m2"},  # 固定车辆道路和停靠能力字段。
    "towers": {"id", "name", "x_m", "y_m", "rotation_matrix"},  # 固定杆塔基座和完整三维旋转字段。
}  # 结束 M1 场景记录字段集合。
M4_SCENE_RECORD_KEYS = {  # 定义式 39 电池状态所需的严格无人机记录字段。
    "uavs": {"id", "name", "carrier_vehicle_id", "maximum_speed_mps", "maximum_thrust_acceleration_mps2", "minimum_clearance_m", "hover_power_w", "mass_kg", "battery_state_of_charge_j", "battery_reserve_ratio"},  # 固定无人机物理能力、当前荷电能量和预留比例字段。
}  # 结束 M4 场景记录字段集合。
STRICT_SCENE_RECORD_KEYS = {**M1_SCENE_RECORD_KEYS, **M4_SCENE_RECORD_KEYS}  # 汇总必须拒绝字段漂移的场景记录契约。
OPTIONAL_ROLLING_RECORD_KEYS = {  # 定义滚动资源状态允许出现的可选字段。
    "vehicles": {"current_road_node_id"},
    "uavs": {"available", "replacement_pack_energy_j", "turnaround_time_s"},
}


@dataclass(frozen=True, slots=True)  # 使用不可变紧凑数据类绑定三份经过验证的 JSON 文档。
class ProjectData:  # 定义 Env 场景装配所消费的 Data 记录集合。
    """保存场景、服务列候选和物理算法配置的原始结构化记录。"""  # 解释数据集合职责。

    scene: dict[str, Any]  # 保存经过顶层和数值检查的场景记录。
    service_candidates: dict[str, Any]  # 保存经过顶层和标识检查的服务列候选记录。
    config: dict[str, Any]  # 保存经过顶层和数值检查的物理算法配置。


class DataValidationError(ValueError):  # 定义 Data 配置结构验证异常。
    """表示配置文件缺失字段、包含未知字段或具有重复标识。"""  # 解释异常触发条件。


def read_json_object(path: Path, required_keys: set[str], allowed_keys: set[str]) -> dict[str, Any]:  # 定义严格 JSON 对象读取函数。
    """读取一个 JSON 对象并拒绝缺失必需字段和未知顶层字段。"""  # 解释配置读取边界。
    if not path.is_file():  # 判断目标配置文件是否存在。
        raise DataValidationError(f"Data 配置文件不存在: {path}")  # 抛出包含目标路径的文件缺失异常。
    try:  # 开始捕获 JSON 语法错误。
        raw_value = json.loads(path.read_text(encoding="utf-8-sig"))  # 按兼容 UTF-8 BOM 的方式读取并解析配置。
    except json.JSONDecodeError as error:  # 捕获 JSON 语法解析异常。
        raise DataValidationError(f"Data 配置 JSON 语法错误: {path}:{error.lineno}:{error.colno}") from error  # 转换为包含行列位置的数据验证异常。
    if not isinstance(raw_value, dict):  # 判断配置顶层值是否属于 JSON 对象。
        raise DataValidationError(f"Data 配置顶层必须为对象: {path}")  # 抛出明确的顶层类型异常。
    actual_keys = set(raw_value)  # 提取配置对象实际顶层字段集合。
    missing_keys = sorted(required_keys - actual_keys)  # 计算配置对象缺失的必需字段。
    unknown_keys = sorted(actual_keys - allowed_keys)  # 计算配置对象包含的未知字段。
    if missing_keys:  # 判断配置对象是否缺少必需字段。
        raise DataValidationError(f"Data 配置缺少必需字段: {','.join(missing_keys)}")  # 抛出包含全部缺失字段的数据验证异常。
    if unknown_keys:  # 判断配置对象是否包含未知字段。
        raise DataValidationError(f"Data 配置包含未知字段: {','.join(unknown_keys)}")  # 抛出包含全部未知字段的数据验证异常。
    return raw_value  # 返回通过顶层结构验证的 JSON 对象。


def require_unique_ids(records: Iterable[dict[str, Any]], collection_name: str) -> None:  # 定义配置记录标识唯一性检查函数。
    """要求给定记录集合中的每条记录具有唯一非空字符串标识。"""  # 解释标识检查规则。
    seen_ids: set[str] = set()  # 创建已经出现的记录标识集合。
    for record_index, record in enumerate(records):  # 按输入顺序遍历配置记录。
        object_id = record.get("id")  # 读取当前配置记录的稳定标识。
        if not isinstance(object_id, str) or not object_id.strip():  # 判断当前记录标识是否为非空字符串。
            raise DataValidationError(f"{collection_name}[{record_index}] 缺少有效 id")  # 抛出包含集合和记录位置的标识异常。
        if object_id in seen_ids:  # 判断当前记录标识是否已经出现。
            raise DataValidationError(f"{collection_name} 存在重复 id: {object_id}")  # 抛出包含冲突标识的重复异常。
        seen_ids.add(object_id)  # 将当前记录标识加入已出现集合。


def require_record_keys(record: Any, required_keys: set[str], allowed_keys: set[str], record_path: str) -> dict[str, Any]:  # 定义嵌套配置记录字段检查函数。
    """要求嵌套记录属于对象且不缺字段也不包含未知字段。"""  # 解释嵌套记录检查边界。
    if not isinstance(record, dict):  # 判断嵌套配置记录是否属于 JSON 对象。
        raise DataValidationError(f"Data 配置记录必须为对象: {record_path}")  # 抛出包含记录路径的类型异常。
    actual_keys = set(record)  # 提取嵌套记录实际字段集合。
    missing_keys = sorted(required_keys - actual_keys)  # 计算嵌套记录缺失字段。
    unknown_keys = sorted(actual_keys - allowed_keys)  # 计算嵌套记录未知字段。
    if missing_keys:  # 判断嵌套记录是否缺少必要字段。
        raise DataValidationError(f"Data 配置记录缺少必要字段: {record_path}:{','.join(missing_keys)}")  # 抛出包含记录路径和字段的异常。
    if unknown_keys:  # 判断嵌套记录是否包含未知字段。
        raise DataValidationError(f"Data 配置记录包含未知字段: {record_path}:{','.join(unknown_keys)}")  # 抛出包含记录路径和字段的异常。
    return record  # 返回通过字段检查的嵌套记录。


def require_record_list(value: Any, collection_name: str) -> list[dict[str, Any]]:  # 定义配置记录列表检查函数。
    """要求给定值属于由 JSON 对象组成的列表。"""  # 解释记录列表检查规则。
    if not isinstance(value, list):  # 判断配置集合是否属于 JSON 数组。
        raise DataValidationError(f"Data 配置集合必须为数组: {collection_name}")  # 抛出包含集合名称的类型异常。
    records: list[dict[str, Any]] = []  # 创建经过对象类型确认的记录列表。
    for record_index, record in enumerate(value):  # 按输入顺序检查每条配置记录。
        if not isinstance(record, dict):  # 判断当前数组成员是否属于 JSON 对象。
            raise DataValidationError(f"Data 配置集合成员必须为对象: {collection_name}[{record_index}]")  # 抛出包含数组位置的类型异常。
        records.append(record)  # 将通过对象类型检查的记录加入结果列表。
    return records  # 返回保持输入顺序的记录列表。


def require_finite_numbers(value: Any, value_path: str) -> None:  # 定义递归有限数值检查函数。
    """拒绝场景和配置结构中出现的正负无穷或非数值浮点值。"""  # 解释有限数值检查目的。
    if isinstance(value, bool) or value is None or isinstance(value, str):  # 判断当前值是否属于无需有限性检查的标量。
        return  # 直接接受布尔值、空值和字符串。
    if isinstance(value, (int, float)):  # 判断当前值是否属于数值标量。
        if not math.isfinite(float(value)):  # 判断数值是否为有限值。
            raise DataValidationError(f"Data 配置数值必须有限: {value_path}")  # 抛出包含数值路径的有限性异常。
        return  # 完成当前数值标量检查。
    if isinstance(value, list):  # 判断当前值是否属于 JSON 数组。
        for item_index, item in enumerate(value):  # 按顺序递归检查数组成员。
            require_finite_numbers(item, f"{value_path}[{item_index}]")  # 递归检查当前数组成员。
        return  # 完成当前数组的递归检查。
    if isinstance(value, dict):  # 判断当前值是否属于 JSON 对象。
        for item_key, item in value.items():  # 按文档顺序递归检查对象字段。
            require_finite_numbers(item, f"{value_path}.{item_key}")  # 递归检查当前对象字段值。
        return  # 完成当前对象的递归检查。
    raise DataValidationError(f"Data 配置包含不支持的值类型: {value_path}")  # 拒绝不能由 JSON 结构表达的异常值类型。


def _validate_scene_collections(scene: dict[str, Any]) -> None:  # 定义场景集合和稳定标识检查函数。
    """检查全部场景记录集合和显式障碍具有局部及全局唯一标识。"""  # 解释场景标识检查范围。
    collection_names = ("road_nodes", "stop_sites", "vehicles", "uavs", "road_legs", "towers", "components", "checkpoints", "observation_poses", "tasks", "weather_states")  # 列出必须由记录数组表达的场景集合。
    all_records: list[dict[str, Any]] = []  # 创建用于全局标识唯一性检查的记录列表。
    for collection_name in collection_names:  # 逐一检查场景记录集合。
        records = require_record_list(scene[collection_name], collection_name)  # 要求当前场景集合属于对象数组。
        require_unique_ids(records, collection_name)  # 要求当前场景集合内部标识唯一。
        if collection_name in STRICT_SCENE_RECORD_KEYS:  # 判断当前集合是否属于严格字段保护的模型对象。
            record_keys = STRICT_SCENE_RECORD_KEYS[collection_name]  # 读取当前集合唯一允许的字段集合。
            for record_index, record in enumerate(records):  # 逐一检查当前 M1 记录字段。
                allowed_keys = record_keys | OPTIONAL_ROLLING_RECORD_KEYS.get(collection_name, set())  # 合并物理必填字段和滚动可选字段。
                require_record_keys(record, record_keys, allowed_keys, f"{collection_name}[{record_index}]")  # 拒绝缺失字段和任何旧字段别名。
        all_records.extend(records)  # 将当前集合记录加入全局标识列表。
    terrain = require_record_keys(scene["terrain"], {"id", "name", "x_coordinates_m", "y_coordinates_m", "heights_m"}, {"id", "name", "x_coordinates_m", "y_coordinates_m", "heights_m"}, "terrain")  # 检查地形记录字段。
    obstacle_field = require_record_keys(scene["obstacle_field"], {"id", "name", "obstacles"}, {"id", "name", "obstacles"}, "obstacle_field")  # 检查障碍场记录字段。
    obstacles = require_record_list(obstacle_field["obstacles"], "obstacle_field.obstacles")  # 要求显式障碍属于对象数组。
    require_unique_ids(obstacles, "obstacle_field.obstacles")  # 要求显式障碍标识在集合内唯一。
    all_records.extend((terrain, obstacle_field, *obstacles))  # 将地形和障碍对象加入全局标识列表。
    require_unique_ids(all_records, "scene_global_objects")  # 要求全部场景对象共享一个唯一标识空间。


def _validate_service_candidates(service_data: dict[str, Any]) -> None:  # 定义服务列候选结构检查函数。
    """检查服务列候选标识、字段和任务位姿序列结构。"""  # 解释服务列候选检查范围。
    candidates = require_record_list(service_data["candidates"], "candidates")  # 要求服务列候选属于对象数组。
    require_unique_ids(candidates, "candidates")  # 要求服务列候选标识唯一。
    base_candidate_keys = {"id", "name", "launch_stop_id", "recovery_stop_id", "uav_id", "weather_state_id"}  # 定义两类服务列候选共享字段。
    explicit_candidate_keys = base_candidate_keys | {"task_pose_sequence"}  # 定义显式任务位姿候选字段。
    enumerated_candidate_keys = base_candidate_keys | {"task_ids", "enumerate_task_orders"}  # 定义外层有限枚举候选字段。
    sequence_keys = {"task_id", "pose_id"}  # 定义任务位姿序列项允许字段。
    for candidate_index, candidate in enumerate(candidates):  # 逐一检查服务列候选记录。
        actual_keys = set(candidate)  # 读取当前候选实际字段以区分显式和枚举模式。
        if actual_keys == explicit_candidate_keys:  # 判断当前候选是否属于显式任务位姿模式。
            sequence = require_record_list(candidate["task_pose_sequence"], f"candidates[{candidate_index}].task_pose_sequence")  # 要求当前任务位姿序列属于对象数组。
            if not sequence:  # 判断当前服务列候选是否没有任务位姿项。
                raise DataValidationError(f"服务列候选至少需要一个任务位姿项: {candidate['id']}")  # 拒绝不能形成服务的空候选。
            for sequence_index, sequence_item in enumerate(sequence):  # 逐一检查当前候选的任务位姿项。
                require_record_keys(sequence_item, sequence_keys, sequence_keys, f"candidates[{candidate_index}].task_pose_sequence[{sequence_index}]")  # 检查当前任务位姿项字段完整性。
            continue  # 完成显式候选验证并处理下一条候选。
        if actual_keys == enumerated_candidate_keys:  # 判断当前候选是否属于任务顺序和位姿外层枚举模式。
            task_ids = candidate["task_ids"]  # 读取枚举模板声明的任务标识集合。
            if not isinstance(task_ids, list) or not task_ids or any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids):  # 判断任务集合是否属于非空有效字符串数组。
                raise DataValidationError(f"服务列枚举模板至少需要一个有效任务标识: {candidate['id']}")  # 拒绝不能形成离散组合的任务集合。
            if len(task_ids) != len(set(task_ids)):  # 判断枚举模板是否重复声明任务。
                raise DataValidationError(f"服务列枚举模板不能包含重复任务: {candidate['id']}")  # 拒绝重复任务模板。
            if not isinstance(candidate["enumerate_task_orders"], bool):  # 判断任务顺序枚举开关是否为布尔值。
                raise DataValidationError(f"服务列任务顺序枚举开关必须为布尔值: {candidate['id']}")  # 拒绝含糊的枚举控制字段。
            continue  # 完成枚举候选验证并处理下一条候选。
        allowed_keys = explicit_candidate_keys | enumerated_candidate_keys  # 汇总用于错误诊断的全部候选字段。
        require_record_keys(candidate, base_candidate_keys, allowed_keys, f"candidates[{candidate_index}]")  # 优先报告共享字段缺失或未知字段。
        raise DataValidationError(f"服务列候选必须且只能使用 task_pose_sequence 或 task_ids 枚举模式: {candidate['id']}")  # 拒绝混合或缺少离散选择模式的候选。


def parse_time_controls(config: dict[str, Any]) -> tuple[float, float]:
    """Return mission and rolling horizons from exactly one supported form."""
    has_legacy = "time_horizon_seconds" in config
    has_mission = "mission_horizon_seconds" in config
    has_rolling = "rolling_window_seconds" in config
    if has_legacy and (has_mission or has_rolling):
        raise DataValidationError("time controls cannot mix legacy and formal fields")
    if has_legacy:
        horizon = config["time_horizon_seconds"]
        rolling = horizon
    elif has_mission and has_rolling:
        horizon = config["mission_horizon_seconds"]
        rolling = config["rolling_window_seconds"]
    else:
        raise DataValidationError(
            "time controls require time_horizon_seconds or the complete formal pair"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in (horizon, rolling)
    ):
        raise DataValidationError("time controls must be positive finite numbers")
    mission_horizon = float(horizon)
    rolling_window = float(rolling)
    if rolling_window > mission_horizon:
        raise DataValidationError("rolling window cannot exceed mission horizon")
    period_count = mission_horizon / rolling_window
    if not math.isclose(period_count, round(period_count), rel_tol=0.0, abs_tol=1e-9):
        raise DataValidationError("mission horizon must be divisible by rolling window")
    return mission_horizon, rolling_window


def load_project_data(scenario_directory: Path, config_path: Path) -> ProjectData:  # 定义完整项目 Data 读取函数。
    """读取 P0 场景、服务列候选和配置并执行不含决策逻辑的严格检查。"""  # 解释完整 Data 读取边界。
    scene_path = scenario_directory / "scene.json"  # 定位场景对象配置文件。
    service_path = scenario_directory / "service_candidates.json"  # 定位服务列候选配置文件。
    scene = read_json_object(scene_path, SCENE_REQUIRED_KEYS, SCENE_REQUIRED_KEYS)  # 读取并验证场景文件顶层字段。
    service_candidates = read_json_object(service_path, SERVICE_REQUIRED_KEYS, SERVICE_REQUIRED_KEYS)  # 读取并验证服务列候选文件顶层字段。
    config = read_json_object(config_path, CONFIG_COMMON_REQUIRED_KEYS, CONFIG_ALLOWED_KEYS)  # 读取并验证旧单时域或正式双时域配置。
    parse_time_controls(config)  # 在任何场景构造前拒绝混合、缺失或不可整除的时间控制。
    _validate_scene_collections(scene)  # 检查场景集合结构和全局稳定标识。
    _validate_service_candidates(service_candidates)  # 检查服务列候选结构和稳定标识。
    require_finite_numbers(scene, str(scene_path))  # 递归拒绝场景文件中的非有限数值。
    require_finite_numbers(service_candidates, str(service_path))  # 递归拒绝服务列候选文件中的非有限数值。
    require_finite_numbers(config, str(config_path))  # 递归拒绝物理算法配置中的非有限数值。
    return ProjectData(scene, service_candidates, config)  # 返回三份经过结构验证的不可变数据引用。
