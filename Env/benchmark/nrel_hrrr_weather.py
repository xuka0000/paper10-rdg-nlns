"""清洗 NREL SRRL/BMS 风观测并构造第五章5.1节的 HRRR 校准天气场景。"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Iterable, Mapping, Sequence

import numpy as np

from Mask.geometry import Vector3
from Mask.weather import WeatherScenario, WeatherScenarioSet, WeatherState


HEIGHTS_FT = (6, 19, 22, 33)
SPLITS = ("train", "validation", "test")
BANDS = ("low", "medium", "high")
FORBIDDEN_FIELD_TOKENS = ("solar", "global_cmp", "direct_", "diffuse_", "uvb")


@dataclass(frozen=True, slots=True)
class NrelMinuteRow:
    """保存一条经过缺测和物理范围检查的 NREL 分钟风观测。"""

    timestamp_mst: datetime
    speeds_mps: tuple[float, float, float, float]
    directions_deg: tuple[float, float, float, float]

    @property
    def speed_33ft_mps(self) -> float:
        """返回约10 m高度的33 ft平均风速。"""
        return self.speeds_mps[-1]

    @property
    def direction_33ft_deg(self) -> float:
        """返回约10 m高度的33 ft来流风向。"""
        return self.directions_deg[-1]


@dataclass(frozen=True, slots=True)
class NrelWindWindow:
    """保存固定时间窗内多高度风速和圆统计风向。"""

    window_start_mst: datetime
    split: str
    sample_count: int
    speeds_mps: tuple[float, float, float, float]
    directions_deg: tuple[float, float, float, float]

    @property
    def year(self) -> int:
        """返回时间窗所属年份。"""
        return self.window_start_mst.year

    @property
    def season(self) -> str:
        """按气象季节返回时间窗类别。"""
        month = self.window_start_mst.month
        if month in (12, 1, 2):
            return "DJF"
        if month in (3, 4, 5):
            return "MAM"
        if month in (6, 7, 8):
            return "JJA"
        return "SON"

    @property
    def speed_33ft_mps(self) -> float:
        """返回时间窗的33 ft平均风速。"""
        return self.speeds_mps[-1]

    @property
    def direction_33ft_deg(self) -> float:
        """返回时间窗的33 ft圆平均来流风向。"""
        return self.directions_deg[-1]


@dataclass(frozen=True, slots=True)
class WeatherScenarioRow:
    """保存一个数据校准天气实现及其条件概率。"""

    split: str
    band: str
    scenario_id: str
    probability: float
    wind_u_mps: float
    wind_v_mps: float
    ambient_temperature_c: float
    nrel_sample_count: int
    nrel_speed_q50_mps: float
    nrel_speed_q80_mps: float
    hrrr_u_error_mps: float
    hrrr_v_error_mps: float
    hrrr_temperature_error_c: float


def circular_mean_degrees(values: Iterable[float]) -> float:
    """计算以正北为零度的风向圆平均值。"""
    angles = tuple(float(value) for value in values)
    if not angles:
        raise ValueError("circular mean requires at least one direction")
    if any(not math.isfinite(value) or value < 0.0 or value >= 360.0 for value in angles):
        raise ValueError("wind directions must be finite values in [0, 360)")
    radians = np.deg2rad(np.asarray(angles, dtype=float))
    sine = float(np.sin(radians).mean())
    cosine = float(np.cos(radians).mean())
    if abs(sine) < 1e-15 and abs(cosine) < 1e-15:
        raise ValueError("circular mean is undefined for cancelling directions")
    result = math.degrees(math.atan2(sine, cosine)) % 360.0
    return 0.0 if math.isclose(result, 360.0, abs_tol=1e-12) else result


def _split_for_year(year: int) -> str:
    """按照时间外推协议把 NREL 年份映射到训练、验证或测试集。"""
    if 2020 <= year <= 2022:
        return "train"
    if year == 2023:
        return "validation"
    if year == 2024:
        return "test"
    raise ValueError(f"NREL year is outside the declared 2020-2024 release: {year}")


def _finite_float(value: object) -> float | None:
    """把合法数值转换为浮点数并把 NREL 缺测码转换为空值。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= -99990.0:
        return None
    return number


def read_nrel_minute_rows(paths: Iterable[Path]) -> tuple[NrelMinuteRow, ...]:
    """读取作者下载的 NREL 原始 CSV，只保留完整的多高度风观测。"""
    rows: list[NrelMinuteRow] = []
    for source in sorted(Path(path) for path in paths):
        if not source.is_file():
            raise ValueError(f"NREL source does not exist: {source}")
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "DATE (MM/DD/YYYY)",
                "MST",
                *(f"Avg Wind Speed @ {height}ft [m/s]" for height in HEIGHTS_FT),
                *(f"Avg Wind Direction @ {height}ft [deg from N]" for height in HEIGHTS_FT),
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                missing = sorted(required.difference(reader.fieldnames or ()))
                raise ValueError(f"NREL source is missing wind fields: {missing}")
            for record in reader:
                speeds = tuple(
                    _finite_float(record[f"Avg Wind Speed @ {height}ft [m/s]"])
                    for height in HEIGHTS_FT
                )
                directions = tuple(
                    _finite_float(record[f"Avg Wind Direction @ {height}ft [deg from N]"])
                    for height in HEIGHTS_FT
                )
                if any(value is None for value in speeds + directions):
                    continue
                numeric_speeds = tuple(float(value) for value in speeds)
                numeric_directions = tuple(float(value) for value in directions)
                if any(value < 0.0 for value in numeric_speeds):
                    continue
                if any(value < 0.0 or value >= 360.0 for value in numeric_directions):
                    continue
                try:
                    timestamp = datetime.strptime(
                        f"{record['DATE (MM/DD/YYYY)']} {record['MST']}",
                        "%m/%d/%Y %H:%M",
                    )
                except ValueError as exc:
                    raise ValueError(f"invalid NREL timestamp in {source}: {record}") from exc
                _split_for_year(timestamp.year)
                rows.append(NrelMinuteRow(timestamp, numeric_speeds, numeric_directions))
    rows.sort(key=lambda item: item.timestamp_mst)
    return tuple(rows)


def aggregate_nrel_windows(
    rows: Iterable[NrelMinuteRow], *, minutes: int = 15
) -> tuple[NrelWindWindow, ...]:
    """把分钟观测聚合为固定时间窗并对风向使用圆统计。"""
    if minutes <= 0 or 60 % minutes != 0:
        raise ValueError("aggregation minutes must be a positive divisor of 60")
    grouped: dict[datetime, list[NrelMinuteRow]] = defaultdict(list)
    for row in rows:
        minute = (row.timestamp_mst.minute // minutes) * minutes
        grouped[row.timestamp_mst.replace(minute=minute, second=0, microsecond=0)].append(row)
    windows: list[NrelWindWindow] = []
    for window_start in sorted(grouped):
        items = grouped[window_start]
        speeds = tuple(fmean(item.speeds_mps[index] for item in items) for index in range(4))
        directions = tuple(
            circular_mean_degrees(item.directions_deg[index] for item in items)
            for index in range(4)
        )
        windows.append(
            NrelWindWindow(
                window_start,
                _split_for_year(window_start.year),
                len(items),
                speeds,
                directions,
            )
        )
    return tuple(windows)


def _wind_components(speed_mps: float, direction_from_deg: float) -> tuple[float, float]:
    """把气象来流风向转换为东向和北向速度分量。"""
    radians = math.radians(direction_from_deg)
    return -speed_mps * math.sin(radians), -speed_mps * math.cos(radians)


def _load_hrrr_temperature_medians(hrrr_root: Path) -> dict[str, float]:
    """从 HRRR 分析记录计算各数据划分的环境温度中位数。"""
    values: dict[str, list[float]] = defaultdict(list)
    path = Path(hrrr_root) / "hrrr_forecast_analysis.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            split = str(row["split"])
            if split in SPLITS:
                value = float(row["temperature_c"])
                if math.isfinite(value):
                    values[split].append(value)
    missing = [split for split in SPLITS if not values[split]]
    if missing:
        raise ValueError(f"HRRR analysis lacks ambient temperature for splits: {missing}")
    return {split: float(median(values[split])) for split in SPLITS}


def build_weather_scenario_rows(
    windows: Sequence[NrelWindWindow], hrrr_root: Path
) -> tuple[WeatherScenarioRow, ...]:
    """组合 NREL 风况代表值、HRRR 环境温度、残差和概率。"""
    train_speeds = np.asarray(
        [window.speed_33ft_mps for window in windows if window.split == "train"],
        dtype=float,
    )
    if train_speeds.size < 3:
        raise ValueError("NREL training split requires at least three wind windows")
    q50, q80 = (float(value) for value in np.quantile(train_speeds, (0.5, 0.8)))
    summary_path = Path(hrrr_root) / "hrrr_band_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    temperatures = _load_hrrr_temperature_medians(Path(hrrr_root))
    result: list[WeatherScenarioRow] = []
    for split in SPLITS:
        split_windows = [window for window in windows if window.split == split]
        if not split_windows:
            continue
        probabilities = summary["probabilities"][split]
        representatives = summary["representatives"][split]
        band_windows = {
            "low": [window for window in split_windows if window.speed_33ft_mps <= q50],
            "medium": [
                window for window in split_windows if q50 < window.speed_33ft_mps <= q80
            ],
            "high": [window for window in split_windows if window.speed_33ft_mps > q80],
        }
        for band in BANDS:
            selected = band_windows[band]
            if not selected:
                raise ValueError(f"NREL split/band has no wind windows: {split}/{band}")
            components = [
                _wind_components(window.speed_33ft_mps, window.direction_33ft_deg)
                for window in selected
            ]
            residual = representatives[band]
            u_error = float(residual["median_u_error_mps"])
            v_error = float(residual["median_v_error_mps"])
            temperature_error = float(residual["median_temperature_error_c"])
            result.append(
                WeatherScenarioRow(
                    split=split,
                    band=band,
                    scenario_id=f"{split}-{band}",
                    probability=float(probabilities[band]),
                    wind_u_mps=float(median(item[0] for item in components)) + u_error,
                    wind_v_mps=float(median(item[1] for item in components)) + v_error,
                    ambient_temperature_c=temperatures[split] + temperature_error,
                    nrel_sample_count=len(selected),
                    nrel_speed_q50_mps=q50,
                    nrel_speed_q80_mps=q80,
                    hrrr_u_error_mps=u_error,
                    hrrr_v_error_mps=v_error,
                    hrrr_temperature_error_c=temperature_error,
                )
            )
    validate_weather_release(result)
    return tuple(result)


def _record_mapping(value: object) -> dict[str, object]:
    """把数据类或映射统一转换为可验证字段字典。"""
    if is_dataclass(value):
        return dict(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise ValueError("weather release rows must be dataclasses or mappings")


def validate_weather_release(rows: Iterable[object]) -> None:
    """验证场景标识、数值、禁止字段和各划分内的概率闭合。"""
    records = [_record_mapping(row) for row in rows]
    if not records:
        raise ValueError("weather release must contain at least one scenario")
    required = {
        "split",
        "band",
        "scenario_id",
        "probability",
        "wind_u_mps",
        "wind_v_mps",
        "ambient_temperature_c",
        "nrel_sample_count",
    }
    identifiers: set[str] = set()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        forbidden = [
            key
            for key in record
            if any(token in key.lower() for token in FORBIDDEN_FIELD_TOKENS)
        ]
        if forbidden:
            raise ValueError(f"weather release contains forbidden fields: {forbidden}")
        missing = required.difference(record)
        if missing:
            raise ValueError(f"weather release row lacks required fields: {sorted(missing)}")
        split = str(record["split"])
        band = str(record["band"])
        identifier = str(record["scenario_id"])
        if split not in SPLITS or band not in BANDS:
            raise ValueError(f"invalid weather split/band: {split}/{band}")
        if identifier in identifiers:
            raise ValueError(f"duplicate weather scenario identifier: {identifier}")
        identifiers.add(identifier)
        numeric = (
            float(record["probability"]),
            float(record["wind_u_mps"]),
            float(record["wind_v_mps"]),
            float(record["ambient_temperature_c"]),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"weather scenario contains nonfinite values: {identifier}")
        if numeric[0] <= 0.0 or int(record["nrel_sample_count"]) <= 0:
            raise ValueError(f"weather scenario has invalid probability or sample count: {identifier}")
        grouped[split].append(record)
    for split, split_rows in grouped.items():
        if {str(row["band"]) for row in split_rows} != set(BANDS):
            raise ValueError(f"weather split must contain low, medium, and high bands: {split}")
        probability_sum = sum(float(row["probability"]) for row in split_rows)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"weather probabilities do not sum to one for {split}: {probability_sum}")


def load_weather_scenario_set(
    scenario_path: Path, split: str, period_index: int, epoch_id: str
) -> WeatherScenarioSet:
    """从正式发布表构造某一数据划分和滚动时段的三实现气象集。"""
    if split not in SPLITS:
        raise ValueError(f"weather scenario split is invalid: {split}")
    if period_index < 0:
        raise ValueError("weather scenario period index cannot be negative")
    with Path(scenario_path).open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("split") == split]
    validate_weather_release(rows)
    scenarios = []
    for row in sorted(rows, key=lambda item: BANDS.index(str(item["band"]))):
        base_id = str(row["scenario_id"])
        state = WeatherState(
            f"{base_id}-p{period_index:02d}",
            f"{split} {row['band']} weather period {period_index}",
            period_index,
            Vector3(float(row["wind_u_mps"]), float(row["wind_v_mps"]), 0.0),
            float(row["ambient_temperature_c"]),
        )
        scenarios.append(WeatherScenario(state, float(row["probability"]), base_id))
    return WeatherScenarioSet(epoch_id, tuple(scenarios))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """以稳定字段顺序写入正式天气发布表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise ValueError(f"stale weather release partial exists: {temporary}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_weather_release(
    nrel_source_root: Path, hrrr_root: Path, output_root: Path
) -> dict[str, Path]:
    """生成5.1场景表、多高度风窗表和不依赖文件哈希的数据说明清单。"""
    nrel_root = Path(nrel_source_root)
    sources = tuple(sorted(nrel_root.glob("20[0-2][0-4]0101.csv")))
    expected_names = {f"{year}0101.csv" for year in range(2020, 2025)}
    if {path.name for path in sources} != expected_names:
        raise ValueError(
            f"NREL release requires the five 2020-2024 CSV files: {sorted(expected_names)}"
        )
    minute_rows = read_nrel_minute_rows(sources)
    windows = aggregate_nrel_windows(minute_rows, minutes=15)
    scenarios = build_weather_scenario_rows(windows, Path(hrrr_root))
    if {row.split for row in scenarios} != set(SPLITS):
        raise ValueError("formal weather release must cover train, validation, and test")

    output = Path(output_root)
    scenario_path = output / "section5_1_weather_scenarios.csv"
    wind_path = output / "nrel_wind_windows.csv"
    manifest_path = output / "weather_release_manifest.json"
    scenario_fields = tuple(WeatherScenarioRow.__dataclass_fields__)
    _write_csv(scenario_path, scenario_fields, (asdict(row) for row in scenarios))

    wind_fields = (
        "window_start_mst",
        "split",
        "year",
        "season",
        "sample_count",
        *(f"speed_{height}ft_mps" for height in HEIGHTS_FT),
        *(f"direction_{height}ft_deg" for height in HEIGHTS_FT),
    )
    wind_rows = []
    for window in windows:
        row: dict[str, object] = {
            "window_start_mst": window.window_start_mst.isoformat(timespec="minutes"),
            "split": window.split,
            "year": window.year,
            "season": window.season,
            "sample_count": window.sample_count,
        }
        row.update(
            {
                f"speed_{height}ft_mps": window.speeds_mps[index]
                for index, height in enumerate(HEIGHTS_FT)
            }
        )
        row.update(
            {
                f"direction_{height}ft_deg": window.directions_deg[index]
                for index, height in enumerate(HEIGHTS_FT)
            }
        )
        wind_rows.append(row)
    _write_csv(wind_path, wind_fields, wind_rows)

    manifest = {
        "schema_version": "section5-1-nrel-hrrr-weather-v1",
        "source_provenance": {
            "nrel_institution": "National Renewable Energy Laboratory",
            "nrel_facility": "Solar Radiation Research Laboratory Baseline Measurement System",
            "nrel_station_latitude_deg_n": 39.742,
            "nrel_station_longitude_deg_w": 105.180,
            "nrel_station_elevation_m_amsl": 1828.8,
            "nrel_download_method": "manual",
            "nrel_observation_period": "2020-01-01 to 2024-05-26",
            "nrel_native_interval": "1 minute",
            "hrrr_provider": "NOAA High-Resolution Rapid Refresh",
        },
        "processing": {
            "aggregation_interval_minutes": 15,
            "direction_aggregation": "circular_mean",
            "model_wind_height_ft": 33,
            "nrel_split": {"train": "2020-2022", "validation": "2023", "test": "2024"},
            "hrrr_split": {"train": "2022", "validation": "2023", "test": "2024"},
            "excluded_variables": ["solar radiation", "UVB-1 Temp"],
        },
        "artifacts": {
            scenario_path.name: {"row_count": len(scenarios)},
            wind_path.name: {"row_count": len(windows)},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"scenarios": scenario_path, "wind_windows": wind_path, "manifest": manifest_path}
