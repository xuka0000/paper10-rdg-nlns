"""定义正式基准配置并执行结构校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "formal-benchmark-v1"
_EXPECTED_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "generator_version",
        "expected_counts",
        "seeds",
        "families",
        "algorithm_seeds",
        "robustness_seeds",
        "time_controls",
        "resources",
        "stop_controls",
        "weather",
        "rolling_regimes",
        "regions",
        "required_source_ids",
    }
)
_EXPECTED_SEED_KEYS = frozenset(
    {
        "controlled_train",
        "controlled_validation",
        "controlled_test",
        "exact",
        "gis_windows",
        "scale",
    }
)
_LOCKED_SEEDS = {
    "controlled_train": (101, 103, 107, 109),
    "controlled_validation": (211,),
    "controlled_test": (307, 311),
    "exact": (401, 409, 419),
    "gis_windows": (503, 509),
    "scale": (601, 607),
}
_LOCKED_FAMILIES = {
    "exact_small": {
        "split": "test",
        "tower_counts": [5, 7, 9],
        "patterns": ["R", "C", "RC"],
        "seed_group": "exact",
    },
    "controlled_main": {
        "splits": ["train", "validation", "test"],
        "tower_counts": [50, 100, 150],
        "patterns": ["R", "C", "RC"],
        "seed_groups": {
            "train": "controlled_train",
            "validation": "controlled_validation",
            "test": "controlled_test",
        },
    },
    "gis_external": {
        "split": "test",
        "tower_counts": [50, 100, 150],
        "pattern": "GIS",
        "seed_group": "gis_windows",
        "regions": ["bay_area", "dallas_fort_worth", "los_angeles_inland"],
    },
    "scale_transfer": {
        "split": "test",
        "tower_counts": [200, 300],
        "patterns": ["R", "C", "RC"],
        "seed_group": "scale",
    },
}
_LOCKED_COUNTS = {
    "total": 120,
    "splits": {"train": 36, "validation": 9, "test": 75},
    "families": {
        "exact_small": 27,
        "controlled_main": 63,
        "gis_external": 18,
        "scale_transfer": 12,
    },
}
_LOCKED_ALGORITHM_SEEDS = (11, 23, 37, 53, 71)
_LOCKED_ROBUSTNESS_SEEDS = (
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
)


@dataclass(frozen=True, slots=True)
class FormalCell:
    family_id: str
    split: str
    tower_count: int
    pattern: str
    geometry_seed: int
    region_id: str | None = None
    window_index: int | None = None

    @property
    def cell_id(self) -> str:
        parts = [
            self.family_id,
            self.split,
            f"n{self.tower_count}",
            self.pattern.lower(),
        ]
        if self.region_id is not None:
            parts.append(self.region_id)
        if self.window_index is not None:
            parts.append(f"w{self.window_index}")
        parts.append(f"s{self.geometry_seed}")
        return "-".join(parts)


@dataclass(frozen=True, slots=True)
class FormalBenchmarkSpec:
    controlled_train_seeds: tuple[int, ...]
    controlled_validation_seeds: tuple[int, ...]
    controlled_test_seeds: tuple[int, ...]
    exact_seeds: tuple[int, ...]
    gis_window_seeds: tuple[int, ...]
    scale_seeds: tuple[int, ...]
    algorithm_seeds: tuple[int, ...]
    robustness_seeds: tuple[int, ...]

    def enumerate_cells(self) -> tuple[FormalCell, ...]:
        cells: list[FormalCell] = []
        for tower_count in (5, 7, 9):
            for pattern in ("R", "C", "RC"):
                for seed in self.exact_seeds:
                    cells.append(
                        FormalCell(
                            "exact_small", "test", tower_count, pattern, seed
                        )
                    )

        controlled_groups = (
            ("train", self.controlled_train_seeds),
            ("validation", self.controlled_validation_seeds),
            ("test", self.controlled_test_seeds),
        )
        for split, seeds in controlled_groups:
            for tower_count in (50, 100, 150):
                for pattern in ("R", "C", "RC"):
                    for seed in seeds:
                        cells.append(
                            FormalCell(
                                "controlled_main",
                                split,
                                tower_count,
                                pattern,
                                seed,
                            )
                        )

        regions = ("bay_area", "dallas_fort_worth", "los_angeles_inland")
        for region_id in regions:
            for tower_count in (50, 100, 150):
                for window_index, seed in enumerate(self.gis_window_seeds):
                    cells.append(
                        FormalCell(
                            "gis_external",
                            "test",
                            tower_count,
                            "GIS",
                            seed,
                            region_id,
                            window_index,
                        )
                    )

        for tower_count in (200, 300):
            for pattern in ("R", "C", "RC"):
                for seed in self.scale_seeds:
                    cells.append(
                        FormalCell(
                            "scale_transfer",
                            "test",
                            tower_count,
                            pattern,
                            seed,
                        )
                    )
        return tuple(cells)


def _require_exact_keys(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details: list[str] = []
        if unknown:
            details.append(f"unknown keys {unknown}")
        if missing:
            details.append(f"missing keys {missing}")
        raise ValueError(f"{name}: {'; '.join(details)}")


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{name} must contain nonnegative integers")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate seeds")
    return result


def _require_locked(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} differs from the locked formal_v1 contract")


def _validate_supporting_contract(raw: dict[str, Any]) -> None:
    _require_locked(raw["benchmark_id"], "formal_v1", "benchmark_id")
    _require_locked(raw["generator_version"], "formal-v1", "generator_version")
    _require_locked(raw["expected_counts"], _LOCKED_COUNTS, "expected_counts")
    _require_locked(raw["families"], _LOCKED_FAMILIES, "families")
    _require_locked(
        raw["time_controls"],
        {"mission_horizon_seconds": 28800.0, "rolling_window_seconds": 600.0},
        "time_controls",
    )
    _require_locked(
        raw["resources"],
        {
            "exact_small": {"vehicle_count": 1, "uavs_per_vehicle": 2},
            "controlled_main": {"vehicle_count": 2, "uavs_per_vehicle": 3},
            "gis_external": {"vehicle_count": 2, "uavs_per_vehicle": 3},
            "scale_transfer": {"vehicle_count": 3, "uavs_per_vehicle": 3},
            "vehicle_cruise_speed_mps": 10.0,
            "uav_max_horizontal_speed_mps": 12.0,
            "observation_speed_cap_mps": 0.5,
        },
        "resources",
    )
    _require_locked(
        raw["stop_controls"],
        {
            "maximum_slope_deg": 5.0,
            "maximum_roughness_m": 0.2,
            "footprint_radius_m": 2.0,
            "minimum_usable_area_m2": 10.0,
            "boundary_margin_m": 2.0,
            "obstacle_clearance_m": 2.0,
        },
        "stop_controls",
    )
    required_source_ids = raw["required_source_ids"]
    if not isinstance(required_source_ids, list) or len(required_source_ids) != len(
        set(required_source_ids)
    ):
        raise ValueError("required_source_ids must be a unique list")


def _validate_enumeration(spec: FormalBenchmarkSpec) -> None:
    cells = spec.enumerate_cells()
    if len(cells) != _LOCKED_COUNTS["total"]:
        raise ValueError("formal benchmark must enumerate exactly 120 cells")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("formal benchmark cell IDs must be unique")
    for split, expected in _LOCKED_COUNTS["splits"].items():
        actual = sum(cell.split == split for cell in cells)
        if actual != expected:
            raise ValueError(f"split {split} count {actual} does not equal {expected}")
    for family_id, expected in _LOCKED_COUNTS["families"].items():
        actual = sum(cell.family_id == family_id for cell in cells)
        if actual != expected:
            raise ValueError(
                f"family {family_id} count {actual} does not equal {expected}"
            )


def load_formal_spec(path: Path) -> FormalBenchmarkSpec:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load formal benchmark config {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("formal benchmark config must be a JSON object")
    _require_exact_keys(raw, _EXPECTED_ROOT_KEYS, "formal benchmark config")
    if raw["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
    if not isinstance(raw["seeds"], dict):
        raise ValueError("seeds must be an object")
    _require_exact_keys(raw["seeds"], _EXPECTED_SEED_KEYS, "seeds")

    parsed_seeds = {
        name: _integer_tuple(raw["seeds"][name], f"seeds.{name}")
        for name in _EXPECTED_SEED_KEYS
    }
    controlled_sets = (
        set(parsed_seeds["controlled_train"]),
        set(parsed_seeds["controlled_validation"]),
        set(parsed_seeds["controlled_test"]),
    )
    if any(
        not left.isdisjoint(right)
        for index, left in enumerate(controlled_sets)
        for right in controlled_sets[index + 1 :]
    ):
        raise ValueError("controlled split seeds must be disjoint")
    for name, expected in _LOCKED_SEEDS.items():
        _require_locked(parsed_seeds[name], expected, f"seeds.{name}")

    algorithm_seeds = _integer_tuple(raw["algorithm_seeds"], "algorithm_seeds")
    robustness_seeds = _integer_tuple(raw["robustness_seeds"], "robustness_seeds")
    _require_locked(algorithm_seeds, _LOCKED_ALGORITHM_SEEDS, "algorithm_seeds")
    _require_locked(
        robustness_seeds, _LOCKED_ROBUSTNESS_SEEDS, "robustness_seeds"
    )
    _validate_supporting_contract(raw)

    spec = FormalBenchmarkSpec(
        controlled_train_seeds=parsed_seeds["controlled_train"],
        controlled_validation_seeds=parsed_seeds["controlled_validation"],
        controlled_test_seeds=parsed_seeds["controlled_test"],
        exact_seeds=parsed_seeds["exact"],
        gis_window_seeds=parsed_seeds["gis_windows"],
        scale_seeds=parsed_seeds["scale"],
        algorithm_seeds=algorithm_seeds,
        robustness_seeds=robustness_seeds,
    )
    _validate_enumeration(spec)
    return spec
