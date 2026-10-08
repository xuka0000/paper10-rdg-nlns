"""审计正式基准数据划分的一致性和泄漏风险。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_REQUIRED_KEYS = {
    "scenario_id",
    "split",
    "parent_ids",
    "weather_dates",
    "window_id",
    "event_seed",
    "generator_seed_lineage",
    "scenario_directory",
    "config_path",
}
_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True, slots=True)
class SplitAuditIssue:
    field: str
    left_scenario_id: str
    right_scenario_id: str
    shared_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitAuditReport:
    issues: tuple[SplitAuditIssue, ...]

    @property
    def issue_count(self) -> int:
        return len(self.issues)


def _validate_manifest(manifest: Mapping[str, Any], index: int) -> None:
    missing = _REQUIRED_KEYS - set(manifest)
    if missing:
        raise ValueError(
            f"split manifest {index} missing fields: {sorted(missing)}"
        )
    if not isinstance(manifest["scenario_id"], str) or not manifest[
        "scenario_id"
    ].strip():
        raise ValueError(f"split manifest {index} has invalid scenario_id")
    if manifest["split"] not in _SPLITS:
        raise ValueError(f"split manifest {index} has invalid split")
    for field in ("parent_ids", "weather_dates"):
        value = manifest[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"split manifest {index} has invalid {field}")
    if manifest["window_id"] is not None and (
        not isinstance(manifest["window_id"], str)
        or not manifest["window_id"]
    ):
        raise ValueError(f"split manifest {index} has invalid window_id")
    if isinstance(manifest["event_seed"], bool) or not isinstance(
        manifest["event_seed"], int
    ):
        raise ValueError(f"split manifest {index} has invalid event_seed")
    lineage = manifest["generator_seed_lineage"]
    if not isinstance(lineage, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        for key, value in lineage.items()
    ):
        raise ValueError(
            f"split manifest {index} has invalid generator_seed_lineage"
        )


def _shared(
    left: Mapping[str, Any], right: Mapping[str, Any], field: str
) -> tuple[str, ...]:
    if field in {"parent_ids", "weather_dates"}:
        return tuple(sorted(set(left[field]) & set(right[field])))
    if field == "window_id":
        value = left[field]
        return (str(value),) if value is not None and value == right[field] else ()
    if field == "event_seed":
        return (str(left[field]),) if left[field] == right[field] else ()
    if field == "generator_seed_lineage":
        left_values = {
            f"{key}={value}" for key, value in left[field].items()
        }
        right_values = {
            f"{key}={value}" for key, value in right[field].items()
        }
        return tuple(sorted(left_values & right_values))
    if field == "source_pair":
        left_pair = (left["scenario_directory"], left["config_path"])
        right_pair = (right["scenario_directory"], right["config_path"])
        return (
            (f"{left_pair[0]}|{left_pair[1]}",)
            if left_pair == right_pair
            else ()
        )
    raise KeyError(field)


def collect_split_issues(
    manifests: Iterable[Mapping[str, Any]],
) -> SplitAuditReport:
    rows = tuple(manifests)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"split manifest {index} must be an object")
        _validate_manifest(row, index)
    scenario_ids = [str(row["scenario_id"]) for row in rows]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("split manifests contain duplicate scenario_id")

    fields = (
        "parent_ids",
        "weather_dates",
        "window_id",
        "event_seed",
        "generator_seed_lineage",
        "source_pair",
    )
    issues: list[SplitAuditIssue] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            for field in fields:
                shared_values = _shared(left, right, field)
                if shared_values:
                    issues.append(
                        SplitAuditIssue(
                            field,
                            str(left["scenario_id"]),
                            str(right["scenario_id"]),
                            shared_values,
                        )
                    )
    return SplitAuditReport(tuple(issues))


def audit_split_manifests(
    manifests: Iterable[Mapping[str, Any]],
) -> SplitAuditReport:
    report = collect_split_issues(manifests)
    if report.issues:
        first = report.issues[0]
        raise ValueError(
            f"semantic split leakage in {first.field}: "
            f"{first.left_scenario_id}/{first.right_scenario_id} "
            f"share {first.shared_values}"
        )
    return report
