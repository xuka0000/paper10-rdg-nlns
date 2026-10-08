"""将正式基准对象写入可执行场景文件。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from Env.benchmark.source_catalog import sha256_file
from Env.data_loader import load_project_data


class ScenarioWriteError(ValueError):
    """Raised when a scenario cannot be validated or promoted atomically."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(value))


def _validate_hashes(values: Mapping[str, str], context: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ScenarioWriteError(f"invalid SHA-256 in {context}: {key!r}")
        normalized[key] = value
    return normalized


def _request_fingerprint(
    scenario_id: str,
    scene: Mapping[str, Any],
    service_candidates: Mapping[str, Any],
    rolling_events: Mapping[str, Any],
    config_path: Path,
    lineage: Mapping[str, Any],
    public_source_hashes: Mapping[str, str],
    public_source_manifest_sha256: str | None,
    generation_diagnostics: Mapping[str, Any],
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "scene_sha256": hashlib.sha256(_canonical_bytes(scene)).hexdigest(),
        "service_candidates_sha256": hashlib.sha256(
            _canonical_bytes(service_candidates)
        ).hexdigest(),
        "rolling_events_sha256": hashlib.sha256(
            _canonical_bytes(rolling_events)
        ).hexdigest(),
        "config_sha256": sha256_file(config_path),
        "lineage": {
            key: lineage[key]
            for key in (
                "split",
                "parent_ids",
                "weather_dates",
                "window_id",
                "event_seed",
                "generator_seed_lineage",
            )
        },
        "public_source_hashes": dict(sorted(public_source_hashes.items())),
        "public_source_manifest_sha256": public_source_manifest_sha256,
        "generation_diagnostics": generation_diagnostics,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _verify_existing(target: Path, fingerprint: str) -> Path:
    manifest_path = target / "generation_manifest.json"
    if not manifest_path.is_file():
        raise ScenarioWriteError(
            f"existing scenario has no generation manifest: {target}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioWriteError(
            f"cannot read existing generation manifest: {manifest_path}"
        ) from exc
    if manifest.get("request_fingerprint") != fingerprint:
        raise ScenarioWriteError(
            f"existing scenario has a different manifest: {target.name}"
        )
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ScenarioWriteError("existing generation manifest has no artifact hashes")
    for name in ("scene.json", "service_candidates.json", "rolling_events.json"):
        if sha256_file(target / name) != artifact_hashes.get(name):
            raise ScenarioWriteError(f"existing scenario artifact hash mismatch: {name}")
    return target


def write_scenario(
    scenario_root: Path,
    scenario_id: str,
    scene: Mapping[str, Any],
    service_candidates: Mapping[str, Any],
    rolling_events: Mapping[str, Any],
    config_path: Path,
    lineage: Mapping[str, Any],
    *,
    public_source_hashes: Mapping[str, str],
    public_source_manifest_path: Path | None = None,
    generation_diagnostics: Mapping[str, Any] | None = None,
) -> Path:
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ScenarioWriteError("scenario_id cannot be empty")
    if scene.get("scene_id") != scenario_id:
        raise ScenarioWriteError("scene_id must match scenario_id")
    required_lineage = {
        "split",
        "parent_ids",
        "weather_dates",
        "window_id",
        "event_seed",
        "generator_seed_lineage",
    }
    missing = required_lineage - set(lineage)
    if missing:
        raise ScenarioWriteError(f"lineage missing fields: {sorted(missing)}")
    source_hashes = _validate_hashes(
        public_source_hashes, "public_source_hashes"
    )
    diagnostics = dict(generation_diagnostics or {})
    retained_source_manifest = (
        Path(public_source_manifest_path).resolve()
        if public_source_manifest_path is not None
        else None
    )
    if source_hashes and (
        retained_source_manifest is None or not retained_source_manifest.is_file()
    ):
        raise ScenarioWriteError(
            "public source hashes require an existing source manifest"
        )
    if not source_hashes and retained_source_manifest is not None:
        raise ScenarioWriteError(
            "public source manifest is invalid without public source hashes"
        )
    retained_source_manifest_sha256 = (
        sha256_file(retained_source_manifest)
        if retained_source_manifest is not None
        else None
    )
    source_config = Path(config_path).resolve()
    if not source_config.is_file():
        raise ScenarioWriteError(f"config file does not exist: {source_config}")
    fingerprint = _request_fingerprint(
        scenario_id,
        scene,
        service_candidates,
        rolling_events,
        source_config,
        lineage,
        source_hashes,
        retained_source_manifest_sha256,
        diagnostics,
    )

    root = Path(scenario_root)
    target = root / scenario_id
    staging = root / f"{scenario_id}.staging"
    if target.exists():
        return _verify_existing(target, fingerprint)
    if staging.exists():
        raise ScenarioWriteError(f"staging directory already exists: {staging}")
    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _write_json(staging / "scene.json", scene)
    _write_json(staging / "service_candidates.json", service_candidates)
    _write_json(staging / "rolling_events.json", rolling_events)
    try:
        load_project_data(staging, source_config)
    except (ValueError, OSError) as exc:
        raise ScenarioWriteError(
            f"staged scenario failed production data validation: {scenario_id}"
        ) from exc

    artifact_hashes = {
        name: sha256_file(staging / name)
        for name in (
            "scene.json",
            "service_candidates.json",
            "rolling_events.json",
        )
    }
    artifact_hashes["config"] = sha256_file(source_config)
    generation_manifest = {
        "schema_version": "generation-manifest-v1",
        "scenario_id": scenario_id,
        "split": lineage["split"],
        "parent_ids": lineage["parent_ids"],
        "weather_dates": lineage["weather_dates"],
        "window_id": lineage["window_id"],
        "event_seed": lineage["event_seed"],
        "generator_seed_lineage": lineage["generator_seed_lineage"],
        "scenario_directory": str(target.resolve()),
        "config_path": str(source_config),
        "public_source_hashes": source_hashes,
        "public_source_manifest_path": (
            str(retained_source_manifest)
            if retained_source_manifest is not None
            else None
        ),
        "public_source_manifest_sha256": retained_source_manifest_sha256,
        "generation_diagnostics": diagnostics,
        "artifact_hashes": artifact_hashes,
        "request_fingerprint": fingerprint,
    }
    _write_json(staging / "generation_manifest.json", generation_manifest)
    staging.replace(target)
    return target
