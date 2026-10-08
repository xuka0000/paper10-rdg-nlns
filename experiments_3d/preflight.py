"""Bounded development admission probe; never consumes formal test outcomes."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import Counter
from pathlib import Path

from Env.inspection_env import InspectionEnv
from experiments_3d.data_upgrade import read_json, sha256, write_once


def evidence_json(value):
    """Preserve unbounded geometric clearance explicitly in strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN in physical admission evidence")
        return {"extended_real": "+infinity" if value > 0 else "-infinity"}
    if isinstance(value, dict):
        return {key: evidence_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [evidence_json(item) for item in value]
    return value


def run(suite_path: Path, output: Path, continuous: bool = False, scenario_id: str | None = None) -> dict:
    suite = read_json(suite_path)
    entry = next(e for e in suite["scenarios"] if e["split"] == "train" and (e["scenario_id"] == scenario_id if scenario_id else "n50-c-s101" in e["scenario_id"]))
    source = suite_path.parent / entry["scenario_directory"]
    scene = read_json(source / "scene.json")
    config = suite_path.parent / entry["config_path"]
    towers = {t["id"] for t in scene["towers"][:3]}
    components = {c["id"] for c in scene["components"] if c["tower_id"] in towers}
    checkpoints = {c["id"] for c in scene["checkpoints"] if c["component_id"] in components}
    for key, allowed in (("components", components), ("checkpoints", checkpoints)):
        scene[key] = [r for r in scene[key] if r["id"] in allowed]
    scene["tasks"] = [t for t in scene["tasks"] if t["checkpoint_id"] in checkpoints]
    scene["observation_poses"] = [p for p in scene["observation_poses"] if p["checkpoint_id"] in checkpoints]
    tasks = {t["id"] for t in scene["tasks"]}
    candidates = read_json(source / "service_candidates.json")
    ids = lambda c: c.get("task_ids") or [n["task_id"] for n in c["task_pose_sequence"]]
    candidates["candidates"] = [c for c in candidates["candidates"] if all(t in tasks for t in ids(c))]
    if continuous:
        # Interface cost probe: deterministic first candidate of each cardinality,
        # selected BEFORE physics results. This is not a coverage benchmark.
        candidates["candidates"] = [next(c for c in candidates["candidates"] if len(ids(c)) == k) for k in (1, 2, 3)]
        # One preselected v0 realization per cardinality for this timing probe.
        for c in candidates["candidates"]:
            if "task_ids" in c:
                c["task_pose_sequence"] = [{"task_id": t, "pose_id": t.replace("task-", "pose-", 1) + "-v0"} for t in c.pop("task_ids")]
                c.pop("enumerate_task_orders")
    write_once(output / "scene.json", scene)
    write_once(output / "service_candidates.json", candidates)
    write_once(output / "input_identity.json", {"source": entry, "scene_sha256": sha256(source / "scene.json"), "config_sha256": sha256(config),
        "tower_ids": sorted(towers), "purpose": "development geometry and continuous-optimizer integration", "continuous": continuous})
    optimizer = None
    if continuous:
        from Agent.optimization.scipy_trajectory_agent import ScipyTrajectoryAgent
        optimizer = ScipyTrajectoryAgent.from_config(config)
    started = time.perf_counter()
    env = InspectionEnv.from_directory(output, config, trajectory_optimizer=optimizer)
    snapshot = env.snapshot()
    columns = snapshot.service_columns
    covered = {t for c in columns for t in c.task_ids}
    report = {"scope": "development_only", "continuous_optimizer": continuous, "elapsed_seconds": time.perf_counter() - started,
        "source_scenario_id": entry["scenario_id"], "task_count": len(tasks), "candidate_count": len(candidates["candidates"]),
        "admitted_column_count": len(columns), "covered_task_count": len(covered), "uncovered_task_ids": sorted(tasks - covered),
        "admitted_cardinalities": dict(Counter(len(c.task_ids) for c in columns)),
        "observation_assessments": [dataclasses.asdict(a) for a in snapshot.observation_assessments],
        "rejections": [dataclasses.asdict(r) for r in snapshot.rejection_records],
        "columns": [{"id": c.object_id, "task_ids": c.task_ids, "pose_ids": c.observation_pattern.pose_ids,
                     "duration_s": c.dynamic_duration, "energy_j": c.energy_mean, "trajectory_status": c.trajectory_status} for c in columns]}
    write_once(output / "report.json", evidence_json(report))
    print(json.dumps({k: v for k, v in report.items() if k not in {"columns", "rejections", "observation_assessments"}}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--scenario", help="Explicit training scene only; final test inputs are forbidden")
    args = parser.parse_args()
    run(args.suite, args.output, args.continuous, args.scenario)
