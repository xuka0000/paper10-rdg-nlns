"""Create a deterministic, separately versioned multicomponent benchmark.

This constructs candidate inputs, not physically validated results. GIS roads,
terrain and tower coordinates are preserved. Component geometry is a template.
No task is removed based on a planner's admission or objective value.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import itertools
from collections import Counter
from pathlib import Path

from Env.benchmark.scenario_writer import write_scenario
from Env.benchmark.split_audit import audit_split_manifests

VERSION = "multicomponent_3d_v2_20260928"
LINEAGE_KEYS = ("split", "parent_ids", "weather_dates", "window_id", "event_seed", "generator_seed_lineage")
PRESERVED_KEYS = ("terrain", "obstacle_field", "road_nodes", "road_legs", "stop_sites", "towers", "vehicles", "uavs", "weather_states")
COMPONENTS = (("tower_id_plate", "tower", 0.35, (0.0, -0.5), 12.5),
              ("glass_insulator", "a", 0.68, (-3.5, 0.0), 20.0),
              ("polymer_insulator", "b", 0.75, (0.0, 0.0), 20.0))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload) -> None:
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def infer_tower_heights(scene: dict) -> dict[str, float]:
    """Invert the original documented local-height template; never guess a height."""
    factors = {"a": .68, "b": .75, "c": .68, "tower": .35, "overhead": .93}
    heights: dict[str, list[float]] = {tower["id"]: [] for tower in scene["towers"]}
    for component in scene["components"]:
        phase = component["id"].rsplit("-", 1)[-1]
        if phase not in factors or component["tower_id"] not in heights:
            raise ValueError(f"unknown legacy component template: {component['id']}")
        heights[component["tower_id"]].append(component["local_origin_m"][2] / factors[phase])
    result = {}
    for tower, values in heights.items():
        if not values or any(not math.isfinite(h) or h <= 0 for h in values):
            raise ValueError(f"no valid height for {tower}")
        if max(values) - min(values) > 1e-6:
            raise ValueError(f"inconsistent source heights for {tower}: {values}")
        result[tower] = values[0]
    return result


def upgrade_scene(scene: dict, candidates: dict) -> tuple[dict, dict, dict]:
    result = copy.deepcopy(scene)
    result["scene_id"] += "-mc3v2"
    result["scene_name"] += " / three template components and three alternative views"
    for key in ("components", "checkpoints", "observation_poses", "tasks"):
        result[key] = []
    components = {c["id"]: c for c in scene["components"]}
    checkpoints = {c["id"]: c for c in scene["checkpoints"]}
    task_towers = {t["id"]: components[checkpoints[t["checkpoint_id"]]["component_id"]]["tower_id"] for t in scene["tasks"]}
    tower_priorities = {}
    for task in scene["tasks"]:
        tower = task_towers[task["id"]]
        tower_priorities.setdefault(tower, []).append(task["priority"])
    heights = infer_tower_heights(scene)
    new_tasks: dict[str, list[tuple[str, list[str]]]] = {}
    for tower in scene["towers"]:
        tower_id = tower["id"]
        # Preserve total priority at each tower when splitting the inspection.
        priority = sum(tower_priorities[tower_id]) / len(COMPONENTS)
        new_tasks[tower_id] = []
        for kind, phase, factor, xy, standoff in COMPONENTS:
            stem = f"{tower_id}-mc3-{kind}"
            component_id, checkpoint_id, task_id = (f"{prefix}-{stem}" for prefix in ("component", "checkpoint", "task"))
            angle = math.pi if phase == "a" else -math.pi / 2
            normal = [math.cos(angle), math.sin(angle), 0.0]
            result["components"].append({"id": component_id, "name": f"template {kind} at {tower_id}", "tower_id": tower_id,
                                          "component_type": kind, "local_origin_m": [*xy, factor * heights[tower_id]]})
            result["checkpoints"].append({"id": checkpoint_id, "name": f"inspect {kind}", "component_id": component_id,
                                           "local_position_m": [0.0, 0.0, 0.0], "surface_normal": normal})
            result["tasks"].append({"id": task_id, "name": f"inspect {kind} at {tower_id}", "checkpoint_id": checkpoint_id,
                                     "priority": priority, "service_duration_s": 8.0})
            pose_ids = []
            # The alternatives differ in azimuth AND elevation, within the
            # original incidence interval. LOS/clearance is evaluated later.
            for view, (azimuth, elevation) in enumerate(((0, 0), (-25, 10), (25, -10))):
                a, e = angle + math.radians(azimuth), math.radians(elevation)
                direction = [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
                pose_id = f"pose-{stem}-v{view}"
                pose_ids.append(pose_id)
                result["observation_poses"].append({"id": pose_id, "name": f"alternative view {view} of {kind}",
                    "checkpoint_id": checkpoint_id, "offset_from_checkpoint_m": [standoff * x for x in direction],
                    "camera_axis": [-x for x in direction]})
            new_tasks[tower_id].append((task_id, pose_ids))

    expanded, seen = [], set()
    for original in candidates["candidates"]:
        old_ids = [x["task_id"] for x in original["task_pose_sequence"]] if "task_pose_sequence" in original else original["task_ids"]
        towers = list(dict.fromkeys(task_towers[t] for t in old_ids))
        patterns = []
        if len(towers) == 1:
            tasks = new_tasks[towers[0]]
            # Seven ordered templates expand to 9 singles + 18 pairs + 54
            # triples. Full pose products keep fixed-pose ablations nested in
            # the same domain; no outcome-dependent combination pruning.
            for order in ((0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2), (2, 1, 0)):
                patterns.append([tasks[k][0] for k in order])
        else:
            if len(towers) > 3:
                raise ValueError("legacy multitower column exceeds the registered three-tower cap")
            for ks in itertools.product(range(3), repeat=len(towers)):
                patterns.append([new_tasks[t][k][0] for t, k in zip(towers, ks)])
        for pattern in patterns:
            key = tuple(original[k] for k in ("launch_stop_id", "recovery_stop_id", "uav_id", "weather_state_id")) + (tuple(pattern),)
            if key in seen:
                continue
            seen.add(key)
            expanded.append({"id": f"mc3-column-{len(expanded):06d}", "name": f"template candidate from {original['id']}",
                **{k: original[k] for k in ("launch_stop_id", "recovery_stop_id", "uav_id", "weather_state_id")},
                "task_ids": pattern, "enumerate_task_orders": False})
    used = {t for c in expanded for t in c["task_ids"]}
    if used != {t["id"] for t in result["tasks"]}:
        raise ValueError("candidate domain does not cover exactly all template tasks")
    for key in PRESERVED_KEYS:
        if result[key] != scene[key]:
            raise AssertionError(f"unexpected source change: {key}")
    diagnostics = {"version": VERSION, "geometry_evidence": "template_components_on_preserved_source_geometry",
        "tower_heights_m": heights, "height_source": "inverse original inspection_templates phase fractions",
        "tower_count": len(heights), "task_count": len(result["tasks"]), "pose_count": len(result["observation_poses"]),
        "candidate_count": len(expanded), "expanded_candidate_count_before_admission": sum(3 ** len(c["task_ids"]) for c in expanded),
        "candidate_lengths": dict(Counter(len(c["task_ids"]) for c in expanded)),
        "cross_stop_candidate_count": sum(c["launch_stop_id"] != c["recovery_stop_id"] for c in expanded),
        "physical_admission": "not_evaluated", "tower_priority_policy": "source tower total divided equally among three tasks",
        "preserved_keys": list(PRESERVED_KEYS)}
    return result, {"candidates": expanded}, diagnostics


def build_suite(source_suite: Path, output_root: Path) -> Path:
    source_suite, output_root = source_suite.resolve(), output_root.resolve()
    source = read_json(source_suite)
    inputs = {str(source_suite): sha256(source_suite)}
    for entry in source["scenarios"]:
        directory = (source_suite.parent / entry["scenario_directory"]).resolve()
        for path in [directory / n for n in ("scene.json", "service_candidates.json", "rolling_events.json", "generation_manifest.json")] + [(source_suite.parent / entry["config_path"]).resolve()]:
            inputs[str(path)] = sha256(path)
    write_once(output_root / "source_inventory.json", {"version": VERSION, "sha256": inputs})
    output_suite = copy.deepcopy(source)
    output_suite["suite_id"] = VERSION
    output_suite["evidence_level"] = "development"
    # Historical learned checkpoints are not compatible with the new task domain.
    output_suite["methods"] = [{"adapter": "alns", "method_id": "alns", "options": {}}]
    if source.get("weather_release_path"):
        output_suite["weather_release_path"] = os.path.relpath((source_suite.parent / source["weather_release_path"]).resolve(), output_root).replace("\\", "/")
    output_suite["scenarios"] = []
    manifests, counts = [], []
    for index, entry in enumerate(source["scenarios"]):
        directory = (source_suite.parent / entry["scenario_directory"]).resolve()
        scene, candidates, diagnostic = upgrade_scene(read_json(directory / "scene.json"), read_json(directory / "service_candidates.json"))
        manifest = read_json(directory / "generation_manifest.json")
        new_id = scene["scene_id"]
        config_path = output_root / "configs" / f"{new_id}.json"
        write_once(config_path, read_json((source_suite.parent / entry["config_path"]).resolve()))
        # Legacy discovery IDs did not refer to real scene tasks. Keep the
        # original source hashed, but never propagate invalid event references.
        events = {"schema_version": "mc3-events-v1", "initial_task_ids": [t["id"] for t in scene["tasks"]], "events": [],
                  "scope": "static benchmark; E3/E4 disturbances are separate event manifests"}
        destination = write_scenario(output_root / "scenarios", new_id, scene, candidates, events, config_path,
            {k: manifest[k] for k in LINEAGE_KEYS}, public_source_hashes=manifest.get("public_source_hashes", {}),
            public_source_manifest_path=Path(manifest["public_source_manifest_path"]) if manifest.get("public_source_manifest_path") else None,
            generation_diagnostics={"upgrade": diagnostic, "source_scenario": entry["scenario_id"], "source_scene_sha256": inputs[str(directory / "scene.json")]})
        manifests.append(read_json(destination / "generation_manifest.json"))
        output_suite["scenarios"].append({**entry, "scenario_id": new_id,
            "scenario_directory": os.path.relpath(destination, output_root).replace("\\", "/"),
            "config_path": os.path.relpath(config_path, output_root).replace("\\", "/"),
            "generation_manifest_path": os.path.relpath(destination / "generation_manifest.json", output_root).replace("\\", "/")})
        counts.append({"scenario_id": new_id, "split": entry["split"], **diagnostic})
        print(json.dumps({"completed": index + 1, "total": len(source["scenarios"]), "scenario_id": new_id,
                          "tasks": diagnostic["task_count"], "candidates": diagnostic["candidate_count"]}), flush=True)
    split_report = audit_split_manifests(manifests)
    changed = [path for path, digest in inputs.items() if sha256(Path(path)) != digest]
    if changed:
        raise ValueError(f"source changed during upgrade: {changed}")
    write_once(output_root / "suite.json", output_suite)
    write_once(output_root / "data_audit.json", {"schema_version": 1, "version": VERSION,
        "source_unchanged": True, "source_file_count": len(inputs), "split_issues": split_report.issue_count,
        "split_counts": dict(Counter(e["split"] for e in counts)), "scenario_count": len(counts),
        "physical_validation": "pending", "counts": counts})
    return output_root / "suite.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_suite(args.source_suite, args.output_root))
