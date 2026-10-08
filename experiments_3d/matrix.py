"""Materialize the approved experiment cells and causal event manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
from pathlib import Path

from experiments_3d.data_upgrade import read_json, sha256, write_once

MODELS = ("M0_planar", "M1_terrain_following", "M2_3d_fixed_speed", "M3_full", "M3_fixed_pose", "M3_distance_energy")
ABLATIONS = {
    "full": {}, "static_graph": {"graph_state_mode": "static_only"},
    "untyped_graph": {"graph_state_mode": "relation_agnostic"},
    "independent_actors": {"repair_policy_mode": "independent"},
    "mean_critic": {"risk_score_mode": "mean"},
    "deterministic_mean": {"risk_score_mode": "mean", "distributional_samples": False},
    "no_sync_repair": {"excluded_repair_modes": ["synchronization_aware"]},
    "uniform_actions": {"policy_mode": "uniform_legal"},
}
REQUIRED_GATES = ("physical_profile_frozen", "independent_executor_validated", "candidate_domain_audited",
                  "model_controls_validated", "public_baselines_adapted", "training_complete")


def visible_event_state(manifest: dict, now_s: float) -> dict:
    """The scheduler sees only released IDs and events observed by current time."""
    if not math.isfinite(now_s) or now_s < 0:
        raise ValueError("invalid decision time")
    tasks = list(manifest["initial_task_ids"])
    observed = []
    for event in manifest["events"]:
        if event["at_s"] <= now_s:
            observed.append(event)
            if event["kind"] == "task_release":
                tasks.extend(event["task_ids"])
    return {"now_s": now_s, "released_task_ids": list(dict.fromkeys(tasks)), "observed_events": observed}


def event_manifest(scene: dict, condition: str, seed: int, high_weather: dict | None = None) -> dict:
    if condition not in {"nominal", "disturbed"}:
        raise ValueError("unknown execution condition")
    task_ids = sorted(t["id"] for t in scene["tasks"])
    # Task release set fixed independently of solver outcomes and execution seed.
    ranked = sorted(task_ids, key=lambda t: hashlib.sha256(("release-v1/" + scene["scene_id"] + "/" + t).encode()).hexdigest())
    delayed = set(ranked[:max(1, len(task_ids) // 10)]) if condition == "disturbed" else set()
    if condition == "disturbed" and high_weather is None:
        raise ValueError("disturbed execution requires an explicit held-out high-wind state")
    weather = {**scene["weather_states"][0], **(high_weather or {})}
    events = []
    if condition == "disturbed":
        events = [{"id": "release-900", "kind": "task_release", "at_s": 900.0, "task_ids": sorted(delayed)},
                  {"id": "weather-1800", "kind": "weather_update", "at_s": 1800.0,
                   "weather_state": weather, "source": "frozen NREL-HRRR test-high weather release"},
                  {"id": "vehicle-delay-1800", "kind": "vehicle_delay", "at_s": 1800.0,
                   "vehicle_id": scene["vehicles"][0]["id"], "duration_s": 60.0}]
    return {"schema_version": "mc3-event-v1", "scene_id": scene["scene_id"], "condition": condition, "seed": seed,
            "initial_task_ids": [t for t in task_ids if t not in delayed], "events": events,
            "mission_horizon_s": 28800.0, "regular_epoch_s": 600.0, "online_wall_limit_s": 30.0,
            "event_epoch_policy": "one additional decision per unique event time; simultaneous events coalesced",
            "commitment_policy": "completed tasks and active sorties locked; update only future operations",
            "information_policy": "future event payloads belong to executor only; scheduler receives visible_event_state"}


def materialize(suite_path: Path, output: Path) -> dict:
    suite = read_json(suite_path)
    weather_path = (suite_path.parent / suite["weather_release_path"]).resolve()
    with weather_path.open(encoding="utf-8-sig", newline="") as stream:
        high = next(row for row in csv.DictReader(stream) if row["split"] == "test" and row["band"] == "high")
    high_weather = {"name": "frozen test-high disturbance", "wind_vector_mps": [float(high["wind_u_mps"]), float(high["wind_v_mps"]), 0.0],
                    "temperature_c": float(high["ambient_temperature_c"])}
    tests = [s for s in suite["scenarios"] if s["split"] == "test"]
    gis = [s for s in tests if s["scenario_id"].startswith("gis_external")]
    gis50 = [s for s in gis if s["labels"]["scale"] == "50"]
    controlled = [s for s in tests if s["scenario_id"].startswith("controlled_main") and "n50-" in s["scenario_id"] and "s307" in s["scenario_id"]]
    if (len(tests), len(gis), len(gis50), len(controlled)) != (75, 18, 6, 3):
        raise ValueError("source suite differs from the approved matrix")
    scenes_e1 = [{"scenario_id": s["scenario_id"], "terrain_relief_factor": a} for s, a in itertools.product(controlled, (0.0, 0.5, 1.0))]
    scenes_e1 += [{"scenario_id": s["scenario_id"], "terrain_relief_factor": 1.0} for s in gis50]
    cells = []
    for scene, model, seed in itertools.product(scenes_e1, MODELS, (11, 23, 37)):
        cells.append({"experiment": "E1", **scene, "model": model, "method": "alns_public", "seed": seed})
    for scene, method, seed in itertools.product(tests, ("rdg_nlns", "alns_public", "nlns_public", "grlos_public", "dac_alns_public"), (11, 23, 37, 53, 71)):
        cells.append({"experiment": "E2", "scenario_id": scene["scenario_id"], "model": "M3_full", "method": method, "seed": seed})
    for scene in tests:
        if scene["scenario_id"].startswith("exact_small"):
            cells.append({"experiment": "E2", "scenario_id": scene["scenario_id"], "model": "M3_full", "method": "milp", "seed": 11})
    for scene, condition, seed, variant in itertools.product(gis, ("nominal", "disturbed"), (11, 23, 37), ABLATIONS):
        cells.append({"experiment": "E3", "scenario_id": scene["scenario_id"], "condition": condition,
                      "model": "M3_full", "method": "rdg_nlns", "variant": variant, "seed": seed})
    case = next(s for s in gis50 if "los_angeles_inland-w1" in s["scenario_id"])
    methods = (("M3_full", "rdg_nlns"), ("M3_full", "alns_public"), ("M1_terrain_following", "alns_public"),
               ("M3_full", "best_public_learning_on_validation"), ("M3_full", "rdg_frozen_plan"))
    for (model, method), condition, seed in itertools.product(methods, ("nominal", "disturbed"), (11, 23, 37)):
        cells.append({"experiment": "E4", "scenario_id": case["scenario_id"], "condition": condition,
                      "model": model, "method": method, "seed": seed, "showcase": seed == 23})
    for number, cell in enumerate(cells):
        cell["cell_id"] = f"{cell['experiment']}-{number:05d}"
    for scene, condition, seed in itertools.product(gis, ("nominal", "disturbed"), (11, 23, 37)):
        data = read_json(suite_path.parent / scene["scenario_directory"] / "scene.json")
        write_once(output / "events" / f"{scene['scenario_id']}-{condition}-s{seed}.json", event_manifest(data, condition, seed, high_weather))
    counts = {e: sum(c["experiment"] == e for c in cells) for e in ("E1", "E2", "E3", "E4")}
    contract = {"schema_version": 1, "suite_path": str(suite_path.resolve()), "suite_sha256": sha256(suite_path),
        "weather_path": str(weather_path), "weather_sha256": sha256(weather_path),
        "cells": cells, "cell_counts": counts, "ablations": ABLATIONS,
        "search_budget": {"wall_clock_s": 300, "neighborhoods": 1000, "stop": "first limit reached", "milp_wall_s": 30},
        "training_budget": {"interactions_per_seed": 10000, "formal_seeds": [11, 23, 37, 53, 71], "ablation_seeds": [11, 23, 37],
                            "checkpoint_selection": "validation only; new domain retraining mandatory"},
        "gates": {g: {"status": "pending", "evidence": None} for g in REQUIRED_GATES},
        "metrics": ["effective_task_coverage", "mission_completion_s", "docking_success_rate", "electrical_energy_j", "vehicle_wait_s",
                    "uav_wait_s", "predicted_vs_executed_time_s", "predicted_vs_executed_energy_j"],
        "statistics": "mean, standard deviation, paired descriptive differences; failures retained in denominator",
        "case_evidence": ["full_3d_scene_routes", "fleet_gantt", "local_dynamics_power_soc", "same_pre_event_state_counterfactuals"],
        "comparison_boundary": "fixed finite raw candidate domain; no full-model prescreening of simpler models"}
    if counts != {"E1": 270, "E2": 1902, "E3": 864, "E4": 30}:
        raise AssertionError(counts)
    write_once(output / "experiment_matrix.json", contract)
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.suite, args.output)["cell_counts"])
