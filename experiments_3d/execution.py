"""Export fixed plans to an independent native rigid-body execution protocol.

This is a controller/component experiment, separate from paper10-live-v2.
No planned positions are written into simulated state after initialization.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path

from Env.scene_builder import SceneBuilder
from Env.data_loader import ProjectData, load_project_data
from Mask.geometry import Vector3, Polyline3D
from experiments_3d.data_upgrade import read_json, sha256, write_once
from experiments_3d.preflight import evidence_json


def vector(v):
    return [v.x, v.y, v.z]


def reference_manifest(snapshot, result, physical_data: ProjectData, model: str):
    physical = SceneBuilder.from_project_data(dataclasses.replace(physical_data, service_candidates={"candidates": []}))
    system = physical.system
    values = dict(result.variable_values)
    flights = []
    uavs = {u["id"]: u for u in physical_data.scene["uavs"]}
    selected = set(result.selected_column_ids)
    by_primitive = {p.object_id: p for p in snapshot.flight_primitives}
    buffer = physical_data.config["physical"].get("flight_tracking_buffer_m",0.)
    for column in snapshot.service_columns:
        if column.object_id not in selected:
            continue
        uav = uavs[column.uav_id]
        actual_launch = system.get_element(column.launch_stop_id).position + Vector3(0, 0, uav["minimum_clearance_m"]+buffer)
        primitives = [by_primitive[pid] for pid in column.primitive_ids]
        # Planar commands use one launch-relative constant altitude. No terrain
        # following, actual target-height correction, or geometric rerouting.
        shift = actual_launch.z - primitives[0].path.points[0].z if model == "M0_planar" else 0.0
        legs = []
        assignments = column.observation_pattern.assignments
        for index, primitive in enumerate(primitives):
            intervals = []
            for interval in primitive.trajectory_intervals():
                position = interval.start_position + Vector3(0, 0, shift)
                intervals.append({"start_s": interval.start_time, "duration_s": interval.duration,
                    "position_m": vector(position), "velocity_mps": vector(interval.start_velocity),
                    "acceleration_mps2": vector(interval.acceleration)})
            assignment = assignments[index] if index < len(assignments) else None
            task = None
            if assignment:
                checkpoint = system.get_element(assignment.checkpoint_id)
                task = {"task_id": assignment.task_id, "checkpoint_id": assignment.checkpoint_id,
                    "pose_id": assignment.pose_id, "target_m": vector(checkpoint.global_position(system)),
                    "normal": vector(checkpoint.global_surface_normal(system)),
                    "camera_axis": vector(assignment.camera_axis), "required_service_s": assignment.service_duration}
            legs.append({"leg_index": index, "duration_s": primitive.duration,
                "endpoint_m": vector(primitive.path.points[-1] + Vector3(0, 0, shift)), "intervals": intervals,
                "service_s": assignment.service_duration if assignment else 0.0, "observation": task})
        recovery = system.get_element(column.recovery_stop_id).position + Vector3(0, 0, uav["minimum_clearance_m"]+buffer)
        carrier = uav["carrier_vehicle_id"]
        flights.append({"flight_id": column.object_id, "uav_id": column.uav_id, "vehicle_id": carrier,
            "launch_stop_id": column.launch_stop_id, "recovery_stop_id": column.recovery_stop_id,
            "planned_launch_s": values[f"column_launch[{carrier},{column.object_id}]"],
            "planned_recovery_s": values[f"column_recovery[{carrier},{column.object_id}]"],
            "planned_duration_s": column.dynamic_duration, "planned_energy_j": column.energy_mean,
            "actual_initial_position_m": vector(actual_launch), "actual_recovery_position_m": vector(recovery),
            "mass_kg": uav["mass_kg"], "maximum_speed_mps": uav["maximum_speed_mps"],
            "pack_energy_j": uav["replacement_pack_energy_j"], "initial_soc": 1.0,
            "reserve_ratio": uav["battery_reserve_ratio"], "minimum_clearance_m": uav["minimum_clearance_m"],
            "legs": legs})
    weather = physical_data.scene["weather_states"][0]
    return {"protocol": "paper10-native-fixed-plan-v1", "scope": "independent native sortie tracking; fleet rendezvous evaluated separately",
        "model": model, "integration_step_s": .01, "telemetry_step_s": .01, "settle_timeout_s": 30.0,
        "position_tolerance_m": .5, "velocity_tolerance_mps": .5, "position_feedback_gain_per_s": .8,
        "camera_model": "ideal stabilized world-frame axis commanded by planning pose; no pixel-quality claim",
        "collision_model": "first terrain/obstacle intersection invalidates execution in geometric telemetry audit; no impact-response claim",
        "timing_policy": "preserve flight reference; settle at each endpoint then fixed service; no rerouting or reoptimization",
        "launch_recovery_boundary": "already airborne at vehicle-relative staging height; takeoff and contact docking are outside this component test",
        "flight_tracking_buffer_m": buffer,
        "wind_mps": weather["wind_vector_mps"], "ambient_temperature_c": weather["temperature_c"],
        "observation_limits": physical_data.config["physical"]["observation_limits"],
        "physical_scene": {"terrain": physical_data.scene["terrain"], "obstacle_field": physical_data.scene["obstacle_field"]},
        "flights": flights, "denominator_task_ids": sorted(t["id"] for t in physical_data.scene["tasks"])}


def audit_execution(manifest_path: Path, telemetry_path: Path, output_path: Path):
    from Mask.terrain import TerrainSurface, ObstacleField, CylinderObstacle
    m = read_json(manifest_path)
    r = m["physical_scene"]["terrain"]
    terrain = TerrainSurface(r["id"], r["name"], tuple(r["x_coordinates_m"]), tuple(r["y_coordinates_m"]), tuple(map(tuple, r["heights_m"])))
    r = m["physical_scene"]["obstacle_field"]
    obstacles = ObstacleField(r["id"], r["name"], tuple(CylinderObstacle(o["id"], o["name"], o["center_x_m"], o["center_y_m"],
        o["radius_m"], o["minimum_elevation_m"], o["maximum_elevation_m"]) for o in r["obstacles"]))
    with telemetry_path.open(encoding="utf-8-sig", newline="") as stream:
        grouped = defaultdict(list)
        for row in csv.DictReader(stream):
            grouped[row["flight_id"]].append(row)
    limits = m["observation_limits"]
    reports, completed = [], set()
    expected_ids = {f["flight_id"] for f in m["flights"]}
    if set(grouped) != expected_ids:
        raise ValueError("missing or unexpected native sortie telemetry")
    for flight in m["flights"]:
        rows = grouped[flight["flight_id"]]
        previous = 0.0
        energy = 0.0
        flight_energy = 0.0
        flight_duration = 0.0
        first_collision = None
        violations = set()
        tracking = []
        observations = defaultdict(float)
        observation_failures = defaultdict(set)
        minimum_agl = math.inf
        for row in rows:
            t = float(row["time_s"])
            dt = t - previous
            if not 0 < dt <= .02001 or not all(math.isfinite(float(row[k])) for k in ("x_m", "y_m", "z_m", "electrical_energy_j", "soc")):
                raise ValueError("invalid or missing telemetry samples")
            previous = t
            pos = Vector3(*(float(row[k]) for k in ("x_m", "y_m", "z_m")))
            speed = math.sqrt(sum(float(row[k])**2 for k in ("vx_mps", "vy_mps", "vz_mps")))
            inside = terrain.contains_xy(pos.x, pos.y)
            agl = pos.z - terrain.height_at(pos.x, pos.y) if inside else -math.inf
            obstacle = obstacles.signed_clearance(pos)
            minimum_agl = min(minimum_agl, agl)
            if not inside or agl < 0 or obstacle < 0:
                first_collision = first_collision if first_collision is not None else t
            if agl < flight["minimum_clearance_m"] - 1e-3:
                violations.add("minimum_clearance")
            if float(row["soc"]) < flight["reserve_ratio"]:
                violations.add("battery_reserve")
            if first_collision is None:
                energy = float(row["electrical_energy_j"])
            if row["phase"] != "rendezvous_wait":
                flight_energy = energy
                flight_duration = t
            tracking.append(float(row["tracking_error_m"]))
            if row["phase"] != "service":
                continue
            task = flight["legs"][int(row["leg_index"])]["observation"]
            target = Vector3(*task["target_m"])
            offset = pos - target
            distance = offset.norm()
            angle = lambda a, b: math.acos(max(-1., min(1., a.normalized().dot(b.normalized()))))
            issues = []
            if first_collision is not None:
                issues.append("prior_collision")
            if not limits["minimum_distance_m"] <= distance <= limits["maximum_distance_m"]:
                issues.append("distance")
            if distance == 0 or angle(Vector3(*task["normal"]), offset) > limits["maximum_incidence_angle_rad"]:
                issues.append("incidence")
            if distance == 0 or angle(Vector3(*task["camera_axis"]), offset.scaled(-1)) > limits["maximum_pointing_error_rad"]:
                issues.append("camera_pointing")
            if agl < limits["minimum_terrain_clearance_m"] or obstacle < limits["minimum_obstacle_clearance_m"]:
                issues.append("observation_clearance")
            if speed > m["velocity_tolerance_mps"]:
                issues.append("observation_speed")
            for p in Polyline3D((pos, target)).sample(limits["line_of_sight_spacing_m"])[1:-1]:
                if not terrain.contains_xy(p.x, p.y) or p.z < terrain.height_at(p.x, p.y) or obstacles.signed_clearance(p) < 0:
                    issues.append("line_of_sight")
                    break
            if not issues:
                observations[task["task_id"]] += dt
            observation_failures[task["task_id"]].update(issues)
        successful = [leg["observation"]["task_id"] for leg in flight["legs"] if leg["observation"] is not None
                      and observations[leg["observation"]["task_id"]] + 1e-5 >= leg["service_s"]]
        completed.update(successful)
        end = rows[-1]
        endpoint = Vector3(*(float(end[k]) for k in ("x_m", "y_m", "z_m")))
        distance = endpoint.distance_to(Vector3(*flight["actual_recovery_position_m"]))
        terminal = end["terminal_status"]
        recovery = terminal == "completed" and first_collision is None and distance <= m["position_tolerance_m"] and "battery_reserve" not in violations
        reports.append({"flight_id": flight["flight_id"], "terminal_status": terminal, "effective_task_ids": successful,
            "observed_valid_service_s": dict(observations), "observation_failures": {k: sorted(v) for k, v in observation_failures.items()},
            "collision_at_s": first_collision, "constraint_violations": sorted(violations), "minimum_agl_m": minimum_agl,
            "recovery_approach_success": recovery, "recovery_position_error_m": distance, "duration_s": previous,
            "flight_duration_s":flight_duration,"rendezvous_wait_s":previous-flight_duration,
            "flight_electrical_energy_j":flight_energy,"rendezvous_wait_electrical_energy_j":energy-flight_energy,
            "electrical_energy_j": energy, "post_collision_integral_j": float(end["electrical_energy_j"]) - energy,
            "time_prediction_error_s": previous - flight["planned_duration_s"], "energy_prediction_error_j": energy - flight["planned_energy_j"],
            "maximum_tracking_error_m": max(tracking), "final_soc": float(end["soc"])})
    report = {"protocol": m["protocol"], "manifest_sha256": sha256(manifest_path), "telemetry_sha256": sha256(telemetry_path),
        "scope": m["scope"], "all_tasks": len(m["denominator_task_ids"]), "effective_tasks": len(completed),
        "effective_coverage": len(completed)/max(1, len(m["denominator_task_ids"])), "flights": reports,
        "fleet_rendezvous_validation": "pending; recovery approach alone is not a vehicle docking certificate"}
    write_once(output_path, evidence_json(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_execution(args.manifest, args.telemetry, args.output)))
