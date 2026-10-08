"""Six planning models evaluated from the same raw candidate templates.

These are planning surrogates. Their feasibility certificates refer to their
own assumptions; execution must use the unmodified physical scene separately.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace
from functools import partial

from Env.data_loader import ProjectData
from Env.inspection_env import InspectionEnv
from Env.scene_builder import SceneBuilder
from Mask.flight_relation import FlightCorridor, FlightDynamicLimits, FlightPrimitive, FlightSegmentProfile, MotionPhase, _clearance_margins
from Mask.geometry import Polyline3D, Vector3
from Mask.service_column import ServiceColumn
from Mask.weather import ProbabilisticEnergy, calculate_service_energy
from experiments_3d.matrix import MODELS


def fixed_speed_primitive(object_id, name, corridor, limits):
    if not corridor.certified:
        return FlightPrimitive.from_corridor(object_id, name, corridor, limits)
    profiles = []
    for start, end in zip(corridor.path.points, corridor.path.points[1:]):
        length = start.distance_to(end)
        if length <= 1e-12:
            continue
        direction = (end - start).scaled(1.0 / length)
        speed = limits.maximum_speed
        phase = MotionPhase("cruise", length / speed, length, speed, speed, direction, Vector3(0, 0, 0))
        profiles.append(FlightSegmentProfile(start, end, direction, length, speed, (phase,)))
    return FlightPrimitive(object_id, name, corridor.source_id, corridor.target_id, corridor.object_id,
                           corridor.path, tuple(profiles), sum(p.duration for p in profiles),
                           corridor.minimum_clearance_margin, True)


def terrain_following_corridor(object_id, name, source_id, target_id, start, end, terrain, obstacle_field, required_clearance, sample_spacing):
    """2.5D transit: sampled XY straight line at a fixed 20 m terrain offset."""
    clearance = max(20.0, required_clearance)
    xy = Polyline3D((Vector3(start.x, start.y, 0), Vector3(end.x, end.y, 0)))
    samples = xy.sample(sample_spacing)
    points = [start]
    points.extend(Vector3(p.x, p.y, terrain.height_at(p.x, p.y) + clearance) for p in samples)
    points.append(end)
    unique = [points[0]]
    for p in points[1:]:
        if p.distance_to(unique[-1]) > 1e-10:
            unique.append(p)
    path = Polyline3D(tuple(unique))
    terrain_margin, obstacle_margin = _clearance_margins(path, terrain, obstacle_field, required_clearance, sample_spacing)
    admitted = terrain_margin >= -1e-10 and obstacle_margin >= 0
    return FlightCorridor(object_id, name, source_id, target_id, path, required_clearance, sample_spacing,
                          min(terrain_margin, obstacle_margin), admitted, "terrain_following_20m",
                          "" if admitted else "terrain_following_clearance_failed")


class DistanceEnergyColumn(ServiceColumn):
    """Distance/nominal-speed hover surrogate, retaining the same chance level."""

    @classmethod
    def assemble(cls, object_id, name, pair, weather, observation_pattern, primitives, uav, parameters):
        base = super().assemble(object_id, name, pair, weather, observation_pattern, primitives, uav, parameters)
        if not all(p.certified for p in primitives):
            return base
        service_s = sum(a.service_duration for a in observation_pattern.assignments)
        distance = sum(p.path.length() for p in primitives)
        mean = uav.hover_power * (distance / uav.maximum_speed + service_s) + parameters.launch_recovery_energy
        uncertainty = calculate_service_energy(primitives, weather, uav, parameters, service_s).probabilistic.standard_deviation
        protected = ProbabilisticEnergy(mean, uncertainty, parameters.violation_probability)
        reasons = tuple(r for r in base.rejection_reasons if r != "insufficient_battery")
        if protected.quantile > uav.available_battery:
            reasons += ("insufficient_battery",)
        return replace(base, energy_mean=mean, energy_standard_deviation=uncertainty, energy_quantile=protected.quantile,
                       admitted=not reasons, rejection_reasons=reasons, weather_scenario_energies=(mean,))

    @classmethod
    def assemble_scenarios(cls, object_id, name, pair, weather_scenarios, observation_pattern, primitives, uav, parameters):
        estimates = [cls.assemble(object_id, name, pair, s.state, observation_pattern, primitives, uav, parameters) for s in weather_scenarios.scenarios]
        base = estimates[0]
        if not all(p.certified for p in primitives):
            return base
        means = tuple(e.energy_mean for e in estimates)
        deviations = tuple(e.energy_standard_deviation for e in estimates)
        aggregate = weather_scenarios.aggregate_energies(means, conditional_standard_deviations=deviations,
                                                        violation_probability=parameters.violation_probability)
        reasons = tuple(r for r in base.rejection_reasons if r != "insufficient_battery")
        if aggregate.quantile > uav.available_battery:
            reasons += ("insufficient_battery",)
        return replace(base, energy_mean=aggregate.mean, energy_standard_deviation=aggregate.standard_deviation,
                       energy_quantile=aggregate.quantile, admitted=not reasons, rejection_reasons=reasons,
                       weather_scenario_ids=weather_scenarios.scenario_ids, weather_scenario_probabilities=weather_scenarios.probabilities,
                       weather_scenario_energies=means, weather_scenario_standard_deviations=deviations, weather_certification_mode="scenario_set")


def transformed_data(data: ProjectData, model: str, relief_factor: float = 1.0) -> ProjectData:
    if model not in MODELS or not math.isfinite(relief_factor) or relief_factor < 0:
        raise ValueError("invalid model or terrain factor")
    scene, candidates, config = copy.deepcopy((data.scene, data.service_candidates, data.config))
    buffer = float(config["physical"].get("flight_tracking_buffer_m", 0.))
    if not math.isfinite(buffer) or buffer < 0:
        raise ValueError("invalid tracking buffer")
    for uav in scene["uavs"]:
        uav["minimum_clearance_m"] += buffer
    heights = scene["terrain"]["heights_m"]
    base = min(min(row) for row in heights)
    scene["terrain"]["heights_m"] = [[base + relief_factor * (h - base) for h in row] for row in heights]
    # Apply the same vertical transform to absolute obstacle elevations.
    for obstacle in scene["obstacle_field"]["obstacles"]:
        height = obstacle["maximum_elevation_m"] - obstacle["minimum_elevation_m"]
        obstacle["minimum_elevation_m"] = base + relief_factor * (obstacle["minimum_elevation_m"] - base)
        obstacle["maximum_elevation_m"] = obstacle["minimum_elevation_m"] + height
    if model == "M0_planar":
        scene["terrain"]["heights_m"] = [[0.0 for _ in row] for row in heights]
        plane = max(u["minimum_clearance_m"] for u in scene["uavs"])
        for component in scene["components"]:
            component["local_origin_m"][2] = plane
        for checkpoint in scene["checkpoints"]:
            checkpoint["local_position_m"][2] = 0.0
        for pose in scene["observation_poses"]:
            pose["offset_from_checkpoint_m"][2] = 0.0
            offset = pose["offset_from_checkpoint_m"]
            norm = math.sqrt(sum(x*x for x in offset))
            pose["camera_axis"] = [-x / norm for x in offset]
        for obstacle in scene["obstacle_field"]["obstacles"]:
            obstacle["minimum_elevation_m"], obstacle["maximum_elevation_m"] = 0.0, 1e6
    if model == "M3_distance_energy":
        # For a fixed geometric candidate the distance energy is independent
        # of flight timing, so continuous optimization minimizes time alone.
        config["physical"]["trajectory_objective"]["expected_energy_weight"] = 0.0
    return ProjectData(scene=scene, service_candidates=candidates, config=config)


def build_model(data: ProjectData, model: str, *, optimizer=None, weather_path=None, weather_split=None, relief_factor=1.0, trajectory_cache=None):
    transformed = transformed_data(data, model, relief_factor)
    if model == "M3_fixed_pose":
        admission_data = replace(transformed, service_candidates={"candidates": []})
        admission = SceneBuilder.from_project_data(admission_data)
        by_checkpoint = {}
        for assessment in admission.observation_assessments:
            if assessment.admissible:
                by_checkpoint.setdefault(assessment.checkpoint_id, []).append(assessment)
        fixed = {key: min(values, key=lambda a: (round(a.distance, 9), a.pose_id)).pose_id for key, values in by_checkpoint.items()}
        # This only restricts the original pose set. Enumerated service
        # templates therefore retain every permissible task order.
        keep = set(fixed.values())
        transformed.scene["observation_poses"] = [p for p in transformed.scene["observation_poses"] if p["id"] in keep]
        if any("task_ids" not in c for c in transformed.service_candidates["candidates"]):
            raise ValueError("fixed-pose control requires the v2 full-product candidate domain")
    dynamic = model.startswith("M3_")
    if dynamic and optimizer is None:
        raise ValueError("M3 controls require an explicit continuous optimizer")
    limits_factory = FlightDynamicLimits
    control_cap = transformed.config["physical"].get("maximum_control_acceleration_mps2")
    if dynamic and control_cap is not None:
        from experiments_3d.controller_optimizer import ControllerAwareOptimizer, ControllerLimits
        limits_factory = partial(ControllerLimits, maximum_control_acceleration=float(control_cap))
        optimizer = ControllerAwareOptimizer(optimizer._config)
    if dynamic and trajectory_cache is not None:
        from experiments_3d.trajectory_cache import AuditedTrajectoryCache
        from experiments_3d.failure_cache import FrozenFailureCache
        optimizer = FrozenFailureCache(AuditedTrajectoryCache(optimizer, trajectory_cache))
    build = SceneBuilder.from_project_data(transformed, trajectory_optimizer=optimizer if dynamic else None,
        weather_scenario_path=weather_path, weather_split=weather_split,
        corridor_factory=terrain_following_corridor if model == "M1_terrain_following" else None,
        primitive_factory=None if dynamic else fixed_speed_primitive,
        column_factory=DistanceEnergyColumn if not dynamic or model == "M3_distance_energy" else ServiceColumn,
        dynamic_limits_factory=limits_factory)
    env = InspectionEnv(build.system, build.time_horizon, build.service_columns, build.rejection_records, build.road_options,
        build.stop_options, build.observation_assessments, build.vehicle_states, build.uav_states, build.state_default_diagnostics,
        rolling_window_seconds=build.rolling_window_seconds, energy_parameters=build.energy_parameters)
    return env, transformed
