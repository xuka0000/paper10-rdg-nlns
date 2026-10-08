"""Exact physical-request reuse with a fresh independent audit on every hit."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import inspect
from collections import OrderedDict
from pathlib import Path

from Mask.geometry import Vector3
from Mask.trajectory_optimization import (TrajectoryDecision, TrajectoryOptimizationResult,
    audit_trajectory_decision, trajectory_expected_energy, trajectory_objective)
from experiments_3d.data_upgrade import read_json, sha256, write_once


def _numeric_object(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _numeric_object(getattr(value, f.name)) for f in dataclasses.fields(value)
                if f.name not in {"object_id", "name", "carrier_vehicle_id"}}
    if isinstance(value, (tuple, list)):
        return [_numeric_object(v) for v in value]
    if isinstance(value, dict):
        return {key: _numeric_object(v) for key, v in value.items()}
    return value


class AuditedTrajectoryCache:
    def __init__(self, optimizer, directory: Path, memory_entries: int = 256):
        if memory_entries < 1:
            raise ValueError("positive memory cache bound required")
        self.optimizer = optimizer
        self.directory = directory
        self.memory_entries = memory_entries
        self.hits = self.misses = 0
        self._memory = OrderedDict()
        self._contexts = OrderedDict()
        root = Path(__file__).resolve().parents[1]
        sources = [Path(__file__), Path(inspect.getfile(type(optimizer))), root / "Agent/optimization/scipy_trajectory_agent.py", *sorted((root / "Mask").glob("*.py"))]
        sources.extend(getattr(optimizer, "cache_source_files", ()))
        self.source_identity = hashlib.sha256("".join(sha256(p) for p in sources).encode()).hexdigest()

    def _context(self, value):
        key = id(value)
        cached = self._contexts.get(key)
        if cached is not None and cached[0] is value:
            self._contexts.move_to_end(key)
            return cached[1]
        # Hold the source reference so Python cannot recycle its id.
        encoded = json.dumps(_numeric_object(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        self._contexts[key] = (value, digest)
        while len(self._contexts) > 32:
            self._contexts.popitem(last=False)
        return digest

    def key(self, request):
        context_names = ("terrain", "obstacle_field", "uav", "weather", "energy_parameters", "dynamic_limits", "objective_weights", "config")
        payload = {name: self._context(getattr(request, name)) for name in context_names}
        payload.update(source=self.source_identity, fixed_node_positions=_numeric_object(request.fixed_node_positions),
            observation_node_indices=request.observation_node_indices, initial_decision=_numeric_object(request.initial_decision),
            service_duration=request.service_duration,
            primitive_slices=[(s.first_interval_index, s.interval_count, s.minimum_clearance_margin) for s in request.primitive_specs])
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def solve(self, request):
        key = self.key(request)
        path = self.directory / key[:2] / f"{key}.json"
        record = self._memory.get(key)
        if record is None and path.exists():
            record = read_json(path)
        if record is not None:
            if record["request_sha256"] != key or record["source_identity"] != self.source_identity:
                raise ValueError("trajectory cache identity mismatch")
            d = record["decision"]
            vectors = lambda name: tuple(Vector3(v["x"], v["y"], v["z"]) for v in d[name])
            decision = TrajectoryDecision(vectors("positions"), vectors("velocities"), vectors("accelerations"), tuple(d["durations"]))
            audit = audit_trajectory_decision(request, decision, True, "exact request cache; independently re-audited")
            if not audit.admitted:
                raise ValueError("cached trajectory failed independent audit")
            cap = getattr(request.dynamic_limits, "maximum_control_acceleration", None)
            if cap is not None and any(a.norm()**2 > cap**2 + request.config.inequality_tolerance for a in decision.accelerations):
                raise ValueError("cached trajectory exceeds native controller envelope")
            objective = trajectory_objective(request, decision)
            primitives = self.optimizer._build_primitives(request, decision, audit, objective)
            self.hits += 1
            result = TrajectoryOptimizationResult(True, "optimized", audit.message, "SLSQP/exact-request-cache",
                decision, objective, trajectory_expected_energy(request, decision), audit, primitives)
        else:
            self.misses += 1
            result = self.optimizer.solve(request)
            if result.success:
                record = {"request_sha256": key, "source_identity": self.source_identity,
                          "decision": dataclasses.asdict(result.decision)}
                write_once(path, record)
            else:
                from experiments_3d.preflight import evidence_json
                failure = self.directory / "failures" / f"{key}.json"
                if not failure.exists():
                    write_once(failure, evidence_json({"request_sha256":key,"source_identity":self.source_identity,
                        "request_id":request.object_id,"solver":result.solver_name if hasattr(result,"solver_name") else type(self.optimizer).__name__,
                        "status":result.status,"audit":dataclasses.asdict(result.audit),
                        "decision":dataclasses.asdict(result.decision) if result.decision is not None else None}))
        if record is not None:
            self._memory[key] = record
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_entries:
                self._memory.popitem(last=False)
        return result
