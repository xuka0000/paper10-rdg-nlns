"""Authors' ALNS control loop with the shared inspection neighborhood interface."""
from __future__ import annotations

import dataclasses
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Agent.baselines.common import initial_solution, result_validator
from Agent.evaluation_runtime import RuntimeDeadline
from Agent.hybrid.rdg_nlns.operators import DestroyOperator, destroy_neighborhood
from Agent.hybrid.rdg_nlns.repair import RepairMode, repair_neighborhood
from experiments_3d.data_upgrade import read_json, sha256


@dataclass(frozen=True)
class InspectionState:
    solution: object
    feasible: bool = True
    action: object = None

    def objective(self):
        return self.solution.objective_value if self.feasible else float("inf")


def load_public_alns(root: Path):
    manifest = read_json(root / "source_manifest.json")
    if manifest["repository"] != "https://github.com/N-Wouda/ALNS":
        raise ValueError("incorrect public ALNS source")
    for record in manifest["files"]:
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("public ALNS source modified")
    if "alns" in sys.modules and Path(sys.modules["alns"].__file__).resolve().parent != (root / "alns").resolve():
        raise RuntimeError("another alns package is already loaded; use an isolated experiment process")
    sys.path.insert(0, str(root.resolve()))
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import SegmentedRouletteWheel
    return ALNS, SimulatedAnnealing, SegmentedRouletteWheel, manifest["commit"]


def solve_public_alns(snapshot, solver, *, source_root: Path, seed=11, iterations=1000, wall_s=300.0, trace_callback=None):
    if iterations <= 0 or wall_s <= 0:
        raise ValueError("positive search bounds required")
    started = time.perf_counter()
    # Include package/source verification, initialization and all repairs.
    # Leave a small return/independent-validation allowance inside the cap.
    reserve = min(.25, wall_s*.05)
    deadline = RuntimeDeadline(wall_s-reserve, started, time.perf_counter)
    ALNS, SimulatedAnnealing, SegmentedRouletteWheel, commit = load_public_alns(source_root)
    initial = initial_solution(snapshot, solver, "greedy", deadline=deadline, route_projection_time_limit_seconds=min(30.0, wall_s))
    native = ALNS(np.random.default_rng(seed))
    ratios = (.1, .2, .3, .4)
    attempted = 0
    corrections = 0
    invalid = 0
    selection = {}

    def make_destroy(operator, ratio):
        def destroy(state, rng):
            # Original operator implementations expect Python Random. Draw
            # its seed from the authors' RNG so one seed controls both layers.
            selection.update(destroy=operator.name, ratio=ratio)
            action = destroy_neighborhood(snapshot, state.solution, operator, ratio, random.Random(int(rng.integers(0, 2**63))))
            return InspectionState(state.solution, action=action)
        return destroy

    def make_repair(mode):
        def repair(state, rng):
            nonlocal attempted, corrections, invalid
            attempted += 1
            outcome = None
            if state.action.removed_column_ids and not deadline.expired():
                outcome = repair_neighborhood(snapshot, state.action, mode, solver, result_validator(snapshot),
                    deadline=deadline, route_projection_time_limit_seconds=min(10.0, max(.001, deadline.remaining_seconds())))
            candidate = outcome.candidate if outcome is not None else None
            valid = bool(candidate is not None and candidate.is_feasible and outcome.validation.is_valid)
            invalid += not valid
            corrections += len(outcome.corrections) if outcome is not None else 0
            if trace_callback is not None:
                trace_callback({"iteration": attempted, **selection, "repair": mode.name,
                    "before_objective": state.solution.objective_value, "candidate_objective": candidate.objective_value if valid else None,
                    "projection_valid": valid, "projection_corrections": list(outcome.corrections) if outcome else [],
                    "issue_codes": list(outcome.validation.issue_codes) if outcome else ["empty_destroy_or_deadline"],
                    "removed_column_ids": list(state.action.removed_column_ids),
                    "selected_column_ids": list(candidate.selected_column_ids) if valid else []})
            return InspectionState(candidate if valid else state.solution, feasible=valid)
        return repair

    for operator in DestroyOperator:
        for ratio in ratios:
            native.add_destroy_operator(make_destroy(operator, ratio), name=f"{operator.name}/{ratio}")
    for mode in RepairMode:
        native.add_repair_operator(make_repair(mode), name=mode.name)
    selector = SegmentedRouletteWheel([8, 4, 2, 0], .8, 10, len(DestroyOperator) * len(ratios), len(RepairMode))
    start_temp = max(abs(initial.objective_value) * .05, 1e-8)
    accept = SimulatedAnnealing(start_temp, 1e-10, .995, method="exponential")
    result = native.iterate(InspectionState(initial), selector, accept,
                          lambda rng, best, current: attempted >= iterations or deadline.expired())
    best = result.best_state
    if not best.feasible:
        raise RuntimeError("authors' ALNS returned an infeasible state")
    diagnostics = (*best.solution.result.diagnostics, ("algorithm", "N-Wouda/ALNS inspection adaptation"),
        ("upstream_commit", commit), ("completed_iterations", str(attempted)), ("projection_rejections", str(invalid)),
        ("projection_corrections", str(corrections)), ("terminal_reason", "wall_clock_limit" if deadline.expired() else "iteration_limit"))
    return dataclasses.replace(best.solution.result, diagnostics=diagnostics, runtime_seconds=time.perf_counter() - started)
