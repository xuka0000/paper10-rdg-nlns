"""为 RDG-NLNS 检查点生成确定性的源码和语义标识。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from Agent.hybrid.rdg_nlns.operators import DestroyOperator
from Agent.hybrid.rdg_nlns.repair import RepairMode


ACTOR_ACTION_CYCLE_SEMANTICS = "exclude-until-current-valid-joint-set-exhausted-v1"
ACTOR_OBJECTIVE_SEMANTICS = "full-joint-risk-critic-expectation-v1"
CANDIDATE_ACCEPTANCE_SEMANTICS = "independently-validated-nonworsening-v1"
DISTRIBUTIONAL_TARGET_SEMANTICS = "conditional-independent-empirical-bellman-v1"
MANUSCRIPT_DESTROY_RATIOS = (0.1, 0.2, 0.3, 0.4)
MANUSCRIPT_DESTROY_ACTIONS = len(DestroyOperator) * len(MANUSCRIPT_DESTROY_RATIOS)
MANUSCRIPT_REPAIR_ACTIONS = len(RepairMode)
MANUSCRIPT_REWARD_SAMPLE_COUNT = 32
MANUSCRIPT_REWARD_TIME_PENALTY = 0.0
MANUSCRIPT_REWARD_ENERGY_WEIGHT = 1.0
MANUSCRIPT_QUANTILE_COUNT = 32
MANUSCRIPT_RISK_ALPHA = 0.25
CONTRACT_KEYS = frozenset({
    "schema_version",
    "destroy_operator_names",
    "destroy_ratios",
    "repair_mode_names",
    "initial_solution_mode",
    "graph_state_mode",
    "risk_score_mode",
    "repair_policy_mode",
    "distributional_samples",
    "reward_sample_count",
    "failure_penalty",
    "time_penalty",
    "time_scale_seconds",
    "energy_weight",
    "insertion_weights",
    "quantile_count",
    "risk_alpha",
    "actor_action_cycle_semantics",
    "actor_objective_semantics",
    "candidate_acceptance_semantics",
    "distributional_target_semantics",
})
INSERTION_WEIGHT_KEYS = frozenset({
    "energy",
    "duration",
    "shared_uav",
    "stop_discontinuity",
})


def algorithm_source_fingerprint(project_root: Path | None = None) -> str:
    """Hash every Python source file that can affect training or deployment."""
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    paths = {
        path.resolve()
        for directory_name in ("Agent", "Env", "Mask")
        for path in (root / directory_name).rglob("*.py")
        if path.is_file()
    }
    paths.update(
        path.resolve()
        for name in ("train_rdg_nlns.py", "run_p0.py", "validate_formal_suite.py")
        if (path := root / name).is_file()
    )
    if not paths:
        raise RuntimeError(f"RDG-NLNS algorithm source set is empty: {root}")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_source_fingerprint(value: object) -> str:
    """Return one canonical SHA-256 source fingerprint."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("algorithm source fingerprint must be a SHA-256 hex string")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("algorithm source fingerprint must be a SHA-256 hex string")
    return normalized


def _finite_float(value: object, field: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"algorithm contract {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        raise ValueError(f"algorithm contract {field} has an invalid value")
    return result


def normalize_algorithm_contract(value: object) -> dict[str, Any]:
    """Validate and normalize the closed checkpoint/runtime semantic contract."""
    if not isinstance(value, Mapping) or set(value) != CONTRACT_KEYS:
        raise ValueError("algorithm contract fields are incomplete or unknown")
    if value["schema_version"] != 2:
        raise ValueError("algorithm contract schema version is unsupported")
    destroy_names = tuple(str(item) for item in value["destroy_operator_names"])
    repair_names = tuple(str(item) for item in value["repair_mode_names"])
    ratios = tuple(_finite_float(item, "destroy_ratios", positive=True) for item in value["destroy_ratios"])
    if (
        destroy_names != tuple(operator.name for operator in DestroyOperator)
        or repair_names != tuple(mode.name for mode in RepairMode)
        or not ratios
        or len(ratios) != len(set(ratios))
        or any(ratio > 1.0 for ratio in ratios)
    ):
        raise ValueError("algorithm contract action space is invalid")
    insertion = value["insertion_weights"]
    if not isinstance(insertion, Mapping) or set(insertion) != INSERTION_WEIGHT_KEYS:
        raise ValueError("algorithm contract insertion weights are invalid")
    string_fields = (
        "initial_solution_mode",
        "graph_state_mode",
        "risk_score_mode",
        "repair_policy_mode",
        "actor_action_cycle_semantics",
        "actor_objective_semantics",
        "candidate_acceptance_semantics",
        "distributional_target_semantics",
    )
    strings = {field: str(value[field]) for field in string_fields}
    if any(not item for item in strings.values()):
        raise ValueError("algorithm contract string field is empty")
    if strings["actor_action_cycle_semantics"] != ACTOR_ACTION_CYCLE_SEMANTICS:
        raise ValueError("algorithm contract Actor action-cycle semantics are unsupported")
    if strings["actor_objective_semantics"] != ACTOR_OBJECTIVE_SEMANTICS:
        raise ValueError("algorithm contract Actor objective semantics are unsupported")
    if strings["candidate_acceptance_semantics"] != CANDIDATE_ACCEPTANCE_SEMANTICS:
        raise ValueError("algorithm contract candidate-acceptance semantics are unsupported")
    if strings["distributional_target_semantics"] != DISTRIBUTIONAL_TARGET_SEMANTICS:
        raise ValueError("algorithm contract distributional target semantics are unsupported")
    distributional = value["distributional_samples"]
    sample_count = value["reward_sample_count"]
    quantile_count = value["quantile_count"]
    if not isinstance(distributional, bool):
        raise ValueError("algorithm contract distributional_samples must be boolean")
    if type(sample_count) is not int or sample_count <= 0 or type(quantile_count) is not int or quantile_count <= 0:
        raise ValueError("algorithm contract sample and quantile counts must be positive integers")
    risk_alpha = _finite_float(value["risk_alpha"], "risk_alpha", positive=True)
    if risk_alpha > 1.0:
        raise ValueError("algorithm contract risk_alpha must lie in (0, 1]")
    return {
        "schema_version": 2,
        "destroy_operator_names": list(destroy_names),
        "destroy_ratios": list(ratios),
        "repair_mode_names": list(repair_names),
        **strings,
        "distributional_samples": distributional,
        "reward_sample_count": sample_count,
        "failure_penalty": _finite_float(value["failure_penalty"], "failure_penalty", nonnegative=True),
        "time_penalty": _finite_float(value["time_penalty"], "time_penalty", nonnegative=True),
        "time_scale_seconds": _finite_float(value["time_scale_seconds"], "time_scale_seconds", positive=True),
        "energy_weight": _finite_float(value["energy_weight"], "energy_weight", nonnegative=True),
        "insertion_weights": {
            key: _finite_float(insertion[key], f"insertion_weights.{key}", nonnegative=True)
            for key in sorted(INSERTION_WEIGHT_KEYS)
        },
        "quantile_count": quantile_count,
        "risk_alpha": risk_alpha,
    }


def build_algorithm_contract(training: object, **overrides: object) -> dict[str, Any]:
    """Build the runtime contract from one resolved training configuration."""
    values = {
        "schema_version": 2,
        "destroy_operator_names": [operator.name for operator in DestroyOperator],
        "destroy_ratios": list(training.destroy_ratios),
        "repair_mode_names": [mode.name for mode in RepairMode],
        "initial_solution_mode": training.initial_solution_mode,
        "graph_state_mode": training.graph_state_mode,
        "risk_score_mode": training.risk_score_mode,
        "repair_policy_mode": training.repair_policy_mode,
        "distributional_samples": training.distributional_samples,
        "reward_sample_count": training.reward_sample_count,
        "failure_penalty": training.failure_penalty,
        "time_penalty": training.time_penalty,
        "time_scale_seconds": training.time_scale_seconds,
        "energy_weight": training.energy_weight,
        "insertion_weights": {
            "energy": training.insertion_energy_weight,
            "duration": training.insertion_duration_weight,
            "shared_uav": training.insertion_shared_uav_weight,
            "stop_discontinuity": training.insertion_stop_discontinuity_weight,
        },
        "quantile_count": training.quantile_count,
        "risk_alpha": training.risk_alpha,
        "actor_action_cycle_semantics": ACTOR_ACTION_CYCLE_SEMANTICS,
        "actor_objective_semantics": ACTOR_OBJECTIVE_SEMANTICS,
        "candidate_acceptance_semantics": CANDIDATE_ACCEPTANCE_SEMANTICS,
        "distributional_target_semantics": DISTRIBUTIONAL_TARGET_SEMANTICS,
    }
    unknown = set(overrides) - CONTRACT_KEYS
    if unknown:
        raise ValueError(f"unknown algorithm contract overrides: {sorted(unknown)}")
    values.update(overrides)
    return normalize_algorithm_contract(values)


def manuscript_algorithm_contract() -> dict[str, Any]:
    """Return the deployed main-method contract stated in the manuscript."""
    class ManuscriptTraining:
        destroy_ratios = MANUSCRIPT_DESTROY_RATIOS
        initial_solution_mode = "greedy"
        graph_state_mode = "static_dynamic"
        risk_score_mode = "lower_tail_cvar"
        repair_policy_mode = "conditional"
        distributional_samples = True
        reward_sample_count = MANUSCRIPT_REWARD_SAMPLE_COUNT
        failure_penalty = 10.0
        time_penalty = MANUSCRIPT_REWARD_TIME_PENALTY
        time_scale_seconds = 600.0
        energy_weight = MANUSCRIPT_REWARD_ENERGY_WEIGHT
        insertion_energy_weight = 1.0
        insertion_duration_weight = 1.0
        insertion_shared_uav_weight = 1000.0
        insertion_stop_discontinuity_weight = 10.0
        quantile_count = MANUSCRIPT_QUANTILE_COUNT
        risk_alpha = MANUSCRIPT_RISK_ALPHA

    return build_algorithm_contract(ManuscriptTraining())


def algorithm_contract_hash(contract: object) -> str:
    normalized = normalize_algorithm_contract(contract)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
