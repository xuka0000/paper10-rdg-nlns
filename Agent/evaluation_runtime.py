"""提供统一的候选方案评估运行时和终止状态记录。"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass


TERMINAL_REASONS = frozenset(
    {
        "iteration_limit",
        "wall_clock_limit",
        "temperature_limit",
        "stagnation_limit",
        "empty_action_space",
        "initialization_failure",
        "constructive_complete",
        "solver_complete",
        "exception",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationRuntimePolicy:
    wall_clock_limit_seconds: float
    projection_time_limit_seconds: float
    direct_milp_time_limit_seconds: float
    initialization_projection_time_limit_seconds: float = 30.0

    def __post_init__(self) -> None:
        values = dataclasses.astuple(self)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("runtime policy limits must be finite and positive")

    @property
    def canonical_payload(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in dataclasses.asdict(self).items()
        }

    @property
    def canonical_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class RuntimeDeadline:
    limit_seconds: float
    started_at: float
    clock: Callable[[], float]

    @classmethod
    def start(
        cls,
        limit_seconds: float,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> "RuntimeDeadline":
        if not math.isfinite(limit_seconds) or limit_seconds <= 0.0:
            raise ValueError("deadline limit must be finite and positive")
        return cls(float(limit_seconds), float(clock()), clock)

    def elapsed_seconds(self) -> float:
        return max(0.0, float(self.clock()) - self.started_at)

    def remaining_seconds(self) -> float:
        return max(0.0, self.limit_seconds - self.elapsed_seconds())

    def expired(self) -> bool:
        return self.remaining_seconds() <= 0.0
