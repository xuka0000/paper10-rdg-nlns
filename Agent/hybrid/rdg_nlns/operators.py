"""Deterministic and stochastic destruction operators for RDG-NLNS. 中文说明。"""

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import math
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import random
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from dataclasses import dataclass
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from enum import IntEnum
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from typing import Any

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.solution import SearchSolution


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class DestroyOperator(IntEnum):
    """Stable action identifiers used by the destroy actor. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    RANDOM = 0
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    WORST = 1
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    RELATED = 2


@dataclass(frozen=True, slots=True)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class DestroyAction:
    """Record an auditable destruction decision. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    operator: DestroyOperator
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    ratio: float
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    removed_column_ids: tuple[str, ...]
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    retained_column_ids: tuple[str, ...]


def _locked_column_ids(snapshot: Any) -> set[str]:
    """Return every column protected by the rolling commitment boundary."""
    return {
        *getattr(snapshot, "locked_column_ids", ()),
        *getattr(snapshot, "committed_column_ids", ()),
        *(state.column_id for state in getattr(snapshot, "airborne_residual_states", ())),
    }


def admissible_destroy_action_mask(
    snapshot: Any,
    solution: SearchSolution,
    destroy_ratios: tuple[float, ...],
) -> tuple[bool, ...]:
    """Return the operator-major mask for ratios executable without touching locks."""
    if not destroy_ratios or any(not 0.0 < ratio <= 1.0 for ratio in destroy_ratios):
        raise ValueError("Destroy ratios must be non-empty and lie in (0, 1]")
    selected = tuple(solution.selected_column_ids)
    locked = _locked_column_ids(snapshot)
    revisable_count = sum(column_id not in locked for column_id in selected)
    ratio_mask = tuple(
        bool(selected)
        and max(1, math.ceil(ratio * len(selected))) <= revisable_count
        for ratio in destroy_ratios
    )
    return ratio_mask * len(DestroyOperator)


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def _relatedness(
    first: Any,
    second: Any,
    first_checkpoint_ids: tuple[str, ...],
    second_checkpoint_ids: tuple[str, ...],
) -> float:
    """说明当前算法或测试步骤。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    first_checkpoints = set(first_checkpoint_ids)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    second_checkpoints = set(second_checkpoint_ids)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    union = first_checkpoints | second_checkpoints
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    checkpoint_score = len(first_checkpoints & second_checkpoints) / max(1, len(union))
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    stop_score = float(first.launch_stop_id == second.launch_stop_id) + float(first.recovery_stop_id == second.recovery_stop_id)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    uav_score = float(first.uav_id == second.uav_id)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return 3.0 * checkpoint_score + stop_score + uav_score


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def destroy_neighborhood(snapshot: Any, solution: SearchSolution, operator: DestroyOperator, ratio: float, rng: random.Random) -> DestroyAction:
    """Remove selected unlocked columns using the requested neighborhood rule. 中文说明。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not 0.0 < ratio <= 1.0:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("Destroy ratio must lie in (0, 1]")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    selected = tuple(solution.selected_column_ids)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    locked = _locked_column_ids(snapshot)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    candidates = tuple(column_id for column_id in selected if column_id not in locked)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not candidates:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return DestroyAction(operator, ratio, (), selected)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    remove_count = max(1, math.ceil(ratio * len(selected)))
    if remove_count > len(candidates):
        return DestroyAction(operator, ratio, (), selected)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    column_by_id = {column.object_id: column for column in snapshot.service_columns}
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if any(column_id not in column_by_id for column_id in candidates):
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("Selected solution references an unknown service column")
    effective_by_id = {
        column_id: snapshot.column_state(column_id) for column_id in candidates
    }

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if operator == DestroyOperator.RANDOM:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        removed = tuple(rng.sample(candidates, remove_count))
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    elif operator == DestroyOperator.WORST:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        removed = tuple(
            sorted(
                candidates,
                key=lambda column_id: (
                    -(
                        float(effective_by_id[column_id].effective_protected_energy)
                        + float(effective_by_id[column_id].effective_duration)
                    ),
                    column_id,
                ),
            )[:remove_count]
        )
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    elif operator == DestroyOperator.RELATED:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        seed_id = rng.choice(candidates)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        seed = column_by_id[seed_id]
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        remainder = tuple(column_id for column_id in candidates if column_id != seed_id)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        related = sorted(
            remainder,
            key=lambda column_id: (
                -_relatedness(
                    seed,
                    column_by_id[column_id],
                    effective_by_id[seed_id].effective_checkpoint_ids,
                    effective_by_id[column_id].effective_checkpoint_ids,
                ),
                column_id,
            ),
        )
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        removed = (seed_id, *related[: remove_count - 1])
    else:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError(f"Unsupported destroy operator: {operator}")

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    removed_set = set(removed)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    retained = tuple(column_id for column_id in selected if column_id not in removed_set)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return DestroyAction(operator, ratio, tuple(removed), retained)
