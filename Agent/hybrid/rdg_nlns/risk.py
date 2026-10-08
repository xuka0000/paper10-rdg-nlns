"""Risk and reward primitives shared by RDG-NLNS training and inference. 中文说明。"""

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import math
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from collections.abc import Sequence


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def normalized_improvement(current_objective: float, candidate_objective: float, epsilon: float = 1e-9) -> float:
    """Return positive normalized gain when a minimization candidate improves. 中文说明。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    values = (current_objective, candidate_objective, epsilon)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not all(math.isfinite(value) for value in values) or epsilon <= 0.0:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("Objectives must be finite and epsilon must be positive")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return (current_objective - candidate_objective) / max(abs(current_objective), epsilon)


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def lower_tail_cvar(quantiles: Sequence[float], alpha: float) -> float:
    """Average the worst alpha fraction of an improvement distribution. 中文说明。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not 0.0 < alpha <= 1.0:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("alpha must lie in (0, 1]")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not quantiles:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("quantiles must not be empty")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    ordered = sorted(float(value) for value in quantiles)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not all(math.isfinite(value) for value in ordered):
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("quantiles must be finite")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    count = max(1, math.ceil(alpha * len(ordered)))
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return sum(ordered[:count]) / count
