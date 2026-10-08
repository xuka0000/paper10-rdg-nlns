"""Immutable solution representation used by RDG-NLNS search transitions. 中文说明。"""

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from dataclasses import dataclass

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.results import AgentResult


@dataclass(frozen=True, slots=True)
class CandidatePlan:  # 中文说明。
    """A deterministic column selection before timing and route projection. 中文说明。"""

    selected_column_ids: tuple[str, ...]  # 中文说明。
    deferred_checkpoint_ids: tuple[str, ...]  # 中文说明。
    insertion_order: tuple[str, ...] = ()  # 中文说明。


@dataclass(frozen=True, slots=True)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class SearchSolution:
    """Bind a common agent result to its independently assessed feasibility. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    result: AgentResult
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    objective_value: float
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    selected_column_ids: tuple[str, ...]
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    is_feasible: bool

    @classmethod
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def from_result(cls, result: AgentResult, is_feasible: bool) -> "SearchSolution":
        """Create a search solution only from a result with a finite objective. 中文说明。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if result.objective_value is None:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("A search solution requires an objective value")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return cls(
            result=result,
            objective_value=float(result.objective_value),
            selected_column_ids=tuple(result.selected_column_ids),
            is_feasible=bool(is_feasible),
        )
