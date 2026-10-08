"""Shared projection boundary for behaviorally distinct baselines. 中文说明。"""

# 中文说明。
import dataclasses
# 中文说明。
from dataclasses import dataclass

# 中文说明。
import torch

# 中文说明。
from Agent.base_agent import BaseAgent
# 中文说明。
from Agent.evaluation_runtime import RuntimeDeadline
# 中文说明。
from Agent.hybrid.rdg_nlns.initialization import (
    build_greedy_initial_solution,
    solver_with_time_limit,
)
from Agent.hybrid.rdg_nlns.risk import normalized_improvement
# 中文说明。
from Agent.hybrid.rdg_nlns.solution import SearchSolution
# 中文说明。
from Agent.results import AgentResult
# 中文说明。
from Env.snapshot import SceneSnapshot
# 中文说明。
from Env.validation import validate_selected_columns


# 中文说明。
def result_validator(snapshot: SceneSnapshot):
    """中文说明。"""
    # 中文说明。
    return lambda result: validate_selected_columns(snapshot, result)


# 中文说明。
def initial_solution(
    snapshot: SceneSnapshot,
    solver: BaseAgent,
    mode: str,
    *,
    deadline: RuntimeDeadline | None = None,
    route_projection_time_limit_seconds: float = 10.0,
) -> SearchSolution:
    """中文说明。"""
    # 中文说明。
    if mode == "greedy":
        # 中文说明。
        return build_greedy_initial_solution(
            snapshot,
            solver,
            result_validator(snapshot),
            deadline=deadline,
            route_projection_time_limit_seconds=route_projection_time_limit_seconds,
        )
    # 中文说明。
    if mode != "exact":
        # 中文说明。
        raise ValueError("initial solution mode must be greedy or exact")
    # 中文说明。
    if deadline is not None and deadline.expired():
        raise RuntimeError("baseline initial solution failed: ('runtime_deadline',)")
    result = solver_with_time_limit(
        solver,
        route_projection_time_limit_seconds,
    ).solve(snapshot)
    # 中文说明。
    validation = result_validator(snapshot)(result)
    # 中文说明。
    if not result.success or result.objective_value is None or not validation.is_valid:
        # 中文说明。
        raise RuntimeError(f"baseline initial solution failed: {validation.issue_codes}")
    # 中文说明。
    return SearchSolution.from_result(result, True)


def normalized_objective_reward(
    current_objective: float,
    candidate_objective: float,
) -> float:
    """Return a scale-free minimization reward bounded by the failure penalty."""
    improvement = normalized_improvement(current_objective, candidate_objective)
    return max(-1.0, min(1.0, improvement))


# 中文说明。
def with_diagnostics(result: AgentResult, runtime_seconds: float, diagnostics: tuple[tuple[str, str], ...]) -> AgentResult:
    """中文说明。"""
    # 中文说明。
    return dataclasses.replace(
        result,
        diagnostics=(*result.diagnostics, *diagnostics),
        runtime_seconds=float(runtime_seconds),
    )


@dataclass(frozen=True, slots=True)
# 中文说明。
class BaselineTrainingEpisode:
    """Return a projected result and its differentiable policy objective. 中文说明。"""

    # 中文说明。
    result: AgentResult
    # 中文说明。
    loss: torch.Tensor | None
    # 中文说明。
    reward_sum: float
    # 中文说明。
    decision_count: int
    # 中文说明。
    gradients_accumulated: bool = False


# 中文说明。
def select_policy_action(
    logits: torch.Tensor,
    *,
    stochastic: bool,
    generator: torch.Generator,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Select one discrete action while retaining log-probability and entropy. 中文说明。"""
    # 中文说明。
    if logits.ndim != 1 or logits.numel() == 0 or not torch.isfinite(logits).all():
        # 中文说明。
        raise ValueError("policy logits must be a finite nonempty vector")
    # 中文说明。
    log_probabilities = torch.log_softmax(logits, dim=-1)
    # 中文说明。
    probabilities = log_probabilities.exp()
    # 中文说明。
    if stochastic:
        # 中文说明。
        action = int(torch.multinomial(probabilities, 1, generator=generator).item())
    else:
        # 中文说明。
        action = int(torch.argmax(logits).item())
    # 中文说明。
    entropy = -(probabilities * log_probabilities).sum()
    # 中文说明。
    return action, log_probabilities[action], entropy
