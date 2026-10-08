"""Risk-distributional dual-stage actor--critic for RDG-NLNS. 中文说明。"""

import math
from dataclasses import dataclass

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import torch
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from torch import nn
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from torch.nn import functional as F

from Agent.reinforcement_learning.algorithm_identity import (
    MANUSCRIPT_DESTROY_ACTIONS,
    MANUSCRIPT_QUANTILE_COUNT,
    MANUSCRIPT_REPAIR_ACTIONS,
)

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.graph_neural_network.heterogeneous_service_graph import DYNAMIC_FEATURE_WIDTH, GLOBAL_FEATURE_WIDTH, STATIC_FEATURE_WIDTH, HeterogeneousServiceGraph


def score_action_quantiles(values: torch.Tensor, mode: str, alpha: float) -> torch.Tensor:
    """Aggregate critic quantiles using the configured risk semantics."""
    if values.ndim < 1 or values.shape[-1] <= 0:
        raise ValueError("Action quantiles must contain a non-empty quantile dimension")
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("risk_alpha must lie in (0, 1]")
    if mode == "mean":
        return values.mean(dim=-1)
    if mode != "lower_tail_cvar":
        raise ValueError("risk_score_mode is invalid")
    tail_count = max(1, math.ceil(alpha * values.shape[-1]))
    return values.sort(dim=-1).values[..., :tail_count].mean(dim=-1)


@dataclass(frozen=True, slots=True)
class ActorActionSelection:
    """Record the action selected by the factorized Actor policy."""

    selected_action: tuple[int, int]
    destroy_probability: float
    repair_probability: float
    joint_probability: float


def actor_action_statistics(
    joint_log_probabilities: torch.Tensor,
    selected_action: tuple[int, int],
) -> ActorActionSelection:
    """Return component and joint probabilities for one Actor action."""
    normalized = F.log_softmax(joint_log_probabilities.reshape(-1), dim=0).reshape(
        joint_log_probabilities.shape
    )
    destroy_id, repair_id = selected_action
    destroy_log_probability = torch.logsumexp(normalized[destroy_id], dim=0)
    joint_log_probability = normalized[destroy_id, repair_id]
    return ActorActionSelection(
        selected_action=selected_action,
        destroy_probability=float(destroy_log_probability.exp().detach()),
        repair_probability=float((joint_log_probability - destroy_log_probability).exp().detach()),
        joint_probability=float(joint_log_probability.exp().detach()),
    )


def graph_destroy_action_mask(
    graph: HeterogeneousServiceGraph,
    action_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a graph-carried Destroy mask, using all actions for legacy graphs."""
    mask = graph.destroy_action_mask
    if mask.numel() == 0:
        return torch.ones(action_count, dtype=torch.bool, device=device)
    if mask.shape != (action_count,) or mask.dtype != torch.bool:
        raise ValueError("Graph destroy action mask has invalid shape or dtype")
    return mask.to(device=device)


def select_actor_action(
    joint_log_probabilities: torch.Tensor,
    excluded_actions: frozenset[tuple[int, int]] = frozenset(),
) -> ActorActionSelection:
    """Select the highest-probability untried joint action from the Actors."""
    if joint_log_probabilities.ndim != 2:
        raise ValueError("Joint actor scores must be a destroy-by-repair matrix")
    if bool(torch.isnan(joint_log_probabilities).any().item()) or bool(
        torch.isposinf(joint_log_probabilities).any().item()
    ):
        raise ValueError("Joint actor scores must not contain NaN or positive infinity")

    destroy_actions, repair_actions = joint_log_probabilities.shape
    invalid_exclusions = tuple(
        action
        for action in excluded_actions
        if (
            len(action) != 2
            or not 0 <= action[0] < destroy_actions
            or not 0 <= action[1] < repair_actions
        )
    )
    if invalid_exclusions:
        raise ValueError("Excluded joint actor actions must be valid action indices")
    available_actions = tuple(
        (destroy_id, repair_id)
        for destroy_id in range(destroy_actions)
        for repair_id in range(repair_actions)
        if (destroy_id, repair_id) not in excluded_actions
        and bool(torch.isfinite(joint_log_probabilities[destroy_id, repair_id]).item())
    )
    if not available_actions:
        raise ValueError("At least one untried joint actor action is required")
    selected = min(
        available_actions,
        key=lambda action: (-float(joint_log_probabilities[action].detach()), action[0], action[1]),
    )
    return actor_action_statistics(joint_log_probabilities, selected)


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def masked_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Replace invalid categorical actions by negative infinity. 中文说明。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if logits.shape != valid_mask.shape:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("Action logits and validity mask must have equal shape")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if valid_mask.dtype != torch.bool:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("Action validity mask must be boolean")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not bool(valid_mask.any()):
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("At least one action must be valid")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return logits.masked_fill(~valid_mask, -torch.inf)


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RelationalMessageLayer(nn.Module):
    """Aggregate relation-specific messages without external graph packages. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __init__(self, hidden_dim: int, relation_count: int) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        super().__init__()
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.relation_weights = nn.Parameter(torch.empty(relation_count, hidden_dim, hidden_dim))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        nn.init.xavier_uniform_(self.relation_weights)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if edge_index.shape[1] == 0:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            return F.relu(self.self_linear(nodes))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        sources, targets = edge_index
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        aggregated = torch.zeros_like(nodes)
        for relation_id in range(self.relation_weights.shape[0]):
            relation_mask = edge_types == relation_id
            if not bool(relation_mask.any()):
                continue
            relation_sources = sources[relation_mask]
            relation_targets = targets[relation_mask]
            relation_messages = F.linear(
                nodes[relation_sources],
                self.relation_weights[relation_id],
            )
            relation_sum = torch.zeros_like(nodes)
            relation_sum.index_add_(0, relation_targets, relation_messages)
            relation_degree = torch.zeros(
                nodes.shape[0], dtype=nodes.dtype, device=nodes.device
            )
            relation_degree.index_add_(
                0,
                relation_targets,
                torch.ones_like(relation_targets, dtype=nodes.dtype),
            )
            aggregated = aggregated + relation_sum / relation_degree.clamp_min(1.0).unsqueeze(-1)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return F.relu(self.self_linear(nodes) + aggregated)


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class StaticDynamicGraphEncoder(nn.Module):
    """Fuse independently encoded physical and rolling schedule graphs. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __init__(self, hidden_dim: int, relation_count: int = 5, node_type_count: int = 4) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        super().__init__()
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.node_type_embedding = nn.Embedding(node_type_count, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.static_input = nn.Linear(STATIC_FEATURE_WIDTH, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.dynamic_input = nn.Linear(DYNAMIC_FEATURE_WIDTH, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.global_input = nn.Linear(GLOBAL_FEATURE_WIDTH, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.static_layers = nn.ModuleList(RelationalMessageLayer(hidden_dim, relation_count) for _ in range(2))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.dynamic_layers = nn.ModuleList(RelationalMessageLayer(hidden_dim, relation_count) for _ in range(2))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.output = nn.Linear(3 * hidden_dim, hidden_dim)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def forward(self, graph: HeterogeneousServiceGraph) -> torch.Tensor:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        device = next(self.parameters()).device
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        node_types = graph.node_types.to(device)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        edge_index = graph.edge_index.to(device)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        edge_types = graph.edge_types.to(device)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        type_features = self.node_type_embedding(node_types)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        static = F.relu(self.static_input(graph.static_features.to(device)) + type_features)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        dynamic = F.relu(self.dynamic_input(graph.dynamic_features.to(device)) + type_features)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        for layer in self.static_layers:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            static = layer(static, edge_index, edge_types)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        for layer in self.dynamic_layers:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            dynamic = layer(dynamic, edge_index, edge_types)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        gate = torch.sigmoid(self.fusion_gate(torch.cat((static, dynamic), dim=-1)))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        fused = gate * static + (1.0 - gate) * dynamic
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        pooled = torch.cat((fused.mean(dim=0), fused.max(dim=0).values), dim=-1)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        global_state = torch.relu(self.global_input(graph.global_features.to(device)))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return torch.tanh(self.output(torch.cat((pooled, global_state), dim=-1)))


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RDGDAC(nn.Module):
    """Dual actors and a joint quantile critic over neighborhood actions. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __init__(
        self,
        hidden_dim: int = 64,
        destroy_actions: int = MANUSCRIPT_DESTROY_ACTIONS,
        repair_actions: int = MANUSCRIPT_REPAIR_ACTIONS,
        quantile_count: int = MANUSCRIPT_QUANTILE_COUNT,
    ) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        super().__init__()
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if min(hidden_dim, destroy_actions, repair_actions, quantile_count) <= 0:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("RDG-DAC dimensions must be positive")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.destroy_actions = destroy_actions
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.repair_actions = repair_actions
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.quantile_count = quantile_count
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.encoder = StaticDynamicGraphEncoder(hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.destroy_actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, destroy_actions))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.destroy_embedding = nn.Embedding(destroy_actions, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.repair_embedding = nn.Embedding(repair_actions, hidden_dim)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.repair_actor = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, repair_actions))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.critic = nn.Sequential(nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, quantile_count))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.register_buffer("quantile_fractions", (torch.arange(quantile_count, dtype=torch.float32) + 0.5) / quantile_count)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def encode(self, graph: HeterogeneousServiceGraph) -> torch.Tensor:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return self.encoder(graph)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def destroy_logits(self, state: torch.Tensor) -> torch.Tensor:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return self.destroy_actor(state)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def repair_logits(self, state: torch.Tensor, destroy_action: torch.Tensor, condition_on_destroy: bool = True) -> torch.Tensor:
        """根据破坏动作条件或零条件消融返回修复策略 logits。"""
        # 根据消融开关读取真实破坏嵌入或同形零向量。
        action_embedding = self.destroy_embedding(destroy_action.to(state.device)) if condition_on_destroy else torch.zeros_like(state)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return self.repair_actor(torch.cat((state, action_embedding), dim=-1))

    def joint_log_probabilities(
        self,
        state: torch.Tensor,
        condition_on_destroy: bool = True,
        destroy_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return log pi(d, r | s) for every destroy-repair action pair."""
        destroy_logits = self.destroy_logits(state)
        if destroy_valid_mask is not None:
            destroy_logits = masked_logits(
                destroy_logits,
                destroy_valid_mask.to(device=destroy_logits.device),
            )
        destroy_log_probabilities = F.log_softmax(destroy_logits, dim=-1)
        repair_log_probabilities = torch.stack(
            tuple(
                F.log_softmax(
                    self.repair_logits(
                        state,
                        torch.tensor(destroy_id, device=state.device),
                        condition_on_destroy=condition_on_destroy,
                    ),
                    dim=-1,
                )
                for destroy_id in range(self.destroy_actions)
            )
        )
        return destroy_log_probabilities[:, None] + repair_log_probabilities

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def action_quantiles(self, state: torch.Tensor) -> torch.Tensor:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        destroy_ids = torch.arange(self.destroy_actions, device=state.device)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        repair_ids = torch.arange(self.repair_actions, device=state.device)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        destroy_grid = destroy_ids[:, None].expand(self.destroy_actions, self.repair_actions).reshape(-1)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        repair_grid = repair_ids[None, :].expand(self.destroy_actions, self.repair_actions).reshape(-1)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        state_grid = state.reshape(1, -1).expand(destroy_grid.shape[0], -1)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        critic_input = torch.cat(
            (state_grid, self.destroy_embedding(destroy_grid), self.repair_embedding(repair_grid)),
            dim=-1,
        )
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        quantiles = self.critic(critic_input)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return quantiles.reshape(self.destroy_actions, self.repair_actions, self.quantile_count)
