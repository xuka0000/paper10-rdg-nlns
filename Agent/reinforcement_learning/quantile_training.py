"""Distributional training and checkpointing for RDG-DAC. 中文说明。"""

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import copy
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import math
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import random
import statistics
# 导入映射协议以验证训练器 checkpoint 状态字段。
from collections.abc import Mapping
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from dataclasses import asdict, dataclass
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from pathlib import Path

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import numpy as np
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import torch
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from torch.nn import functional as F

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.risk import normalized_improvement
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.solution import SearchSolution
from Agent.reinforcement_learning.algorithm_identity import (
    DISTRIBUTIONAL_TARGET_SEMANTICS,
    MANUSCRIPT_DESTROY_ACTIONS,
    MANUSCRIPT_QUANTILE_COUNT,
    MANUSCRIPT_REPAIR_ACTIONS,
    MANUSCRIPT_REWARD_ENERGY_WEIGHT,
    MANUSCRIPT_REWARD_TIME_PENALTY,
)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.reinforcement_learning.rdg_dac import RDGDAC, graph_destroy_action_mask, masked_logits, score_action_quantiles, select_actor_action
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.reinforcement_learning.replay import SearchTransition


@dataclass(frozen=True, slots=True)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RewardConfig:
    """说明当前算法或测试步骤。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    failure_penalty: float = 1.0
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    time_penalty: float = MANUSCRIPT_REWARD_TIME_PENALTY
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    time_scale_seconds: float = 1.0
    # 设置每次可行修复的 Monte Carlo 收益样本数。
    sample_count: int = 32
    # 将 M4 能耗预测误差映射到式 50 目标扰动。
    energy_weight: float = MANUSCRIPT_REWARD_ENERGY_WEIGHT
    # 固定局部随机数种子以支持实验复现。
    seed: int = 0
    # 控制收益是否保留气象或高斯分布而非确定性均值消融。
    distributional_samples: bool = True

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __post_init__(self) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        values = (self.failure_penalty, self.time_penalty, self.time_scale_seconds, self.energy_weight)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if not all(math.isfinite(value) for value in values) or self.failure_penalty < 0.0 or self.time_penalty < 0.0 or self.time_scale_seconds <= 0.0 or self.sample_count <= 0 or self.energy_weight < 0.0 or not isinstance(self.distributional_samples, bool):
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Reward configuration values are invalid")


@dataclass(frozen=True, slots=True)  # 使用不可变数据类保存经验收益分布和来源。
class RewardDistribution:  # 定义 critic 训练使用的结构化收益对象。
    """保存固定宽度收益样本及其气象、高斯、失败或消融来源。"""  # 解释日志能够区分真实不确定性和确定性消融。

    samples: tuple[float, ...]  # 保存供分位 critic 学习的经验收益样本。
    source: str  # 保存收益分布生成机制的稳定标识。


@dataclass(frozen=True, slots=True)
class EnergyWeightStatistic:
    """Summarize the reward mean-to-CVaR separation for one candidate weight."""

    energy_weight: float
    trial_count: int
    median_absolute_mean_reward: float
    median_absolute_mean_cvar_gap: float
    ratio: float
    valid: bool


@dataclass(frozen=True, slots=True)
class EnergyWeightCalibration:
    """Record the smallest fixed weight that satisfies the calibration band."""

    energy_weight: float
    minimum_ratio: float
    maximum_ratio: float
    statistics: tuple[EnergyWeightStatistic, ...]


def select_energy_weight(
    trials: Mapping[float, tuple[tuple[float, float], ...]],
    minimum_ratio: float,
    maximum_ratio: float,
) -> EnergyWeightCalibration:
    """Select the smallest weight whose median mean-to-CVaR gap is informative."""
    if (
        not math.isfinite(minimum_ratio)
        or not math.isfinite(maximum_ratio)
        or minimum_ratio < 0.0
        or maximum_ratio < minimum_ratio
    ):
        raise ValueError("Calibration ratio bounds must be finite and ordered")
    if not isinstance(trials, Mapping) or not trials:
        raise ValueError("Calibration trials must be a non-empty mapping")

    candidate_statistics: list[EnergyWeightStatistic] = []
    for raw_weight, raw_trials in sorted(trials.items(), key=lambda item: float(item[0])):
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)) or not math.isfinite(float(raw_weight)) or float(raw_weight) <= 0.0:
            raise ValueError("Calibration weights must be finite positive numbers")
        observations = tuple(raw_trials)
        if not observations:
            raise ValueError("Each energy weight requires at least one calibration trial")
        parsed: list[tuple[float, float]] = []
        for observation in observations:
            if not isinstance(observation, tuple) or len(observation) != 2:
                raise ValueError("Each calibration trial must contain mean and CVaR")
            mean_reward, cvar = observation
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in observation):
                raise ValueError("Calibration trial values must be finite numbers")
            parsed.append((float(mean_reward), float(cvar)))
        median_mean = float(statistics.median(abs(mean_reward) for mean_reward, _ in parsed))
        median_gap = float(statistics.median(abs(mean_reward - cvar) for mean_reward, cvar in parsed))
        ratio = median_gap / max(median_mean, 1e-6)
        candidate_statistics.append(
            EnergyWeightStatistic(
                energy_weight=float(raw_weight),
                trial_count=len(parsed),
                median_absolute_mean_reward=median_mean,
                median_absolute_mean_cvar_gap=median_gap,
                ratio=ratio,
                valid=minimum_ratio <= ratio <= maximum_ratio,
            )
        )

    selected = next((item for item in candidate_statistics if item.valid), None)
    if selected is None:
        raise ValueError("No energy weight satisfies the calibration ratio band")
    return EnergyWeightCalibration(
        energy_weight=selected.energy_weight,
        minimum_ratio=minimum_ratio,
        maximum_ratio=maximum_ratio,
        statistics=tuple(candidate_statistics),
    )


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def post_repair_reward(current: SearchSolution, candidate: SearchSolution | None, repair_seconds: float, config: RewardConfig) -> float:
    """Assign both actors one reward only after repair and projection. 中文说明。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if not math.isfinite(repair_seconds) or repair_seconds < 0.0:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("Repair runtime must be finite and nonnegative")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if candidate is None or not candidate.is_feasible:
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return -config.failure_penalty
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    gain = normalized_improvement(current.objective_value, candidate.objective_value)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return gain


# 说明利用 M4 服务列能耗标准差生成同一状态动作的经验收益分布。
def post_repair_reward_samples(snapshot, current: SearchSolution, candidate: SearchSolution | None, repair_seconds: float, config: RewardConfig) -> RewardDistribution:
    """对可行候选的概率能耗误差采样，并返回归一化搜索收益。"""
    # 首先计算由完整修复和可行性投影决定的确定性基础收益。
    base_reward = post_repair_reward(current, candidate, repair_seconds, config)
    # 在确定性均值消融中重复基础收益并跳过全部不确定性来源。
    if not config.distributional_samples:
        # 返回固定宽度单点分布以保持 critic 接口不变。
        return RewardDistribution((base_reward,) * config.sample_count, "deterministic_mean")
    # 对不可行修复重复失败惩罚以保持固定经验分布尺寸。
    if candidate is None or not candidate.is_feasible:
        # 返回不含虚假随机改善的失败样本。
        return RewardDistribution((base_reward,) * config.sample_count, "projection_failure")
    # 建立服务列索引以读取 M4 已认证的逐场景能耗追踪。
    column_by_id = {column.object_id: column for column in snapshot.service_columns}
    action_seed = config.seed + sum(ord(character) for column_id in candidate.selected_column_ids for character in column_id)
    generator = random.Random(action_seed)
    selected_ids = tuple(dict.fromkeys((*current.selected_column_ids, *candidate.selected_column_ids)))  # 汇总当前解和候选解涉及的全部服务列。
    if any(column_id not in column_by_id for column_id in selected_ids):  # 判断奖励是否引用快照外服务列。
        raise ValueError("reward references an unknown service column")  # 拒绝不能追踪能耗的奖励转移。
    selected_columns = tuple(column_by_id[column_id] for column_id in selected_ids)  # 读取奖励涉及的完整服务列对象。
    scenario_columns = tuple(column for column in selected_columns if column.weather_certification_mode == "scenario_set")  # 提取正式场景集认证服务列。
    if scenario_columns and len(scenario_columns) != len(selected_columns):  # 判断一次奖励是否混用正式与兼容能耗证据。
        raise ValueError("reward cannot mix scenario-set and legacy weather certification")  # 拒绝混合证据来源。
    if scenario_columns:  # 使用服务列认证时保存的同一场景身份、概率和能耗。
        scenario_ids = scenario_columns[0].weather_scenario_ids  # 读取规范场景顺序。
        probabilities = scenario_columns[0].weather_scenario_probabilities  # 读取规范场景概率。
        for column in scenario_columns[1:]:  # 逐一核对所有奖励相关服务列的场景追踪。
            if column.weather_scenario_ids != scenario_ids or column.weather_scenario_probabilities != probabilities:  # 判断场景顺序或概率是否漂移。
                raise ValueError("selected service columns have different weather scenario traces")  # 失败关闭而不重排或重新归一化。

        def realized_objective(solution: SearchSolution, scenario_index: int) -> float:  # 定义一个已认证天气实现下的方案目标。
            """用逐场景能耗相对期望值的偏差修正确定性投影目标。"""  # 解释不会重新构造风速代理。
            deviation = math.fsum(column_by_id[column_id].weather_scenario_energies[scenario_index] - column_by_id[column_id].energy_mean for column_id in solution.selected_column_ids)  # 汇总所选列在同一实现下的能耗偏差。
            return solution.objective_value + config.energy_weight * deviation  # 使用训练奖励能耗权重形成实现目标。

        sampled_indices = generator.choices(range(len(scenario_ids)), weights=probabilities, k=config.sample_count)  # 按认证概率采样场景标识索引。
        samples = tuple(normalized_improvement(realized_objective(current, index), realized_objective(candidate, index)) for index in sampled_indices)  # 计算每个天气实现下的归一化搜索改善。
        return RewardDistribution(samples, "weather_scenarios")  # 返回与服务列认证同源的经验奖励分布。
    # 汇总候选方案所有已选服务列的独立能耗方差。
    energy_variance = sum(float(column_by_id[column_id].energy_standard_deviation) ** 2 for column_id in candidate.selected_column_ids if column_id in column_by_id)
    # 创建本次转移专用且可复现的局部随机数发生器。
    # 使用当前目标绝对值作为目标扰动的无量纲归一化尺度。
    objective_scale = max(abs(current.objective_value), 1.0)
    # 计算所选列聚合能耗误差的标准差。
    energy_sigma = math.sqrt(energy_variance)
    # 生成反映概率能耗误差对搜索改善影响的经验收益样本。
    samples = tuple(base_reward - config.energy_weight * generator.gauss(0.0, energy_sigma) / objective_scale for _ in range(config.sample_count))
    return RewardDistribution(samples, "gaussian_fallback")  # 中文说明。


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
def quantile_huber_loss(prediction: torch.Tensor, target: torch.Tensor, fractions: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Compute pairwise quantile regression with a Huber residual. 中文说明。"""
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    if (prediction.ndim != 2 or target.ndim != 2
            or prediction.shape[0] != target.shape[0]
            or prediction.shape[1] != fractions.numel()
            or target.shape[1] == 0):
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        raise ValueError("Quantile tensors have incompatible shapes")
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    residual = target.unsqueeze(1) - prediction.unsqueeze(2)
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    absolute = residual.abs()
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    huber = torch.where(absolute <= kappa, 0.5 * residual.square(), kappa * (absolute - 0.5 * kappa))
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    weights = torch.abs(fractions.view(1, -1, 1) - (residual.detach() < 0.0).to(prediction.dtype))
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    return (weights * huber / kappa).mean()


def empirical_bellman_samples(
    rewards: torch.Tensor,
    continuation: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    """Construct the conditional-independent empirical Bellman distribution.

    Reward draws and continuation atoms are paired by their Cartesian product,
    with equal mass on every pair. This assumes independent reward disturbances
    across search transitions conditional on the encoded state. Dependence among
    missions within one weather realization is retained by the reward sampler.
    A persistent unobserved weather variable would require a different state and
    joint sampler. Matching marginal quantile ranks does not implement this law.
    """
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("Bellman rewards must be a non-empty vector")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("Bellman discount must lie in [0, 1]")
    if not bool(torch.isfinite(rewards).all().item()):
        raise FloatingPointError("Bellman rewards are not finite")
    if continuation is None or gamma == 0.0:
        return rewards.detach()
    if continuation.ndim != 1 or continuation.numel() == 0:
        raise ValueError("Bellman continuation must be a non-empty vector")
    if not bool(torch.isfinite(continuation).all().item()):
        raise FloatingPointError("Bellman continuation is not finite")
    target = (
        rewards.to(continuation).unsqueeze(1)
        + gamma * continuation.unsqueeze(0)
    ).reshape(-1)
    if not bool(torch.isfinite(target).all().item()):
        raise FloatingPointError("Bellman target samples are not finite")
    return target.detach()


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RDGDACTrainer:
    """Train dual actors and a target-backed quantile critic. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __init__(self, hidden_dim: int = 64, destroy_actions: int = MANUSCRIPT_DESTROY_ACTIONS, repair_actions: int = MANUSCRIPT_REPAIR_ACTIONS, quantile_count: int = MANUSCRIPT_QUANTILE_COUNT, learning_rate: float = 3e-4, gamma: float = 0.95, entropy_coefficient: float = 0.01, target_tau: float = 0.01, risk_alpha: float = 0.25, seed: int = 0, risk_score_mode: str = "lower_tail_cvar", condition_on_destroy: bool = True) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if not 0.0 <= gamma <= 1.0 or not 0.0 < target_tau <= 1.0 or not 0.0 < risk_alpha <= 1.0:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Trainer discount, target rate, or risk level is invalid")
        # 要求训练风险评分属于完整 CVaR 或均值消融。
        if risk_score_mode not in {"lower_tail_cvar", "mean"}:
            # 拒绝未知 Actor 和目标动作选择行为。
            raise ValueError("Trainer risk_score_mode is invalid")
        # 要求修复 Actor 条件开关保持布尔语义。
        if not isinstance(condition_on_destroy, bool):
            # 拒绝隐式真值造成消融漂移。
            raise ValueError("Trainer condition_on_destroy must be boolean")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        random.seed(seed)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        np.random.seed(seed)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        torch.manual_seed(seed)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.model = RDGDAC(hidden_dim, destroy_actions, repair_actions, quantile_count)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.target_model = copy.deepcopy(self.model)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.target_model.eval()
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.gamma = gamma
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.entropy_coefficient = entropy_coefficient
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.target_tau = target_tau
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.risk_alpha = risk_alpha
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self.seed = seed
        # 保存 Actor 优势和目标动作选择使用的风险评分模式。
        self.risk_score_mode = risk_score_mode
        # 保存修复 Actor 是否读取破坏动作嵌入。
        self.condition_on_destroy = condition_on_destroy
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._architecture = {
            "hidden_dim": hidden_dim,
            "destroy_actions": destroy_actions,
            "repair_actions": repair_actions,
            "quantile_count": quantile_count,
            # 将风险评分行为纳入 checkpoint 结构兼容门禁。
            "risk_score_mode": risk_score_mode,
            # 将修复 Actor 条件行为纳入 checkpoint 结构兼容门禁。
            "condition_on_destroy": condition_on_destroy,
            "distributional_target_semantics": DISTRIBUTIONAL_TARGET_SEMANTICS,
        }

    # 说明计算 Actor 与推理阶段共同采用的经验下尾 CVaR。
    def empirical_cvar(self, samples: tuple[float, ...]) -> float:
        """返回收益样本最差 alpha 比例的均值。"""
        # 拒绝无法定义经验风险的空样本集合。
        if not samples:
            # 抛出明确的经验分布异常。
            raise ValueError("CVaR requires at least one reward sample")
        # 按下尾比例向上取整以保证至少使用一个样本。
        tail_count = max(1, math.ceil(self.risk_alpha * len(samples)))
        # 返回较低收益尾部的算术均值。
        return sum(sorted(float(value) for value in samples)[:tail_count]) / tail_count

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def _target_quantiles(self, transition: SearchTransition) -> torch.Tensor:
        """Return empirical Bellman samples using the actor-selected continuation."""
        device = next(self.model.parameters()).device
        reward_samples = torch.tensor(transition.reward_samples, dtype=torch.float32, device=device)
        if transition.done:
            return empirical_bellman_samples(reward_samples, None, self.gamma)
        with torch.no_grad():
            online_state = self.model.encode(transition.next_state_graph)
            next_destroy_mask = graph_destroy_action_mask(
                transition.next_state_graph,
                self.model.destroy_actions,
                online_state.device,
            )
            selection = select_actor_action(
                self.model.joint_log_probabilities(
                    online_state,
                    condition_on_destroy=self.condition_on_destroy,
                    destroy_valid_mask=next_destroy_mask,
                ),
            )
            target_state = self.target_model.encode(transition.next_state_graph)
            next_quantiles = self.target_model.action_quantiles(target_state)[selection.selected_action]
            return empirical_bellman_samples(reward_samples, next_quantiles, self.gamma)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def update(self, transitions: tuple[SearchTransition, ...]) -> dict[str, float]:
        """Fit Bellman return samples and optimize the actors' lower-tail scores."""
        if not transitions:
            raise ValueError("At least one transition is required")
        critic_predictions: list[torch.Tensor] = []
        critic_targets: list[torch.Tensor] = []
        actor_objectives: list[torch.Tensor] = []
        destroy_entropies: list[torch.Tensor] = []
        repair_entropies: list[torch.Tensor] = []
        joint_entropies: list[torch.Tensor] = []
        target_scores: list[torch.Tensor] = []
        predicted_scores: list[torch.Tensor] = []
        empirical_means: list[float] = []
        empirical_cvars: list[float] = []
        maximum_entropies: list[float] = []
        for transition in transitions:
            state = self.model.encode(transition.state_graph)
            destroy_mask = graph_destroy_action_mask(
                transition.state_graph,
                self.model.destroy_actions,
                state.device,
            )
            if not bool(destroy_mask[transition.destroy_action].item()):
                raise ValueError("Replay transition uses an inadmissible Destroy action")
            all_quantiles = self.model.action_quantiles(state)
            prediction = all_quantiles[transition.destroy_action, transition.repair_action]
            target = self._target_quantiles(transition).to(prediction.device)
            critic_predictions.append(prediction)
            critic_targets.append(target)

            joint_log_probabilities = self.model.joint_log_probabilities(
                state,
                condition_on_destroy=self.condition_on_destroy,
                destroy_valid_mask=destroy_mask,
            )
            joint_probabilities = joint_log_probabilities.exp()
            action_scores = score_action_quantiles(
                all_quantiles,
                self.risk_score_mode,
                self.risk_alpha,
            ).detach()
            valid_joint_mask = destroy_mask.unsqueeze(-1).expand_as(action_scores)
            valid_scores = action_scores[valid_joint_mask]
            centered_scores = action_scores - valid_scores.mean()
            score_scale = valid_scores.std(unbiased=False)
            normalized_scores = (
                centered_scores / score_scale
                if float(score_scale) > 1e-6
                else centered_scores
            )
            normalized_scores = normalized_scores.masked_fill(~valid_joint_mask, 0.0)
            actor_objectives.append((joint_probabilities * normalized_scores).sum())
            finite_joint_logs = torch.where(
                torch.isfinite(joint_log_probabilities),
                joint_log_probabilities,
                torch.zeros_like(joint_log_probabilities),
            )
            joint_entropy = -(joint_probabilities * finite_joint_logs).sum()
            destroy_log_probabilities = F.log_softmax(
                masked_logits(self.model.destroy_logits(state), destroy_mask),
                dim=-1,
            )
            destroy_probabilities = destroy_log_probabilities.exp()
            finite_destroy_logs = torch.where(
                torch.isfinite(destroy_log_probabilities),
                destroy_log_probabilities,
                torch.zeros_like(destroy_log_probabilities),
            )
            destroy_entropy = -(destroy_probabilities * finite_destroy_logs).sum()
            joint_entropies.append(joint_entropy)
            destroy_entropies.append(destroy_entropy)
            repair_entropies.append(joint_entropy - destroy_entropy)

            target_scores.append(score_action_quantiles(target, self.risk_score_mode, self.risk_alpha))
            predicted_scores.append(score_action_quantiles(prediction, self.risk_score_mode, self.risk_alpha))
            empirical_means.append(sum(transition.reward_samples) / len(transition.reward_samples))
            empirical_cvars.append(self.empirical_cvar(transition.reward_samples))
            maximum_entropies.append(
                math.log(int(destroy_mask.sum().item()))
                + math.log(self.model.repair_actions)
            )

        predictions = torch.stack(critic_predictions)
        if (not bool(torch.isfinite(predictions).all().item())
                or any(not bool(torch.isfinite(target).all().item()) for target in critic_targets)):
            raise FloatingPointError("RDG-DAC critic distributions are not finite")
        # Terminal transitions have only reward samples. Normalize each loss
        # before averaging so longer target vectors do not reweight transitions.
        critic_loss = torch.stack([
            quantile_huber_loss(
                prediction.unsqueeze(0), target.unsqueeze(0), self.model.quantile_fractions,
            )
            for prediction, target in zip(critic_predictions, critic_targets)
        ]).mean()

        raw_advantages = torch.stack(target_scores) - torch.stack(predicted_scores).detach()
        if len(transitions) == 1:
            normalized_advantages = raw_advantages
        else:
            centered_advantages = raw_advantages - raw_advantages.mean()
            advantage_scale = centered_advantages.std(unbiased=False)
            normalized_advantages = (
                centered_advantages / advantage_scale
                if float(advantage_scale.detach()) > 1e-6
                else centered_advantages
            )
        # Optimize the complete discrete policy against the critic's risk ranking.
        # A sampled-action TD residual is an off-policy critic diagnostic, not a
        # policy-gradient advantage; using it here can reinforce a collapsed
        # behavior policy even when the critic ranks other actions higher.
        actor_loss = -torch.stack(actor_objectives).mean()

        destroy_entropy = torch.stack(destroy_entropies).mean()
        repair_entropy = torch.stack(repair_entropies).mean()
        joint_entropy = torch.stack(joint_entropies).mean()
        maximum_entropy = sum(maximum_entropies) / len(maximum_entropies)
        entropy_ratio = (
            joint_entropy / maximum_entropy
            if maximum_entropy > 0.0
            else joint_entropy.new_zeros(())
        )
        total_loss = (
            critic_loss
            + actor_loss
            - self.entropy_coefficient * joint_entropy
        )
        if not bool(torch.isfinite(total_loss).item()):
            raise FloatingPointError("RDG-DAC total loss is not finite")
        self.optimizer.zero_grad()
        total_loss.backward()

        def component_gradient_norm(parameters) -> torch.Tensor:
            squares = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
            return torch.stack(squares).sum().sqrt() if squares else torch.tensor(0.0)

        destroy_actor_gradient_norm = component_gradient_norm(self.model.destroy_actor.parameters())
        repair_actor_gradient_norm = component_gradient_norm(self.model.repair_actor.parameters())
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        gradient_values = (gradient_norm, destroy_actor_gradient_norm, repair_actor_gradient_norm)
        if any(not bool(torch.isfinite(value).item()) for value in gradient_values):
            self.optimizer.zero_grad()
            raise FloatingPointError("RDG-DAC gradient norm is not finite")
        self.optimizer.step()
        if any(not bool(torch.isfinite(parameter).all().item()) for parameter in self.model.parameters()):
            raise FloatingPointError("RDG-DAC model parameters are not finite")
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_model.parameters(), self.model.parameters()):
                target_parameter.mul_(1.0 - self.target_tau).add_(parameter, alpha=self.target_tau)

        target_score = torch.stack(target_scores).mean()
        predicted_score = torch.stack(predicted_scores).mean()
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "entropy": float(joint_entropy.detach()),
            "destroy_entropy": float(destroy_entropy.detach()),
            "repair_entropy": float(repair_entropy.detach()),
            "joint_entropy": float(joint_entropy.detach()),
            "entropy_ratio": float(entropy_ratio.detach()),
            "total_loss": float(total_loss.detach()),
            "empirical_mean_return": float(sum(empirical_means) / len(empirical_means)),
            "empirical_cvar": float(sum(empirical_cvars) / len(empirical_cvars)),
            "target_cvar": float(target_score.detach()),
            "predicted_cvar": float(predicted_score.detach()),
            "raw_advantage_mean": float(raw_advantages.mean().detach()),
            "normalized_advantage_mean": float(normalized_advantages.mean().detach()),
            "destroy_actor_gradient_norm": float(destroy_actor_gradient_norm.detach()),
            "repair_actor_gradient_norm": float(repair_actor_gradient_norm.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    # 导出在线网络、目标网络和 optimizer 的完整训练器状态。
    def state_dict(self) -> dict[str, object]:
        """返回可由完整训练 checkpoint 恢复的训练器状态。"""
        # 返回结构标识、两个网络、optimizer 和构造种子。
        return {
            "architecture": dict(self._architecture),
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "seed": self.seed,
        }

    # 加载与当前网络结构兼容的完整训练器状态。
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """严格检查字段和网络结构后恢复模型与 optimizer。"""
        # 定义训练器状态必须且只能包含的字段。
        expected_keys = {"architecture", "model", "target_model", "optimizer", "seed"}
        # 要求载荷属于映射且没有缺失或未知字段。
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            # 拒绝语义不明确的训练器状态。
            raise ValueError("Checkpoint trainer state has missing or unknown fields")
        # 要求网络结构与当前训练器完全一致。
        if state["architecture"] != self._architecture:
            # 拒绝维度或分位数量漂移。
            raise ValueError("Checkpoint architecture does not match the trainer")
        # 恢复在线网络参数和缓冲区。
        self.model.load_state_dict(state["model"])
        # 恢复 Polyak 目标网络参数和缓冲区。
        self.target_model.load_state_dict(state["target_model"])
        # 恢复 Adam 动量和步数状态。
        self.optimizer.load_state_dict(state["optimizer"])
        # 恢复训练器构造种子元数据。
        self.seed = int(state["seed"])
        # 保持目标网络处于确定性评估模式。
        self.target_model.eval()

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def save(self, path: Path, step: int, protocol: str = "single_instance_smoke") -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        path.parent.mkdir(parents=True, exist_ok=True)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        # 复用正式训练器状态并增加旧接口需要的步数元数据。
        payload = self.state_dict()
        # 写入旧单模型 checkpoint 使用的训练步数。
        payload["step"] = int(step)
        payload["protocol"] = str(protocol)  # 记录检查点产生所遵循的实验协议。
        # 保存兼容既有 smoke 推理入口的轻量 checkpoint。
        torch.save(payload, path)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def load(self, path: Path) -> dict[str, int | str]:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        # 提取不含旧接口步数的正式训练器状态。
        trainer_state = {key: payload[key] for key in ("architecture", "model", "target_model", "optimizer", "seed")}
        # 复用严格状态加载逻辑恢复模型和 optimizer。
        self.load_state_dict(trainer_state)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return {"step": int(payload["step"]), "seed": int(payload["seed"]), "protocol": str(payload.get("protocol", "legacy_unknown"))}  # 返回检查点来源协议。
