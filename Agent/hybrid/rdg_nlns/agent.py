"""Complete RDG-NLNS search, inference, and online training loop. 中文说明。"""

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import dataclasses
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import random
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import time
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from dataclasses import dataclass

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
import torch

# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.base_agent import BaseAgent
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.evaluation_runtime import EvaluationRuntimePolicy, RuntimeDeadline
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.graph_neural_network.heterogeneous_service_graph import build_service_graph
# 导入非最优贪婪初始化器以保留邻域搜索改进空间。
from Agent.hybrid.rdg_nlns.initialization import (
    build_greedy_initial_solution,
    solver_with_time_limit,
)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.operators import DestroyOperator, admissible_destroy_action_mask, destroy_neighborhood
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.repair import InsertionWeights, RepairMode, repair_neighborhood
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.hybrid.rdg_nlns.solution import SearchSolution
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.results import AgentResult
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.reinforcement_learning.algorithm_identity import MANUSCRIPT_DESTROY_RATIOS
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.reinforcement_learning.quantile_training import RDGDACTrainer, RewardConfig, post_repair_reward_samples
# 共享 Actor 动作统计和风险聚合，确保训练与推理使用同一策略语义。
from Agent.reinforcement_learning.rdg_dac import ActorActionSelection, actor_action_statistics, masked_logits, score_action_quantiles, select_actor_action
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Agent.reinforcement_learning.replay import SearchTransition
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Env.snapshot import SceneSnapshot
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
from Env.validation import validate_selected_columns


@dataclass(frozen=True, slots=True)  # 使用不可变数据类定义可审计消融行为。
class RDGNLNSAblationConfig:  # 定义图状态、风险评分、条件策略和收益分布开关。
    """保存四项会改变真实算法数据流的消融设置。"""  # 解释未知设置必须失败而不能退回完整方法。

    graph_state_mode: str = "static_dynamic"  # 保存完整静态动态图或静态图模式。
    risk_score_mode: str = "lower_tail_cvar"  # 保存下尾 CVaR 或均值动作评分模式。
    repair_policy_mode: str = "conditional"  # 保存条件修复 Actor 或独立修复 Actor 模式。
    distributional_samples: bool = True  # 保存分布收益或确定性均值收益模式。

    def __post_init__(self) -> None:  # 定义消融配置初始化后验证方法。
        """拒绝未知字符串模式和非布尔收益开关。"""  # 解释清单拼写错误不能静默改变实验组。
        if self.graph_state_mode not in {"static_dynamic", "static_only", "relation_agnostic"}:
            raise ValueError("graph_state_mode 无效")  # 拒绝未知图输入行为。
        if self.risk_score_mode not in {"lower_tail_cvar", "mean"}:  # 判断风险评分消融模式是否受支持。
            raise ValueError("risk_score_mode 无效")  # 拒绝未知动作评分行为。
        if self.repair_policy_mode not in {"conditional", "independent"}:  # 判断修复策略消融模式是否受支持。
            raise ValueError("repair_policy_mode 无效")  # 拒绝未知 Actor 条件行为。
        if not isinstance(self.distributional_samples, bool):  # 判断收益分布开关是否保持布尔语义。
            raise ValueError("distributional_samples 必须为布尔值")  # 拒绝隐式真值造成的实验漂移。


@dataclass(frozen=True, slots=True)  # 使用不可变数据类定义公开单步训练结果。
class RDGTrainingStep:  # 定义一次破坏、修复、投影和状态更新的结构化结果。
    """保存 post-repair transition、下一解、动作和投影诊断。"""  # 解释跨场景会话无需复制 Agent 私有流程。

    transition: SearchTransition  # 保存供 Replay 和 RDG-DAC 使用的完整转移。
    next_solution: SearchSolution  # 保存接受规则后的下一可行搜索解。
    destroy_action: int  # 保存破坏 Actor 动作索引。
    repair_action: int  # 保存修复 Actor 动作索引。
    accepted: bool  # 保存候选是否按目标非劣规则接受。
    projection_valid: bool  # 保存候选是否通过 Env 独立投影验证。
    repair_seconds: float  # 保存仅用于诊断的墙钟修复耗时。
    issue_codes: tuple[str, ...]  # 保存投影失败或空破坏的稳定问题代码。
    reward_source: str  # 保存收益样本来自气象、高斯、投影失败或确定性消融。
    candidate_solution: SearchSolution | None  # 保存投影后的候选供训练集奖励校准复算。
    selection: ActorActionSelection  # 保存实际进入 LNS 的双 Actor 联合动作及其概率。


@dataclass(frozen=True, slots=True)
# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RDGNLNSConfig:
    """Search controls shared by heuristic and learned RDG-NLNS modes. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    iterations: int = 50
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    destroy_ratios: tuple[float, ...] = MANUSCRIPT_DESTROY_RATIOS
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    risk_alpha: float = 0.25
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    seed: int = 0
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    policy_mode: str = "learned"
    # 说明默认使用非最优贪婪初始解以保留邻域搜索空间。
    initial_solution_mode: str = "greedy"
    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    reward: RewardConfig = RewardConfig()
    # 保存会改变真实图、策略、风险和收益路径的消融设置。
    ablation: RDGNLNSAblationConfig = RDGNLNSAblationConfig()
    insertion_weights: InsertionWeights = InsertionWeights()
    repair_modes: tuple[int, ...] = (0, 1, 2)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __post_init__(self) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if self.iterations <= 0 or not self.destroy_ratios:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("RDG-NLNS requires positive iterations and destroy ratios")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if any(not 0.0 < ratio <= 1.0 for ratio in self.destroy_ratios):
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Destroy ratios must lie in (0, 1]")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if not 0.0 < self.risk_alpha <= 1.0:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Risk controls are invalid")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if not self.repair_modes or len(set(self.repair_modes)) != len(self.repair_modes) or any(type(mode) is not int or mode not in tuple(int(item) for item in RepairMode) for mode in self.repair_modes):
            raise ValueError("repair_modes must contain distinct supported repair identifiers")
        if self.policy_mode not in {"heuristic", "learned"}:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Policy mode must be heuristic or learned")
        # 检查初始解模式是否属于公开支持的构造方式。
        if self.initial_solution_mode not in {"greedy", "exact"}:
            # 拒绝无法复现的隐式初始化策略。
            raise ValueError("Initial solution mode must be greedy or exact")


# 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
class RDGNLNSAgent(BaseAgent):
    """Risk-aware graph-guided large neighborhood search agent. 中文说明。"""

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def __init__(
        self,
        repair_solver: BaseAgent,
        config: RDGNLNSConfig = RDGNLNSConfig(),
        trainer: RDGDACTrainer | None = None,
        *,
        runtime_policy: EvaluationRuntimePolicy | None = None,
    ) -> None:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        expected_destroy_actions = len(DestroyOperator) * len(config.destroy_ratios)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if config.policy_mode == "learned" and trainer is None:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Learned RDG-NLNS mode requires an RDG-DAC trainer or loaded model")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if trainer is not None and (trainer.model.destroy_actions != expected_destroy_actions or trainer.model.repair_actions != len(config.repair_modes)):
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("RDG-DAC action dimensions do not match the search configuration")
        # 确认训练器风险评分行为与 Agent 消融配置一致。
        if trainer is not None and trainer.risk_score_mode != config.ablation.risk_score_mode:
            # 拒绝训练和推理使用不同风险语义。
            raise ValueError("RDG-DAC risk score mode does not match the search ablation")
        # 确认训练器修复 Actor 条件行为与 Agent 消融配置一致。
        if trainer is not None and trainer.condition_on_destroy != (config.ablation.repair_policy_mode == "conditional"):
            # 拒绝训练和推理使用不同修复策略条件语义。
            raise ValueError("RDG-DAC repair policy mode does not match the search ablation")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._repair_solver = repair_solver
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._config = config
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._trainer = trainer
        self._runtime_policy = runtime_policy
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._rng = random.Random(config.seed)

    # 构造与当前消融模式一致的静态动态图输入。
    def _build_graph(self, snapshot: SceneSnapshot, solution: SearchSolution):
        """从快照和当前解构造完整图或静态图消融。"""
        # 仅在完整模式下把动态节点和全局目标通道送入编码器。
        include_dynamic = self._config.ablation.graph_state_mode != "static_only"
        # 委托唯一图构造器保持节点和边结构一致。
        graph = build_service_graph(snapshot, solution.selected_column_ids, solution.result, include_dynamic=include_dynamic)
        if self._config.ablation.graph_state_mode == "relation_agnostic":
            # Preserve nodes, features, directed edges and action masks while
            # pooling all messages as one relation. Unused relation matrices
            # remain allocated for checkpoint compatibility, not extra capacity.
            graph = dataclasses.replace(graph, edge_types=torch.zeros_like(graph.edge_types))
        return dataclasses.replace(
            graph,
            destroy_action_mask=torch.tensor(
                admissible_destroy_action_mask(
                    snapshot,
                    solution,
                    self._config.destroy_ratios,
                ),
                dtype=torch.bool,
            ),
        )

    # 导出 Agent 破坏算子私有 RNG 状态供完整 checkpoint 使用。
    def rng_state(self) -> object:
        """返回下一次随机破坏选择所需的 Python RNG 状态。"""
        # 返回 random.Random 的不可变状态元组。
        return self._rng.getstate()

    # 从完整 checkpoint 恢复 Agent 破坏算子私有 RNG。
    def load_rng_state(self, state: object) -> None:
        """验证候选状态后恢复随机破坏采样位置。"""
        # 创建临时随机数生成器以避免坏状态污染当前 Agent。
        candidate = random.Random()
        # 预验证 Python RNG 状态格式。
        try:
            # 将候选状态加载到临时生成器。
            candidate.setstate(state)
        # 统一转换损坏 RNG 状态错误。
        except (TypeError, ValueError) as error:
            # 保留原异常并拒绝半恢复。
            raise ValueError("Agent RNG 状态无效") from error
        # 验证通过后恢复真实 Agent 私有 RNG。
        self._rng.setstate(candidate.getstate())

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def _validator(self, snapshot: SceneSnapshot):
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return lambda result: validate_selected_columns(snapshot, result)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def _initial_solution(
        self,
        snapshot: SceneSnapshot,
        *,
        deadline: RuntimeDeadline | None = None,
    ) -> SearchSolution:
        """说明当前算法或测试步骤。"""
        initialization_limit = (
            self._runtime_policy.initialization_projection_time_limit_seconds
            if self._runtime_policy is not None
            else 10.0
        )
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if self._config.initial_solution_mode == "greedy":
            # 用逐任务服务列选择加 M5 投影构造非最优可行起点。
            return build_greedy_initial_solution(
                snapshot,
                self._repair_solver,
                self._validator(snapshot),
                deadline=deadline,
                route_projection_time_limit_seconds=initialization_limit,
            )
        # 精确初始化仅保留给小规模诊断和求解器基准。
        if deadline is not None and deadline.expired():
            raise RuntimeError("RDG-NLNS initialization failed: runtime_deadline")
        result = solver_with_time_limit(
            self._repair_solver,
            initialization_limit,
        ).solve(snapshot)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        validation = self._validator(snapshot)(result)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if not result.success or result.objective_value is None or not validation.is_valid:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise RuntimeError(f"RDG-NLNS could not construct a valid initial solution: {validation.issue_codes}")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return SearchSolution.from_result(result, True)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def _decode_destroy(self, action_id: int) -> tuple[DestroyOperator, float]:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        ratio_count = len(self._config.destroy_ratios)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return DestroyOperator(action_id // ratio_count), self._config.destroy_ratios[action_id % ratio_count]

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def _heuristic_action(self, iteration: int) -> tuple[int, int]:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        destroy_count = len(DestroyOperator) * len(self._config.destroy_ratios)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return iteration % destroy_count, iteration % len(self._config.repair_modes)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def _select_learned_action(
        self,
        graph,
        excluded_actions: frozenset[tuple[int, int]] = frozenset(),
    ) -> ActorActionSelection:
        """Return the highest-probability untried action selected by the Actors."""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        assert self._trainer is not None
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        model = self._trainer.model
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        model.eval()
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        with torch.no_grad():
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            state = model.encode(graph)
            return select_actor_action(
                model.joint_log_probabilities(
                    state,
                    condition_on_destroy=(self._config.ablation.repair_policy_mode == "conditional"),
                    destroy_valid_mask=graph.destroy_action_mask,
                ),
                excluded_actions,
            )

    def _learned_action(
        self,
        graph,
        excluded_actions: frozenset[tuple[int, int]] = frozenset(),
    ) -> tuple[int, int]:
        """保留仅返回动作元组的兼容接口。"""
        return self._select_learned_action(graph, excluded_actions).selected_action

    # 从当前在线策略随机采样一个用于探索的联合动作。
    def _sample_training_action(self, graph) -> ActorActionSelection:
        """按当前消融配置从两个 Actor 依次采样训练动作。"""
        # 要求学习模式已经提供 RDG-DAC 训练器。
        if self._trainer is None:
            # 拒绝没有神经策略的训练动作请求。
            raise RuntimeError("Training requires an RDG-DAC trainer")
        # 将在线模型切换到训练模式。
        self._trainer.model.train()
        # 采样动作无需构建梯度图。
        with torch.no_grad():
            # 编码当前完整或静态消融图状态。
            state = self._trainer.model.encode(graph)
            # 从破坏 Actor 分类分布采样动作。
            destroy_distribution = torch.distributions.Categorical(
                logits=masked_logits(
                    self._trainer.model.destroy_logits(state),
                    graph.destroy_action_mask.to(device=state.device),
                )
            )
            # 读取采样得到的普通整数动作索引。
            destroy_id = int(destroy_distribution.sample().item())
            # 根据条件或独立修复策略构造第二个分类分布。
            repair_logits = self._trainer.model.repair_logits(state, torch.tensor(destroy_id, device=state.device), condition_on_destroy=self._config.ablation.repair_policy_mode == "conditional")
            # 从修复 Actor 分类分布采样动作。
            repair_distribution = torch.distributions.Categorical(logits=repair_logits)
            # 读取采样得到的普通整数动作索引。
            repair_id = int(repair_distribution.sample().item())
            joint_log_probabilities = self._trainer.model.joint_log_probabilities(
                state,
                condition_on_destroy=self._config.ablation.repair_policy_mode == "conditional",
                destroy_valid_mask=graph.destroy_action_mask,
            )
        return actor_action_statistics(joint_log_probabilities, (destroy_id, repair_id))

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def _attempt(
        self,
        snapshot: SceneSnapshot,
        current: SearchSolution,
        destroy_id: int,
        repair_id: int,
        *,
        deadline: RuntimeDeadline | None = None,
    ):
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        operator, ratio = self._decode_destroy(destroy_id)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        action = destroy_neighborhood(snapshot, current, operator, ratio, self._rng)
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if not action.removed_column_ids:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            return action, None, 0.0
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        started = time.perf_counter()
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        outcome = repair_neighborhood(
            snapshot,
            action,
            RepairMode(self._config.repair_modes[repair_id]),
            self._repair_solver,
            self._validator(snapshot),
            weights=self._config.insertion_weights,
            plan_callback=getattr(self, '_record_repair_plan', None),
            deadline=deadline,
            route_projection_time_limit_seconds=(
                self._runtime_policy.projection_time_limit_seconds
                if self._runtime_policy is not None
                else 10.0
            ),
        )
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        elapsed = time.perf_counter() - started
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return action, outcome, elapsed

    # 公开一个 episode 的可行初始解构造入口。
    def start_episode(self, snapshot: SceneSnapshot) -> SearchSolution:
        """按配置的贪婪或精确模式返回经独立验证的初始解。"""
        # 复用唯一初始化流程以保持推理和训练边界一致。
        return self._initial_solution(snapshot)

    # 公开一次完整破坏、修复、投影和转移构造入口。
    def training_step(self, snapshot: SceneSnapshot, current: SearchSolution, step_index: int, final_step: bool, *, deadline: RuntimeDeadline | None = None) -> RDGTrainingStep:
        """执行一个 post-repair 训练步并返回结构化投影证据。"""
        # 要求训练步下标为非负整数。
        if type(step_index) is not int or step_index < 0:
            # 拒绝无法写入稳定日志主键的步数。
            raise ValueError("step_index 必须为非负整数")
        # 要求终止标记保持布尔语义。
        if not isinstance(final_step, bool):
            # 拒绝隐式真值进入 Replay。
            raise ValueError("final_step 必须为布尔值")
        # 构造与当前消融一致的状态图。
        graph = self._build_graph(snapshot, current)
        # 按破坏 Actor 与条件修复 Actor 依次采样实际训练动作。
        selection = self._sample_training_action(graph)
        destroy_id, repair_id = selection.selected_action
        # 执行真实破坏、确定性修复和 Env 投影。
        _, outcome, elapsed = self._attempt(snapshot, current, destroy_id, repair_id, deadline=deadline)
        # 读取存在时的投影候选解。
        candidate = outcome.candidate if outcome is not None else None
        # 用消融开关覆盖收益分布行为并保留其他奖励参数。
        reward_config = dataclasses.replace(self._config.reward, distributional_samples=self._config.ablation.distributional_samples)
        # 墙钟耗时只进入诊断，训练 reward 使用零秒以保证跨恢复确定性。
        reward_distribution = post_repair_reward_samples(snapshot, current, candidate, 0.0, reward_config)
        # 仅接受通过投影且目标不劣于当前解的候选。
        accepted = candidate is not None and candidate.is_feasible and candidate.objective_value <= current.objective_value
        # 使用接受候选或保持当前可行解形成下一状态。
        next_solution = candidate if accepted else current
        # 构造与当前消融一致的下一状态图。
        next_graph = self._build_graph(snapshot, next_solution)
        # 创建供 Replay 使用的 post-repair 转移。
        transition = SearchTransition(graph, destroy_id, repair_id, reward_distribution.samples, next_graph, final_step)
        # 读取投影验证问题或记录空破坏动作。
        issue_codes = tuple(outcome.validation.issue_codes) if outcome is not None else ("empty_destroy_action",)
        # 只有存在候选且 Env 报告有效时标记投影通过。
        projection_valid = bool(outcome is not None and outcome.validation.is_valid and candidate is not None and candidate.is_feasible)
        # 返回完整单步训练和审计结果。
        return RDGTrainingStep(
            transition,
            next_solution,
            destroy_id,
            repair_id,
            accepted,
            projection_valid,
            elapsed,
            issue_codes,
            reward_distribution.source,
            candidate,
            selection,
        )

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def solve(self, snapshot: SceneSnapshot) -> AgentResult:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        started = time.perf_counter()
        deadline = (
            RuntimeDeadline.start(self._runtime_policy.wall_clock_limit_seconds)
            if self._runtime_policy is not None
            else None
        )
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        current = self._initial_solution(snapshot, deadline=deadline)
        initial_objective = current.objective_value
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        best = current
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        attempted = 0
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        feasible_repairs = 0
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        accepted = 0
        distinct_joint_actions: set[tuple[int, int]] = set()
        examined_actor_actions: set[tuple[int, int]] = set()
        action_cycle_resets = 0
        selected_actor_probability_sum = 0.0
        learned_action_count = 0
        correction_count = 0  # 初始化投影自动修正计数。
        projection_rejections = 0  # 初始化投影拒绝计数。
        fallback_count = 0  # 初始化 MILP 投影回退计数。
        completed_iterations = 0
        terminal_reason = "iteration_limit"
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        for iteration in range(self._config.iterations):
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            graph = self._build_graph(snapshot, current)
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            if self._config.policy_mode == "learned":
                assert self._trainer is not None
                valid_destroy_ids = tuple(
                    int(index)
                    for index in graph.destroy_action_mask.nonzero().flatten().tolist()
                )
                if not valid_destroy_ids:
                    terminal_reason = "empty_action_space"
                    break
                valid_joint_actions = {
                    (destroy_id, repair_id)
                    for destroy_id in valid_destroy_ids
                    for repair_id in range(self._trainer.model.repair_actions)
                }
                examined_actor_actions.intersection_update(valid_joint_actions)
                if examined_actor_actions == valid_joint_actions:
                    examined_actor_actions.clear()
                    action_cycle_resets += 1
                selection = self._select_learned_action(
                    graph,
                    frozenset(examined_actor_actions),
                )
                destroy_id, repair_id = selection.selected_action
                examined_actor_actions.add(selection.selected_action)
                selected_actor_probability_sum += selection.joint_probability
                learned_action_count += 1
            else:
                destroy_id, repair_id = self._heuristic_action(iteration)
            distinct_joint_actions.add((destroy_id, repair_id))
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            action, outcome, _ = self._attempt(
                snapshot,
                current,
                destroy_id,
                repair_id,
                deadline=deadline,
            )
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            if not action.removed_column_ids:
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                terminal_reason = "empty_action_space"
                break
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            attempted += 1
            if outcome is not None and "runtime_deadline" in outcome.validation.issue_codes:
                terminal_reason = "wall_clock_limit"
                break
            if outcome is not None:  # 判断当前邻域是否进入投影阶段。
                correction_count += len(outcome.corrections)  # 累加投影自动修正数量。
                fallback_count += int(outcome.used_milp_fallback)  # 累加 MILP 回退次数。
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            if outcome is None or outcome.candidate is None or not outcome.candidate.is_feasible:
                projection_rejections += 1  # 记录当前投影未产生可行候选。
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            else:
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                feasible_repairs += 1
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                candidate = outcome.candidate
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                if candidate.objective_value <= current.objective_value:
                    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                    current = candidate
                    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                    accepted += 1
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                if candidate.objective_value < best.objective_value:
                    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                    best = candidate
            completed_iterations += 1
            if deadline is not None and deadline.expired():
                terminal_reason = "wall_clock_limit"
                break
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        runtime = time.perf_counter() - started
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        diagnostics = (*best.result.diagnostics,
             ("algorithm", "RDG-NLNS"),
             ("policy_mode", self._config.policy_mode),
             ("attempted_neighborhoods", str(attempted)),
             ("feasible_repairs", str(feasible_repairs)),
             ("accepted_candidates", str(accepted)),
             ("initial_objective", str(initial_objective)),
             ("objective_improvement", str(initial_objective - best.objective_value)),
             ("mean_selected_actor_probability", str(selected_actor_probability_sum / learned_action_count if learned_action_count else 0.0)),
             ("distinct_joint_actions", str(len(distinct_joint_actions))),
             ("action_cycle_resets", str(action_cycle_resets)),
             ("projection_corrections", str(correction_count)),
             ("projection_rejections", str(projection_rejections)),
             ("projection_fallbacks", str(fallback_count)),
             ("completed_iterations", str(completed_iterations)),
             ("terminal_reason", terminal_reason),
             ("iteration_limit", str(self._config.iterations)),
             ("risk_alpha", str(self._config.risk_alpha)),
             ("graph_state_mode", self._config.ablation.graph_state_mode),
             ("risk_score_mode", self._config.ablation.risk_score_mode),
             ("repair_policy_mode", self._config.ablation.repair_policy_mode),
             ("distributional_samples", str(self._config.ablation.distributional_samples)))
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return dataclasses.replace(best.result, diagnostics=diagnostics, runtime_seconds=runtime)

    # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
    def train(self, snapshot: SceneSnapshot, episodes: int, steps_per_episode: int) -> dict[str, int | float]:
        """说明当前算法或测试步骤。"""
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if self._trainer is None:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise RuntimeError("Training requires an RDG-DAC trainer")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        if episodes <= 0 or steps_per_episode <= 0:
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            raise ValueError("Training episodes and steps must be positive")
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        updates = 0
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        reward_sum = 0.0
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        self._trainer.model.train()
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        for _ in range(episodes):
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            current = self.start_episode(snapshot)
            # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
            for step in range(steps_per_episode):
                # 通过公开单步 API 执行完整破坏、修复、投影和转移构造。
                training_step = self.training_step(snapshot, current, step, step == steps_per_episode - 1)
                # 使用公开结果推进下一可行搜索状态。
                current = training_step.next_solution
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                self._trainer.update((training_step.transition,))
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                updates += 1
                # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
                reward_sum += training_step.transition.mean_reward
        # 说明当前语句在 RDG-NLNS 算法或验证流程中的作用。
        return {"updates": updates, "mean_reward": reward_sum / updates}
