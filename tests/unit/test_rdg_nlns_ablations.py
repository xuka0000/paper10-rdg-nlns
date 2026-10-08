"""验证 RDG-NLNS 四项消融开关会改变真实算法行为。"""  # 说明本文件拒绝仅修改方法名称的伪消融。

from pathlib import Path  # 导入路径类型以定位 P0 场景和配置。
from types import SimpleNamespace

import pytest  # 导入异常断言和近似比较工具。
import torch  # 导入张量工具以检查图通道和策略输出。

from Agent.graph_neural_network.heterogeneous_service_graph import build_service_graph  # 导入支持静态图消融的服务图构造器。
from Agent.hybrid.rdg_nlns.agent import RDGNLNSAgent, RDGNLNSAblationConfig, RDGNLNSConfig, score_action_quantiles  # 导入待实现的消融配置和风险评分函数。
from Agent.hybrid.rdg_nlns.solution import SearchSolution  # 导入可行搜索解包装器。
from Agent.optimization.scipy_milp_agent import ScipyMilpAgent  # 导入 P0 可行解构造算法。
from Agent.reinforcement_learning.quantile_training import RewardConfig, post_repair_reward_samples  # 导入收益分布和确定性消融接口。
from Agent.reinforcement_learning.rdg_dac import RDGDAC  # 导入条件双 Actor 网络。
from Env.inspection_env import InspectionEnv  # 导入完整巡检环境。


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 计算 project 根目录以避免依赖当前工作目录。
SCENARIO = PROJECT_ROOT / "Data" / "scenarios" / "p0_minimal"  # 定位 P0 场景目录。
CONFIG = PROJECT_ROOT / "Data" / "configs" / "p0_minimal.json"  # 定位 P0 算法配置。


def _snapshot_and_solution() -> tuple[object, SearchSolution]:  # 定义真实快照和可行解夹具构造器。
    """从 P0 环境返回带完整目标分解的可行搜索解。"""  # 解释消融测试使用生产模型字段。
    environment = InspectionEnv.from_directory(SCENARIO, CONFIG)  # 装配快速接口环境。
    snapshot = environment.snapshot()  # 获取不可变 M1 至 M5 快照。
    result = ScipyMilpAgent.from_config(CONFIG).solve(snapshot)  # 求解可行调度结果。
    solution = SearchSolution.from_result(result, environment.validate_result(result).is_valid)  # 封装经独立验证的搜索解。
    return snapshot, solution  # 返回快照和解供多项行为测试复用。


def test_rdg_nlns_config_rejects_removed_critic_candidate_parameter() -> None:
    """The public LNS config must not expose the rejected Critic reranking budget."""
    with pytest.raises(TypeError, match="actor_candidate_count"):
        RDGNLNSConfig(actor_candidate_count=2)


def test_static_only_graph_removes_dynamic_and_global_schedule_channels() -> None:  # 定义静态图消融测试。
    """要求 static_only 保留物理结构但清零全部调度状态和目标上下文。"""  # 解释消融必须阻断动态信息路径。
    snapshot, solution = _snapshot_and_solution()  # 构造带非零调度结果的真实状态。

    graph = build_service_graph(snapshot, solution.selected_column_ids, solution.result, include_dynamic=False)  # 构造静态图消融输入。

    assert torch.count_nonzero(graph.static_features).item() > 0  # 要求 M1 至 M4 静态物理信号仍然存在。
    assert torch.count_nonzero(graph.dynamic_features).item() == 0  # 要求节点级调度状态完全清零。
    assert torch.count_nonzero(graph.global_features).item() == 0  # 要求目标分解和滚动上下文完全清零。


def test_independent_repair_policy_does_not_change_with_destroy_action() -> None:  # 定义修复 Actor 条件消融测试。
    """要求 independent 模式使用同一零条件嵌入而不读取破坏动作。"""  # 解释消融真实切断 Actor 条件依赖。
    model = RDGDAC(hidden_dim=16, destroy_actions=3, repair_actions=2, quantile_count=8)  # 创建小型双 Actor 网络。
    state = torch.randn(16)  # 创建固定维度编码状态。

    first = model.repair_logits(state, torch.tensor(0), condition_on_destroy=False)  # 使用第一个破坏动作请求独立修复 logits。
    second = model.repair_logits(state, torch.tensor(2), condition_on_destroy=False)  # 使用另一个破坏动作请求独立修复 logits。

    assert torch.equal(first, second)  # 要求独立模式输出完全不随破坏动作变化。


def test_mean_and_lower_tail_scoring_choose_different_joint_actions() -> None:  # 定义风险评分消融测试。
    """要求同一 critic 分布在均值和下尾 CVaR 下能够选择不同动作。"""  # 解释风险消融改变决策而非标签。
    quantiles = torch.tensor([[[4.0, 4.0, 4.0, 4.0], [-5.0, 8.0, 8.0, 8.0]]])  # 构造均值 4.75 但下尾为负的第二联合动作。

    mean_choice = score_action_quantiles(quantiles, "mean", 0.25).reshape(-1).argmax().item()  # 按均值选择第二动作。
    risk_choice = score_action_quantiles(quantiles, "lower_tail_cvar", 0.25).reshape(-1).argmax().item()  # 按最差四分位选择第一动作。

    assert mean_choice == 1  # 要求普通均值偏好高平均收益动作。
    assert risk_choice == 0  # 要求下尾 CVaR 回避显著负尾部动作。


def test_learned_action_is_controlled_only_by_actor_logits() -> None:
    """The Critic must not be read while the Actor selects an LNS action."""

    class InferenceModel:
        destroy_actions = 3
        repair_actions = 3
        logits = torch.tensor(
            [
                [9.0, 8.0, 0.0],
                [7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0],
            ]
        )

        def eval(self) -> None:
            return None

        def encode(self, graph: object) -> torch.Tensor:
            return torch.zeros(4)

        def joint_log_probabilities(
            self,
            state: torch.Tensor,
            condition_on_destroy: bool = True,
            destroy_valid_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert torch.equal(destroy_valid_mask, torch.ones(3, dtype=torch.bool))
            return torch.log_softmax(self.logits.reshape(-1), dim=0).reshape(3, 3)

        def action_quantiles(self, state: torch.Tensor) -> torch.Tensor:
            raise AssertionError("Critic values cannot participate in Actor action selection")

    class InferenceTrainer:
        model = InferenceModel()
        risk_score_mode = "lower_tail_cvar"
        condition_on_destroy = True

    agent = RDGNLNSAgent(
        repair_solver=object(),
        config=RDGNLNSConfig(
            iterations=2,
            destroy_ratios=(0.5,),
            policy_mode="learned",
        ),
        trainer=InferenceTrainer(),
    )

    graph = SimpleNamespace(destroy_action_mask=torch.ones(3, dtype=torch.bool))
    first = agent._learned_action(graph, frozenset())
    agent._trainer.model.logits = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 9.0],
        ]
    )
    second = agent._learned_action(graph, frozenset())

    assert first == (0, 0)
    assert second == (2, 2)


def test_deterministic_mean_reward_removes_monte_carlo_spread() -> None:  # 定义收益分布消融测试。
    """要求关闭 distributional_samples 后重复确定性基础收益。"""  # 解释消融不再读取气象场景或高斯误差扩散。
    snapshot, solution = _snapshot_and_solution()  # 构造包含多气象状态和服务列不确定性的真实场景。
    config = RewardConfig(sample_count=8, distributional_samples=False)  # 关闭收益分布采样并保留固定样本宽度。

    distribution = post_repair_reward_samples(snapshot, solution, solution, 0.0, config)  # 计算确定性收益消融结果。

    assert distribution.source == "deterministic_mean"  # 要求证据字段明确记录确定性来源。
    assert len(distribution.samples) == 8  # 要求 critic 输入宽度仍由配置控制。
    assert len(set(distribution.samples)) == 1  # 要求全部经验收益样本完全相同。


@pytest.mark.parametrize(("kwargs", "message"), (({"graph_state_mode": "bad"}, "graph_state_mode"), ({"risk_score_mode": "bad"}, "risk_score_mode"), ({"repair_policy_mode": "bad"}, "repair_policy_mode")))  # 参数化三个字符串消融字段的非法值。
def test_ablation_config_rejects_unknown_behavior_modes(kwargs: dict[str, object], message: str) -> None:  # 定义消融配置失败关闭测试。
    """要求未知消融值不能静默退回完整方法。"""  # 解释正式清单中的拼写错误必须终止运行。
    with pytest.raises(ValueError, match=message):  # 要求错误信息指向对应消融字段。
        RDGNLNSAblationConfig(**kwargs)  # 尝试构造未知行为模式。
