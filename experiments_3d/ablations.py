"""Actual action-space and policy interventions for the eight E3 variants."""
from __future__ import annotations

from dataclasses import replace

from Agent.hybrid.rdg_nlns.agent import RDGNLNSAgent, RDGNLNSConfig, RDGNLNSAblationConfig
from Agent.reinforcement_learning.quantile_training import RDGDACTrainer
from experiments_3d.matrix import ABLATIONS


class UniformLegalAgent(RDGNLNSAgent):
    """Uniform joint sampling over the same currently legal action set."""

    def _build_graph(self, snapshot, solution):
        graph = super()._build_graph(snapshot, solution)
        self._legal_destroy = tuple(int(i) for i in graph.destroy_action_mask.nonzero().flatten().tolist())
        return graph

    def _heuristic_action(self, iteration):
        if not self._legal_destroy:
            raise RuntimeError("uniform policy has no legal destroy action")
        return self._rng.choice(self._legal_destroy), self._rng.randrange(len(self._config.repair_modes))

    def solve(self, snapshot):
        result = super().solve(snapshot)
        return replace(result, diagnostics=(*result.diagnostics, ("experimental_policy", "uniform_legal")))


def variant_config(base: RDGNLNSConfig, variant: str) -> RDGNLNSConfig:
    if variant not in ABLATIONS:
        raise ValueError("unknown registered ablation")
    options = ABLATIONS[variant]
    ablation = RDGNLNSAblationConfig(**{k: v for k, v in options.items() if k in {
        "graph_state_mode", "risk_score_mode", "repair_policy_mode", "distributional_samples"}})
    return replace(base, ablation=ablation, policy_mode="heuristic" if variant == "uniform_actions" else "learned",
                   repair_modes=(0, 1) if variant == "no_sync_repair" else (0, 1, 2))


def make_variant(solver, base: RDGNLNSConfig, variant: str, *, trainer_options=None, runtime_policy=None, trace_callback=None):
    config = variant_config(base, variant)
    from experiments_3d.mechanism_trace import MechanismTraceMixin, TracedRDGAgent
    if variant == "uniform_actions":
        class TracedUniform(MechanismTraceMixin,UniformLegalAgent):
            pass
        agent=(TracedUniform if trace_callback else UniformLegalAgent)(solver,config,runtime_policy=runtime_policy)
        agent.trace_callback=trace_callback
        return agent,None
    options = dict(trainer_options or {})
    for key in ("repair_actions", "seed", "risk_score_mode", "condition_on_destroy", "risk_alpha", "destroy_actions"):
        if key in options:
            raise ValueError(f"variant controls cannot be overridden: {key}")
    trainer = RDGDACTrainer(**options, destroy_actions=3 * len(config.destroy_ratios), repair_actions=len(config.repair_modes),
        seed=config.seed, risk_alpha=config.risk_alpha, risk_score_mode=config.ablation.risk_score_mode,
        condition_on_destroy=config.ablation.repair_policy_mode == "conditional")
    agent=(TracedRDGAgent if trace_callback else RDGNLNSAgent)(solver, config, trainer, runtime_policy=runtime_policy)
    agent.trace_callback=trace_callback
    return agent,trainer
