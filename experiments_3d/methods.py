"""One common search boundary for formal and development experiment runners."""
from pathlib import Path
from dataclasses import replace
import torch

from Agent.evaluation_runtime import EvaluationRuntimePolicy
from Agent.hybrid.rdg_nlns.agent import RDGNLNSConfig
from Agent.optimization.scipy_milp_agent import ScipyMilpAgent
from experiments_3d.ablations import make_variant
from experiments_3d.public_alns import solve_public_alns
from experiments_3d.public_search import PublicLearningSearch
from Agent.reinforcement_learning.quantile_training import RewardConfig
from Agent.hybrid.rdg_nlns.repair import InsertionWeights

SOURCES=Path(__file__).resolve().parents[1]/'Data/external/multicomponent_3d_public_sources_20260928'


def make_learner(method,config,*,seed,variant='full',iterations=1000,wall_s=300.,trace_callback=None,training_settings=None):
    solver=ScipyMilpAgent.from_config(config,min(10.,wall_s))
    if method=='rdg_nlns':
        policy=EvaluationRuntimePolicy(wall_s,min(10.,wall_s),min(30.,wall_s),min(30.,wall_s))
        settings=training_settings or {}
        reward=RewardConfig(**{k:settings[k] for k in ('failure_penalty','time_penalty','time_scale_seconds','energy_weight') if k in settings},
                            sample_count=settings.get('reward_sample_count',32),seed=seed)
        base=RDGNLNSConfig(iterations=iterations,seed=seed,reward=reward,
            destroy_ratios=tuple(settings.get('destroy_ratios',(.1,.2,.3,.4))),risk_alpha=settings.get('risk_alpha',.25),
            initial_solution_mode=settings.get('initial_solution_mode','greedy'),
            insertion_weights=InsertionWeights(**{k:settings['insertion_'+k+'_weight'] for k in
                ('energy','duration','shared_uav','stop_discontinuity') if 'insertion_'+k+'_weight' in settings}))
        trainer_options={k:settings[k] for k in ('hidden_dim','quantile_count','learning_rate','gamma','entropy_coefficient','target_tau') if k in settings}
        agent,trainer=make_variant(solver,base,variant,trainer_options=trainer_options,
                                  runtime_policy=policy,trace_callback=trace_callback)
        return agent,trainer,solver
    if variant!='full':raise ValueError('public baseline does not accept RDG ablations')
    return PublicLearningSearch(method,seed=seed),None,solver


def solve_method(snapshot,config,method,*,seed,iterations=1000,wall_s=300.,variant='full',learner=None,trace_callback=None):
    if method=='milp':
        return ScipyMilpAgent.from_config(config,min(30.,wall_s)).solve(snapshot)
    if method=='alns_public':
        return solve_public_alns(snapshot,ScipyMilpAgent.from_config(config,min(10.,wall_s)),source_root=SOURCES/'alns',
                                 seed=seed,iterations=iterations,wall_s=wall_s,trace_callback=trace_callback)
    if learner is None:
        if not(method=='rdg_nlns' and variant=='uniform_actions'):
            raise ValueError('learned inference requires an explicitly loaded target-task checkpoint')
        learner,_,_=make_learner(method,config,seed=seed,variant=variant,iterations=iterations,wall_s=wall_s,trace_callback=trace_callback)
    if method=='rdg_nlns':
        # Online epochs and validation use the caller's bound, including when
        # reusing a learner created for the 300-second static experiment.
        old_config,old_policy=learner._config,learner._runtime_policy
        old_solver=learner._repair_solver
        try:
            learner._config=replace(old_config,iterations=iterations)
            learner._runtime_policy=EvaluationRuntimePolicy(wall_s,min(10.,wall_s),min(30.,wall_s),min(30.,wall_s))
            learner._repair_solver=ScipyMilpAgent.from_config(config,min(10.,wall_s))
            return learner.solve(snapshot)
        finally:
            learner._config,learner._runtime_policy=old_config,old_policy
            learner._repair_solver=old_solver
    return learner.solve(snapshot,ScipyMilpAgent.from_config(config,min(10.,wall_s)),iterations=iterations,wall_s=wall_s,
                         training=False,trace_callback=trace_callback)
