"""New-domain training with 10,000 interactions and validation-only selection.

Formal test scenes are not opened by this module. Invoke under supervision.
Partial attempts retain journals/checkpoints and cannot pass the formal gate.
"""
import argparse
import dataclasses
import datetime
import json
import math
import time
from pathlib import Path

import torch
from Agent.baselines.common import initial_solution
from Agent.evaluation_runtime import RuntimeDeadline
from Env.validation import validate_selected_columns
from experiments_3d.data_upgrade import read_json,sha256,write_once
from experiments_3d.methods import make_learner,solve_method
from experiments_3d.preflight import evidence_json
from experiments_3d.snapshot_cache import prepare_snapshot,source_identity
from experiments_3d.supervision import atomic_json
from experiments_3d.random_state import isolated_evaluation
from Agent.reinforcement_learning.replay import ReplayBuffer
from experiments_3d.release_identity import dataset_identity


def train(suite_path,output,cache,method,variant,seed,total_wall_s=None,resume_state=None):
    if variant=='uniform_actions':raise ValueError('uniform ablation does not train')
    if resume_state is None: output.mkdir(parents=True,exist_ok=False)
    elif not output.is_dir(): raise ValueError('resume output directory missing')
    started=time.monotonic()
    suite=read_json(suite_path)
    train_entries=[s for s in suite['scenarios'] if s['split']=='train']
    validation=[s for s in suite['scenarios'] if s['split']=='validation']
    # Existing suite uses 'val'; accept that explicit role only.
    if not validation:validation=[s for s in suite['scenarios'] if s['split']=='val']
    if (len(train_entries),len(validation))!=(36,9):raise ValueError('training split differs from approved 36/9 protocol')
    audit=read_json(suite_path.parent/'data_audit.json')
    profile=read_json(suite_path.parent/'physical_profile.json')
    if audit['split_issues'] or not audit['source_unchanged'] or not profile.get('controller_alignment'):
        raise ValueError('audited controller-aligned data release required')
    torch.set_num_threads(1)
    config=suite_path.parent/train_entries[0]['config_path']
    settings=suite['training']
    learner,trainer,solver=make_learner(method,config,seed=seed,variant=variant,training_settings=settings)
    replay=ReplayBuffer(settings['replay_capacity'],seed)
    identity={'suite_sha256':sha256(suite_path),'method':method,'variant':variant,'seed':seed,'sources_sha256':source_identity(),
        'training_data_sha256':dataset_identity(suite_path,training_only=True)}
    interactions=episodes=0
    best_score=math.inf
    selected=None
    lineage={'segments':[{'adaptation_version':__import__('experiments_3d.public_search',fromlist=['ADAPTATION_VERSION']).ADAPTATION_VERSION,
        'first_interaction':1,'last_interaction':None}]}
    if resume_state is None:
        write_once(output/'identity.json',identity)
    else:
        if method=='rdg_nlns': raise ValueError('RDG resume state is unsupported')
        # Resume files are private artifacts from this campaign, never public inputs.
        state=torch.load(resume_state,map_location='cpu',weights_only=False)
        run=state.get('training',{})
        if (run.get('suite_sha256')!=identity['suite_sha256'] or
            run.get('training_data_sha256')!=identity['training_data_sha256'] or
            run.get('journal_sha256')!=sha256(output/'training.jsonl') or
            run.get('interactions')!=state.get('interactions') or
            not isinstance(run.get('episodes'),int) or run['episodes']<1 or
            not 0<run['interactions']<=10000):
            raise ValueError('resume lineage, journal, or episode counter mismatch')
        with (output/'training.jsonl').open(encoding='utf-8') as recorded:
            if sum(1 for _ in recorded)!=run['interactions']:
                raise ValueError('resume journal row count mismatch')
        old_identity=read_json(output/'identity.json')
        if old_identity!=identity and old_identity!=run.get('prefix_identity'):
            raise ValueError('resume original identity mismatch')
        if old_identity!=identity and (not run.get('prefix_journal_sha256') or
            not run.get('prefix_interactions')):
            raise ValueError('mixed-version resume requires verified prefix lineage')
        learner.restore_resume_state(state,identity,expected_interactions=run['interactions'])
        interactions,episodes=run['interactions'],run['episodes']
        lineage=state['lineage']
        best_score=run.get('best_score',math.inf)
        selected=run.get('selected')
        write_once(output/'identity_resume.json',identity)
    def guard():
        if total_wall_s is not None and time.monotonic()-started>=total_wall_s:
            raise TimeoutError('Training attempt deadline; no automatic continuation')
        if source_identity()!=identity['sources_sha256']:
            raise RuntimeError('Experiment sources changed during frozen training attempt')
        if dataset_identity(suite_path,training_only=True)!=identity['training_data_sha256']:
            raise RuntimeError('Training data or configuration changed during frozen attempt')
    def save_resume():
        if method=='rdg_nlns':return
        state=learner.resume_state(identity,lineage)
        state['training']={'suite_sha256':identity['suite_sha256'],
            'training_data_sha256':identity['training_data_sha256'],
            'journal_sha256':sha256(output/'training.jsonl'),
            'interactions':interactions,'episodes':episodes,
            'best_score':best_score,'selected':selected,
            'prefix_identity':read_json(output/'identity.json'),
            'prefix_journal_sha256':lineage.get('prefix_journal_sha256'),
            'prefix_interactions':lineage.get('prefix_interactions')}
        temporary=output/'resume_state.pt.partial'
        if temporary.exists():temporary.unlink()
        torch.save(state,temporary)
        temporary.replace(output/'resume_state.pt')
    def validate_boundary():
        nonlocal selected,best_score
        candidate=output/f'candidate_{interactions:05d}.pt'
        if not candidate.exists():
            if method=='rdg_nlns':trainer.save(candidate,interactions,protocol='multicomponent_3d_targetmatched_v2')
            else:learner.save(candidate,identity)
        evidence=output/f'validation_{interactions:05d}.json'
        if evidence.exists():
            records=read_json(evidence)
        else:
            records=[]
            for val in validation:
                guard()
                vs,vmeta=prepare_snapshot(suite_path,val,'M3_full',cache)
                config=suite_path.parent/val['config_path']
                if method=='rdg_nlns':learner._repair_solver=ScipyMilpAgent.from_config(config,10.)
                guard()
                remaining=total_wall_s-(time.monotonic()-started) if total_wall_s is not None else 300.
                with isolated_evaluation(learner,seed):
                    result=solve_method(vs,config,method,seed=seed,variant=variant,learner=learner,
                        wall_s=min(300.,remaining))
                valid=validate_selected_columns(vs,result)
                records.append({'scenario_id':val['scenario_id'],'objective':result.objective_value,
                    'valid':valid.is_valid,'issues':list(valid.issue_codes),'runtime_s':result.runtime_seconds,
                    'normalized_objective':result.objective_value/max(1,len(vs.tasks)) if valid.is_valid else None})
            write_once(evidence,records)
        if len(records)!=9:raise ValueError('incomplete validation boundary')
        score=sum(r['normalized_objective'] for r in records)/len(records) if all(r['valid'] for r in records) else math.inf
        if score<best_score:
            best_score=score
            selected={'checkpoint':candidate.name,'checkpoint_sha256':sha256(candidate),'selected_checkpoint_step':interactions,
                'validation_file':evidence.name,'validation_sha256':sha256(evidence),'validation_score':score}
        save_resume()
    with (output/'training.jsonl').open('a' if resume_state else 'x',encoding='utf-8') as log:
        if resume_state is not None and interactions>0 and interactions%2500==0:
            validate_boundary()
        while interactions<10000:
            guard()
            entry=train_entries[episodes%36]
            snapshot,cache_record=prepare_snapshot(suite_path,entry,'M3_full',cache)
            guard()
            config=suite_path.parent/entry['config_path']
            from Agent.optimization.scipy_milp_agent import ScipyMilpAgent
            solver=ScipyMilpAgent.from_config(config,10.)
            count=min(settings['steps_per_episode'],10000-interactions,2500-interactions%2500)
            deadline=RuntimeDeadline.start(min(300.,max(.001,total_wall_s-(time.monotonic()-started))) if total_wall_s is not None else 300.)
            before=interactions
            if method=='rdg_nlns':
                learner._repair_solver=solver
                current=initial_solution(snapshot,solver,'greedy',deadline=deadline,route_projection_time_limit_seconds=min(30.,deadline.remaining_seconds()))
                for index in range(count):
                    if deadline.expired():break
                    step=learner.training_step(snapshot,current,index,index==count-1,deadline=deadline)
                    replay.append(step.transition)
                    if len(replay)>=max(settings['replay_warmup'],settings['batch_size']) and (interactions+1)%settings['update_frequency']==0:
                        trainer.update(replay.sample(settings['batch_size']))
                    current=step.next_solution
                    interactions+=1
                    log.write(json.dumps({'interaction':interactions,'scenario_id':entry['scenario_id'],
                        'destroy':step.destroy_action,'repair':step.repair_action,'projection_valid':step.projection_valid,
                        'issues':list(step.issue_codes),'reward_source':step.reward_source,'mean_reward':step.transition.mean_reward,
                        'objective':current.objective_value})+'\n')
                    log.flush()
            else:
                before_native=learner.interactions
                def trace(record):
                    log.write(json.dumps({'scenario_id':entry['scenario_id'],**record})+'\n');log.flush()
                learner.solve(snapshot,solver,iterations=count,wall_s=deadline.remaining_seconds(),training=True,trace_callback=trace)
                interactions+=learner.interactions-before_native
            episodes+=1
            atomic_json(output/'status.json',{'state':'training','interactions':interactions,'target':10000,
                'episode':episodes,'last_scenario':entry['scenario_id'],'elapsed_s':time.monotonic()-started,
                'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            if interactions==before:raise RuntimeError('No trainable action in episode; diagnose before continuing')
            if method!='rdg_nlns':
                log.flush()
                save_resume()
            if interactions%2500!=0:continue
            validate_boundary()
        if selected is None:raise RuntimeError('No checkpoint passed validation; formal release remains closed')
        guard()
        write_once(output/'checkpoint.json',{**identity,**selected,'formal_training_interactions':interactions,'selected_on':'validation',
            'validation_count':9,'training_scenarios':36,'checkpoint_identity':identity,
            'training_lineage':lineage if method!='rdg_nlns' else None,
            'elapsed_s':time.monotonic()-started})
        atomic_json(output/'status.json',{'state':'completed','interactions':interactions,'selected':selected,'elapsed_s':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('suite','output','cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['rdg_nlns','nlns_public','grlos_public','dac_alns_public'],required=True)
    p.add_argument('--variant',default='full')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--deadline-seconds',type=float)
    p.add_argument('--resume-state',type=Path)
    args=p.parse_args()
    if args.deadline_seconds is not None and args.deadline_seconds<=0:p.error('deadline must be positive')
    train(args.suite,args.output,args.cache,args.method,args.variant,args.seed,args.deadline_seconds,args.resume_state)
