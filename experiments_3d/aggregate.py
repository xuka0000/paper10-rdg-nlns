"""Descriptive matched comparisons; failures remain in registered denominators."""
import argparse
from collections import defaultdict
import math
import statistics
from pathlib import Path
from experiments_3d.data_upgrade import read_json,write_once,sha256


def summary(values):
    finite=[v for v in values if v is not None and math.isfinite(v)]
    return {'n':len(finite),'mean':statistics.mean(finite) if finite else None,
        'std':statistics.stdev(finite) if len(finite)>1 else (0. if finite else None)}


def paired_comparisons(rows):
    """Pair methods/ablations on identical scenes, seeds and event conditions."""
    paired={name:[] for name in ('E2','E3','E4')}
    baseline={}
    key=lambda r:(r['experiment'],r['scenario_id'],r['seed'],r.get('condition','static'))
    for row in rows:
        if row['method']=='rdg_nlns' and row['model']=='M3_full' and row.get('variant','full')=='full':
            baseline[key(row)]=row
    for row in rows:
        full=baseline.get(key(row))
        if row['experiment'] not in paired or full is None or row is full:continue
        pair={'comparison':row.get('variant',row['method']) if row['experiment']=='E3' else row['model']+'/'+row['method'],
            'scenario_id':row['scenario_id'],'seed':row['seed'],'condition':row.get('condition','static'),
            'both_recorded':row['cell_state']==full['cell_state']=='completed',
            'both_planning_valid':row['planning_valid'] is True and full['planning_valid'] is True,
            'both_missions_complete':row['mission_complete'] is True and full['mission_complete'] is True}
        for metric in ('objective','coverage','completion_s','energy_j','vehicle_wait_s','uav_wait_s','rendezvous_rate'):
            eligible=pair['both_recorded']
            if metric=='objective':eligible=eligible and pair['both_planning_valid']
            elif metric not in ('coverage','rendezvous_rate'):eligible=eligible and pair['both_missions_complete']
            left,right=full[metric],row[metric]
            pair['full_minus_comparator_'+metric]=left-right if eligible and left is not None and right is not None else None
        paired[row['experiment']].append(pair)
    return paired


def collect(matrix_path,index_path,output):
    matrix=read_json(matrix_path);index=read_json(index_path)
    suite=read_json(Path(matrix['suite_path']))
    scenes={s['scenario_id']:s for s in suite['scenarios']}
    # Explicit cell->attempt binding prevents timestamp selection or hiding
    # failed attempts when several directories contain the same cell.
    if index['matrix_sha256']!=sha256(matrix_path):raise ValueError('result index belongs to another matrix')
    unknown=set(index['cells'])-{c['cell_id'] for c in matrix['cells']}
    if unknown:raise ValueError('unknown result-index cells')
    rows=[]
    for cell in matrix['cells']:
        row={**cell,'cell_state':'not_run','planning_valid':None,'objective':None,'coverage':None,
            'mission_complete':None,'completion_s':None,'energy_j':None,'vehicle_wait_s':None,'uav_wait_s':None,'rendezvous_rate':None}
        row.update(scene_family=cell['scenario_id'].split('-')[0],tower_scale=scenes[cell['scenario_id']]['labels'].get('scale'))
        binding=index['cells'].get(cell['cell_id'])
        if binding:
            path=Path(binding['result'])
            if sha256(path)!=binding['sha256']:raise ValueError('result evidence changed')
            if not binding.get('all_attempts'):raise ValueError('all-attempt provenance required')
            for attempt in binding['all_attempts']:
                if sha256(Path(attempt['path']))!=attempt['sha256']:raise ValueError('attempt provenance changed')
            result=read_json(path)
            if binding.get('artifact_type')=='process_failure':
                row.update(cell_state='infrastructure_failure',failure=result,all_attempts=binding['all_attempts'])
                rows.append(row);continue
            if result['cell']!=cell:raise ValueError('result-cell binding mismatch')
            if 'validation' in result:
                row.update(planning_valid=result['validation']['is_valid'],objective=result['result']['objective_value'],cell_state='completed')
            execution=result.get('execution')
            if execution is None and result.get('independent_execution'):execution=result['independent_execution']['fleet']
            if execution is not None:
                row.update(cell_state='completed',coverage=execution['effective_task_coverage'],mission_complete=execution['mission_complete'],
                    completion_s=execution['mission_completion_s'],energy_j=execution.get('electrical_energy_j',execution.get('native_electrical_energy_j')),
                    vehicle_wait_s=execution['vehicle_wait_s'],uav_wait_s=execution['uav_wait_s'],rendezvous_rate=execution['rendezvous_success_rate'])
                if execution.get('failure'):row['failure']=execution['failure']
            row['all_attempts']=binding['all_attempts']
        rows.append(row)
    groups=defaultdict(list)
    for row in rows:groups[(row['experiment'],row['scene_family'],row['tower_scale'],row['model'],row['method'],row.get('variant','full'),row.get('condition','static'))].append(row)
    aggregates=[]
    for key,records in groups.items():
        completed=[r for r in records if r['cell_state']=='completed'];safe=[r for r in completed if r['mission_complete'] is True]
        aggregates.append({'group':key,'registered_cells':len(records),'recorded_cells':len(completed),
            'unrun_or_infrastructure_unknown':len(records)-len(completed),'successful_missions':len(safe),
            'success_rate_over_registered':len(safe)/len(records) if len(completed)==len(records) and key[0]!='E2' else None,
            'planning_valid_cells':sum(r['planning_valid'] is True for r in completed),
            'planning_objective_valid_only':summary([r['objective'] for r in completed if r['planning_valid']]),
            'coverage_all_recorded':summary([r['coverage'] for r in completed]),
            'completion_s_successful_only':summary([r['completion_s'] for r in safe]),
            'energy_j_successful_only':summary([r['energy_j'] for r in safe]),
            'vehicle_wait_s_all_recorded':summary([r['vehicle_wait_s'] for r in completed])})
    paired=[]
    for row in rows:
        if row['experiment']!='E1' or row['model']=='M3_full':continue
        match=next(r for r in rows if r['experiment']=='E1' and r['model']=='M3_full' and
            (r['scenario_id'],r['seed'],r['terrain_relief_factor'])==(row['scenario_id'],row['seed'],row['terrain_relief_factor']))
        if row['cell_state']!='completed' or match['cell_state']!='completed':continue
        pair={'comparison':row['model'],'scenario_id':row['scenario_id'],'seed':row['seed'],'relief':row['terrain_relief_factor'],
            'full_minus_control_coverage':match['coverage']-row['coverage'] if match['coverage'] is not None and row['coverage'] is not None else None,
            'both_missions_complete':bool(match['mission_complete'] and row['mission_complete'])}
        pair['full_minus_control_energy_j']=match['energy_j']-row['energy_j'] if pair['both_missions_complete'] else None
        pair['full_minus_control_completion_s']=match['completion_s']-row['completion_s'] if pair['both_missions_complete'] else None
        paired.append(pair)
    report={'matrix_sha256':sha256(matrix_path),'index_sha256':sha256(index_path),'rows':rows,'groups':aggregates,'paired_E1':paired,
        **{'paired_'+name:records for name,records in paired_comparisons(rows).items()},
        'statistics':'descriptive mean/std and paired differences; no hypothesis tests',
        'cost_boundary':'energy/time comparisons conditional on both missions completing; failures never rewarded for early termination'}
    write_once(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('matrix','index','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();collect(a.matrix,a.index,a.output)
