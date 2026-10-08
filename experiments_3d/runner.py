"""Execute registered static cells; all formal evidence gates are mandatory."""
import argparse
import dataclasses
import datetime
import traceback
from pathlib import Path

import torch
from Env.data_loader import load_project_data
from Env.validation import validate_selected_columns
from experiments_3d.data_upgrade import read_json,sha256,write_once
from experiments_3d.execution import reference_manifest
from experiments_3d.gates import require_gates,require_checkpoint
from experiments_3d.methods import make_learner,solve_method
from experiments_3d.model_controls import transformed_data
from experiments_3d.native_executor import execute_fleet
from experiments_3d.preflight import evidence_json
from experiments_3d.snapshot_cache import prepare_snapshot,source_identity
from experiments_3d.supervision import atomic_json


def run_cell(matrix_path,gate_path,cell_id,output,cache,checkpoint_metadata=None,editor=None,native_project=None):
    matrix=read_json(matrix_path)
    cell=next(c for c in matrix['cells'] if c['cell_id']==cell_id)
    require_gates(matrix_path,gate_path,cell)
    if cell['experiment'] not in {'E1','E2'}:
        raise ValueError('Rolling E3/E4 cells require the causal fleet runner, not the static runner')
    suite_path=Path(matrix['suite_path'])
    suite=read_json(suite_path)
    entry=next(s for s in suite['scenarios'] if s['scenario_id']==cell['scenario_id'])
    if entry['split']!='test':raise ValueError('formal cell is not bound to final test split')
    if cell['experiment']=='E1' and (editor is None or native_project is None):
        raise ValueError('E1 requires independent native execution paths')
    output.mkdir(parents=True,exist_ok=False)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    identity={'cell':cell,'matrix_sha256':sha256(matrix_path),'gates_sha256':sha256(gate_path),
              'suite_sha256':sha256(suite_path),'source_sha256':source_identity(),'started_at':started}
    write_once(output/'identity.json',identity)
    atomic_json(output/'status.json',{'state':'preparing','started_at':started})
    try:
        snapshot,cache_record=prepare_snapshot(suite_path,entry,cell['model'],cache,cell.get('terrain_relief_factor',1.))
        config=suite_path.parent/entry['config_path']
        trace=[]
        learner=trainer=None
        method=cell['method']
        variant=cell.get('variant','full')
        if method not in {'milp','alns_public'}:
            if checkpoint_metadata is None:raise ValueError('formal learned baseline checkpoint metadata required')
            path,metadata=require_checkpoint(checkpoint_metadata,suite_path=suite_path,method=method,variant=variant,seed=cell['seed'])
            if metadata.get('sources_sha256')!=source_identity():raise ValueError('checkpoint experiment sources changed')
            learner,trainer,_=make_learner(method,config,seed=cell['seed'],variant=variant,
                iterations=matrix['search_budget']['neighborhoods'],wall_s=matrix['search_budget']['wall_clock_s'],training_settings=suite['training'])
            if method=='rdg_nlns':trainer.load(path)
            else:learner.load(path,metadata['checkpoint_identity'])
        atomic_json(output/'status.json',{'state':'solving','started_at':started,'preparation':cache_record})
        torch.set_num_threads(1)
        result=solve_method(snapshot,config,method,seed=cell['seed'],variant=variant,learner=learner,
            iterations=matrix['search_budget']['neighborhoods'],wall_s=matrix['search_budget']['wall_clock_s'],trace_callback=trace.append)
        validation=validate_selected_columns(snapshot,result)
        record={'cell':cell,'scope':'formal static planning result','result':dataclasses.asdict(result),
            'validation':dataclasses.asdict(validation),'preparation':cache_record,'search_trace':trace,
            'checkpoint_metadata_sha256':sha256(checkpoint_metadata) if checkpoint_metadata else None,
            'independent_execution':None}
        write_once(output/'planning.json',evidence_json(record))
        if not validation.is_valid:raise RuntimeError('M5 validation failed; planning evidence retained')
        if cell['experiment']=='E1':
            physical=load_project_data(suite_path.parent/entry['scenario_directory'],config)
            physical=transformed_data(physical,'M3_full',cell.get('terrain_relief_factor',1.))
            # Remove only the planning tracking buffer from the execution
            # safety threshold; actual terrain-relief transform is retained.
            buffer=physical.config['physical'].get('flight_tracking_buffer_m',0.)
            for u in physical.scene['uavs']:u['minimum_clearance_m']-=buffer
            manifest=output/'execution_manifest.json'
            write_once(manifest,reference_manifest(snapshot,result,physical,cell['model']))
            record['independent_execution']=execute_fleet(snapshot,result,physical,manifest,output/'native',editor=editor,project=native_project)
        if source_identity()!=identity['source_sha256']:raise RuntimeError('sources changed during registered cell')
        require_gates(matrix_path,gate_path,cell)
        write_once(output/'result.json',evidence_json(record))
        atomic_json(output/'status.json',{'state':'completed','finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'meaning':'cell completed; failure/coverage metrics remain in result; not scientific acceptance'})
        return record
    except BaseException as error:
        atomic_json(output/'status.json',{'state':'failed','error':str(error),'traceback':traceback.format_exc(),
            'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'continuation':'diagnosis required; no retry'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('matrix','gates','output','cache'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cell',required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--native-editor',type=Path)
    p.add_argument('--native-project',type=Path)
    a=p.parse_args()
    run_cell(a.matrix,a.gates,a.cell,a.output,a.cache,a.checkpoint,a.native_editor,a.native_project)
