"""Content-addressed local snapshot cache with source and payload verification.

Pickles are private outputs of this process, never public dataset inputs.
Only load the cache directory owned by this experiment; hashes detect changes.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path

from Env.data_loader import load_project_data
from Agent.optimization.scipy_trajectory_agent import ScipyTrajectoryAgent
from experiments_3d.data_upgrade import read_json, sha256, write_once
from experiments_3d.model_controls import build_model


def source_identity():
    root=Path(__file__).resolve().parents[1]
    paths=[*sorted((root/'Mask').rglob('*.py')),*sorted((root/'Env').rglob('*.py')),
           *sorted((root/'Agent').rglob('*.py')),*sorted((root/'experiments_3d').glob('*.py'))]
    native=root/'Unreal/IRLiveS140/Plugins/IRLiveS140Platform/Source/EmbodiedAgents'
    paths += [native/p for p in ('Public/IRDroneDynamicsComponent.h','Public/IRDroneBatteryComponent.h',
        'Private/IRDroneDynamicsComponent.cpp','Private/IRDroneBatteryComponent.cpp',
        'Private/Tests/IRPaper10ExecutionTests.cpp','Private/Tests/IRPaper10FleetTests.cpp')]
    # Bind pinned upstream source as well as the local task adapters. Ignore
    # interpreter caches, which may be generated merely by importing a module.
    external=root/'Data/external/multicomponent_3d_public_sources_20260928'
    paths += sorted(p for p in external.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.toml','.yml','.yaml','.txt'})
    return hashlib.sha256(''.join(str(p.relative_to(root))+sha256(p) for p in paths).encode()).hexdigest()


def scene_identity(suite_path,entry,model,relief):
    root=suite_path.parent
    return {'suite_sha256':sha256(suite_path),'scenario_id':entry['scenario_id'],
        'scene_sha256':sha256(root/entry['scenario_directory']/'scene.json'),
        'candidates_sha256':sha256(root/entry['scenario_directory']/'service_candidates.json'),
        'config_sha256':sha256(root/entry['config_path']), 'model':model,'relief':float(relief),'sources_sha256':source_identity()}


def prepare_snapshot(suite_path,entry,model,directory,relief=1.):
    suite=read_json(suite_path)
    identity=scene_identity(suite_path,entry,model,relief)
    weather=(suite_path.parent/suite['weather_release_path']).resolve()
    identity['weather_sha256']=sha256(weather)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    metadata=directory/f'{key}.json'
    payload=directory/f'{key}.pkl'
    if metadata.exists():
        record=read_json(metadata)
        if record['identity']!=identity or sha256(payload)!=record['payload_sha256']:
            raise ValueError('snapshot cache identity or payload mismatch')
        with payload.open('rb') as stream:
            snapshot=pickle.load(stream)
        return snapshot,record
    started=time.perf_counter()
    config=suite_path.parent/entry['config_path']
    data=load_project_data(suite_path.parent/entry['scenario_directory'],config)
    optimizer=ScipyTrajectoryAgent.from_config(config) if model.startswith('M3_') else None
    env,_=build_model(data,model,optimizer=optimizer,relief_factor=relief,weather_path=weather,weather_split=entry['split'],
                      trajectory_cache=directory/'trajectories')
    snapshot=env.snapshot()
    directory.mkdir(parents=True,exist_ok=True)
    with payload.open('xb') as stream:
        pickle.dump(snapshot,stream,protocol=5)
    record={'identity':identity,'payload_sha256':sha256(payload),'build_seconds':time.perf_counter()-started,
        'task_count':len(snapshot.tasks),'admitted_columns':len(snapshot.service_columns),'rejections':len(snapshot.rejection_records)}
    write_once(metadata,record)
    return snapshot,record
