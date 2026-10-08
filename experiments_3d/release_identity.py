"""Bind indirect data/config/event files, not just the suite's path list."""
import hashlib
import json
from pathlib import Path
from experiments_3d.data_upgrade import read_json,sha256


def digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def dataset_identity(suite_path,*,training_only=False):
    suite_path=Path(suite_path);suite=read_json(suite_path);root=suite_path.parent
    records={'suite':sha256(suite_path),'weather':sha256((root/suite['weather_release_path']).resolve()),
        'physical_profile':sha256(root/'physical_profile.json'),'scope':'train_validation' if training_only else 'all_scenes'}
    for entry in suite['scenarios']:
        if training_only and entry['split'] not in {'train','validation','val'}:continue
        folder=root/entry['scenario_directory'];name=entry['scenario_id']
        records[name]={'scene':sha256(folder/'scene.json'),'candidates':sha256(folder/'service_candidates.json'),
            'config':sha256(root/entry['config_path'])}
    return digest(records)


def execution_identity(matrix_path):
    matrix_path=Path(matrix_path);matrix=read_json(matrix_path)
    return digest({'matrix':sha256(matrix_path),'dataset':dataset_identity(Path(matrix['suite_path'])),
        'events':{p.name:sha256(p) for p in sorted((matrix_path.parent/'events').glob('*.json'))}})
