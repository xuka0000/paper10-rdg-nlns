"""Fail closed before any formal test cell or old-checkpoint reuse."""
from pathlib import Path
from experiments_3d.data_upgrade import read_json,sha256
from experiments_3d.matrix import REQUIRED_GATES
from experiments_3d.release_identity import dataset_identity,execution_identity


def required_gates(cell=None):
    if cell is None:return REQUIRED_GATES
    names=['physical_profile_frozen','candidate_domain_audited','model_controls_validated']
    if cell['experiment'] in {'E1','E3','E4'}:names.append('independent_executor_validated')
    if cell['method']!='milp':names.append('public_baselines_adapted')
    if cell['method'] not in {'milp','alns_public'} and cell.get('variant')!='uniform_actions':names.append('training_complete')
    return tuple(names)


def require_gates(matrix_path: Path,gate_path: Path,cell=None):
    matrix=read_json(matrix_path)
    release=read_json(gate_path)
    if release.get('matrix_sha256')!=sha256(matrix_path):
        raise ValueError('gate release belongs to another experiment matrix')
    root=Path(__file__).resolve().parents[2]
    failures=[]
    for name in required_gates(cell):
        record=release.get('gates',{}).get(name,{})
        if record.get('status')!='pass' or not record.get('evidence'):
            failures.append(name);continue
        if cell and record.get('experiments') and cell['experiment'] not in record['experiments']:
            failures.append(name+':experiment_not_covered')
        if cell and record.get('methods') and cell['method'] not in record['methods']:
            failures.append(name+':method_not_covered')
        for evidence in record['evidence']:
            path=(root/evidence['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256(path)!=evidence['sha256']:
                failures.append(name+':stale_or_invalid_evidence')
    if failures:
        raise RuntimeError('Formal execution gates remain closed: '+', '.join(failures))
    if sha256(Path(matrix['suite_path']))!=matrix['suite_sha256']:
        raise ValueError('experiment suite changed after registration')
    from experiments_3d.snapshot_cache import source_identity
    if release.get('sources_sha256')!=source_identity():
        raise ValueError('formal release does not match current experiment sources')
    if release.get('inputs_sha256')!=execution_identity(matrix_path):
        raise ValueError('formal release data, configuration or event inputs changed')
    if cell and cell['experiment'] in {'E1','E3','E4'}:
        binary=release.get('native_binary',{})
        path=(root/binary.get('path','')).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path)!=binary.get('sha256'):
            raise ValueError('native execution binary is absent or changed since validation')
    return matrix


def require_checkpoint(metadata_path: Path,*,suite_path: Path,method: str,variant: str,seed: int):
    metadata=read_json(metadata_path)
    expected={'suite_sha256':sha256(suite_path),'method':method,'variant':variant,'seed':seed}
    if any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('checkpoint method/variant/seed/domain mismatch')
    if metadata.get('formal_training_interactions')!=10000 or metadata.get('selected_on')!='validation':
        raise ValueError('checkpoint lacks complete matched training and validation selection')
    if metadata.get('training_data_sha256')!=dataset_identity(suite_path,training_only=True):
        raise ValueError('checkpoint training data or configuration changed')
    path=(metadata_path.parent/metadata['checkpoint']).resolve()
    if sha256(path)!=metadata['checkpoint_sha256']:
        raise ValueError('checkpoint payload changed')
    return path,metadata
