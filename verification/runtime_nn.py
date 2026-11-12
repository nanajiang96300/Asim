"""Source/evaluator/library/physical-config identity for supervised NN acceptance."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

def file_sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fingerprint(repo,binary,config,cwd):
    repo=Path(repo);binary=Path(binary);paths=set()
    for directory,suffixes in [('src',{'.h','.cc'}),('verification',{'.py'}),('scripts',{'.py'})]:
        paths.update(p for p in (repo/directory).rglob('*') if p.is_file() and p.suffix in suffixes)
    paths.update(repo/p for p in ['CMakeLists.txt','dependencies/lock.json','configs/ascend_910b_nn_v1.json','configs/ascend_910b_quiet.json','orchestrator/codex_operators.json'])
    files=[dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(paths)]
    hardware=dict(config);physical={}
    for k in ['dram_config_path','icnt_config_path']:
        physical[k]=file_sha((Path(cwd)/config[k]).resolve());hardware.pop(k,None)
    library=binary.parent.parent/'lib/libramulator.so'
    return {'schema':'nn-acceptance-source-fingerprint-v1','files':files,'binary_sha256':file_sha(binary),'library_sha256':file_sha(library),'python_version':sys.version.split()[0],'numpy_version':np.__version__,'hardware_config':hardware,'physical_config_sha256':physical,'notice':'locator normalization for this acceptance snapshot only; legacy comparison_identity unchanged'}

def common_identity(a,b):
    return all(a[k]==b[k] for k in ['files','binary_sha256','library_sha256','python_version','numpy_version','physical_config_sha256'])
