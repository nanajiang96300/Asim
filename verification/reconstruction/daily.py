"""Pinned, supervised daily reconstruction entry. No automatic search or retries."""
import argparse,hashlib,json,os
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
from verification.contracts.core import Invalid,ScreenPolicy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_runtime_files(spec,repo):
    for e in spec['files']:
        p=repo/e['path']
        if not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise Invalid('pinned runtime file changed/missing: '+e['path'])
    required={spec['binary'],spec['config'],*spec['dependencies']}
    if not required<={e['path'] for e in spec['files']}:raise Invalid('runtime manifest did not pin every required file')
    if not (repo/spec['cwd']).is_dir():raise Invalid('pinned runtime working directory missing')



def load_runtime(path,repo):
    raw=Path(path).read_bytes();spec=json.loads(raw)
    if spec.get('schema')!='supervised-reconstruction-runtime-v1':raise Invalid('unknown pinned runtime schema')
    from verification.schedule.runtime import HEADER
    if spec['runtime_contract']!=HEADER:raise Invalid('pinned observer contract differs from evaluator')
    verify_runtime_files(spec,repo)
    spec['_loaded_sha256']=hashlib.sha256(raw).hexdigest();return spec


def baseline(path,operator,iterations,binary_sha,runtime_spec_sha=None):
    path=Path(path);r=json.loads(path.read_text())
    if r['status'] not in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') or r.get('schedule_status')!='MODEL_RW_CHECKED_CONSERVATIVE' or r['numeric_status']!='EXECUTED' or r['failed_batches']:
        raise Invalid('baseline must have passed numeric and runtime checks')
    if r['operator']!=operator or r['algorithm']['iterations']!=iterations or r['asim']['binary_sha256']!=binary_sha:raise Invalid('baseline operator/parameters/binary differs')
    if runtime_spec_sha is not None:
        pin=path.parent/'runtime_pin.json'
        if not pin.is_file() or json.loads(pin.read_text()).get('spec_sha256')!=runtime_spec_sha:raise Invalid('baseline was not produced by the same pinned daily runtime')
    required={'evaluation_result.json','runtime_pin.json','asim.json','schedule.json','outputs.npy','inputs.npz','references.npz','channel.npz','instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','config.json','models.json','simulator.log'}
    names=[e['path'] for e in r['artifacts']]
    if len(set(names))!=len(names) or not required<=set(names):raise Invalid('baseline required artifact set incomplete/duplicate')
    for e in r['artifacts']:
        p=path.parent/e['path']
        if not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise Invalid('baseline artifact changed/missing: '+e['path'])
    original=json.loads((path.parent/'evaluation_result.json').read_text())
    if any(r.get(k)!=v for k,v in original.items() if k!='artifacts'):raise Invalid('baseline result differs from sealed evaluation')
    current={e['path']:e for e in r['artifacts']}
    if any(current.get(e['path'])!=e for e in original['artifacts']):raise Invalid('baseline changed original evidence manifest')
    receipt=json.loads((path.parent/'asim.json').read_text())
    if receipt['status']!='COMPLETED' or receipt['returncode']!=0 or receipt['binary_sha256']!=binary_sha or receipt['finish_cycle']!=r['asim']['finish_cycle']:raise Invalid('baseline differs from actual Asim receipt')
    return r['asim']['finish_cycle'],r['comparison_identity']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--operator',default='ns_corrected_v1')
    ap.add_argument('--profile',choices=['structure_f64_real_block','modeled_fp16_fp32_real_block_v1'],default='modeled_fp16_fp32_real_block_v1')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--iterations',type=int,default=16);ap.add_argument('--baseline',type=Path)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[2]
    root=(a.output or repo/'results/daily_reconstruction'/(datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])).resolve()
    if root.exists():raise Invalid('refusing existing output directory')
    phase="PREFLIGHT";r=None
    try:
        spec_path=a.runtime or repo/'orchestrator/codex_runtime.json'
        spec=load_runtime(spec_path,repo);spec_sha=spec['_loaded_sha256']
        if a.profile not in spec.get('profiles',[]):raise Invalid('profile not accepted by pinned runtime')
        if a.operator not in spec['operators']:raise Invalid('operator not accepted by pinned runtime')
        if spec.get('library_dir'):os.environ['LD_LIBRARY_PATH']=str(repo/spec['library_dir'])
        os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
        from verification.reconstruction.runner import evaluate
        from verification.contracts.mmse import MMSETask
        from verification.profiles.numeric import policy_for
        import sys,numpy as np
        if spec.get('python_version')!=sys.version.split()[0] or spec.get('numpy_version')!=np.__version__:raise Invalid('verifier Python/NumPy version differs from accepted runtime')
        config=json.loads((repo/spec['config']).read_text());config['precision']=policy_for(a.profile).bytes
        for key in ('dram_config_path','icnt_config_path'):
            if config.get(key):config[key]=str((repo/spec['cwd']/config[key]).resolve())
        policy=identity=None
        if a.baseline:
            cycles,identity=baseline(a.baseline,a.operator,a.iterations,sha(repo/spec['binary']),spec_sha);policy=ScreenPolicy(cycles,1,10)
        phase='EVALUATE'
        r=evaluate(a.operator,MMSETask(profile=a.profile),a.iterations,a.seed,repo/spec['binary'],config,root,root,policy,identity)
        phase='SEAL'
        if sha(spec_path)!=spec_sha:raise Invalid('runtime manifest changed during evaluation')
        verify_runtime_files(spec,repo)
        if r.get('asim') and r['asim']['binary_sha256']!=next(e['sha256'] for e in spec['files'] if e['path']==spec['binary']):raise Invalid('evaluated binary differs from pinned binary')
        (root/'evaluation_result.json').write_bytes((root/'result.json').read_bytes())
        (root/'runtime_pin.json').write_text(json.dumps({'spec_sha256':spec_sha,'files':spec['files'],'runtime_contract':spec['runtime_contract'],'binary_source_commit':spec['binary_source_commit']},indent=2))
        for name in ('evaluation_result.json','runtime_pin.json'):
            path=root/name;r['artifacts'].append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
        r['entrypoint']='supervised_daily_v1'
        pending=root/('result-'+uuid4().hex+'.tmp');pending.write_text(json.dumps(r,indent=2,allow_nan=False));pending.replace(root/'result.json')
    except (Invalid,KeyError,TypeError,ValueError,OSError) as e:
        root.mkdir(parents=True,exist_ok=True);r={'status':'PREFLIGHT_REJECTED' if phase=='PREFLIGHT' else 'RUN_OR_ARCHIVE_FAILED','numeric_status':r.get('numeric_status','NOT_RUN') if r else 'NOT_RUN','score':None,'reason':str(e),'phase':phase}
        (root/'result.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({'run_dir':str(root),**{k:v for k,v in r.items() if k not in ('artifacts','batches','asim')}}))
    return 0 if r['status'] in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') else 1

if __name__=='__main__':raise SystemExit(main())
