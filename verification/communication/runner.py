"""Read-only QPSK communication gate after approved actual reconstruction.

The caller approves exact prior result hashes; not a signature or OS boundary.
No candidate-specific algorithm is used to generate communication outputs.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from dataclasses import fields

import numpy as np

from verification.contracts.core import Invalid, Unsupported, digest
from verification.contracts.mmse import MMSETask,fixture,task_reference,decode_checked
from verification.profiles.numeric import policy_for,bit_equal
from verification.operators.catalog import adapter
from verification.reconstruction.daily import load_runtime,sha
from verification.reconstruction.runner import error
from verification.communication.metrics import paired,summarize
from verification.planning.handoff import verify as verify_plan

REQUIRED_ARTIFACTS = {'outputs.npy','channel.npz','inputs.npz','references.npz','asim.json',
                      'schedule.json','instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl',
                      'config.json','models.json','simulator.log'}


def validate(contract):
    required={'schema','policy_level','required_metrics','runtime','cases','family_alpha',
              'max_channel_ber_increase','max_channel_evm_increase','max_outage_probability','min_independent_channels'}
    if set(contract)!=required or contract['schema']!='rayleigh-qpsk-quality-v1':
        raise Invalid('unknown/incomplete communication contract')
    if contract['policy_level']!='engineering_trial': raise Unsupported('production promotion policy not implemented')
    if contract['required_metrics']!=['paired_uncoded_qpsk_ber','raw_symbol_evm']:
        raise Unsupported('SE/BLER/CDL-B/high-order QAM are not supported; no metric substitution')
    for name in ('family_alpha','max_outage_probability'):
        v=contract[name]
        if type(v) not in (int,float) or not math.isfinite(v) or not 0<v<1: raise Invalid('invalid probability '+name)
    for name in ('max_channel_ber_increase','max_channel_evm_increase'):
        v=contract[name]
        if type(v) not in (int,float) or not math.isfinite(v) or v<0: raise Invalid('invalid tolerance '+name)
    n=contract['min_independent_channels']
    if type(n) is not int or n<2: raise Invalid('at least two independent channels required')
    cases=contract['cases']
    if not isinstance(cases,list) or not 1<=len(cases)<=32: raise Invalid('1..32 approved results per contract')
    for item in [contract['runtime'],*cases]:
        keys={'path','sha256'} if item is contract['runtime'] else {'path','sha256','role'}
        if set(item)!=keys or not isinstance(item['path'],str) or not item['path'] or not isinstance(item['sha256'],str) or len(item['sha256'])!=64:
            raise Invalid('exact artifact path/SHA-256 required')
        if 'role' in item and item['role'] not in ('holdout','historical_diagnostic'):
            raise Invalid('calibration data cannot certify communication quality')


def check_artifacts(directory,result):
    seen=set()
    for a in result['artifacts']:
        name=a['path'];p=directory/name
        if name in seen or Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(directory.resolve()) or not p.is_file() or p.stat().st_size!=a['bytes'] or sha(p)!=a['sha256']:
            raise Invalid('changed/missing/unsafe reconstruction artifact: '+name)
        seen.add(name)
    if not REQUIRED_ARTIFACTS<=seen: raise Invalid('incomplete Asim/reconstruction evidence')


def load_case(item,runtime,repo):
    path=(repo/item['path']).resolve();directory=path.parent
    if sha(path)!=item['sha256']: raise Invalid('approved reconstruction result changed')
    r=json.loads(path.read_text())
    if r.get('status')!='MODELED_NUMERIC_CHECKED' or r.get('numeric_status')!='EXECUTED' or r.get('failed_batches')!=[] or r.get('schedule_status')!='MODEL_RW_CHECKED_CONSERVATIVE' or r.get('structural_reference_passed') is not True or r.get('task_reference_passed') is not True:
        raise Invalid('communication requires successful actual numeric and schedule gates')
    if r.get('plan_quality_status','PASSED')!='PASSED': raise Invalid('plan quality prerequisite failed')
    task=MMSETask(**{f.name:r['task'][f.name] for f in fields(MMSETask)});task.validate()
    if task.profile!='modeled_fp16_fp32_real_block_v1' or task.scope!='mmse_h_y_to_xhat_v1':
        raise Unsupported('communication requires full-scope explicit FP16 task')
    precision=policy_for(task.profile)
    if r['task']!=task.frozen() or r['task_hash']!=task.identity or r['precision_policy']!=precision.frozen() or r['runtime_contract']!=runtime['runtime_contract']:
        raise Invalid('task/precision/runtime identity mismatch')
    _,params,models,_=adapter(r['operator'],task,r['algorithm'].get('iterations',1))
    if params!=r['algorithm'] or r['operator'] not in runtime['operators']: raise Invalid('unsupported actual operator/parameters')
    if type(r['seed']) is not int or r['seed']<0: raise Invalid('invalid input seed')
    check_artifacts(directory,r)
    receipt=json.loads((directory/'asim.json').read_text())
    schedule=json.loads((directory/'schedule.json').read_text())
    if any(r['asim'].get(k)!=v for k,v in receipt.items()): raise Invalid('result differs from actual Asim receipt')
    if receipt['status']!='COMPLETED' or receipt['returncode']!=0 or receipt['binary_sha256']!=sha(repo/runtime['binary']) or receipt['finish_cycle']!=r['asim']['finish_cycle'] or schedule['status']!=r['schedule_status']:
        raise Invalid('actual Asim receipt/schedule/runtime mismatch')
    if json.loads((directory/'models.json').read_text())!=models: raise Invalid('actual Asim model differs from task')
    required_holdout_paths=[]
    if item['role']=='holdout':
        plan=json.loads((directory.parent/'approved_plan.json').read_text())
        verify_plan(plan,repo)
        binding=json.loads((directory/'plan_binding.json').read_text())
        job=binding['job']
        if ('plan_binding.json' not in {a['path'] for a in r['artifacts']} or
            binding['plan_content_sha256']!=digest(plan) or job not in plan['jobs'] or
            job['split']!='holdout' or job['seed']!=r['seed'] or MMSETask(**job['task']).frozen()!=task.frozen() or
            plan['algorithm']!=params):
            raise Invalid('heldout role/task/parameter plan binding mismatch')
        required_holdout_paths=[str((directory.parent/j['id']/'result.json').resolve())
                                for j in plan['jobs'] if j['split']=='holdout']
    data=fixture(task,r['seed'])
    with np.load(directory/'channel.npz',allow_pickle=False) as stored:
        if set(stored.files)!={'H','Y','symbols'} or any(not np.array_equal(stored[k],data[k]) for k in stored.files):
            raise Invalid('channel/symbol/noise fixture mismatch')
    with np.load(directory/'inputs.npz',allow_pickle=False) as stored:
        if set(stored.files)!=set(data['inputs']) or any(not np.array_equal(stored[k],data['inputs'][k]) for k in stored.files):
            raise Invalid('actual quantized input mismatch')
    target=task_reference(data,task)
    actual=np.load(directory/'outputs.npy',allow_pickle=False)
    if actual.shape!=(task.batch_size,2*task.nt,2*task.nrhs): raise Invalid('incomplete actual output shape')
    with np.load(directory/'references.npz',allow_pickle=False) as stored:
        if stored['task'].shape!=target.shape or not np.allclose(stored['task'],target,rtol=1e-12,atol=1e-12):
            raise Invalid('stored MMSE differs from recomputed high precision reference')
        if not bit_equal(actual,stored['algorithm_block']): raise Invalid('actual replay differs from sealed algorithm reference')
    output=np.stack([decode_checked(x,task.nt,task.nrhs,precision.quadrant_tolerance) for x in actual])
    if any(error(x,y)>precision.task_tolerance for x,y in zip(output,target)): raise Invalid('actual task quality gate failed on recheck')
    rows=paired(output,target,data['symbols'])
    for row,h in zip(rows,data['H']):
        row.update(channel_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
                   source_result=item['path'],seed=r['seed'])
    # Only diagnostics use the independent task solve. Candidate metrics use actual outputs.
    return dict(identity=digest({'task':task.frozen(),'algorithm':params}),task=task.frozen(),
                algorithm=params,channels=rows,finish_cycle=receipt['finish_cycle'],sample_role=item['role'],
                required_holdout_paths=required_holdout_paths)


def evaluate(contract,repo):
    validate(contract);repo=Path(repo).resolve()
    own_sources={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'verification/communication').glob('*.py'))}
    ref=contract['runtime'];runtime_path=repo/ref['path']
    if sha(runtime_path)!=ref['sha256']: raise Invalid('approved runtime manifest changed')
    runtime=load_runtime(runtime_path,repo)
    if runtime['python_version']!=platform.python_version() or runtime['numpy_version']!=np.__version__:
        raise Invalid('communication/runtime toolchain mismatch')
    groups={}
    approved_holdouts={str((repo/c['path']).resolve()) for c in contract['cases'] if c['role']=='holdout'}
    for item in contract['cases']:
        g=load_case(item,runtime,repo);key=g['identity']
        if not set(g['required_holdout_paths'])<=approved_holdouts:
            raise Invalid('communication contract omitted heldout jobs from the approved plan')
        if key not in groups: groups[key]={k:v for k,v in g.items() if k!='channels'} | dict(channels=[])
        if groups[key]['sample_role']!=g['sample_role']: raise Invalid('cannot mix historical and heldout samples')
        seen={r['channel_sha256'] for r in groups[key]['channels']}
        for row in g['channels']:
            if row['channel_sha256'] in seen: raise Invalid('duplicate channel cannot increase statistical sample size')
            seen.add(row['channel_sha256']);groups[key]['channels'].append(row)
    result=summarize(list(groups.values()),contract)
    if any(g['sample_role']!='holdout' for g in groups.values()):
        result['statistical_status']=result['status'];result['status']='DIAGNOSTIC_ONLY'
        for g in result['groups']:
            if g['sample_role']!='holdout':
                g['statistical_status']=g['status'];g['status']='DIAGNOSTIC_ONLY'
    # Detect source/runtime changes over the complete read/compute interval.
    for item in contract['cases']:
        path=repo/item['path']
        if sha(path)!=item['sha256']: raise Invalid('result changed during communication check')
        check_artifacts(path.parent,json.loads(path.read_text()))
    load_runtime(runtime_path,repo)
    if sha(runtime_path)!=ref['sha256']: raise Invalid('runtime changed during communication check')
    if own_sources!={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'verification/communication').glob('*.py'))}:
        raise Invalid('communication evaluator changed during check')
    return result | dict(schema='rayleigh-qpsk-quality-result-v1',contract_sha256=digest(contract),
                         policy_level='engineering_trial',se_status='UNSUPPORTED',score=None,
                         full_communication_certified=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--contract-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists(): raise Invalid('refusing existing communication directory')
    start=time.perf_counter();repo=Path(__file__).resolve().parents[2];contract=None
    try:
        if sha(a.contract)!=a.contract_sha256: raise Invalid('approved communication contract changed')
        contract=json.loads(a.contract.read_text());r=evaluate(contract,repo)
        if sha(a.contract)!=a.contract_sha256: raise Invalid('contract changed during evaluation')
    except (Invalid,KeyError,ValueError,TypeError,OSError,FloatingPointError) as exc:
        r=dict(status='UNSUPPORTED' if isinstance(exc,Unsupported) else 'PREREQUISITE_REJECTED',
               reason=str(exc),score=None,full_communication_certified=False,se_status='UNSUPPORTED')
    r.update(elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,
             approved_contract_sha256=a.contract_sha256,
             provenance=dict(python=platform.python_version(),numpy=np.__version__,
                             sources={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'verification/communication').glob('*.py'))}))
    a.output.mkdir(parents=True,exist_ok=False)
    if contract is not None: (a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('status','reason','se_status','elapsed_seconds')}))
    return 0 if r['status']=='QPSK_QUALITY_CHECKED' else 1


if __name__=='__main__': raise SystemExit(main())
