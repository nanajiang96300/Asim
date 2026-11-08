"""Replay I/Q response probes only after successful actual Asim reconstruction.

The original timing result stays immutable. Probe costs are offline validation
overhead, not production cycles. Every probe is checked against an independent
same-algorithm precision oracle; candidate responses always come from the trace.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from verification.communication.response import identify
from verification.communication.runner import load_case,check_artifacts
from verification.contracts.core import Invalid,Unsupported,digest
from verification.contracts.mmse import MMSETask,fixture,decode_checked
from verification.fixtures.rayleigh import real_block
from verification.operators.catalog import adapter
from verification.profiles.numeric import policy_for,bit_equal
from verification.reconstruction.binding import validate_actual
from verification.reconstruction.daily import load_runtime,sha
from verification.reconstruction.source_views import execute_batch

LIMITS={'max_prediction_error','max_complex_leakage','max_cross_rhs_leakage','max_zero_offset','max_rate_loss'}


def validate(c):
    if set(c)!={'schema','runtime','cases','batch_indices','probe_seed','max_replay_calls','limits'} or c['schema']!='actual-response-se-proxy-v1':
        raise Invalid('unknown/incomplete response probe contract')
    if set(c['limits'])!=LIMITS: raise Invalid('incomplete response limits')
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in c['limits'].values()):
        raise Invalid('finite nonnegative response limits required')
    if type(c['probe_seed']) is not int or c['probe_seed']<0 or type(c['max_replay_calls']) is not int or not 1<=c['max_replay_calls']<=2048:
        raise Invalid('invalid probe seed/budget')
    ids=c['batch_indices']
    if not isinstance(ids,list) or not 1<=len(ids)<=4 or any(type(b) is not int or b<0 for b in ids) or ids!=sorted(set(ids)):
        raise Invalid('freeze 1..4 distinct batch indices before observing responses')
    if not isinstance(c['cases'],list) or not 1<=len(c['cases'])<=16: raise Invalid('1..16 approved source cases')
    for item in [c['runtime'],*c['cases']]:
        keys={'path','sha256'} if item is c['runtime'] else {'path','sha256','role'}
        if set(item)!=keys or not isinstance(item['path'],str) or not isinstance(item['sha256'],str) or len(item['sha256'])!=64:
            raise Invalid('source path/SHA-256 missing')
        if 'role' in item and item['role'] not in ('holdout','historical_diagnostic'): raise Invalid('invalid source role')


class ActualResponse:
    def __init__(self,directory,result,task,batch):
        self.directory=directory;self.result=result;self.task=task;self.batch=batch;self.ledger=[];self.arrays={}
        records=[json.loads(line) for line in (directory/'instructions.jsonl').read_text().splitlines()]
        if len(records)!=1: raise Invalid('one actual operation required')
        self.record=records[0];dispatch=[json.loads(line) for line in (directory/'dispatch.jsonl').read_text().splitlines()]
        self.data=fixture(task,result['seed']);config=json.loads((directory/'config.json').read_text())
        self.header,self.groups,_,_=validate_actual(self.record,dispatch,task,result['algorithm'],self.data['inputs'],config)
        _,_,_,self.reference=adapter(result['operator'],task,result['algorithm'].get('iterations',1))
        self.policy=policy_for(task.profile)
        self.original=np.load(directory/'outputs.npy',allow_pickle=False)[batch]
        self.H=self.data['H'][batch]
        self.Y=self.data['Y'][batch]

    def __call__(self,label,y):
        t=self.task;b=self.batch
        if y.shape!=(t.nr,t.nrhs) or not np.isfinite(y).all(): raise Invalid('invalid probe Y')
        external={**self.data['inputs'],'Y':self.data['inputs']['Y'].copy()}
        external['Y'][b]=self.policy.store(real_block(y))
        inputs=[external[v['name'].split('.')[-1]] for v in self.header['inputs']]
        oracle=self.reference(self.H[None],y[None],t.regularization,self.result['algorithm'].get('iterations',1))
        expected=oracle.get('output_block')
        if expected is None: expected=np.stack([real_block(z) for z in oracle['output']])
        actual=execute_batch(self.header,self.groups[b],inputs,b,expected[0],return_output=True)
        value=actual.pop('_output',None)
        if actual['status']!='STRUCTURE_MATCH' or actual['executed']!=actual['total'] or actual['cost_issues'] or value is None or not bit_equal(value,expected[0]):
            raise Invalid('actual probe replay/independent reference mismatch: '+label+' '+str(actual))
        if label=='original' and not bit_equal(value,self.original): raise Invalid('normal payload no longer reproduces archived output')
        self.arrays['input_'+label]=np.asarray(y).copy();self.arrays['raw_'+label]=value.copy()
        self.ledger.append(dict(label=label,instructions_executed=actual['executed'],precision_bit_equal=True,
                                quantized_y_sha256=hashlib.sha256(external['Y'][b].tobytes()).hexdigest(),
                                oracle_output_sha256=hashlib.sha256(expected[0].astype(np.float64).tobytes()).hexdigest(),
                                output_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
        return decode_checked(value,t.nt,t.nrhs,self.policy.quadrant_tolerance)


def evaluate(c,repo,output):
    validate(c);repo=Path(repo).resolve();output=Path(output)
    own={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'verification/communication').glob('*.py'))}
    rp=repo/c['runtime']['path']
    if sha(rp)!=c['runtime']['sha256']: raise Invalid('approved runtime changed')
    runtime=load_runtime(rp,repo)
    if runtime['python_version']!=platform.python_version() or runtime['numpy_version']!=np.__version__:
        raise Invalid('probe runtime toolchain mismatch')
    estimates=[];total=0;seen=set()
    # Budget is checked before any numerical replay, even on later source cases.
    for item in c['cases']:
        path=repo/item['path']
        if sha(path)!=item['sha256']: raise Invalid('approved source changed')
        r=json.loads(path.read_text());t=MMSETask(**{f.name:r['task'][f.name] for f in fields(MMSETask)});t.validate()
        if any(b>=t.batch_size for b in c['batch_indices']): raise Invalid('probe batch outside actual task')
        total+=len(c['batch_indices'])*(2*math.ceil(t.nr/t.nrhs)+9)
        estimates.append((item,path,r,t))
        if str(path.resolve()) in seen: raise Invalid('duplicate source case')
        seen.add(str(path.resolve()))
    if total>c['max_replay_calls']: raise Invalid('probe replay budget exceeded')
    heldouts={str((repo/i['path']).resolve()) for i in c['cases'] if i['role']=='holdout'}
    rows=[];calls=0
    for index,(item,path,r,t) in enumerate(estimates):
        prerequisites=load_case(item,runtime,repo)
        if not set(prerequisites['required_holdout_paths'])<=heldouts: raise Invalid('probe omitted approved heldout case coverage')
        for b in c['batch_indices']:
            response=ActualResponse(path.parent,r,t,b)
            measurements,arrays=identify(response,response.H,response.Y,t.regularization,
                                         [c['probe_seed'],r['seed'],b],c['limits'])
            arrays.update(H=response.H,original_Y=response.Y,original_raw_output=response.original)
            artifact=output/f'case{index:03d}-batch{b:03d}.npz';np.savez(artifact,**arrays,**response.arrays)
            calls+=len(response.ledger)
            rows.append(dict(source_result=item['path'],source_role=item['role'],batch=b,operator=r['operator'],
                             algorithm=r['algorithm'],regularization=t.regularization,
                             original_finish_cycle=r['asim']['finish_cycle'],actual_instruction_sha256=sha(path.parent/'instructions.jsonl'),
                             measurements=measurements,probe_ledger=response.ledger,
                             artifact=dict(path=artifact.name,bytes=artifact.stat().st_size,sha256=sha(artifact))))
            print(json.dumps(dict(operator=r['operator'],batch=b,status=measurements['status'],calls=len(response.ledger))),flush=True)
        check_artifacts(path.parent,r)
    if calls!=total: raise Invalid('unexpected probe replay count')
    for item,path,r,t in estimates:
        if sha(path)!=item['sha256']: raise Invalid('source changed during probes')
        check_artifacts(path.parent,r)
    load_runtime(rp,repo)
    if sha(rp)!=c['runtime']['sha256'] or own!={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'verification/communication').glob('*.py'))}:
        raise Invalid('runtime/probe sources changed')
    passed=all(r['measurements']['status']=='SE_PROXY_SAMPLE_CHECKED' for r in rows)
    return dict(schema='actual-response-se-proxy-result-v1',status='SE_PROXY_SAMPLES_CHECKED' if passed else 'PROBE_GATE_FAILED',
                contract_sha256=digest(c),actual_replay_calls=calls,rows=rows,source_hashes=own,
                performance_claim=False,population_quality_certified=False,full_communication_certified=False,score=None,
                se_definition='Gaussian-input linear receiver proxy of identified W; FP16 nonlinear effects screened only on frozen probes')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contract',required=True,type=Path);ap.add_argument('--contract-sha256',required=True)
    ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if a.output.exists(): raise Invalid('refusing existing probe output directory')
    started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    try:
        if sha(a.contract)!=a.contract_sha256: raise Invalid('approved probe contract changed')
        c=json.loads(a.contract.read_text());r=evaluate(c,Path(__file__).resolve().parents[2],a.output)
        if sha(a.contract)!=a.contract_sha256: raise Invalid('contract changed during probes')
    except (Invalid,ValueError,TypeError,KeyError,OSError,IndexError,FloatingPointError) as exc:
        r=dict(status='UNSUPPORTED' if isinstance(exc,Unsupported) else 'PROBE_REJECTED',reason=str(exc),score=None,
               full_communication_certified=False,population_quality_certified=False)
    r.update(elapsed_seconds=time.perf_counter()-started,pure_model_thinking_seconds=None,
             approved_contract_sha256=a.contract_sha256,python=platform.python_version(),numpy=np.__version__)
    (a.output/'contract.json').write_bytes(a.contract.read_bytes())
    (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('status','reason','actual_replay_calls','elapsed_seconds')}))
    return 0 if r['status']=='SE_PROXY_SAMPLES_CHECKED' else 1


if __name__=='__main__':raise SystemExit(main())
