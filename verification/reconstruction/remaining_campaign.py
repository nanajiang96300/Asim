"""Bounded blocked/merged mapping and actual factor-memory witnesses."""
import argparse,copy,json,gc,subprocess
from pathlib import Path
import numpy as np
from verification.reconstruction.runner import evaluate
from verification.reconstruction.source_views import execute_batch
from verification.reconstruction.engine import full_view
from verification.reconstruction.binding import validate_actual
from verification.contracts.mmse import MMSETask
from verification.contracts.core import Invalid
from verification.profiles.numeric import policy_for,bit_equal
from verification.asim.runner import file_hash

FAMILIES=('cholesky_block','ldl_block','cholesky_merge')
def fingerprints():
    root=Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)):file_hash(p) for p in sorted((root/'verification').rglob('*.py')) if 'tests' not in p.parts}

def run(binary,config,cwd,output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);before=fingerprints();cases=[]
    for family in FAMILIES:
        cases.extend([(family+'_f64',family+'_explicit_v1','structure_f64_real_block',42,1,{}),(family+'_f16_42',family+'_explicit_v1','modeled_fp16_fp32_real_block_v1',42,1,{}),(family+'_f16_123',family+'_explicit_v1','modeled_fp16_fp32_real_block_v1',123,1,{}),(family+'_shape2',family+'_explicit_v1','structure_f64_real_block',42,1,dict(batch_size=48,nr=32,nt=8,nrhs=4))])
    for op,k in [('ns_corrected_v1',16),('cholesky_explicit_v1',1),('ldl_explicit_v1',1),('bri_explicit_v1',192)]:cases.append((op+'_regression',op,'modeled_fp16_fp32_real_block_v1',42,k,{}))
    rows=[];stages=[];negative=[]
    for name,op,profile,seed,k,dims in cases:
        cfg=dict(config);cfg['precision']=policy_for(profile).bytes;task=MMSETask(profile=profile,**dims);dest=root/name
        r=evaluate(op,task,k,seed,binary,cfg,cwd,dest);ok=r['status'] in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED')
        rows.append(dict(case=name,status=r['status'],accepted=ok,operator=op,batch_size=task.batch_size,cycles=r.get('asim',{}).get('finish_cycle'),reason=r.get('reason')))
        print(name,r['status'],rows[-1]['cycles'],r.get('reason'),flush=True)
        if not ok or not (name.endswith('_f64') or name.endswith('_f16_42')):continue
        recs=[json.loads(line) for line in (dest/'instructions.jsonl').read_text().splitlines()]
        if len(recs)!=1:raise ValueError('one actual operation required')
        rec=recs[0];header=rec['header'];regions={x['name']:x for x in header['regions']};n=2*task.nt
        with np.load(dest/'inputs.npz') as f:inputs={key:f[key] for key in f.files}
        bound=[inputs[t['name'].split('.')[-1]] for t in header['inputs']]
        with np.load(dest/'references.npz') as f:refs={key:f[key] for key in f.files}
        by_batch={b:[] for b in range(task.batch_size)}
        for i in rec['instructions']:by_batch[i['batch']].append(i)
        witnesses=[];snapshots={}
        for b,ops in by_batch.items():
            index=next(i for i,x in enumerate(ops) if x['id'].endswith('forward0.solve'))-1
            capture={}
            for role in ('L','D') if op=='ldl_block_explicit_v1' else ('L',):
                reg=regions[role];capture[role]=full_view(reg['base'],(reg['rows'],reg['cols']))
            result=execute_batch(header,ops,bound,b,refs['algorithm_block'][b],captures={index:capture});observed=result.pop('_observations')
            checks={}
            for role,v in observed.items():
                expected=refs['factor_'+role+'_block'][b];actual=v.get('value');shape=(1,n) if role=='D' else (n,n)
                if v['status']!='CAPTURED' or actual.shape!=shape or expected.shape!=shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('invalid stage shape/finite or not reached')
                checks[role]=bit_equal(actual,expected) if policy_for(profile).modeled else bool(np.allclose(actual,expected,rtol=1e-10,atol=1e-12));snapshots[str(b)+'.'+role]=actual
            if set(observed)!=set(capture):raise ValueError('missing actual factor role')
            if 'D' in observed:checks['unit_diagonal']=bool(np.array_equal(np.diag(observed['L']['value']),np.ones(n)));checks['positive_D']=bool((observed['D']['value']>0).all())
            witnesses.append(dict(batch=b,passed=result['status']=='STRUCTURE_MATCH' and all(checks.values()),checks=checks))
        np.savez(dest/'factor_snapshots.npz',**snapshots);(dest/'factor_witnesses.json').write_text(json.dumps(witnesses,indent=2))
        stages.append(dict(case=name,batch_size=task.batch_size,passed=all(w['passed'] for w in witnesses)));print(name,'factor witnesses',stages[-1]['passed'],flush=True)
        if name=='cholesky_merge_f16_42':
            dispatch=[json.loads(line) for line in (dest/'dispatch.jsonl').read_text().splitlines()];bad=copy.deepcopy(rec)
            scalar=next(i for i in bad['instructions'] if 17<=i['opcode']<=21);scalar['compute_size']=16
            try:validate_actual(bad,dispatch,task,r['algorithm'],inputs,cfg)
            except Invalid as e:negative.append(dict(kind='scalar_width16_cost_admission',status='REJECTED',reason=str(e),actual_mutant_simulated=False))
            else:raise ValueError('illegal scalar width accepted')
            bad=copy.deepcopy(by_batch[0]);dot=next(i for i in bad if i['id'].endswith('factor2.diagonalDot'));dot['semantic']['src'][0]['cols']=1;dot['semantic']['src'][1]['cols']=1;dot['k']=1
            value=execute_batch(header,bad,bound,0,refs['algorithm_block'][0]);negative.append(dict(kind='dropped_reduction_term',status=value['status'],relative_error=value.get('relative_error'),actual_mutant_simulated=False))
        del rec,recs,by_batch,snapshots,refs;gc.collect()
    if fingerprints()!=before:raise ValueError('verification code changed during frozen campaign')
    passed=len(rows)==16 and all(r['accepted'] for r in rows) and len(stages)==6 and all(s['passed'] for s in stages) and len(negative)==2 and all(n['status'] in ('REJECTED','NUMERIC_FAILED') for n in negative)
    summary=dict(schema='remaining-mapping-campaign-v1',status='REMAINING_MAPPING_GATES_PASSED' if passed else 'GATES_FAILED',cases=rows,stage_cases=stages,negative_cases=negative,source_hashes=before,score=None,binary_sha256=file_hash(binary),runner_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (root/'summary.json').write_text(json.dumps(summary,indent=2));return passed

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('binary','config','cwd','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();raise SystemExit(0 if run(a.binary,json.loads(a.config.read_text()),a.cwd,a.output) else 1)
