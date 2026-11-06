"""Frozen LDL/BRI mapping campaign; serial Asim-first, no optimization search."""
import argparse,copy,json,subprocess
from pathlib import Path
import numpy as np
from verification.contracts.mmse import MMSETask
from verification.reconstruction.runner import evaluate
from verification.reconstruction.source_views import execute_batch
from verification.profiles.numeric import policy_for
from verification.asim.runner import file_hash


def fingerprints():
    root=Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)):file_hash(p) for p in sorted((root/'verification').rglob('*.py')) if 'tests' not in p.parts}


def run(binary,config,cwd,output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);sources=fingerprints()
    cases=[]
    for family in ('ldl','bri'):
        k=1 if family=='ldl' else 192
        cases.extend([(family+'_f64',family+'_explicit_v1','structure_f64_real_block',42,k,{}),
                      (family+'_f16_42',family+'_explicit_v1','modeled_fp16_fp32_real_block_v1',42,k,{}),
                      (family+'_f16_123',family+'_explicit_v1','modeled_fp16_fp32_real_block_v1',123,k,{}),
                      (family+'_shape2',family+'_explicit_v1','structure_f64_real_block',42,k,dict(batch_size=48,nr=32,nt=8,nrhs=4))])
    cases.extend([('bri_low_k','bri_explicit_v1','modeled_fp16_fp32_real_block_v1',42,1,dict(batch_size=24,nr=32,nt=8,nrhs=4)),
                  ('ns_regression','ns_corrected_v1','modeled_fp16_fp32_real_block_v1',42,16,{}),
                  ('chol_regression','cholesky_explicit_v1','modeled_fp16_fp32_real_block_v1',42,1,{})])
    rows=[];negatives=[]
    for name,op,profile,seed,k,dims in cases:
        c=dict(config);c['precision']=policy_for(profile).bytes;r=evaluate(op,MMSETask(profile=profile,**dims),k,seed,binary,c,cwd,root/name)
        expected='ALGORITHM_QUALITY_FAILED' if name=='bri_low_k' else 'MODELED_NUMERIC_CHECKED' if policy_for(profile).modeled else 'STRUCTURE_CHECKED'
        row={key:value for key,value in r.items() if key not in ('artifacts','asim','batches')};row.update(case=name,expected_status=expected,finish_cycle=r.get('asim',{}).get('finish_cycle'),accepted=r['status']==expected)
        if name=='bri_low_k':row['accepted']=row['accepted'] and r.get('structural_reference_passed') is True and r.get('task_reference_passed') is False
        rows.append(row);print(name,r['status'],row['finish_cycle'],r.get('reason'),flush=True)
        if name in ('ldl_f16_42','bri_f16_42') and row['accepted']:
            records=[json.loads(line) for line in (root/name/'instructions.jsonl').read_text().splitlines()];assert len(records)==1;rec=records[0]
            ops=copy.deepcopy([i for i in rec['instructions'] if i['batch']==0])
            with np.load(root/name/'inputs.npz') as values:inputs=[values[t['name'].split('.')[-1]] for t in rec['header']['inputs']]
            with np.load(root/name/'references.npz') as refs:target=refs['algorithm_block'][0]
            if name.startswith('ldl'):
                i=next(i for i in ops if i['id'].endswith('rowWeight1'));i['semantic']['src'][1]['col']=0;mutation='D_wrong_row_index'
            else:
                i=next(i for i in ops if i['id'].endswith('iteration191.update'));i['opcode']=11;i['semantic']['opcode']=11;mutation='iteration_ADD_replaced_with_MUL'
            bad=execute_batch(rec['header'],ops,inputs,0,target)
            negatives.append(dict(case=name,mutation=mutation,status=bad['status'],relative_error=bad.get('relative_error'),actual_mutant_simulated=False))
    if fingerprints()!=sources:raise ValueError('verification code changed during frozen campaign')
    accepted=all(r['accepted'] for r in rows) and len(negatives)==2 and all(n['status']=='NUMERIC_FAILED' for n in negatives)
    summary=dict(schema='solver-mapping-campaign-v1',status='SUPPORTED_MAPPING_GATES_PASSED' if accepted else 'GATES_FAILED',cases=rows,offline_negatives=negatives,score=None,
                 binary_sha256=file_hash(binary),runner_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes=sources,
                 limitations=['Rayleigh/declared model only','BRI real block size2/K192; low K intentionally fails quality','offline arithmetic negatives have no mutant-cycle claim','old implementations remain unchanged and unverified'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','config','cwd','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();r=run(a.binary,json.loads(a.config.read_text()),a.cwd,a.output);raise SystemExit(0 if r['status']=='SUPPORTED_MAPPING_GATES_PASSED' else 1)
