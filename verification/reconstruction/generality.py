"""Bounded cross-algorithm campaign. One simulator at a time, no optimization search."""
import argparse,copy,json,subprocess,time
from pathlib import Path
from verification.contracts.mmse import MMSETask
from verification.reconstruction.runner import evaluate
from verification.reconstruction.source_views import execute_batch
from verification.profiles.numeric import policy_for
from verification.asim.runner import file_hash
import numpy as np


def run(binary,config,cwd,output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    cases=[('chol_f64_42','cholesky_explicit_v1','structure_f64_real_block',42,{}),
           ('chol_f16_42','cholesky_explicit_v1','modeled_fp16_fp32_real_block_v1',42,{}),
           ('chol_f16_123','cholesky_explicit_v1','modeled_fp16_fp32_real_block_v1',123,{}),
           ('chol_shape2','cholesky_explicit_v1','structure_f64_real_block',42,dict(batch_size=48,nr=32,nt=8,nrhs=4)),
           ('ns_regression','ns_corrected_v1','modeled_fp16_fp32_real_block_v1',42,{})]
    rows=[];negative=[]
    for name,operator,profile,seed,dims in cases:
        c=dict(config);c['precision']=policy_for(profile).bytes;task=MMSETask(profile=profile,**dims)
        before=time.monotonic();r=evaluate(operator,task,16 if operator=='ns_corrected_v1' else 1,seed,binary,c,cwd,root/name)
        row={k:v for k,v in r.items() if k not in ('asim','artifacts','batches')};row.update(case=name,finish_cycle=r.get('asim',{}).get('finish_cycle'),command_wall_seconds=time.monotonic()-before)
        rows.append(row);print(name,r['status'],row['finish_cycle'],r.get('reason'),flush=True)
        if r['status'] not in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED'):continue
        if name in ('chol_f16_42','chol_shape2'):
            rec=json.loads((root/name/'instructions.jsonl').read_text());assert len(rec)==1
            rec=rec[0];ops=[i for i in rec['instructions'] if i['batch']==0]
            with np.load(root/name/'inputs.npz') as values:inputs=[values[t['name'].split('.')[-1]] for t in rec['header']['inputs']]
            with np.load(root/name/'references.npz') as ref:expected=ref['algorithm_block'][0]
            # Offline counterfactual of a real row division. It is not evidence
            # of running a faulty binary, and contributes no performance data.
            bad=copy.deepcopy(ops);i=next(i for i in bad if i['id'].endswith('forward1.divide'));i['semantic']['src'][1].update(row=0,col=0)
            result=execute_batch(rec['header'],bad,inputs,0,expected)
            negative.append(dict(case=name,mutation='offline_wrong_diagonal',status=result['status'],relative_error=result.get('relative_error'),actual_mutant_simulated=False))
            bad=copy.deepcopy(ops);bad[0]['opcode']=99;bad[0]['semantic']['opcode']=99
            result=execute_batch(rec['header'],bad,inputs,0,expected)
            negative.append(dict(case=name,mutation='offline_unknown_opcode',status=result['status'],actual_mutant_simulated=False))
    passed=all(r['status'] in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') for r in rows) and len(negative)==4 and all(n['status']!='STRUCTURE_MATCH' for n in negative)
    result=dict(schema='cross-algorithm-generality-v1',status='SUPPORTED_SCOPE_PASSED' if passed else 'GATES_FAILED',cases=rows,negative_cases=negative,score=None,
                binary_sha256=file_hash(binary),runner_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                limits=['two algorithm families only, not arbitrary algorithms','Rayleigh only, no CDL-B','modeled precision, not device certification','offline negatives are not simulated mutant performance'])
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('binary','config','cwd','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();r=run(a.binary,json.loads(a.config.read_text()),a.cwd,a.output);raise SystemExit(0 if r['status']=='SUPPORTED_SCOPE_PASSED' else 1)
