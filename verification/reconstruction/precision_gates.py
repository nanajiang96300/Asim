"""Bounded actual-Asim numerical-model conformance, not hardware certification."""
import argparse,json,subprocess
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import numpy as np
from verification.contracts.mmse import MMSETask
from verification.reconstruction.runner import evaluate
from verification.reconstruction.binding import validate_actual
from verification.asim.runner import file_hash
from verification.contracts.core import Invalid


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('binary','config','cwd','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);config=json.loads(a.config.read_text());config['precision']=2
    task=MMSETask(profile='modeled_fp16_fp32_real_block_v1');cases=[]
    for seed in (42,123,456):
        r=evaluate('ns_corrected_v1',task,16,seed,a.binary,config,a.cwd,root/f'seed{seed}')
        cases.append({'case':f'seed{seed}','status':r['status'],'cycle':r.get('asim',{}).get('finish_cycle'),'batches':len(r.get('batches',[])),
            'bit_equal_batches':sum(x.get('precision_bit_equal') is True for x in r.get('batches',[])),
            'max_task_error':max((x.get('task_error',1) for x in r.get('batches',[])),default=None),
            'passed':r['status']=='MODELED_NUMERIC_CHECKED' and len(r['batches'])==96 and all(x.get('precision_bit_equal') is True for x in r['batches'])})
    c64=deepcopy(config);c64['precision']=8
    r=evaluate('ns_corrected_v1',replace(task,profile='structure_f64_real_block'),16,42,a.binary,c64,a.cwd,root/'f64_regression')
    cases.append({'case':'f64_regression','status':r['status'],'cycle':r.get('asim',{}).get('finish_cycle'),'passed':r['status']=='STRUCTURE_CHECKED'})
    r=evaluate('ns_corrected_v1',task,1,42,a.binary,config,a.cwd,root/'lowK')
    cases.append({'case':'lowK','status':r['status'],'passed':r['status']=='ALGORITHM_QUALITY_FAILED' and r['structural_reference_passed']})
    d=root/'seed42';record=json.loads((d/'instructions.jsonl').read_text());ds=[json.loads(x) for x in (d/'dispatch.jsonl').read_text().splitlines()]
    with np.load(d/'inputs.npz') as z:inputs=dict(z)
    bad=deepcopy(record);i=next(x for x in bad['instructions'] if x['opcode']==10);i['compute_size']=i['m']*i['n']*2
    try:validate_actual(bad,ds,task,record['header']['algorithm'],inputs,config)
    except Invalid as e:probe={'name':'storage_width_used_for_fp32_vector','status':'REJECTED','reason':str(e)}
    else:raise Invalid('Vector cost undercount accepted')
    source=Path(__file__).resolve().parents[2]
    result={'schema':'precision-model-gates-v1','status':'GATES_PASSED' if all(c['passed'] for c in cases) else 'GATES_FAILED','score':None,'hardware_calibration':'NOT_DONE','G4':'NOT_DONE','quality_suite':'NOT_DONE','cases':cases,'cost_probe':probe,
        'runner_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'binary_source_commit':subprocess.check_output(['git','-C',str(a.cwd),'rev-parse','HEAD'],text=True).strip(),
        'binary_sha256':file_hash(a.binary),'sources':[{'path':str(p.relative_to(source)),'sha256':file_hash(p)} for folder in ('src','verification','orchestrator') for p in sorted((source/folder).rglob('*')) if p.suffix in ('.py','.cc','.h','.json')],
        'artifacts':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(root.rglob('*')) if p.is_file()]}
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifacts')},ensure_ascii=False));return 0 if result['status']=='GATES_PASSED' else 1

if __name__=='__main__':raise SystemExit(main())
