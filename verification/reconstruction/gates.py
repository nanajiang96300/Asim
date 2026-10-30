"""Bounded common-runner gates, actual NS and offline mutation diagnostics."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import numpy as np
from verification.asim.runner import file_hash
from verification.contracts.core import Invalid,ScreenPolicy
from verification.contracts.mmse import MMSETask
from verification.reconstruction.runner import evaluate
from verification.reconstruction.source_views import execute_batch
from verification.fixtures.rayleigh import real_block


def legacy_regression(source_root,output):
    frozen=source_root/'experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/run_manifest.json'
    manifest=json.loads(frozen.read_text());rows=[]
    for item in manifest['artifacts']:
        p=source_root/item['path']
        if p.stat().st_size!=item['bytes'] or file_hash(p)!=item['sha256']:raise Invalid('legacy frozen evidence hash mismatch')
    raw=Path(manifest['raw_root'])
    for p in sorted(raw.glob('*/reconstruction.json')):
        old=json.loads(p.read_text());d=p.parent;record=json.loads((d/'instructions.jsonl').read_text());g=defaultdict(list)
        for i in record['instructions']:g[i['batch']].append(i)
        with np.load(d/'inputs.npz') as inp:values=[inp[str(i)] for i in range(len(record['header']['inputs']))]
        with np.load(d/'task_reference.npz') as refs:expected=refs['xhat' if d.name=='block_richardson' else 'inverse']
        results=[execute_batch(record['header'],g[b],values,b,expected[b]) for b in range(96)]
        counts=dict(Counter(r['status'] for r in results));passed=counts==old['status_counts']
        rows.append({'operator':d.name,'status_counts':counts,'same_as_frozen':passed})
        (output/(d.name+'.json')).write_text(json.dumps(results,indent=2))
    if len(rows)!=7 or not all(r['same_as_frozen'] for r in rows):raise Invalid('shared replay changed legacy diagnostics')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('binary','config','cwd','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.config.read_text());config['precision']=8;task=MMSETask();rows=[]
    for seed in (42,123,456):
        r=evaluate('ns_corrected_v1',task,16,seed,a.binary,config,a.cwd,root/f'seed{seed}')
        rows.append({'case':f'seed{seed}','expected':'STRUCTURE_CHECKED','actual':r['status'],'passed':r['status']=='STRUCTURE_CHECKED','cycle':r.get('asim',{}).get('finish_cycle'),
                     'max_algorithm_error':max((b.get('algorithm_error',1) for b in r.get('batches',[])),default=None),'max_task_error':max((b.get('task_error',1) for b in r.get('batches',[])),default=None)})
    base=rows[0]['cycle'];baseline_identity=json.loads((root/'seed42/result.json').read_text())['comparison_identity']
    for name,k,policy,want in [('lowK',1,None,'ALGORITHM_QUALITY_FAILED'),('not_competitive',16,ScreenPolicy(base*.1,1,4),'SKIPPED_NOT_COMPETITIVE'),('implausible',16,ScreenPolicy(base*100,1,4),'QUARANTINED_IMPLAUSIBLE_SPEEDUP')]:
        r=evaluate('ns_corrected_v1',task,k,42,a.binary,config,a.cwd,root/name,policy,baseline_identity if policy else None)
        rows.append({'case':name,'expected':want,'actual':r['status'],'numeric_status':r['numeric_status'],'passed':r['status']==want and (r['numeric_status']=='NOT_RUN' if policy else r.get('structural_reference_passed') is True)})
    d=root/'seed42';record=json.loads((d/'instructions.jsonl').read_text());h=record['header'];ops=[i for i in record['instructions'] if i['batch']==0]
    with np.load(d/'inputs.npz') as inp:values=[inp[t['name'].split('.')[-1]] for t in h['inputs']]
    with np.load(d/'references.npz') as refs:expected=real_block(refs['algorithm'][0])
    mutations=[]
    for kind in ('wrong_arithmetic','missing_load','output_squared','cross_batch'):
        changed=deepcopy(ops)
        if kind=='wrong_arithmetic':
            i=next(i for i in changed if i['id'].endswith('iter0.negate'));i['opcode']=i['semantic']['opcode']=10
        elif kind=='missing_load':changed.pop(0)
        elif kind=='output_squared':
            i=next(i for i in changed if i['id'].endswith('.apply'));i['src_addrs'][1]=i['src_addrs'][0];i['semantic']['src'][1]=deepcopy(i['semantic']['src'][0])
        else:
            i=changed[0];stride=task.nr*task.nt*4*8;i['src_addrs']=[x+stride for x in i['src_addrs']];i['semantic']['src'][0]['base']+=stride
        r=execute_batch(h,changed,values,0,expected);mutations.append({'case':kind,'result':r,'passed':r['status']!='STRUCTURE_MATCH','scope':'OFFLINE_ACTUAL_RECORD_MUTATION; no simulated mutant performance claim'})
    legacy_dir=root/'legacy_replay';legacy_dir.mkdir();source_root=Path(__file__).resolve().parents[2]
    legacy=legacy_regression(source_root,legacy_dir)
    summary={'status':'GATES_PASSED' if all(r['passed'] for r in rows+mutations) else 'GATES_FAILED','score':None,'cases':rows,'offline_mutations':mutations,'legacy_shared_replay':legacy,'scope':'structure f64; no FP16/G4/communication/G6 certification',
      'runner_commit':subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip(),'binary_source_commit':subprocess.check_output(['git','-C',str(a.cwd),'rev-parse','HEAD'],text=True).strip(),
      'binary_sha256':file_hash(a.binary),'sources':[{'path':str(q.relative_to(source_root)),'sha256':file_hash(q)} for basepath in ('src','verification','orchestrator') for q in sorted((source_root/basepath).rglob('*')) if q.suffix in ('.py','.h','.cc','.json')],
      'artifacts':[{'path':str(q.relative_to(root)),'bytes':q.stat().st_size,'sha256':file_hash(q)} for q in sorted(root.rglob('*')) if q.is_file()]}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in summary.items() if k not in ('artifacts','sources')},ensure_ascii=False));return 0 if summary['status']=='GATES_PASSED' else 1

if __name__=='__main__':raise SystemExit(main())
