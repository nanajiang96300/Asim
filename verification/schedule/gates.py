"""Real Core event gates and supervised reconstruction repeatability checks."""
import argparse,csv,json,subprocess,gc
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from verification.asim.runner import simulate,file_hash
from verification.reconstruction.runner import evaluate
from verification.contracts.mmse import MMSETask
from verification.contracts.core import Invalid,Unsupported,ScreenPolicy
from verification.schedule.runtime import check_runtime
from verification.reconstruction.source_views import execute_batch


def data(path):
    record=json.loads((path/'instructions.jsonl').read_text())
    ds=[json.loads(s) for s in (path/'dispatch.jsonl').read_text().splitlines()]
    return record,ds


def probe(binary,config,cwd,path,scenario):
    receipt=simulate(binary,config,{'models':[{'name':'rw_probe','scenario':scenario}]},'rw_probe_test',cwd,path)
    if receipt['status']!='COMPLETED':raise Invalid('probe Asim did not complete')
    record,ds=data(path);r=check_runtime(record,ds,path/'rw_events.jsonl',path/'trace.csv',config,receipt['finish_cycle'])
    (path/'schedule.json').write_text(json.dumps(r,indent=2))
    value=None;overlap=0
    if r['status']=='MODEL_RW_CHECKED_CONSERVATIVE':
        A=np.arange(1,257,dtype=np.float64).reshape(16,16)/100;B=np.diag(np.linspace(2,3,16));expected=np.ones((16,16)) if scenario=='serialized' else A@B
        numeric=execute_batch(record['header'],record['instructions'],[A[None],B[None]],0,expected,return_output=True);value=numeric.pop('_output',None)
        (path/'numeric.json').write_text(json.dumps(numeric,indent=2));np.savez(path/'probe_inputs_reference.npz',A=A,B=B,expected=expected)
        if numeric['status']!='STRUCTURE_MATCH':raise Invalid('positive probe numeric reconstruction failed')
        spans={row['name']:(int(row['start_cycle']),int(row['end_cycle'])) for row in csv.DictReader(open(path/'trace.csv')) if row['name'] in ('cube','vector')}
        overlap=max(0,min(spans['cube'][1],spans['vector'][1])-max(spans['cube'][0],spans['vector'][0]))
    wanted='MODEL_RW_CHECKED_CONSERVATIVE' if scenario in ('serialized','independent') else 'MODEL_RW_REJECTED'
    return {'case':scenario,'status':r['status'],'cycle':receipt['finish_cycle'],'faults':[f for b in r['batches'] for f in b['faults']+b['epoch_faults']],'actual_cube_vector_overlap':overlap,
            'passed':r['status']==wanted and (scenario!='independent' or overlap>0)}


def corruption_probes(path,config,output):
    record,ds=data(path);finish=json.loads((path/'asim.json').read_text())['finish_cycle'];rows=[]
    original=[json.loads(x) for x in (path/'rw_events.jsonl').read_text().splitlines()];header=original[0]
    for name in ('missing_compute','duplicate_response','wrong_buffer','wrong_fragment','response_time','compute_issue_time','missing_reset','tail_payload'):
        es=deepcopy(original[1:]);rec=deepcopy(record)
        if name=='missing_compute':es.pop(next(n for n,e in enumerate(es) if e['kind']=='compute_write_visible'))
        elif name=='duplicate_response':es.append(deepcopy(next(e for e in es if e['kind']=='dma_response')))
        elif name=='wrong_buffer':next(e for e in es if e['kind']=='dma_response')['buffer']^=1
        elif name=='wrong_fragment':next(e for e in es if e['kind']=='dma_response')['fragment']=999999
        elif name=='response_time':
            next(e for e in es if e['kind']=='dma_response')['cycle']+=1;es.sort(key=lambda e:(e['cycle'],e['order']))
        elif name=='compute_issue_time':next(e for e in es if e['kind']=='compute_read_begin')['cycle']+=1;es.sort(key=lambda e:(e['cycle'],e['order']))
        elif name=='missing_reset':es.pop(next(n for n,e in enumerate(es) if e['kind']=='buffer_reset'))
        else:
            rec['instructions'][0]['semantic']['src'][0]['rows']=15;rec['instructions'][0]['semantic']['dst']['rows']=15
        for n,e in enumerate(es):e['order']=n
        dest=output/(name+'.jsonl');dest.write_text('\n'.join(json.dumps(e) for e in [header]+es)+'\n')
        try:check_runtime(rec,ds,dest,path/'trace.csv',config,finish)
        except (Invalid,Unsupported,ValueError,KeyError) as error:rows.append({'case':name,'status':'REJECTED','reason':str(error),'scope':'offline actual artifact corruption; not a simulated mutant'})
        else:raise Invalid('corrupt actual observer artifact accepted: '+name)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('binary','config','cwd','output'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);c=json.loads(a.config.read_text());cases=[]
    small=deepcopy(c);small.update(num_cores=1,precision=8);small['core_config']={'core_0':c['core_config']['core_0']}
    try:
        for scenario in ('serialized','independent','missing_barrier','overwrite_input'):cases.append(probe(a.binary,small,a.cwd,root/scenario,scenario))
        mut=root/'corruptions';mut.mkdir();corruptions=corruption_probes(root/'serialized',small,mut)
        task=MMSETask(profile='modeled_fp16_fp32_real_block_v1');low=deepcopy(c);low['precision']=2
        for name,profile,seed in [('ns16_a',task,42),('ns16_b',task,42),('ns16_seed123',task,123),('ns64',MMSETask(),42)]:
            cfg=low if profile==task else deepcopy(c);cfg['precision']=2 if profile==task else 8
            r=evaluate('ns_corrected_v1',profile,16,seed,a.binary,cfg,a.cwd,root/name)
            wanted='MODELED_NUMERIC_CHECKED' if profile==task else 'STRUCTURE_CHECKED'
            cases.append({'case':name,'status':r['status'],'schedule':r.get('schedule_status'),'cycle':r.get('asim',{}).get('finish_cycle'),'passed':r['status']==wanted and r.get('schedule_status')=='MODEL_RW_CHECKED_CONSERVATIVE'})
            gc.collect()
        one=json.loads((root/'ns16_a/result.json').read_text());two=json.loads((root/'ns16_b/result.json').read_text())
        deterministic={'cycle':one['asim']['finish_cycle']==two['asim']['finish_cycle'],'outputs':file_hash(root/'ns16_a/outputs.npy')==file_hash(root/'ns16_b/outputs.npy'),'observer':file_hash(root/'ns16_a/rw_events.jsonl')==file_hash(root/'ns16_b/rw_events.jsonl')}
        if not all(deterministic.values()):raise Invalid('same-input repeatability changed')
        # Real process followed by a trusted test-harness artifact fault, verifying
        # that the normal runner preserves failure and never starts numeric replay.
        def faulty(*args,**kwargs):
            receipt=simulate(*args,**kwargs);path=Path(args[5]);lines=(path/'rw_events.jsonl').read_text().splitlines();(path/'rw_events.jsonl').write_text('\n'.join(lines[:-1])+'\n');return receipt
        with patch('verification.reconstruction.runner.simulate',side_effect=faulty):
            r=evaluate('ns_corrected_v1',task,16,42,a.binary,low,a.cwd,root/'runner_corruption')
        cases.append({'case':'runner_corruption','status':r['status'],'numeric_status':r['numeric_status'],'passed':r['status']=='INVALID' and r['numeric_status']=='NOT_RUN' and not (root/'runner_corruption/numerics_started.json').exists(),'scope':'actual Asim then trusted postprocess fault injection'})
        r=evaluate('ns_corrected_v1',task,16,42,a.binary,low,a.cwd,root/'not_competitive',ScreenPolicy(one['asim']['finish_cycle']*.1,1,4),one['comparison_identity'])
        cases.append({'case':'not_competitive','status':r['status'],'numeric_status':r['numeric_status'],'passed':r['status']=='SKIPPED_NOT_COMPETITIVE' and r['numeric_status']=='NOT_RUN'})
        r=evaluate('ns_corrected_v1',MMSETask(nr=3,nt=2,nrhs=1,profile=task.profile),16,42,a.binary,low,a.cwd,root/'unsupported_tail')
        cases.append({'case':'unsupported_tail','status':r['status'],'passed':r['status']=='UNSUPPORTED' and 'asim' not in r})
        repo=Path(__file__).resolve().parents[2]
        result={'schema':'runtime-rw-gates-v1','status':'GATES_PASSED' if all(x['passed'] for x in cases) else 'GATES_FAILED','score':None,'hardware_G4':'NOT_CERTIFIED','deterministic_repeat':deterministic,'cases':cases,'corruptions':corruptions,
          'runner_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'binary_source_commit':subprocess.check_output(['git','-C',str(a.cwd),'rev-parse','HEAD'],text=True).strip(),'binary_sha256':file_hash(a.binary),
          'sources':[{'path':str(p.relative_to(repo)),'sha256':file_hash(p)} for base in ('src','verification') for p in sorted((repo/base).rglob('*')) if p.suffix in ('.h','.cc','.py')],
          'artifacts':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(root.rglob('*')) if p.is_file()]}
        (root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifacts')},ensure_ascii=False));return 0 if result['status']=='GATES_PASSED' else 1
    except (Invalid,ValueError,KeyError,TypeError,OSError,RuntimeError) as error:
        result={'schema':'runtime-rw-gates-v1','status':'GATES_FAILED','score':None,'reason':str(error),'cases':cases,
                'artifacts':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(root.rglob('*')) if p.is_file()]}
        (root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}));return 1

if __name__=='__main__':raise SystemExit(main())
