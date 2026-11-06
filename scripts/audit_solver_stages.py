#!/usr/bin/env python3
"""Read-only actual-memory witnesses after completed LDL/BRI runs.

Stage references never write replay memory. This is an auditor, not a solver path.
"""
import argparse,json,gc
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from verification.asim.runner import file_hash
from verification.reconstruction.source_views import execute_batch
from verification.reconstruction.engine import full_view
from verification.profiles.numeric import bit_equal
from scripts.archive_checks import indexed,nested,verify_files,CASE_REQUIRED,PROCESS_REQUIRED


def require_array(value,shape,role):
    if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid stage shape/finite: '+role)


def audit(campaign,output):
    root=Path(campaign);out=Path(output);out.mkdir(parents=True,exist_ok=False);summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='SUPPORTED_MAPPING_GATES_PASSED':raise ValueError('main gates must pass first')
    cases=[]
    for row in summary['cases']:
        if not row['case'].startswith(('ldl_','bri_')) or row['case']=='bri_low_k':continue
        run=root/row['case'];r=json.loads((run/'result.json').read_text());index=indexed(r['artifacts'],CASE_REQUIRED)
        if r['status'] not in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') or r['failed_batches'] or r['asim']['status']!='COMPLETED' or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE':raise ValueError('missing passed evidence')
        verify_files(run,index);receipt=json.loads((run/'asim.json').read_text())
        if any(receipt[k]!=r['asim'][k] for k in ('status','returncode','binary_sha256','finish_cycle','mode')):raise ValueError('actual receipt mismatch')
        nested(index,'',receipt['artifacts'],PROCESS_REQUIRED)
        records=[json.loads(line) for line in (run/'instructions.jsonl').read_text().splitlines()];assert len(records)==1;rec=records[0];h=rec['header'];regions={v['name']:v for v in h['regions']}
        with np.load(run/'inputs.npz') as values:inputs=[values[t['name'].split('.')[-1]] for t in h['inputs']];input_map={k:values[k] for k in values.files}
        with np.load(run/'references.npz') as values:refs={k:values[k] for k in values.files}
        groups={b:[] for b in range(r['task']['batch_size'])}
        for i in rec['instructions']:groups[i['batch']].append(i)
        results=[];snapshots={};modeled=r['status']=='MODELED_NUMERIC_CHECKED';n=2*r['task']['nt'];rhs=2*r['task']['nrhs']
        require_array(refs['algorithm_block'],(r['task']['batch_size'],n,rhs),'algorithm block')
        for b,ops in groups.items():
            plan={}
            def capture(index,name,region):
                v=regions[region];plan.setdefault(index,{})[name]=full_view(v['base'],(v['rows'],v['cols']))
            if row['case'].startswith('ldl'):
                n=regions['L']['rows'];at=next(k for k,i in enumerate(ops) if i['id'].endswith('factor'+str(n-1)+'.D'));capture(at,'L','L');capture(at,'D','D')
            else:
                at=next(k for k,i in enumerate(ops) if i['id'].endswith('.damp'));capture(at,'B','B')
                at=next(k for k,i in enumerate(ops) if i['id'].endswith('.initializeX'));capture(at,'X0','X')
            replay=execute_batch(h,ops,inputs,b,refs['algorithm_block'][b],captures=plan);observed=replay.pop('_observations')
            if replay['status']!='STRUCTURE_MATCH' or any(v['status']!='CAPTURED' for v in observed.values()):raise ValueError('actual stage not reached')
            actual={k:v['value'] for k,v in observed.items()};checks={}
            roles={'L':(n,n),'D':(1,n)} if row['case'].startswith('ldl') else {'B':(n,n),'X0':(n,rhs)}
            if set(actual)!=set(roles):raise ValueError('missing stage role')
            for role,shape in roles.items():require_array(actual[role],shape,role)

            if 'L' in actual:
                l,d=actual['L'],actual['D'];checks['unit_diagonal']=bool(np.array_equal(np.diag(l),np.ones(l.shape[0])))
                checks['positive_D']=bool((d>0).all())
                if modeled:
                    require_array(refs['factor_L_block'][b],(n,n),'L reference');require_array(refs['factor_D_block'][b],(1,n),'D reference')
                    checks['L_bit_equal']=bit_equal(l,refs['factor_L_block'][b]);checks['D_bit_equal']=bit_equal(d,refs['factor_D_block'][b])
                else:
                    hreal=input_map['H'][b];a=hreal.T@hreal+input_map['RegI'];error=float(np.linalg.norm((l*d)@l.T-a)/np.linalg.norm(a));checks['factorization_error']=error;checks['factorization_match']=error<=1e-10
            else:
                checks['zero_initializer']=bool(np.count_nonzero(actual['X0'])==0)
                require_array(refs['preconditioner_block'][b],(n,n),'B reference')
                checks['B_match']=bit_equal(actual['B'],refs['preconditioner_block'][b]) if modeled else bool(np.allclose(actual['B'],refs['preconditioner_block'][b],rtol=1e-10,atol=1e-12))
            passed=all(v for k,v in checks.items() if k!='factorization_error')
            results.append(dict(batch=b,status='STAGE_WITNESSES_MATCH' if passed else 'STAGE_MISMATCH',checks=checks))
            for name,value in actual.items():snapshots[str(b)+'.'+name]=value
        dest=out/row['case'];dest.mkdir();np.savez(dest/'actual_stage_snapshots.npz',**snapshots);(dest/'batches.json').write_text(json.dumps(results,indent=2));cases.append(dict(case=row['case'],batch_size=len(results),passed=all(v['status']=='STAGE_WITNESSES_MATCH' for v in results)))
        print(row['case'],'stages',cases[-1]['passed'],flush=True);del rec,records,groups,snapshots;gc.collect()
    passed=len(cases)==8 and all(c['passed'] for c in cases);report=dict(schema='solver-stage-witness-v1',status='STAGE_WITNESSES_PASSED' if passed else 'STAGE_WITNESSES_FAILED',cases=cases,campaign_summary_sha256=file_hash(root/'summary.json'),scope='actual pre-existing instruction memory; no synthetic STORE or continuation after error')
    (out/'summary.json').write_text(json.dumps(report,indent=2));return passed

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--campaign',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();raise SystemExit(0 if audit(a.campaign,a.output) else 1)
