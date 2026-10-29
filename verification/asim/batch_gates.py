"""24-core real-Asim structure gates for batched Rayleigh MMSE kernels.

Given A/b/X0/C. No claim of timed H/y preprocessing, calibrated FP16 or G4.
"""
import argparse
from collections import Counter,defaultdict
import csv
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import numpy as np
from verification.asim.batch import recover_batch,check_events,evaluate_batch
from verification.asim.runner import simulate,file_hash
from verification.contracts.core import ScreenPolicy,Invalid,digest
from verification.fixtures.rayleigh import mmse_batch
from verification.fixtures.solve_program import inverse_apply


def validate_run(directory,program,batch_size,cores,receipt):
    if receipt['status']!='COMPLETED':raise Invalid('Asim calibration did not complete')
    records=[json.loads(s) for s in (directory/'instructions.jsonl').read_text().splitlines()]
    ds=[json.loads(s) for s in (directory/'dispatch.jsonl').read_text().splitlines()]
    if len(records)!=1:raise Invalid('calibration operation count')
    _,expected,dma,mapping=recover_batch(records[0],program,ds,batch_size,cores)
    with (directory/'trace.csv').open() as stream:check_events(list(csv.DictReader(stream)),expected,dma,receipt['finish_cycle'],mapping)


def diagnostics(path,data):
    values=np.load(path/'outputs.npy');nt=data['metadata']['Nt'];nrhs=data['metadata']['Nrhs']
    residuals=[];candidate_errors=reference_errors=decision_mismatches=0
    for i,c in enumerate(data['channels']):
        x=values[i,:nt,:nrhs]+1j*values[i,nt:,:nrhs]
        residuals.append(float(np.linalg.norm(c['A']@x-c['b'])/(np.linalg.norm(c['A'],2)*np.linalg.norm(x)+np.linalg.norm(c['b'])+1e-15)))
        def errors(v):return int(np.count_nonzero((np.sign(v.real)!=np.sign(c['symbols'].real)) | (np.sign(v.imag)!=np.sign(c['symbols'].imag))))
        candidate_errors+=errors(x);reference_errors+=errors(c['x_reference'])
        decision_mismatches+=int(np.count_nonzero((np.sign(x.real)!=np.sign(c['x_reference'].real)) | (np.sign(x.imag)!=np.sign(c['x_reference'].imag))))
    changes=defaultdict(list)
    with (path/'trace.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['unit'].endswith('_Cube'):
                changes[float(row['start_cycle'])].append((row['unit'],1))
                changes[float(row['end_cycle'])].append((row['unit'],-1))
    active=Counter();peak=0
    for t in sorted(changes):
        for core,delta in changes[t]:active[core]+=delta
        peak=max(peak,sum(v>0 for v in active.values()))
    report={'max_normalized_residual':max(residuals),'residuals':residuals,'symbol_count':len(values)*nt*nrhs,
            'candidate_symbol_errors':candidate_errors,'reference_symbol_errors':reference_errors,
            'decision_parity':decision_mismatches==0,'decision_mismatches':decision_mismatches,'peak_cores_with_cube_instructions':peak,
            'scope':'small fixed-sample decision diagnostic; not formal communication acceptance or PE utilization'}
    (path/'diagnostics.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--cwd',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());config['precision']=8
    cube=config['ascend_cube_model']
    if config['num_cores']!=24 or not cube['enabled'] or [cube[k] for k in ('cube_m','cube_n','cube_k')]!=[16,16,16]:raise Invalid('expected Ascend-style24/core16cube configuration')
    program=inverse_apply();(root/'program.json').write_text(json.dumps(program,indent=2));cases=[];fixtures={}
    binary_hash=file_hash(args.binary)
    for batch_size,cores in [(24,24),(96,24),(96,1)]:
        key=f'b{batch_size}_c{cores}';c=deepcopy(config);c['num_cores']=cores
        if cores==1:c['core_config']={'core_0':c['core_config']['core_0']}
        data=fixtures.setdefault(batch_size,mmse_batch(20261029,batch_size=batch_size))
        calibration=root/(key+'_calibration')
        receipt=simulate(args.binary,c,{'models':[{'name':'control','program':program,'batch_size':batch_size}]},'semantic_program_test',args.cwd,calibration,max_cycles=1000000)
        validate_run(calibration,program,batch_size,cores,receipt)
        policy=ScreenPolicy(receipt['finish_cycle'],1,4)
        path=root/key
        r=evaluate_batch(program,data['inputs'],args.binary,c,args.cwd,path,policy,data['task'],data['algorithm'])
        row={'case':key,'batch_size':batch_size,'cores':cores,'status':r['status'],'numeric_status':r['numeric_status'],
             'cycle':r.get('asim',{}).get('finish_cycle'),'max_task_error':r.get('maximum_task_error'),'max_algorithm_error':r.get('maximum_algorithm_error'),
             'core_tile_counts':r.get('core_tile_counts'),'compute_barrier_completions':r.get('compute_barrier_completions'),'dma_completions':r.get('dma_completions')}
        if r['status']=='STRUCTURE_CHECKED':
            d=diagnostics(path,data);row['diagnostics']=d
            row['gate_passed']=d['max_normalized_residual']<=1e-10 and d['decision_parity'] and len(r['batches'])==batch_size and set(r['core_tile_counts'].values())=={batch_size//cores}
        else:row['gate_passed']=False
        cases.append(row)
    data=fixtures[96];base=next(r['cycle'] for r in cases if r['case']=='b96_c24')
    probes=[('not_competitive',data['inputs'],ScreenPolicy(base*.1,1,4),'SKIPPED_NOT_COMPETITIVE'),
            ('implausible',data['inputs'],ScreenPolicy(base*100,1,4),'QUARANTINED_IMPLAUSIBLE_SPEEDUP')]
    mixed={k:a.copy() for k,a in data['inputs'].items()};mixed['b'][-1]=mixed['b'][0]
    probes.append(('last_batch_input_mixing',mixed,ScreenPolicy(base,1,4),'NUMERIC_FAILED'))
    for name,inputs,policy,wanted in probes:
        r=evaluate_batch(program,inputs,args.binary,config,args.cwd,root/name,policy,data['task'],data['algorithm'])
        row={'case':name,'status':r['status'],'expected_status':wanted,'numeric_status':r['numeric_status'],'failed_batches':r.get('failed_batches'),'gate_passed':r['status']==wanted}
        if name=='last_batch_input_mixing':row['gate_passed'] &= r.get('failed_batches')==[95]
        else:row['gate_passed'] &= r['numeric_status']=='NOT_RUN'
        cases.append(row)
    fast=next(r['cycle'] for r in cases if r['case']=='b96_c24');serial=next(r['cycle'] for r in cases if r['case']=='b96_c1')
    if file_hash(args.binary)!=binary_hash:raise Invalid('binary changed across gate suite')
    source_root=Path(__file__).resolve().parents[2]
    sources=[{'path':str(p.relative_to(source_root)),'sha256':file_hash(p)} for basepath in ('src','verification') for p in sorted((source_root/basepath).rglob('*')) if p.suffix in ('.py','.cc','.h')]
    summary={'schema_version':'batch-gates-v1','status':'GATES_PASSED' if all(r['gate_passed'] for r in cases) else 'GATES_FAILED',
        'score':None,'G4':'NOT_RUN','fixture':data['metadata'],'precision_profile':'structure_f64; precision8 Asim cost uncalibrated',
        'timed_scope':data['metadata']['measurement_scope'],'excluded_costs':['H/y preparation','Gram and regularization','X0 spectral norm and initialization'],
        'same_input_batch96_model_cycle_speedup_24_vs_1':serial/fast,'cases':cases,'binary_sha256':binary_hash,
        'binary_source_commit':subprocess.check_output(['git','-C',str(args.cwd),'rev-parse','HEAD'],text=True).strip(),
        'runner_commit':subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip(),'sources':sources,
        'artifacts':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(root.rglob('*')) if p.is_file()]}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sources','artifacts')},ensure_ascii=False))
    return 0 if summary['status']=='GATES_PASSED' else 1

if __name__=='__main__':raise SystemExit(main())
