"""Common Asim-first evaluation for source-view operators and trusted references."""
import argparse
from collections import Counter
import csv
import json
import hashlib
from time import perf_counter
from pathlib import Path
import numpy as np
from verification.asim.runner import simulate,file_hash
from verification.asim.batch import check_events
from verification.contracts.core import Invalid,Unsupported,ScreenPolicy,screen_asim,digest
from verification.contracts.mmse import MMSETask,fixture,task_reference,decode_checked
from verification.fixtures.rayleigh import real_block
from verification.profiles.numeric import policy_for,bit_equal
from verification.schedule.runtime import check_runtime,HEADER as RUNTIME_CONTRACT
from verification.shadow.metrics import relative_error
from verification.operators.catalog import adapter
from verification.reconstruction.binding import validate_actual
from verification.reconstruction.source_views import execute_batch
from verification.reporting.pipeline import statistics,write_detail
from verification.reconstruction.stream import load_operation


def error(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise Invalid('reference shape/finite mismatch')
    return relative_error(np.stack((a.real,a.imag)),np.stack((b.real,b.imag)))


def compare_stage_captures(observed,names,evidence,batch):
    checks=[]
    for name in sorted(names):
        entry=observed.get(name,{});value=entry.get('value');expected=evidence.get(name)
        def fingerprint(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float16).tobytes()).hexdigest() if a is not None else None
        checks.append({'name':name,'index':entry.get('index'),'bit_equal':value is not None and expected is not None and bit_equal(value,expected[batch]),'actual_fp16_sha256':fingerprint(value),'expected_fp16_sha256':fingerprint(expected[batch] if expected is not None else None)})
    return checks


def evaluate(operator,task,iterations,seed,binary,config,cwd,directory,policy=None,baseline_identity=None,task_backend=None):
    """Trusted catalog and references are outside the candidate modification scope."""
    # Only trusted task integrations supply callbacks; candidate JSON cannot select code.
    backend=task_backend or {
        'adapter':adapter,'fixture':fixture,'task_reference':task_reference,
        'dma_shapes':lambda t:((2*t.nr,2*t.nt),(2*t.nr,2*t.nrhs),(2*t.nt,2*t.nt),(2*t.nt,2*t.nrhs)),
        'algorithm_reference':lambda fn,d,t,i:fn(d['H'],d['Y'],t.regularization,i),
        'blocks':lambda a:a.get('output_block',np.stack([real_block(z) for z in a['output']])),
        'decode':lambda v,t,p:decode_checked(v,t.nt,t.nrhs,p.quadrant_tolerance),
        'diagnostics':lambda d:('channel.npz',{k:d[k] for k in ('H','Y','symbols')})
    }
    directory=Path(directory).resolve()
    if directory.exists():raise Invalid('refusing existing directory')
    started=last_time=perf_counter();phase='setup';timings={}
    def mark(next_phase):
        nonlocal last_time,phase
        now=perf_counter();timings[phase]=timings.get(phase,0.)+now-last_time;last_time=now;phase=next_phase
    result={'status':'INVALID','numeric_status':'NOT_RUN','score':None,'G4':'NOT_CERTIFIED','precision':'structure_f64_real_block','operator':operator}
    try:
        task.validate();precision=policy_for(task.profile);result['precision']=task.profile;result['precision_policy']=precision.frozen()
        if backend.get('hardware_check'):backend['hardware_check'](config)
        entry,parameters,models,reference=backend['adapter'](operator,task,iterations)
        if config['num_cores']!=24 or config['precision']!=precision.bytes or config['dram_req_size']!=64:raise Invalid('expected24core/request64/precision8')
        cube=config.get('ascend_cube_model',{})
        if not cube.get('enabled') or [cube.get(k) for k in ('cube_m','cube_n','cube_k')]!=[16,16,16]:raise Invalid('expected enabled Cube16x16x16 cost model')
        if any(rows*cols*precision.bytes%64 for rows,cols in backend['dma_shapes'](task)):raise Unsupported('runtime observer tail DMA semantics not implemented')
        data=backend['fixture'](task,seed)
        comparison=digest({'runtime_contract':RUNTIME_CONTRACT,'numeric_policy':precision.frozen(),'task':task.frozen(),'hardware_config':config,'inputs':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in data['inputs'].items()}})
        if policy is not None and baseline_identity!=comparison:raise Invalid('baseline task/input/hardware/measurement identity mismatch')
        result['comparison_identity']=comparison;result['runtime_contract']=RUNTIME_CONTRACT
        mark('asim')
        receipt=simulate(binary,config,models,entry['mode'],cwd,directory,timeout=entry.get('asim_timeout_seconds',120),max_cycles=1000000)
        mark('artifact_and_schedule_checks')
        result.update(asim=receipt,task=task.frozen(),task_hash=task.identity,algorithm=parameters,seed=seed)
        np.savez(directory/'inputs.npz',**data['inputs']);diagnostic_name,diagnostics=backend['diagnostics'](data);np.savez(directory/diagnostic_name,**diagnostics)
        if receipt['status']!='COMPLETED':result['status']='SKIPPED_ASIM_FAILED'
        else:
            receipt_files={e['path'] for e in receipt['artifacts']}
            if 'rw_events.jsonl' not in receipt_files:raise Unsupported('Asim binary did not produce required Core observer stream')
            for e in receipt['artifacts']:
                path=directory/e['path']
                if not path.is_file() or path.stat().st_size!=e['bytes'] or file_hash(path)!=e['sha256']:raise Invalid('Asim artifact changed after process receipt: '+e['path'])
            records=[load_operation(directory/'instructions.jsonl')]
            dispatch=[json.loads(line) for line in (directory/'dispatch.jsonl').read_text().splitlines()]
            if len(records)!=1:raise Invalid('one actual operation required')
            h,groups,bound,counts=validate_actual(records[0],dispatch,task,parameters,data['inputs'],config)
            if backend.get('workload_check'):backend['workload_check'](groups)
            events={};dma=Counter();mapping={d['batch']:d for d in dispatch}
            for i in records[0]['instructions']:
                core=mapping[i['batch']]['core']
                if i['opcode'] in (0,1):dma[(core,'Load' if i['opcode']==0 else 'Store')]+=len(i['src_addrs'])
                else:events[i['id']]=f"Core{core}_"+('Cube' if i['opcode'] in (3,4) else 'Scalar' if 17<=i['opcode']<=21 else 'Vector')
            # The common event checker validates unique compute IDs (including
            # Scalar), issue order and all DMA fragments before any numerics.
            with (directory/'trace.csv').open() as f:check_events(csv.DictReader(f),events,dma,receipt['finish_cycle'],mapping)
            del events
            stats=statistics(directory/'trace.csv',records[0]['instructions'],receipt['finish_cycle'],24,dispatch)
            with (directory/'pipeline_statistics.json').open('w') as f:json.dump(stats,f,indent=2)
            write_detail(directory/'instruction_pipeline_detail.csv',stats)
            if stats['status']!='TRACE_VALID':raise Invalid('trace pipeline coverage failed')
            del stats
            schedule=check_runtime(records[0],dispatch,directory/'rw_events.jsonl',directory/'trace.csv',config,receipt['finish_cycle'],stream_limits=entry.get('observer_limits'))
            (directory/'schedule.json').write_text(json.dumps(schedule,indent=2))
            result['schedule_status']=schedule['status']
            if schedule['status']!='MODEL_RW_CHECKED_CONSERVATIVE':raise Invalid('actual Core schedule rejected by declared conservative model')
            # Large diagnostic dictionaries are no longer needed after the gates.
            # Keep the actual instructions/binding for numerical reconstruction.
            binding=digest({'task':task.frozen(),'algorithm':parameters,'record_sha256':file_hash(directory/'instructions.jsonl'),'seed':seed})
            receipt['program_hash']=binding
            screen=policy or ScreenPolicy(receipt['finish_cycle'],1,1)
            state=screen_asim(receipt,binding,screen)
            result.update(status=state,core_tile_counts=counts,policy=vars(screen),baseline_creation=policy is None)
            if state=='ELIGIBLE_FOR_NUMERICS':
                result['numeric_status']='FAILED';(directory/'numerics_started.json').write_text(json.dumps({'binding':binding}))
                mark('independent_references')
                alg=backend['algorithm_reference'](reference,data,task,iterations)
                target=backend['task_reference'](data,task)
                alg_blocks=backend['blocks'](alg)
                evidence={**alg.get('evidence',{}),**({'X0':alg['X0']} if 'X0' in alg else {})}
                np.savez(directory/'references.npz',algorithm=alg['output'],algorithm_block=alg_blocks,task=target,**evidence)
                mark('instruction_replay_and_comparison')
                rows=[];outputs=[]
                for b in range(task.batch_size):
                    capture_map=backend['captures'](groups[b]) if backend.get('captures') else None
                    r=execute_batch(h,groups[b],bound,b,alg_blocks[b],return_output=True,captures=capture_map)
                    if capture_map is not None:
                        observed=r.pop('_observations',{})
                        names={key for entries in capture_map.values() for key in entries}
                        checks=compare_stage_captures(observed,names,alg.get('evidence',{}),b)
                        if backend.get('inspect_stages'):
                            properties=backend['inspect_stages'](observed);r['stage_properties']=properties
                            if not properties['passed']:r['status']='PRECISION_MISMATCH'
                        r['stage_checks']=checks
                        if not checks or not all(x['bit_equal'] for x in checks):r['status']='PRECISION_MISMATCH'

                    value=r.pop('_output',None)
                    if value is not None:
                        complex_value=backend['decode'](value,task,precision);
                        if precision.modeled:
                            r['precision_bit_equal']=bit_equal(value,alg_blocks[b])
                            if not r['precision_bit_equal']:r['status']='PRECISION_MISMATCH'
                        outputs.append(value)
                        r.update(algorithm_error=error(complex_value,alg['output'][b]),task_error=error(complex_value,target[b]),algorithm_task_error=error(alg['output'][b],target[b]))
                    rows.append(r)
                if len(outputs)==task.batch_size:np.save(directory/'outputs.npy',np.stack(outputs))
                structural=all(r['status']=='STRUCTURE_MATCH' and r.get('algorithm_error',1)<=1e-10 for r in rows)
                quality=structural and all(r['task_error']<=precision.task_tolerance for r in rows)
                result.update(numeric_status='EXECUTED',status=('MODELED_NUMERIC_CHECKED' if precision.modeled else 'STRUCTURE_CHECKED') if quality else 'ALGORITHM_QUALITY_FAILED' if structural else 'REPLAY_MISMATCH',batches=rows,
                    failed_batches=[r['batch'] for r in rows if r['status']!='STRUCTURE_MATCH' or r.get('algorithm_error',1)>1e-10 or r.get('task_error',1)>precision.task_tolerance],
                    structural_reference_passed=structural,task_reference_passed=quality)
    except (Invalid,ValueError,KeyError,TypeError,OSError,IndexError,FloatingPointError) as e:
        result.update(status='UNSUPPORTED' if isinstance(e,Unsupported) else 'INVALID',reason=str(e))
    mark('archive')
    result['phase_timings_seconds']=timings
    result['evaluation_wall_seconds_before_archive']=last_time-started
    directory.mkdir(parents=True,exist_ok=True)
    result['artifacts']=[{'path':str(p.relative_to(directory)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(directory.rglob('*')) if p.is_file()]
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('binary','config','cwd','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--profile',choices=['structure_f64_real_block','modeled_fp16_fp32_real_block_v1','modeled_fp16_fp32_v1','modeled_fp16_fp32_nn_v1']);ap.add_argument('--operator',default='ns_corrected_v1');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--iterations',type=int);ap.add_argument('--task',choices=['mmse','affine','nn'],default='mmse');ap.add_argument('--shape',type=int,nargs=3)
    ap.add_argument('--nn-kind',choices=['softmax','gelu','mlp','attention'],default='attention')
    a=ap.parse_args();backend=None
    if a.task=='nn':
        from verification.tasks.tiny_nn import NNTask,backend as nn_backend
        if a.profile is not None and a.profile!='modeled_fp16_fp32_nn_v1':ap.error('NN task requires its frozen nonlinear precision profile')
        l,d,f=a.shape or (32,32,64);task=NNTask(kind=a.nn_kind,length=l,hidden=d,ffn=f);backend=nn_backend(task);a.profile=task.profile;iterations=1 if a.iterations is None else a.iterations
    elif a.task=='affine':
        from verification.tasks.affine import AffineTask,BACKEND
        m,k,n=a.shape or (32,32,32);a.profile=a.profile or 'modeled_fp16_fp32_v1';task=AffineTask(m=m,k=k,n=n,profile=a.profile);backend=BACKEND;iterations=1 if a.iterations is None else a.iterations
    else:
        if a.shape is not None:ap.error('--shape is only valid for affine task')
        a.profile=a.profile or 'structure_f64_real_block';task=MMSETask(profile=a.profile);iterations=16 if a.iterations is None else a.iterations
    c=json.loads(a.config.read_text());c['precision']=policy_for(a.profile).bytes
    r=evaluate(a.operator,task,iterations,a.seed,a.binary,c,a.cwd,a.output,task_backend=backend)
    print(json.dumps({k:v for k,v in r.items() if k not in ('artifacts','batches','asim')},ensure_ascii=False));return 0 if r['status'] in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') else 1

if __name__=='__main__':raise SystemExit(main())
