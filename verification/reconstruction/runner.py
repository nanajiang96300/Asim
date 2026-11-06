"""Common Asim-first evaluation for source-view operators and trusted references."""
import argparse
from collections import Counter
import csv
import json
import hashlib
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


def error(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise Invalid('reference shape/finite mismatch')
    return relative_error(np.stack((a.real,a.imag)),np.stack((b.real,b.imag)))


def evaluate(operator,task,iterations,seed,binary,config,cwd,directory,policy=None,baseline_identity=None):
    """Trusted catalog and references are outside the candidate modification scope."""
    directory=Path(directory).resolve()
    if directory.exists():raise Invalid('refusing existing directory')
    result={'status':'INVALID','numeric_status':'NOT_RUN','score':None,'G4':'NOT_CERTIFIED','precision':'structure_f64_real_block','operator':operator}
    try:
        task.validate();precision=policy_for(task.profile);result['precision']=task.profile;result['precision_policy']=precision.frozen()
        entry,parameters,models,reference=adapter(operator,task,iterations)
        if config['num_cores']!=24 or config['precision']!=precision.bytes or config['dram_req_size']!=64:raise Invalid('expected24core/request64/precision8')
        cube=config.get('ascend_cube_model',{})
        if not cube.get('enabled') or [cube.get(k) for k in ('cube_m','cube_n','cube_k')]!=[16,16,16]:raise Invalid('expected enabled Cube16x16x16 cost model')
        if any(rows*cols*precision.bytes%64 for rows,cols in ((2*task.nr,2*task.nt),(2*task.nr,2*task.nrhs),(2*task.nt,2*task.nt),(2*task.nt,2*task.nrhs))):raise Unsupported('runtime observer tail DMA semantics not implemented')
        data=fixture(task,seed)
        comparison=digest({'runtime_contract':RUNTIME_CONTRACT,'numeric_policy':precision.frozen(),'task':task.frozen(),'hardware_config':config,'inputs':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in data['inputs'].items()}})
        if policy is not None and baseline_identity!=comparison:raise Invalid('baseline task/input/hardware/measurement identity mismatch')
        result['comparison_identity']=comparison;result['runtime_contract']=RUNTIME_CONTRACT
        receipt=simulate(binary,config,models,entry['mode'],cwd,directory,max_cycles=1000000)
        result.update(asim=receipt,task=task.frozen(),task_hash=task.identity,algorithm=parameters,seed=seed)
        np.savez(directory/'inputs.npz',**data['inputs']);np.savez(directory/'channel.npz',H=data['H'],Y=data['Y'],symbols=data['symbols'])
        if receipt['status']!='COMPLETED':result['status']='SKIPPED_ASIM_FAILED'
        else:
            receipt_files={e['path'] for e in receipt['artifacts']}
            if 'rw_events.jsonl' not in receipt_files:raise Unsupported('Asim binary did not produce required Core observer stream')
            for e in receipt['artifacts']:
                path=directory/e['path']
                if not path.is_file() or path.stat().st_size!=e['bytes'] or file_hash(path)!=e['sha256']:raise Invalid('Asim artifact changed after process receipt: '+e['path'])
            records=[json.loads(line) for line in (directory/'instructions.jsonl').read_text().splitlines()]
            dispatch=[json.loads(line) for line in (directory/'dispatch.jsonl').read_text().splitlines()]
            if len(records)!=1:raise Invalid('one actual operation required')
            h,groups,bound,counts=validate_actual(records[0],dispatch,task,parameters,data['inputs'],config)
            events={};dma=Counter();mapping={d['batch']:d for d in dispatch}
            for i in records[0]['instructions']:
                core=mapping[i['batch']]['core']
                if i['opcode'] in (0,1):dma[(core,'Load' if i['opcode']==0 else 'Store')]+=len(i['src_addrs'])
                else:events[i['id']]=f"Core{core}_"+('Cube' if i['opcode'] in (3,4) else 'Scalar' if 17<=i['opcode']<=21 else 'Vector')
            # The common event checker validates unique compute IDs (including
            # Scalar), issue order and all DMA fragments before any numerics.
            with (directory/'trace.csv').open() as f:trace=list(csv.DictReader(f))
            check_events(trace,events,dma,receipt['finish_cycle'],mapping)
            stats=statistics(directory/'trace.csv',records[0]['instructions'],receipt['finish_cycle'],24,dispatch)
            (directory/'pipeline_statistics.json').write_text(json.dumps(stats,indent=2));write_detail(directory/'instruction_pipeline_detail.csv',stats)
            if stats['status']!='TRACE_VALID':raise Invalid('trace pipeline coverage failed')
            schedule=check_runtime(records[0],dispatch,directory/'rw_events.jsonl',directory/'trace.csv',config,receipt['finish_cycle'])
            (directory/'schedule.json').write_text(json.dumps(schedule,indent=2))
            result['schedule_status']=schedule['status']
            if schedule['status']!='MODEL_RW_CHECKED_CONSERVATIVE':raise Invalid('actual Core schedule rejected by declared conservative model')
            binding=digest({'task':task.frozen(),'algorithm':parameters,'record_sha256':file_hash(directory/'instructions.jsonl'),'seed':seed})
            receipt['program_hash']=binding
            screen=policy or ScreenPolicy(receipt['finish_cycle'],1,1)
            state=screen_asim(receipt,binding,screen)
            result.update(status=state,core_tile_counts=counts,policy=vars(screen),baseline_creation=policy is None)
            if state=='ELIGIBLE_FOR_NUMERICS':
                result['numeric_status']='FAILED';(directory/'numerics_started.json').write_text(json.dumps({'binding':binding}))
                alg=reference(data['H'],data['Y'],task.regularization,iterations)
                target=task_reference(data,task)
                alg_blocks=alg.get('output_block',np.stack([real_block(z) for z in alg['output']]))
                evidence={**alg.get('evidence',{}),**({'X0':alg['X0']} if 'X0' in alg else {})}
                np.savez(directory/'references.npz',algorithm=alg['output'],algorithm_block=alg_blocks,task=target,**evidence)
                rows=[];outputs=[]
                for b in range(task.batch_size):
                    r=execute_batch(h,groups[b],bound,b,alg_blocks[b],return_output=True)
                    value=r.pop('_output',None)
                    if value is not None:
                        complex_value=decode_checked(value,task.nt,task.nrhs,precision.quadrant_tolerance);
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
                    failed_batches=[r['batch'] for r in rows if r['status']!='STRUCTURE_MATCH' or r.get('task_error',1)>precision.task_tolerance],
                    structural_reference_passed=structural,task_reference_passed=quality)
    except (Invalid,ValueError,KeyError,TypeError,OSError,IndexError,FloatingPointError) as e:
        result.update(status='UNSUPPORTED' if isinstance(e,Unsupported) else 'INVALID',reason=str(e))
    directory.mkdir(parents=True,exist_ok=True)
    result['artifacts']=[{'path':str(p.relative_to(directory)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(directory.rglob('*')) if p.is_file()]
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('binary','config','cwd','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--profile',default='structure_f64_real_block',choices=['structure_f64_real_block','modeled_fp16_fp32_real_block_v1']);ap.add_argument('--operator',default='ns_corrected_v1');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--iterations',type=int,default=16)
    a=ap.parse_args();c=json.loads(a.config.read_text());c['precision']=policy_for(a.profile).bytes
    r=evaluate(a.operator,MMSETask(profile=a.profile),a.iterations,a.seed,a.binary,c,a.cwd,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('artifacts','batches','asim')},ensure_ascii=False));return 0 if r['status'] in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') else 1

if __name__=='__main__':raise SystemExit(main())
