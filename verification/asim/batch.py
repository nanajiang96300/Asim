"""Batched actual-instruction recovery and supervised Asim-first replay.

Static independent tiles; this is not asynchronous G4 certification.
"""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from verification.contracts.core import Invalid, Unsupported, digest, screen_asim, json_safe
from verification.asim.runner import recover, simulate, file_hash
from verification.shadow.program import validate
from verification.shadow.execute import _replay
from verification.shadow.metrics import relative_error


def execution_policy(batch_size, cores):
    if type(batch_size) is not int or not 1<=batch_size<=384 or type(cores) is not int or cores not in (1,24):
        raise Unsupported('prototype requires cores=1/24 and batch=1..384')
    if cores==24 and batch_size!=1 and batch_size%24:
        raise Unsupported('production batches must be a multiple of 24')
    return {'batch_size':batch_size,'num_cores':cores,'placement':'simple_round_robin_v1'}


def prefix(batch, batch_size):
    return '' if batch_size==1 else f'batch{batch}.'


def recover_batch(record, requested, dispatch, batch_size, cores):
    policy=execution_policy(batch_size,cores)
    validate(requested)
    if record.get('execution')!=policy:
        raise Invalid('actual execution policy mismatch')
    grouped={b:[] for b in range(batch_size)}
    for inst in record['instructions']:
        b=inst['batch']
        if type(b) is not int or b not in grouped:raise Invalid('unexpected batch identity')
        if inst.get('planned_core')!=b%cores:raise Invalid('planned core mismatch')
        grouped[b].append(inst)
    if len(dispatch)!=batch_size:raise Invalid('missing/extra dispatch tile')
    by_batch={}
    for d in dispatch:
        b=d['batch']
        if type(b) is not int or b not in grouped or b in by_batch:raise Invalid('duplicate/unexpected dispatch batch')
        if any(type(d[k]) is not int for k in ('core','spad_buffer','accum_buffer')) or d['core']!=(b+1)%cores or d['spad_buffer'] not in (0,1) or d['accum_buffer'] not in (0,1):
            raise Invalid('runtime core/buffer mismatch')
        if type(d['cycle']) is not int or d['cycle']<0:raise Invalid('invalid dispatch cycle')
        if d['ids']!=[i['id'] for i in grouped[b]]:raise Invalid('runtime instruction identities mismatch')
        by_batch[b]=d
    stride=sum(((r['shape'][0]*r['shape'][1]*8+63)//64)*64 for r in requested['regions'].values() if r['space']=='DRAM')
    actual=[];expected_events={};dma=Counter()
    for b in range(batch_size):
        if not grouped[b]:raise Invalid('missing batch operations')
        normalized=deepcopy(grouped[b]);tag=prefix(b,batch_size)
        for physical,i in zip(grouped[b],normalized):
            name=i['id']
            if not name.startswith(tag):raise Invalid('batch operation prefix mismatch')
            i['id']=name[len(tag):];i['batch']=0
            if i['opcode'] in (0,1):
                # DMA src_addrs identify physical DRAM requests, not local SRAM.
                i['src_addrs']=[address-b*stride for address in i['src_addrs']]
                dma[(by_batch[b]['core'],'Load' if i['opcode']==0 else 'Store')]+=len(i['src_addrs'])
            else:
                unit='Cube' if i['opcode']==3 else 'Vector'
                if name in expected_events:raise Invalid('duplicate global operation id')
                expected_events[name]=f"Core{by_batch[b]['core']}_{unit}"
        one=dict(record,instructions=normalized)
        actual.append(recover(one,requested))
    return actual,expected_events,dma,by_batch


def check_events(events, expected, dma, finish_cycle, dispatch=None):
    seen=[];transfers=Counter();issued={}
    for d in (dispatch or {}).values():
        if d["cycle"]>finish_cycle:raise Invalid("dispatch after simulation finish")
        for name in d["ids"]:
            if name in issued:raise Invalid("duplicate dispatched ID")
            issued[name]=d["cycle"]
    for e in events:
        start=float(e['start_cycle']);end=float(e['end_cycle'])
        if not math.isfinite(start) or not math.isfinite(end) or not 0<=start<end<=finish_cycle:
            raise Invalid('invalid completion interval')
        if re.fullmatch(r'Core\d+_(Cube|Vector)',e['unit']):
            if expected.get(e['name'])!=e['unit']:raise Invalid('unexpected compute/barrier completion')
            if dispatch is not None and (e['name'] not in issued or start<issued[e['name']]):raise Invalid('completion before tile dispatch')
            seen.append(e['name'])
        match=re.fullmatch(r'Core(\d+)_MTE([23])',e['unit'])
        if match:
            name='Load' if match[2]=='2' else 'Store'
            if e['name']!=name:raise Invalid('unexpected DMA event')
            transfers[(int(match[1]),name)]+=1
    if sorted(seen)!=sorted(expected):raise Invalid('missing/duplicate compute/barrier event')
    if transfers!=dma:raise Invalid('DMA fragment completion count mismatch')


def evaluate_batch(program, inputs, binary, config, cwd, run_dir, policy, expected, algorithm_expected):
    """Trusted caller supplies independent references and immutable evaluation policy.

    Batched arrays are persisted before simulation; no numeric operations execute
    until real Asim and all static bindings/performance screening have passed.
    """
    directory=Path(run_dir).resolve()
    if directory.exists():raise Invalid('refusing existing run directory')
    stage='NOT_RUN';result={'score':None,'G4':'NOT_RUN'}
    try:
        validate(program);policy.validate()
        if config.get('num_cores') not in (1,24) or config.get('precision')!=8 or config.get('dram_req_size')!=64 or config.get('scheduler')!='simple':
            raise Unsupported('batch prototype requires simple scheduler, precision8, request64, cores1/24')
        if set(inputs)!={op['src'][0] for op in program['ops'] if op['opcode']=='LOAD'}:
            raise Invalid('external input set mismatch')
        if any(op['opcode']=='COPY' for op in program['ops']) or program['ops'][-1]['opcode']!='STORE' or sum(op['opcode']=='STORE' for op in program['ops'])!=1:
            raise Unsupported('batch prototype requires supported arithmetic and one terminal STORE')
        arrays={k:np.asarray(v).copy() for k,v in inputs.items()}
        batches={a.shape[0] for a in arrays.values() if a.ndim==3}
        if len(batches)!=1 or any(a.ndim!=3 for a in arrays.values()):raise Invalid('batch input rank mismatch')
        batch_size=batches.pop();cores=config['num_cores'];ep=execution_policy(batch_size,cores)
        for k,a in arrays.items():
            if a.dtype!=np.float64 or np.iscomplexobj(a) or list(a.shape[1:])!=program['regions'][k]['shape'] or not np.isfinite(a).all():
                raise Invalid('batch input shape/dtype/finite mismatch')
        output_shape=(batch_size,*program['regions'][program['output']]['shape'])
        refs=[np.asarray(ref).copy() for ref in (expected,algorithm_expected)]
        if any(ref.shape!=output_shape or ref.dtype!=np.float64 or np.iscomplexobj(ref) or not np.isfinite(ref).all() for ref in refs):raise Invalid('independent reference mismatch')
        # Static validation/serialization only; candidate numeric replay starts later.
        receipt=simulate(binary,config,{'models':[{'name':'batched_semantic_fixture','program':program,'batch_size':batch_size}]},'semantic_program_test',cwd,directory,max_cycles=1000000)
        result['asim']=receipt
        binding={'program':digest(program),'execution':ep,'inputs':{k:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()} for k,a in arrays.items()}}
        receipt['program_hash']=digest(binding)
        np.savez(directory/'inputs.npz',**arrays)
        np.savez(directory/'references.npz',task=refs[0],algorithm=refs[1])
        if receipt['status']!='COMPLETED':result['status']='SKIPPED_ASIM_FAILED'
        else:
            records=[json.loads(s) for s in (directory/'instructions.jsonl').read_text().splitlines()]
            dispatch=[json.loads(s) for s in (directory/'dispatch.jsonl').read_text().splitlines()]
            if len(records)!=1:raise Invalid('unexpected export count')
            actual,events,dma,mapping=recover_batch(records[0],program,dispatch,batch_size,cores)
            with (directory/'trace.csv').open() as stream:check_events(list(csv.DictReader(stream)),events,dma,receipt['finish_cycle'],mapping)
            state=screen_asim(receipt,digest(binding),policy);result['status']=state
            result['batch_size']=batch_size;result['core_tile_counts']=dict(sorted(Counter(d['core'] for d in dispatch).items()))
            result['compute_barrier_completions']=len(events);result['dma_completions']=sum(dma.values())
            if state=='ELIGIBLE_FOR_NUMERICS':
                stage='FAILED';(directory/'numerics_started.json').write_text(json.dumps(binding))
                values=[];metrics=[]
                for b in range(batch_size):
                    replay=_replay(actual[b],{k:a[b] for k,a in arrays.items()})
                    value=np.asarray(replay['output']);values.append(value)
                    metrics.append({'batch':b,'task_error':relative_error(value,refs[0][b]),
                                    'algorithm_error':relative_error(value,refs[1][b])})
                output=np.stack(values);np.save(directory/'outputs.npy',output)
                passed=all(max(m['task_error'],m['algorithm_error'])<=1e-10 for m in metrics)
                stage='EXECUTED';result['failed_batches']=[m['batch'] for m in metrics if max(m['task_error'],m['algorithm_error'])>1e-10]
                result.update(status='STRUCTURE_CHECKED' if passed else 'NUMERIC_FAILED',batches=metrics,
                    maximum_task_error=max(m['task_error'] for m in metrics),maximum_algorithm_error=max(m['algorithm_error'] for m in metrics))
    except (Invalid,ValueError,TypeError,KeyError,OSError,IndexError) as error:
        result.update(status='UNSUPPORTED' if isinstance(error,Unsupported) else 'INVALID',reason=str(error))
    directory.mkdir(parents=True,exist_ok=True)
    result['numeric_status']=stage;result['policy']=json_safe(vars(policy))
    result['artifacts']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(directory.iterdir()) if p.is_file()]
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
