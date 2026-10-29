"""Supervised local trusted runner: real Asim first, then actual-object replay.

Not an OS security boundary. Never run a candidate-selected binary/config/runner.
"""
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from verification.contracts.core import Invalid, Unsupported, digest, screen_asim
from verification.shadow.program import validate
from verification.shadow.execute import _replay
from verification.shadow.metrics import relative_error


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(record, requested, request_bytes=64):
    """Recover only typed semantics embedded in actual C++ Instructions."""
    if record.get('format') != 'actual-instructions-v1' or record.get('header') is None:
        raise Unsupported('LEGACY_MISSING_NUMERIC_SEMANTICS: no executable header/views')
    if record['header'].get('schema')!='structure-program-v1':
        raise Unsupported('NUMERIC_SCHEMA_REQUIRES_SEPARATE_ADAPTER: '+str(record['header'].get('schema')))
    actual=deepcopy(record['header']); actual['ops']=[]
    if actual != {k:v for k,v in requested.items() if k!='ops'} | {'ops': []}:
        raise Invalid('actual program header mismatch')
    next_addr={'DRAM':0x10000,'SPAD':0x10000000,'ACCUM':0x20000000}
    addresses={}
    def count(name):
        m,n=actual['regions'][name]['shape']; return (m*n*8+request_bytes-1)//request_bytes
    for name,r in sorted(actual['regions'].items()):
        addresses[name]=next_addr[r['space']]; next_addr[r['space']]+=count(name)*request_bytes
    barriers=[]
    previous=None
    expecting_barrier=False
    for i in record['instructions']:
        if i['barrier']:
            if i['semantic'] is not None or i['opcode']!=25:
                raise Invalid('invalid auxiliary barrier')
            if not expecting_barrier or previous is None:
                raise Invalid('unexpected auxiliary barrier')
            expected_id='semantic_barrier_'+str(len(barriers))
            if (i['id'],i['dest_addr'],i['src_addrs'],i['size'],i['compute_size'],i['m'],i['k'],i['n'],i['base_addr'],i['batch'],i['src_from_accum']) != (expected_id,next_addr['SPAD'],[previous],1,0,0,0,0,0,0,True):
                raise Invalid('auxiliary barrier projection mismatch')
            next_addr['SPAD']+=request_bytes
            expecting_barrier=False
            barriers.append(i);continue
        if expecting_barrier:raise Invalid('missing auxiliary barrier')
        op=i['semantic']
        if op is None:
            raise Unsupported('untyped actual instruction')
        actual['ops'].append(op)
        code=op['opcode']; dst=op['dst']; src=op['src']
        m,n=actual['regions'][dst]['shape']; k=n
        if i['id']!=op['id'] or i['batch']!=0 or i['base_addr']!=0:
            raise Invalid('instruction identity mismatch')
        if code in ('LOAD','STORE'):
            local=dst if code=='LOAD' else src[0];dram=src[0] if code=='LOAD' else dst
            expect_src=[addresses[dram]+j*request_bytes for j in range(count(dram))]
            expect_dest=addresses[local];size=count(local);compute=0;opcode=0 if code=='LOAD' else 1
            accum=expect_dest>=0x20000000
        else:
            expect_src=[addresses[s] for s in src];expect_dest=addresses[dst];size=count(dst);accum=True
            opcode={'GEMM':3,'ADD':10,'SUB':10,'MUL':11}.get(code)
            if opcode is None:raise Unsupported('Asim primitive not supported')
            if code=='GEMM':k=actual['regions'][src[0]]['shape'][1];compute=n
            else:compute=m*n*8
        if (i['opcode'],i['dest_addr'],i['src_addrs'],i['size'],i['compute_size'],i['m'],i['k'],i['n'],i['src_from_accum']) != (opcode,expect_dest,expect_src,size,compute,m,k,n,accum):
            raise Invalid('actual semantic/timing projection mismatch at '+op['id'])
        previous=i['dest_addr'];expecting_barrier=code!='STORE'
    if expecting_barrier:raise Invalid('missing terminal barrier')
    validate(actual)
    if actual!=requested:raise Invalid('actual instruction stream differs from frozen request')
    return actual


def simulate(binary, config, models, mode, cwd, run_dir, timeout=120, max_cycles=100000):
    """Fresh artifact directory; export comes from this subprocess only."""
    run_dir=Path(run_dir).resolve();run_dir.mkdir(parents=True,exist_ok=False)
    binary=Path(binary).resolve();config=deepcopy(config)
    if config.get('dram_config_path'):
        config['dram_config_path']=str((Path(cwd)/config['dram_config_path']).resolve())
    (run_dir/'config.json').write_text(json.dumps(config,indent=2))
    (run_dir/'models.json').write_text(json.dumps(models,indent=2))
    env=dict(os.environ,ONNXIM_HOME=str(Path(cwd).parent),ONNXIM_MAX_CORE_CYCLES=str(max_cycles),
             ASIM_DISPATCH_EXPORT=str(run_dir/'dispatch.jsonl'),
             ASIM_INSTRUCTION_EXPORT=str(run_dir/'instructions.jsonl'),
             ONNXIM_TRACE_CSV=str(run_dir/'trace.csv'),ONNXIM_FORMULA_JSON=str(run_dir/'formula.json'))
    command=[str(binary),'--config',str(run_dir/'config.json'),'--models_list',str(run_dir/'models.json'),'--mode',mode,'--log_level','info']
    binary_before=file_hash(binary)
    with (run_dir/'simulator.log').open('w') as log:
        try:
            process=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            rc=process.returncode
        except subprocess.TimeoutExpired:rc=124
    if file_hash(binary)!=binary_before:raise Invalid('binary changed during run')
    text=(run_dir/'simulator.log').read_text()
    matches=re.findall(r'Simulation Finished at (\d+) cycle',text)
    complete=rc==0 and len(matches)==1 and 'Simulation aborted' not in text
    receipt={'status':'COMPLETED' if complete else 'FAILED','returncode':rc,
             'finish_cycle':int(matches[0]) if complete else None,'mode':mode,'command':command,
             'binary_sha256':binary_before,'max_core_cycles':max_cycles,'artifacts':[]}
    for name in ('config.json','models.json','simulator.log','instructions.jsonl','trace.csv','formula.json','dispatch.jsonl'):
        p=run_dir/name
        if p.is_file():receipt['artifacts'].append({'path':name,'bytes':p.stat().st_size,'sha256':file_hash(p)})
    (run_dir/'asim.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def _evaluate(program, inputs, binary, config, cwd, run_dir, policy, expected):
    validate(program)
    policy.validate()
    # Conservative prototype: no multiple outstanding STORE or duplicate LOAD allocation.
    if any(op['opcode']=='COPY' for op in program['ops']):raise Unsupported('COPY Asim mapping not implemented')
    if len(program['ops'])>256:raise Unsupported('prototype instruction budget')
    if program['ops'][-1]['opcode']!='STORE' or sum(op['opcode']=='STORE' for op in program['ops'])!=1:
        raise Unsupported('prototype requires one terminal STORE')
    if config.get('precision')!=8:raise Invalid('structure adapter requires explicit precision=8')
    if config.get('dram_req_size')!=64:raise Unsupported('prototype requires 64 byte requests')
    config=deepcopy(config)
    receipt=simulate(binary,config,{'models':[{'name':'semantic_fixture','program':program}]},'semantic_program_test',cwd,run_dir)
    run_dir=Path(run_dir)
    receipt['program_hash']=digest(program)
    result={'status':'INVALID','score':None,'numeric_status':'NOT_RUN','G4':'NOT_RUN','asim':receipt}
    if receipt['status']!='COMPLETED':result['status']='SKIPPED_ASIM_FAILED'
    else:
        records=[json.loads(s) for s in (run_dir/'instructions.jsonl').read_text().splitlines()]
        if len(records)!=1:raise Invalid('unexpected operation export count')
        actual=recover(records[0],program)
        with (run_dir/'trace.csv').open() as stream:events=list(csv.DictReader(stream))
        expected_events={op['id']:('Core0_Cube' if op['opcode']=='GEMM' else 'Core0_Vector') for op in actual['ops'] if op['opcode'] not in ('LOAD','STORE')}
        expected_events.update({i['id']:'Core0_Vector' for i in records[0]['instructions'] if i['barrier']})
        seen=[]
        for e in events:
            start=float(e['start_cycle']);end=float(e['end_cycle'])
            if not math.isfinite(start) or not math.isfinite(end) or not 0<=start<end<=receipt['finish_cycle']:
                raise Invalid('invalid completion event interval')
            if e['unit'] in ('Core0_Cube','Core0_Vector'):
                if expected_events.get(e['name'])!=e['unit']:raise Invalid('unexpected compute completion event')
                seen.append(e['name'])
        if sorted(seen)!=sorted(expected_events):raise Invalid('incomplete/duplicate compute/barrier events')
        state=screen_asim(receipt,digest(actual),policy)
        result['status']=state
        if state=='ELIGIBLE_FOR_NUMERICS':
            (run_dir/'numerics_started.json').write_text(json.dumps({'program_hash':digest(actual)}))
            replay=_replay(actual,inputs)
            observed=np.array(replay['output']);expected=np.asarray(expected)
            if observed.shape!=expected.shape or not np.isfinite(expected).all():raise Invalid('invalid independent expected output')
            error=relative_error(observed,expected)
            result.update(status='STRUCTURE_CHECKED' if error<=1e-12 else 'NUMERIC_FAILED',numeric_status='EXECUTED',relative_error=error,replay=replay)
        (run_dir/'actual_program.json').write_text(json.dumps(actual,indent=2))
    (run_dir/'inputs.json').write_text(json.dumps({k:np.asarray(v).tolist() for k,v in inputs.items()}))
    (run_dir/'expected.json').write_text(json.dumps(np.asarray(expected).tolist()))
    result['policy']=vars(policy)
    result['artifacts']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(run_dir.iterdir()) if p.is_file()]
    (run_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def evaluate(program, inputs, binary, config, cwd, run_dir, policy, expected):
    run_dir=Path(run_dir)
    if run_dir.exists():raise Invalid('refusing existing run directory')
    try:
        return _evaluate(program,inputs,binary,config,cwd,run_dir,policy,expected)
    except (Invalid,KeyError,ValueError,TypeError,OSError) as error:
        run_dir.mkdir(parents=True,exist_ok=True)
        result={'status':'UNSUPPORTED' if isinstance(error,Unsupported) else 'INVALID','reason':str(error),'numeric_status':'NOT_RUN','score':None,'G4':'NOT_RUN'}
        if (run_dir/'numerics_started.json').exists():result['numeric_status']='FAILED'
        result['artifacts']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(run_dir.iterdir()) if p.is_file()]
        (run_dir/'result.json').write_text(json.dumps(result,indent=2))
        return result
