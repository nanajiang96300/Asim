"""Trusted physical workload checks for source-view programs, before numerics."""
from collections import Counter,defaultdict
from verification.contracts.core import Invalid,Unsupported
from verification.profiles.numeric import policy_for


def validate_actual(record,dispatch,task,parameters,inputs,config):
    h=record['header'];B=task.batch_size;p=config['precision'];req=config['dram_req_size']
    policy=policy_for(task.profile)
    if h.get('precision_policy',policy.frozen())!=policy.frozen():raise Invalid('numeric policy binding mismatch')
    if h.get('schema')!='source-views-program-v1' or h.get('task_contract')!=task.frozen() or h.get('profile')!=policy.name or h['precision_bytes']!=p or p!=policy.bytes or req!=64:
        raise Invalid('actual task/profile/measurement binding mismatch')
    if h.get('algorithm')!=parameters:raise Invalid('algorithm parameters differ from frozen candidate')
    if len(h.get('capacities',[]))!=24 or {c['core'] for c in h['capacities']}!=set(range(24)):raise Invalid('missing core capacity evidence')
    if len(h['inputs'])!=len(inputs):raise Invalid('external input count mismatch')
    bound=[]
    for t in h['inputs']:
        role=t['name'].split('.')[-1]
        if role not in inputs or list(inputs[role].shape)!=t['shape'] or t['bytes']!=inputs[role].size*p:raise Invalid('external input role/shape/size mismatch')
        bound.append(inputs[role])
    if len(set(t['name'].split('.')[-1] for t in h['inputs']))!=len(inputs):raise Invalid('duplicate input role')
    output_shape=task.output_shape if hasattr(task,'output_shape') else [B,2*task.nt,2*task.nrhs]
    output_elements=1
    for extent in output_shape:output_elements*=extent
    if len(h['outputs'])!=1 or h['outputs'][0]['shape']!=output_shape or h['outputs'][0]['bytes']!=output_elements*p:raise Invalid('output declaration mismatch')
    groups=defaultdict(list);seen=set()
    for i in record['instructions']:
        if i['batch'] not in range(B) or not i['id'] or i['id'] in seen:raise Invalid('unexpected/duplicate instruction identity')
        seen.add(i['id']);groups[i['batch']].append(i)
        s=i.get('semantic')
        if not s or s.get('schema')!='source-views-v1' or 'invalid_reason' in s or s['opcode']!=i['opcode']:raise Unsupported('missing/ambiguous actual source semantics')
        if i['opcode']==25:continue
        d=s['dst'];local=d if i['opcode']!=1 else s['src'][0];rows,cols=local['rows'],local['cols']
        if i['size']!=(rows*cols*p+req-1)//req:raise Invalid('allocation/transfer size not derived from view')
        code=i['opcode']
        if code in (0,1):
            if len(i['src_addrs'])!=i['size'] or i['base_addr']!=0:raise Invalid('DMA request count/base mismatch')
        elif code in (3,4):
            left,right=s['src'];lm,lk=(left['cols'],left['rows']) if left['transpose'] else (left['rows'],left['cols']);rk,rn=(right['cols'],right['rows']) if right['transpose'] else (right['rows'],right['cols'])
            if (lm,lk,rn)!=(rows,i['k'],cols) or lk!=rk or (i['m'],i['n'])!=(rows,cols) or i['compute_size']!=cols:raise Invalid('GEMM workload mismatch')
        elif code in (8,22):
            from verification.primitives.nn import validate_instruction
            validate_instruction(i,task.profile)
        elif 17<=code<=21:
            if rows*cols!=1 or i['compute_size']!=1 or any(v['rows']*v['cols']!=1 for v in s['src']):raise Invalid('batched SCALAR is not a scalar instruction')
        elif code in (10,11,13,16):
            if (i['m'],i['n'],i['compute_size'])!=(rows,cols,rows*cols*policy.vector_bytes):raise Invalid('vector workload mismatch')
        else:raise Unsupported('unimplemented raw opcode')
    if set(groups)!=set(range(B)) or len(dispatch)!=B:raise Invalid('missing batch instructions/dispatch')
    by_batch={d['batch']:d for d in dispatch}
    if set(by_batch)!=set(range(B)):raise Invalid('duplicate/missing dispatch batch')
    counts=Counter()
    for b,d in by_batch.items():
        if d['ids']!=[i['id'] for i in groups[b]] or d['core'] not in range(24) or d['spad_buffer'] not in (0,1) or d['accum_buffer'] not in (0,1):raise Invalid('dispatch/core/buffer binding mismatch')
        counts[d['core']]+=1
    if len(counts)!=24 or set(counts.values())!={B//24}:raise Invalid('24-core placement incomplete')
    return h,groups,bound,dict(counts)
