"""Bind actual Core observer events to actual source-view instructions.

Certificate applies to the declared conservative Asim model, not device timing.
"""
from bisect import bisect_right
from collections import defaultdict
import csv,json,re
from collections import Counter
from pathlib import Path
from verification.contracts.core import Invalid,Unsupported
from verification.schedule.hazards import Access
from verification.schedule.intervals import check_schedule_fast

HEADER={'kind':'header','schema':'core-rw-v4','producer':'asim_core_observer_v4','clock':'homogeneous_core_cycle_v1','compute_reads':'conservative_issue_to_complete_v1','dma_reads':'request_snapshot_v1','dma_writes':'fragment_response_commit_v1','dma_identity':'trace_request_op_fragment_v1','compute_identity':'trace_issue_commit_cycle_v1','dma_descriptor':'per_fragment_address_v1'}
FIELDS=('id','opcode','dest_addr','src_addrs','size','compute_size','m','k','n','base_addr','src_from_accum','operand_id')


def stamp(e,phase=0):return e['cycle'],e['order']*4+phase

def time_event(name,kind,e,phase=0):
    c,q=stamp(e,phase);return {'op':name,'kind':kind,'cycle':c,'order':q}


def iter_stream(path,finish,max_bytes=256*1024*1024,max_events=1000000):
    """Bounded streaming decoder; no list of full repeated descriptors."""
    if not Path(path).is_file():raise Unsupported('missing actual Core read/write event stream')
    if Path(path).stat().st_size>max_bytes:raise Unsupported('observer byte budget exceeded')
    with open(path) as f:
        first=json.loads(next(f,'{}'))
        if first!=HEADER:raise Unsupported('unknown Core observer semantics or clock')
        clocks={}
        for n,line in enumerate(f):
            if n>=max_events:raise Unsupported('observer event budget exceeded')
            e=json.loads(line)
            if not isinstance(e,dict) or type(e.get('order')) is not int or e.get('order')!=n or type(e.get('cycle')) is not int or not 0<=e['cycle']<=finish or type(e.get('core')) is not int or e['core']<0:raise Invalid('observer order/clock/identity invalid')
            if e['cycle']<clocks.get(e['core'],0):raise Invalid('nonmonotonic per-core observer clock')
            clocks[e['core']]=e['cycle'];yield e


def read_stream(path,finish):
    # Compatibility callers retain the original limits and eager error behavior.
    return list(iter_stream(path,finish))


def within_epoch(access,event,batch,resets,reset_by_key):
    if access.space=='DRAM':return True
    reset=resets[batch,access.space];moments=reset_by_key[access.core,access.space,access.buffer]
    at=bisect_right(moments,stamp(reset));next_reset=moments[at] if at<len(moments) else None
    return stamp(event)>stamp(reset) and (next_reset is None or stamp(event)<next_reset)


def check_runtime(record,dispatch,path,trace_path,config,finish,stream_limits=None):
    if record['header'].get('schema')!='source-views-program-v1':raise Unsupported('runtime checking requires explicit source-view program')
    if any(c['core_type']!='systolic_ws' for c in config['core_config'].values()):raise Unsupported('observer supports homogeneous SystolicWS only')
    h=record['header'];p=h['precision_bytes'];req=config['dram_req_size']
    if req!=64 or p!=config['precision']:raise Unsupported('unsupported observer request/profile size')
    groups=defaultdict(list);by_id={};mapping={d['batch']:d for d in dispatch}
    if any(type(d['core']) is not int or d['core'] not in range(config['num_cores']) or d['spad_buffer'] not in (0,1) or d['accum_buffer'] not in (0,1) for d in dispatch):raise Invalid('invalid dispatch resource identity')
    if len(mapping)!=len(dispatch):raise Invalid('duplicate dispatch batch')
    for i in record['instructions']:
        if not isinstance(i['id'],str) or not re.fullmatch(r'[A-Za-z0-9_.:#-]+',i['id']) or i['id'] in by_id or i['batch'] not in mapping:raise Unsupported('missing/duplicate actual instruction id')
        by_id[i['id']]=i;groups[i['batch']].append(i)
    if set(groups)!=set(mapping):raise Invalid('missing dispatched instructions')
    for b,d in mapping.items():
        if d['ids']!=[i['id'] for i in groups[b]]:raise Invalid('runtime dispatch order mismatch')
    regions={r['base']:r for r in h['regions']}
    if len(regions)!=len(h['regions']):raise Invalid('aliased region declarations')
    def span(v,d):
        r=regions.get(v['base'])
        if r is None or any(type(v[k]) is not int for k in ('row','col','rows','cols')) or min(v['row'],v['col'])<0 or min(v['rows'],v['cols'])<=0 or v['row']+v['rows']>r['rows'] or v['col']+v['cols']>r['cols']:raise Invalid('invalid runtime source view')
        space='ACCUM' if v['base']>=0x20000000 else 'SPAD';buffer=d['accum_buffer'] if space=='ACCUM' else d['spad_buffer']
        if v['cols']==r['cols']:
            lo=v['base']+v['row']*r['cols']*p;return [Access(space,d['core'],buffer,lo,lo+v['rows']*r['cols']*p)]
        return [Access(space,d['core'],buffer,v['base']+((v['row']+j)*r['cols']+v['col'])*p,v['base']+((v['row']+j)*r['cols']+v['col']+v['cols'])*p) for j in range(v['rows'])]
    limits=stream_limits or {};events=iter_stream(path,finish,**limits);event_count=0;compute=defaultdict(dict);requests={};responses={};resets={};reset_by_key=defaultdict(list)
    for e in events:
        event_count+=1
        kind=e['kind']
        if kind=='buffer_reset':
            b=e['batch'];space=e['space']
            if b not in mapping or space not in ('SPAD','ACCUM') or e['core']!=mapping[b]['core'] or e['layer']!=mapping[b]['layer'] or e['cycle']!=mapping[b]['cycle'] or e['buffer']!=mapping[b]['spad_buffer' if space=='SPAD' else 'accum_buffer'] or (b,space) in resets:raise Invalid('buffer reset/dispatch mismatch')
            resets[b,space]=e;reset_by_key[e['core'],space,e['buffer']].append(stamp(e));continue
        i=by_id.get(e.get('id'))
        if i is None:raise Invalid('observer event not in actual program')
        d=mapping[i['batch']]
        fields=tuple(k for k in FIELDS if k!='src_addrs') if i['opcode'] in (0,1) else FIELDS
        if i['opcode'] in (0,1) and 'src_addrs' in e:raise Unsupported('redundant/noncanonical DMA observer descriptor')
        if any(e.get(k)!=i[k] for k in fields) or (e['core'],e['spad_buffer'],e['accum_buffer'])!=(d['core'],d['spad_buffer'],d['accum_buffer']) or e['cycle']<d['cycle']:raise Invalid('actual observer instruction/core/buffer binding mismatch')
        if kind in ('compute_read_begin','compute_read_done','compute_write_visible'):
            if i['opcode'] in (0,1) or kind in compute[i['id']]:raise Invalid('duplicate/wrong compute event')
            if kind=='compute_read_begin' and e.get('semantic')!=i['semantic']:raise Invalid('runtime views differ from exported instruction')
            compute[i['id']][kind]={k:e[k] for k in ('core','cycle','order')}
        elif kind in ('dma_request','dma_response'):
            if i['opcode'] not in (0,1) or type(e['fragment']) is not int or not 0<=e['fragment']<len(i['src_addrs']):raise Invalid('invalid DMA owner/fragment')
            buf=d['accum_buffer'] if i['dest_addr']>=0x20000000 else d['spad_buffer']
            if (e['dram_addr'],e['local_addr'],e['bytes'],e['write'],e['buffer'])!=(i['src_addrs'][e['fragment']]+i['base_addr'],i['dest_addr'],req,i['opcode']==1,buf):raise Invalid('DMA payload address/size/buffer mismatch')
            key=e['request_id'];target=requests if kind=='dma_request' else responses
            if type(key) is not int or key<0 or key in target:raise Invalid('duplicate/invalid DMA request identity')
            target[key]=e
        else:raise Unsupported('unknown Core observer event kind')
    if len(resets)!=2*len(mapping):raise Unsupported('missing resets or unsupported shared-accumulation tile')
    if set(requests)!=set(responses):raise Invalid('missing DMA request/response')
    fragments={}
    for key,e in requests.items():
        r=responses[key]
        if any(r.get(k)!=e.get(k) for k in (*(k for k in FIELDS if k!='src_addrs'),'spad_buffer','accum_buffer','fragment','dram_addr','local_addr','bytes','buffer','write','core')) or stamp(r)<stamp(e):raise Invalid('DMA response changed owner/payload or preceded request')
        token=e['id'],e['fragment']
        if token in fragments:raise Invalid('duplicate owner DMA fragment')
        fragments[token]=e,r
    expected_compute={i['id'] for i in by_id.values() if i['opcode'] not in (0,1)}
    if set(compute)!=expected_compute or any(set(es)!={'compute_read_begin','compute_read_done','compute_write_visible'} for es in compute.values()):raise Invalid('compute observer coverage incomplete')
    expected_fragments={(i['id'],n) for i in by_id.values() if i['opcode'] in (0,1) for n in range(len(i['src_addrs']))}
    if set(fragments)!=expected_fragments:raise Invalid('DMA fragment coverage incomplete')
    trace={};dma_trace={}
    with open(trace_path) as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or [])!={'name','unit','start_cycle','end_cycle','request_id','op_id','fragment','issue_cycle','commit_cycle'}:raise Unsupported('missing DMA trace identity columns')
        for row in reader:
            start,end=int(row['start_cycle']),int(row['end_cycle'])
            if not 0<=start<=end<=finish:raise Invalid('invalid pipeline trace clock')
            match=re.fullmatch(r'Core(\d+)_(Cube|Vector|Scalar|MTE2|MTE3|Wait)',row['unit'])
            if not match or int(match[1]) not in range(config['num_cores']):raise Invalid('unknown pipeline resource')
            unit=match[2]
            if unit in ('MTE2','MTE3'):
                key=int(row['request_id'])
                if key in dma_trace:raise Invalid('duplicate DMA pipeline request identity')
                dma_trace[key]=(int(match[1]),unit=='MTE3',start,end,row['op_id'],int(row['fragment']))
                if row['name']!=('Store' if unit=='MTE3' else 'Load'):raise Invalid('wrong DMA pipeline kind')
            elif row['request_id'] or row['op_id'] or row['fragment']:raise Invalid('non-DMA event has transfer identity')
            elif unit!='Wait' and row['name'] not in expected_compute:raise Invalid('extra compute pipeline event')
            if row['name'] in expected_compute:
                if row['name'] in trace:raise Invalid('duplicate compute pipeline event')
                trace[row['name']]=row
    expected_dma={key:(e['core'],e['write'],e['cycle'],responses[key]['cycle'],e['id'],e['fragment']) for key,e in requests.items()}
    if dma_trace!=expected_dma:raise Invalid('DMA request/response lifetime or identity differs from pipeline trace')
    if set(trace)!=expected_compute:raise Invalid('pipeline/observer compute set mismatch')
    for name,es in compute.items():
        a,b,c=(es[k] for k in ('compute_read_begin','compute_read_done','compute_write_visible'))
        if not stamp(a)<=stamp(b)<stamp(c) or b['cycle']!=c['cycle'] or c['order']!=b['order']+1:raise Invalid('compute event order invalid')
        unit='Cube' if by_id[name]['opcode'] in (3,4) else 'Scalar' if 17<=by_id[name]['opcode']<=21 else 'Vector'
        row=trace[name]
        if int(row['issue_cycle'])!=a['cycle'] or int(row['commit_cycle'])!=b['cycle']:raise Invalid('compute issue/commit differs from pipeline trace')
        if row['unit']!=f"Core{a['core']}_{unit}" or not a['cycle']<=float(row['start_cycle'])<float(row['end_cycle'])<=b['cycle']:raise Invalid('pipeline interval outside observed read protection')
    tensors=h['inputs']+h['outputs']
    for n,t in enumerate(tensors):
        if t['bytes']<=0 or any(max(t['base'],o['base'])<min(t['base']+t['bytes'],o['base']+o['bytes']) for o in tensors[:n]):raise Unsupported('overlapping external tensor aliases')
    initial=[Access('DRAM',-1,-1,t['base'],t['base']+t['bytes']) for t in h['inputs']]
    for value in reset_by_key.values():value.sort()
    reports=[]
    for batch,instructions in groups.items():
        d=mapping[batch];ops=[];ev=[];epoch_faults=[]
        def append(name,reads,writes,rb,rd,wv):
            ops.append({'id':name,'reads':reads,'writes':writes})
            if reads:ev.extend([time_event(name,'read_begin',rb,0),time_event(name,'read_done',rd,1)])
            if writes:ev.append(time_event(name,'write_visible',wv,2))
            for a,e in [(x,rb) for x in reads+writes]+[(x,rd) for x in reads]+[(x,wv) for x in writes]:
                if a.space=='DRAM':continue
                if not within_epoch(a,e,batch,resets,reset_by_key):epoch_faults.append({'hazard':'BUFFER_EPOCH_CROSSING','op':name,'space':a.space})
        for i in instructions:
            name=i['id'];s=i['semantic']
            if not s or s.get('schema')!='source-views-v1' or s.get('opcode')!=i['opcode'] or 'invalid_reason' in s:raise Unsupported('missing/ambiguous arithmetic views')
            if i['opcode'] in (0,1):
                if len(s['src'])!=1 or i['size']!=len(i['src_addrs']) or i['base_addr']!=0:raise Invalid('DMA arity/count/base mismatch')
                load=i['opcode']==0;remote=s['src'][0] if load else s['dst'];local=s['dst'] if load else s['src'][0]
                if any(local[k] for k in ('row','col','transpose','conjugate')):raise Unsupported('DMA local offsets/transforms not implemented')
                if local['base']!=i['dest_addr'] or (local['rows'],local['cols'])!=(remote['rows'],remote['cols']) or any(remote[k] for k in ('row','col','transpose','conjugate')):raise Invalid('noncontiguous or mismatched DMA views')
                local_spans=span(local,d)
                if len(local_spans)!=1:raise Unsupported('DMA noncontiguous local view')
                lo,hi=remote['base'],remote['base']+remote['rows']*remote['cols']*p;targets=h['inputs'] if load else h['outputs'];slot=i['operand_id']-(100 if load else 200)
                if not 0<=slot<len(targets):raise Invalid('DMA operand slot invalid')
                t=targets[slot]
                if not t['base']<=lo<hi<=t['base']+t['bytes']:raise Invalid('DMA crossed tensor boundary')
                if len(t['shape'])!=3 and not load:raise Unsupported('unbatched output')
                if len(t['shape'])==3:
                    stride=t['shape'][1]*t['shape'][2]*p
                    if not t['base']+batch*stride<=lo<hi<=t['base']+(batch+1)*stride:raise Invalid('DMA crossed batch boundary')
                coverage=[]
                for n,addr in enumerate(i['src_addrs']):
                    rq,rp=fragments[name,n]
                    if addr%req or addr<lo or addr+req>hi:raise Unsupported('DMA tail/padding byte semantics not implemented')
                    left,right=addr,addr+req
                    if left>=right:raise Invalid('DMA request outside logical payload')
                    coverage.append((left,right));a=local_spans[0];offset=left-lo
                    loc=Access(a.space,a.core,a.buffer,a.lo+offset,a.lo+offset+right-left);dram=Access('DRAM',-1,-1,left,right)
                    append(name+f'#dma{n}',[dram] if load else [loc],[loc] if load else [dram],rq,rq,rp)
                end=lo
                for left,right in sorted(coverage):
                    if left!=end:raise Invalid('DMA logical gaps/overlaps')
                    end=right
                if end!=hi:raise Invalid('DMA missing logical bytes')
            else:
                es=compute[name];a,b,c=(es[k] for k in ('compute_read_begin','compute_read_done','compute_write_visible'))
                if i['opcode']==25:
                    if s['src'] or s['dst'] is not None or i['size']!=1:raise Invalid('unsupported barrier payload')
                    addr=i['dest_addr']
                    if not any(r['base']<=addr and addr+req<=r['base']+r['rows']*r['cols']*p for r in regions.values()):raise Invalid('auxiliary barrier outside declared memory')
                    space='ACCUM' if addr>=0x20000000 else 'SPAD';buf=d['accum_buffer'] if space=='ACCUM' else d['spad_buffer']
                    writes=[Access(space,d['core'],buf,addr,addr+req)];reads=[]
                else:
                    if [v['base'] for v in s['src']]!=i['src_addrs'] or s['dst']['base']!=i['dest_addr']:raise Invalid('compute physical view binding mismatch')
                    reads=[x for v in s['src'] for x in span(v,d)];writes=span(s['dst'],d)
                    if any(x.space=='ACCUM' for x in reads) and not i['src_from_accum']:raise Invalid('wrong SRAM source bank flag')
                append(name,reads,writes,a,b,c)
        r=check_schedule_fast(ops,ev,initial)
        r.update(batch=batch,fragment_operations=sum('#dma' in o['id'] for o in ops),epoch_faults=epoch_faults)
        if epoch_faults:r['status']='SCHEDULE_MODEL_REJECTED'
        reports.append(r)
    failed=[r['batch'] for r in reports if r['status']!='SCHEDULE_MODEL_CHECKED']
    return {'status':'MODEL_RW_REJECTED' if failed else 'MODEL_RW_CHECKED_CONSERVATIVE','failed_batches':failed,'batches':reports,'event_count':event_count,'compute_operations':len(compute),'dma_fragments':len(fragments),'buffer_resets':len(resets),'scope':HEADER,'hardware_G4':'NOT_CERTIFIED'}
