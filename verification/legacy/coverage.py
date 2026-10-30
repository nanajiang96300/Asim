"""Exhaustive static source-view audit, including instructions after first fault.

No execution, mutation or inference of missing indices. Counts are not passes.
"""
from collections import Counter
from verification.reconstruction.source_views import Memory
from verification.reconstruction.engine import ReplayFault

CODES={3:'GEMM',4:'GEMM',10:'ADD',17:'ADD',18:'SUB',11:'MUL',19:'MUL',13:'DIV',20:'DIV',16:'SQRT',21:'SQRT'}

def audit(header,instructions,inputs,batch,expected_shape):
    mem=Memory(header,inputs);counts=Counter();issues=[]
    def issue(index,i,code,reason):
        counts[code]+=1
        issues.append({'index':index,'id':i['id'],'batch':batch,'code':code,'site':(i.get('semantic') or {}).get('site'),'reason':reason})
    for index,i in enumerate(instructions):
        counts['instructions']+=1;s=i.get('semantic')
        try:
            if not s:raise ReplayFault('MISSING_SEMANTICS','missing actual instruction views')
            if s.get('schema')!='source-views-v1' or s['opcode']!=i['opcode']:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','schema/opcode')
            if 'invalid_reason' in s:raise ReplayFault('SEMANTIC_AMBIGUITY',s['invalid_reason'])
            code=i['opcode'];src=s['src'];dst=s['dst'];broadcast=s.get('broadcast','none')
            if code==25:
                if src or dst is not None:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','barrier views')
            elif code in (0,1):
                if len(src)!=1:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','DMA arity')
                remote=src[0] if code==0 else dst;local=dst if code==0 else src[0]
                mem.slice(local)
                if any(local[k] for k in ('row','col','transpose','conjugate')):raise ReplayFault('DMA_VIEW_UNSUPPORTED','local offset/transform')
                if local['base']!=i['dest_addr']:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','DMA local base')
                tensors=header['inputs'] if code==0 else header['outputs']
                slot,offset,count=mem.tensor_slice(remote,tensors)
                if slot!=i['operand_id']-(100 if code==0 else 200):raise ReplayFault('INPUT_BINDING_MISMATCH','actual operand and physical tensor differ')
                shape=tensors[slot]['shape']
                if len(shape)==3 and (offset//(shape[1]*shape[2])!=batch or (offset+count-1)//(shape[1]*shape[2])!=batch):raise ReplayFault('BATCH_BINDING_MISMATCH','DMA batch offset')
                reqs=set(i['src_addrs'])
                if any(not any(a<=byte<a+64 for a in reqs) for byte in range(remote['base'],remote['base']+count*mem.p,mem.p)):raise ReplayFault('DMA_COVERAGE_ERROR','uncovered element')
                if (local['rows'],local['cols'])!=(remote['rows'],remote['cols']):raise ReplayFault('SHAPE_MISMATCH','DMA shape')
                if code==1 and (slot!=0 or offset!=batch*expected_shape[0]*expected_shape[1] or count!=expected_shape[0]*expected_shape[1]):raise ReplayFault('OUTPUT_COVERAGE_UNSUPPORTED','STORE not complete task batch')
            else:
                if code not in CODES:raise ReplayFault('UNKNOWN_OPCODE',str(code))
                if len(src)!=len(i['src_addrs']) or [v['base'] for v in src]!=i['src_addrs'] or dst['base']!=i['dest_addr']:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','physical operands')
                mem.slice(dst)
                for v in src:mem.slice(v)
                if dst['transpose'] or dst['conjugate']:raise ReplayFault('VIEW_INVALID','transformed destination')
                shapes=[(v['cols'],v['rows']) if v['transpose'] else (v['rows'],v['cols']) for v in src];want=(dst['rows'],dst['cols'])
                primitive=CODES[code];arity=1 if primitive=='SQRT' else 2
                if len(src)!=arity:raise ReplayFault('SHAPE_MISMATCH','primitive arity')
                if broadcast not in ('none','scalar_rhs'):raise ReplayFault('BROADCAST_UNSUPPORTED',str(broadcast))
                if broadcast=='scalar_rhs' and (primitive not in ('ADD','SUB','MUL','DIV') or shapes[1]!=(1,1)):raise ReplayFault('SHAPE_MISMATCH','scalar_rhs')
                if primitive=='GEMM':
                    if broadcast!='none' or shapes[0][1]!=shapes[1][0] or want!=(shapes[0][0],shapes[1][1]) or (i['m'],i['k'],i['n'])!=(shapes[0][0],shapes[0][1],shapes[1][1]):raise ReplayFault('SHAPE_MISMATCH','GEMM dimensions')
                elif shapes[0]!=want or (len(src)==2 and broadcast=='none' and shapes[0]!=shapes[1]):raise ReplayFault('SHAPE_MISMATCH','elementwise shape/implicit broadcast')
                if 17<=code<=21 and (want!=(1,1) or i['compute_size']!=1 or any(sh!=(1,1) for sh in shapes)):
                    issue(index,i,'SCALAR_WORKLOAD_UNSUPPORTED','fixed-latency scalar cannot price a vector or reduction')
            counts['explicit_views_valid']+=1
        except (ReplayFault,KeyError,TypeError,ValueError) as e:
            issue(index,i,getattr(e,'code','INVALID_DESCRIPTOR'),str(e))
    return {'schema':'legacy-coverage-v1','status':'DIAGNOSTIC_ONLY','counts':dict(counts),'issues':issues,'numeric_execution_claim':False}
