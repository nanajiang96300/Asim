"""Actual source-view adapter; no operator-name dispatch or reference algorithm."""
import numpy as np
from verification.reconstruction.engine import ArrayMemory,ReplayFault,execute_views
from verification.profiles.numeric import policy_for

class Memory(ArrayMemory):
    def __init__(self,header,inputs):
        self.header=header;self.inputs=inputs;self.p=header['precision_bytes']
        if not header.get('capacities'):raise ReplayFault('CAPACITY_MISSING','missing configured bank capacities')
        self.regions={r['base']:r for r in header['regions']}
        if len(self.regions)!=len(header['regions']):raise ReplayFault('REGION_ALIAS','duplicate region base')
        ranges=[]
        for base,r in self.regions.items():
            if min(r['rows'],r['cols'])<=0:raise ReplayFault('VIEW_INVALID','invalid region shape')
            end=base+r['rows']*r['cols']*self.p
            if any(base<hi and lo<end for lo,hi in ranges):raise ReplayFault('REGION_ALIAS','overlapping physical local regions')
            ranges.append((base,end))
            bank=0x10000000 if base<0x20000000 else 0x20000000
            if base<bank or end>bank+0x10000000:raise ReplayFault('CAPACITY_ERROR','region crosses address bank')
            key='spad_bytes' if bank==0x10000000 else 'accum_bytes'
            for cap in header.get('capacities',[]):
                if end-bank>cap[key]:raise ReplayFault('CAPACITY_ERROR','region exceeds configured Asim bank capacity')
        precision=policy_for(header['profile']) if header.get('schema')=='source-views-program-v1' else None
        super().__init__(self.regions,np.float64 if precision else np.complex128,precision)
    def tensor_slice(self,v,tensors):
        count=v['rows']*v['cols'];address=v['base'];nbytes=count*self.p
        if v['row'] or v['col'] or v['transpose'] or v['conjugate']:raise ReplayFault('VIEW_INVALID','DMA must be contiguous view')
        for index,t in enumerate(tensors):
            if t['base']<=address and address+nbytes<=t['base']+t['bytes'] and (address-t['base'])%self.p==0:
                return index,(address-t['base'])//self.p,count
        raise ReplayFault('TENSOR_BOUNDS_ERROR','DMA view outside declared tensors')


def execute_batch(header,instructions,inputs,batch,expected,precision='structure_complex128',return_output=False,captures=None):
    if header.get('schema') not in ('legacy-source-views-v1','source-views-program-v1'):raise ReplayFault('SCHEMA_MISMATCH','missing source-view schema')
    expected=np.asarray(expected,dtype=np.complex128 if np.iscomplexobj(expected) else np.float64)
    observed={}
    def capture(index,mem):
        for name,v in (captures or {}).get(index,{}).items():
            try:observed[name]={'status':'CAPTURED','value':mem.read(v),'index':index}
            except (ReplayFault,KeyError,TypeError,ValueError) as e:observed[name]={'status':getattr(e,'code','CAPTURE_INVALID'),'reason':str(e),'index':index}
    def observations():return {'_observations':observed} if captures is not None else {}
    executed=0;cost_issues=[];last=None;output=None;stores=0
    try:
        mem=Memory(header,inputs)
        for index,i in enumerate(instructions):
            last={'index':index,'id':i['id'],'batch':batch,'opcode':i['opcode']}
            s=i.get('semantic')
            if s is None:raise ReplayFault('MISSING_SEMANTICS','unannotated actual Instruction')
            last['site']=s.get('site')
            if s.get('schema')!='source-views-v1':raise ReplayFault('SCHEMA_MISMATCH','unknown per-instruction semantic schema')
            if s['opcode']!=i['opcode']:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','arithmetic identity mismatch')
            if 'invalid_reason' in s:raise ReplayFault('SEMANTIC_AMBIGUITY',s['invalid_reason'])
            code=i['opcode'];src=s['src'];dst=s['dst']
            if code==25:
                if src or dst is not None:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','barrier views')
                capture(index,mem);executed+=1;continue
            if code in (0,1):
                if len(src)!=1:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','DMA arity')
                remote=src[0] if code==0 else dst;local=dst if code==0 else src[0]
                if any(local[k] for k in ('row','col','transpose','conjugate')):raise ReplayFault('DMA_VIEW_UNSUPPORTED','local offset or transform requires explicit physical DMA support')
                if local['base']!=i['dest_addr']:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','DMA local address')
                targets=header['inputs'] if code==0 else header['outputs']
                slot,offset,count=mem.tensor_slice(remote,targets)
                declared_slot=i['operand_id']-(100 if code==0 else 200)
                if slot!=declared_slot:raise ReplayFault('INPUT_BINDING_MISMATCH','DMA crossed into another tensor')
                tshape=targets[slot]['shape']
                if len(tshape)==3 and (offset//(tshape[1]*tshape[2])!=batch or (offset+count-1)//(tshape[1]*tshape[2])!=batch):
                    raise ReplayFault('BATCH_BINDING_MISMATCH','DMA crossed logical batch boundary')
                covered=set(i['src_addrs']);request=64
                for byte in range(remote['base'],remote['base']+count*mem.p,mem.p):
                    if not any(a<=byte<a+request for a in covered):raise ReplayFault('DMA_COVERAGE_ERROR','logical value not loaded/stored')
                if code==0:
                    value=inputs[slot].reshape(-1)[offset:offset+count].reshape(remote['rows'],remote['cols'])
                    mem.write(dst,value)
                else:
                    if slot!=0 or offset!=batch*expected.size or count!=expected.size:raise ReplayFault('OUTPUT_COVERAGE_UNSUPPORTED','STORE must cover precisely one complete task output')
                    stores+=1
                    if stores!=1:raise ReplayFault('OUTPUT_COVERAGE_UNSUPPORTED','multiple STORE reconstruction not implemented')
                    output=mem.read(src[0]);
                    if output.shape!=(dst['rows'],dst['cols']):raise ReplayFault('SHAPE_MISMATCH','STORE shape')
                capture(index,mem);executed+=1;continue
            if len(src)!=len(i['src_addrs']) or [v['base'] for v in src]!=i['src_addrs'] or dst['base']!=i['dest_addr']:
                raise ReplayFault('SEMANTIC_BINDING_MISMATCH','view/actual operand address mismatch')
            codes={8:'SOFTMAX',22:'GELU',3:'GEMM',4:'GEMM',10:'ADD',17:'ADD',18:'SUB',11:'MUL',19:'MUL',13:'DIV',20:'DIV',16:'SQRT',21:'SQRT'}
            if code not in codes:raise ReplayFault('UNKNOWN_OPCODE',str(code))
            if code in (3,4):
                shapes=[mem.read(v).shape for v in src]
                if [*shapes[0],shapes[1][1]]!=[i['m'],i['k'],i['n']]:raise ReplayFault('SHAPE_MISMATCH','GEMM dimensions')
            if code in (8,22):
                from verification.primitives.nn import validate_instruction
                validate_instruction(i,header['profile'])
            value=execute_views(mem,codes[code],src,dst,s.get('broadcast','none'))
            if 17<=code<=21 and value.size!=1:cost_issues.append({'index':index,'id':i['id'],'reason':'batched SCALAR fixed latency','elements':value.size})
            capture(index,mem);executed+=1
        if output is None:raise ReplayFault('MISSING_OUTPUT','no STORE executed')
        if output.shape!=expected.shape:raise ReplayFault('OUTPUT_SHAPE_MISMATCH',f'output {output.shape}, task {expected.shape}')
        # Stable complex error via real/imag concatenation after scaling.
        scale=max(1.,float(np.max(np.abs(output))),float(np.max(np.abs(expected))))
        error=float(np.linalg.norm(output/scale-expected/scale)/max(np.linalg.norm(expected/scale),1e-15/scale))
        if not np.isfinite(error):raise ReplayFault('NONFINITE_METRIC','relative error is not finite')
        return observations() | ({'_output':output} if return_output else {}) | {'batch':batch,'status':'STRUCTURE_MATCH' if error<=1e-10 else 'NUMERIC_FAILED','relative_error':error,'executed':executed,'total':len(instructions),'cost_issues':cost_issues}
    except (ReplayFault,FloatingPointError,ValueError) as e:
        return observations() | {'batch':batch,'status':getattr(e,'code','NUMERIC_FAILED'),'reason':str(e),'first_bad':last,'executed':executed,'total':len(instructions),'cost_issues':cost_issues}
