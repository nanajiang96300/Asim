"""Execute source-attached views on actual legacy Instructions, without FormulaLogger."""
from collections import Counter
import numpy as np
from verification.contracts.core import Invalid


class ReplayFault(Invalid):
    def __init__(self,code,message):super().__init__(message);self.code=code


def external_inputs(header,seed=42,regularization=.1):
    """Bind actual Model tensors. Never replace Model.H by the A the Op intended."""
    slots=header['inputs'];shape=slots[0]['shape']
    batch,m,u=shape
    rng=np.random.default_rng(seed)
    h=(rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2*m)
    a=h.conj().transpose(0,2,1)@h+regularization*np.eye(u)
    inverse=np.linalg.solve(a,np.broadcast_to(np.eye(u),a.shape))
    symbols=((2*rng.integers(0,2,(batch,u,u))-1)+1j*(2*rng.integers(0,2,(batch,u,u))-1))/np.sqrt(2)
    y=h@symbols+np.sqrt(regularization/2)*(rng.standard_normal((batch,m,u))+1j*rng.standard_normal((batch,m,u)))
    x0=np.stack([np.eye(u)/np.linalg.norm(x,2) for x in a])
    arrays=[]
    for t in slots:
        name=t['name'].split('.')[-1]
        values={'H':h,'RegI':regularization*np.eye(u),'X_init':x0,'C':2*np.eye(u),'Y':y}
        if name not in values:raise ReplayFault('INPUT_ROLE_UNKNOWN',name)
        value=np.asarray(values[name],dtype=np.complex128)
        if list(value.shape)!=t['shape']:raise ReplayFault('INPUT_SHAPE_MISMATCH',name)
        arrays.append(value)
    return arrays,{'H':h,'A':a,'inverse':inverse,'Y':y,'xhat':np.linalg.solve(a,h.conj().transpose(0,2,1)@y),'lambda':regularization}


class Memory:
    def __init__(self,header,inputs):
        self.header=header;self.inputs=inputs;self.p=header['precision_bytes']
        self.regions={r['base']:r for r in header['regions']}
        self.values={b:np.empty((r['rows'],r['cols']),dtype=np.complex128) for b,r in self.regions.items()}
        self.valid={b:np.zeros(v.shape,dtype=bool) for b,v in self.values.items()}
    def slice(self,v):
        base=v['base'];r=v['row'];c=v['col'];nr=v['rows'];nc=v['cols']
        if base not in self.regions or any(type(x) is not int for x in (r,c,nr,nc)) or min(r,c)<0 or min(nr,nc)<=0:
            raise ReplayFault('VIEW_INVALID','unknown region or invalid coordinates')
        shape=self.values[base].shape
        if r+nr>shape[0] or c+nc>shape[1]:raise ReplayFault('BOUNDS_ERROR',f'view exceeds region {self.regions[base]["name"]}: {v}')
        return base,(slice(r,r+nr),slice(c,c+nc))
    def read(self,v):
        base,sl=self.slice(v)
        if not self.valid[base][sl].all():raise ReplayFault('UNINITIALIZED_READ',self.regions[base]['name'])
        value=self.values[base][sl].copy()
        if v['conjugate']:value=value.conj()
        if v['transpose']:value=value.T
        return value
    def write(self,v,value):
        if v['transpose'] or v['conjugate']:raise ReplayFault('VIEW_INVALID','transformed destination')
        base,sl=self.slice(v);shape=self.values[base][sl].shape
        if value.shape!=shape:raise ReplayFault('SHAPE_MISMATCH',f'write {value.shape} into {shape}')
        if not np.isfinite(value).all():raise ReplayFault('NONFINITE_RESULT','nonfinite arithmetic')
        self.values[base][sl]=value;self.valid[base][sl]=True
    def tensor_slice(self,v,tensors):
        count=v['rows']*v['cols'];address=v['base'];nbytes=count*self.p
        if v['row'] or v['col'] or v['transpose'] or v['conjugate']:raise ReplayFault('VIEW_INVALID','DMA must be contiguous view')
        for index,t in enumerate(tensors):
            if t['base']<=address and address+nbytes<=t['base']+t['bytes'] and (address-t['base'])%self.p==0:
                return index,(address-t['base'])//self.p,count
        raise ReplayFault('TENSOR_BOUNDS_ERROR','DMA view outside declared tensors')


def execute_batch(header,instructions,inputs,batch,expected,precision='structure_complex128'):
    if header.get('schema')!='legacy-source-views-v1':raise ReplayFault('SCHEMA_MISMATCH','missing source-view schema')
    mem=Memory(header,inputs);executed=0;cost_issues=[];last=None;output=None
    try:
        for index,i in enumerate(instructions):
            last={'index':index,'id':i['id'],'batch':batch,'opcode':i['opcode']}
            s=i.get('semantic')
            if s is None:raise ReplayFault('MISSING_SEMANTICS','unannotated actual Instruction')
            last['site']=s.get('site')
            if s['opcode']!=i['opcode']:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','arithmetic identity mismatch')
            if 'invalid_reason' in s:raise ReplayFault('SEMANTIC_AMBIGUITY',s['invalid_reason'])
            code=i['opcode'];src=s['src'];dst=s['dst']
            if code==25:
                if src or dst is not None:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','barrier views')
                executed+=1;continue
            if code in (0,1):
                if len(src)!=1:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','DMA arity')
                remote=src[0] if code==0 else dst;local=dst if code==0 else src[0]
                if local['base']!=i['dest_addr']:raise ReplayFault('SEMANTIC_BINDING_MISMATCH','DMA local address')
                targets=header['inputs'] if code==0 else header['outputs']
                slot,offset,count=mem.tensor_slice(remote,targets)
                declared_slot=i['operand_id']-(100 if code==0 else 200)
                if slot!=declared_slot:raise ReplayFault('INPUT_BINDING_MISMATCH','DMA crossed into another tensor')
                tshape=targets[slot]['shape']
                if len(tshape)==3 and offset//(tshape[1]*tshape[2])!=batch:
                    raise ReplayFault('BATCH_BINDING_MISMATCH','DMA crossed logical batch boundary')
                covered=set(i['src_addrs']);request=64
                for byte in range(remote['base'],remote['base']+count*mem.p,mem.p):
                    if not any(a<=byte<a+request for a in covered):raise ReplayFault('DMA_COVERAGE_ERROR','logical value not loaded/stored')
                if code==0:
                    value=inputs[slot].reshape(-1)[offset:offset+count].reshape(remote['rows'],remote['cols'])
                    mem.write(dst,value)
                else:
                    output=mem.read(src[0]);
                    if output.shape!=(dst['rows'],dst['cols']):raise ReplayFault('SHAPE_MISMATCH','STORE shape')
                executed+=1;continue
            if len(src)!=len(i['src_addrs']) or [v['base'] for v in src]!=i['src_addrs'] or dst['base']!=i['dest_addr']:
                raise ReplayFault('SEMANTIC_BINDING_MISMATCH','view/actual operand address mismatch')
            values=[mem.read(v) for v in src]
            with np.errstate(divide='raise',invalid='raise',over='raise'):
                if code in (3,4):
                    if [*values[0].shape,values[1].shape[1]]!=[i['m'],i['k'],i['n']]:raise ReplayFault('SHAPE_MISMATCH','GEMM dimensions')
                    value=values[0]@values[1]
                elif code in (10,17):value=values[0]+values[1]
                elif code==18:value=values[0]-values[1]
                elif code in (11,19):value=values[0]*values[1]
                elif code in (13,20):value=values[0]/values[1]
                elif code in (16,21):value=np.sqrt(values[0])
                else:raise ReplayFault('UNKNOWN_OPCODE',str(code))
            if 17<=code<=21 and value.size!=1:cost_issues.append({'index':index,'id':i['id'],'reason':'batched SCALAR fixed latency','elements':value.size})
            mem.write(dst,value);executed+=1
        if output is None:raise ReplayFault('MISSING_OUTPUT','no STORE executed')
        # Stable complex error via real/imag concatenation after scaling.
        scale=max(1.,float(np.max(np.abs(output))),float(np.max(np.abs(expected))))
        error=float(np.linalg.norm(output/scale-expected/scale)/max(np.linalg.norm(expected/scale),1e-15/scale))
        if not np.isfinite(error):raise ReplayFault('NONFINITE_METRIC','relative error is not finite')
        return {'batch':batch,'status':'STRUCTURE_MATCH' if error<=1e-10 else 'NUMERIC_FAILED','relative_error':error,'executed':executed,'total':len(instructions),'cost_issues':cost_issues}
    except (ReplayFault,FloatingPointError,ValueError) as e:
        return {'batch':batch,'status':getattr(e,'code','NUMERIC_FAILED'),'reason':str(e),'first_bad':last,'executed':executed,'total':len(instructions),'cost_issues':cost_issues}
