"""Algorithm-independent numeric primitives and initialized view memory.

Adapters bind actual instructions to these operations; this module has no solver.
"""
import numpy as np
from verification.contracts.core import Invalid, Unsupported

class ReplayFault(Invalid):
    def __init__(self,code,message):super().__init__(code+': '+message);self.code=code

class ArrayMemory:
    def __init__(self,regions,dtype=np.complex128,precision=None):
        self.precision=precision
        self.regions=regions
        self.values={b:np.empty((r['rows'],r['cols']),dtype=dtype) for b,r in regions.items()}
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
        if np.iscomplexobj(value) and self.values[base].dtype==np.float64:raise ReplayFault('PRECISION_UNSUPPORTED','complex values in real profile')
        if self.precision is not None:value=self.precision.store(value)
        self.values[base][sl]=value;self.valid[base][sl]=True

def full_view(base,shape):
    return dict(base=base,row=0,col=0,rows=shape[0],cols=shape[1],transpose=False,conjugate=False)

def arithmetic(code,values,broadcast='none',precision=None):
    arities={'LOAD':1,'STORE':1,'COPY':1,'SQRT':1,'SOFTMAX':1,'GELU':1,'GEMM':2,'ADD':2,'SUB':2,'MUL':2,'DIV':2}
    if code not in arities:raise Unsupported('unknown numeric primitive: '+str(code))
    if len(values)!=arities[code]:raise ReplayFault('SHAPE_MISMATCH','primitive arity')
    if broadcast not in ('none','scalar_rhs','row_rhs'):raise Unsupported('unknown broadcast semantics')
    if broadcast=='scalar_rhs' and (code not in ('ADD','SUB','MUL','DIV') or values[1].shape!=(1,1)):
        raise ReplayFault('SHAPE_MISMATCH','explicit scalar_rhs requires scalar right operand')
    if broadcast=='row_rhs' and (code not in ('ADD','MUL') or values[1].shape!=(1,values[0].shape[1])):raise ReplayFault('SHAPE_MISMATCH','row_rhs needs a matching row')
    if code in ('ADD','SUB','MUL','DIV') and broadcast=='none' and values[0].shape!=values[1].shape:
        raise ReplayFault('SHAPE_MISMATCH','implicit broadcasting forbidden')
    if code=='GEMM' and values[0].shape[1]!=values[1].shape[0]:raise ReplayFault('SHAPE_MISMATCH','GEMM inner dimension')
    if code in ('SOFTMAX','GELU') and (precision is None or not precision.modeled):raise Unsupported('NN explicit modeled precision required')
    if precision is not None and precision.modeled:return precision.calculate(code,values)
    with np.errstate(divide='raise',invalid='raise',over='raise'):
        if code in ('LOAD','STORE','COPY'):result=values[0].copy()
        elif code=='GEMM':
            if values[0].shape[1]!=values[1].shape[0]:raise ReplayFault('SHAPE_MISMATCH','GEMM inner dimension')
            result=values[0]@values[1]
        elif code=='ADD':result=values[0]+values[1]
        elif code=='SUB':result=values[0]-values[1]
        elif code=='MUL':result=values[0]*values[1]
        elif code=='DIV':result=values[0]/values[1]
        elif code=='SQRT':result=np.sqrt(values[0])
    if not np.isfinite(result).all():raise ReplayFault('NONFINITE_RESULT','nonfinite arithmetic')
    return result

def execute_views(memory,code,src,dst,broadcast='none'):
    result=arithmetic(code,[memory.read(v) for v in src],broadcast,memory.precision)
    memory.write(dst,result)
    return result
