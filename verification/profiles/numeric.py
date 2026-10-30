"""Frozen numerical models. Conformance is not Ascend hardware calibration."""
from dataclasses import dataclass,asdict
import numpy as np
from verification.contracts.core import Invalid,Unsupported

@dataclass(frozen=True)
class NumericPolicy:
    name:str
    storage:str
    compute:str
    accumulation:str
    gemm_order:str
    rounding:str='RNE'
    subnormal:str='preserve'
    fusion:str='none'
    task_tolerance:float=1e-10
    quadrant_tolerance:float=1e-10
    def frozen(self):return {'schema':'numeric-policy-v1',**asdict(self)}
    @property
    def bytes(self):return 2 if self.storage=='float16' else 8
    @property
    def vector_bytes(self):return 4 if self.modeled else 8
    @property
    def modeled(self):return self.name=='modeled_fp16_fp32_v1'
    def store(self,value):
        a=np.asarray(value)
        if np.iscomplexobj(a) or not np.isfinite(a).all():raise Invalid('nonfinite/complex input to real precision policy')
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            q=a.astype(np.float16 if self.modeled else np.float64)
        if not np.isfinite(q).all():raise Invalid('NONFINITE_WRITEBACK: storage overflow')
        return q
    def calculate(self,code,values):
        a=[np.asarray(x,dtype=np.float32) for x in values]
        if not all(np.isfinite(x).all() for x in a):raise Invalid('NONFINITE_COMPUTE_INPUT')
        with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
            if code in ('LOAD','STORE','COPY'):return a[0].copy()
            if code=='GEMM':
                result=np.zeros((a[0].shape[0],a[1].shape[1]),np.float32)
                for k in range(a[0].shape[1]):
                    product=np.multiply(a[0][:,k:k+1],a[1][k:k+1,:],dtype=np.float32)
                    result=np.add(result,product,dtype=np.float32)
                return result
            if code=='SQRT':return np.sqrt(a[0],dtype=np.float32)
            return {'ADD':np.add,'SUB':np.subtract,'MUL':np.multiply,'DIV':np.divide}[code](*a,dtype=np.float32)

F64=NumericPolicy('structure_f64','float64','float64','float64','host_matmul')
F16=NumericPolicy('modeled_fp16_fp32_v1','float16','float32','float32','k_ascending_multiply_then_add',task_tolerance=.02,quadrant_tolerance=.02)

def policy_for(profile):
    if profile in ('structure_f64','structure_f64_real_block'):return F64
    if profile in ('modeled_fp16_fp32_v1','modeled_fp16_fp32_real_block_v1'):return F16
    raise Unsupported('unknown or uncalibrated precision policy: '+str(profile))

def bit_equal(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return False
    ah,bh=a.astype(np.float16),b.astype(np.float16)
    if not np.array_equal(a,ah.astype(np.float64)) or not np.array_equal(b,bh.astype(np.float64)):return False
    return bool(np.array_equal(ah.view(np.uint16),bh.view(np.uint16)))
