"""Trusted real affine task, independent of the candidate C++ generator."""
from dataclasses import dataclass,asdict
from collections import Counter
import numpy as np
from verification.contracts.core import Invalid,Unsupported,digest
from verification.profiles.numeric import policy_for
from verification.reference.affine import simulate_affine

PARAMETERS={'id':'affine_gemm_v1','formula':'Z=half(half(A@B)+C)','gemm_order':'k_ascending_multiply_then_add'}
@dataclass(frozen=True)
class AffineTask:
    batch_size:int=96
    m:int=32
    k:int=32
    n:int=32
    profile:str='modeled_fp16_fp32_v1'
    def validate(self):
        if type(self.batch_size) is not int or self.batch_size!=96 or any(type(x) is not int or x<16 or x>64 or x%16 for x in (self.m,self.k,self.n)):raise Invalid('affine demo requires batch96 and 16-aligned dimensions16..64')
        if self.profile!='modeled_fp16_fp32_v1':raise Unsupported('affine demo FP16 profile only')
    @property
    def output_shape(self):return [self.batch_size,self.m,self.n]
    @property
    def identity(self):return digest(self.frozen())
    def frozen(self):
        self.validate();return {'schema':'affine-task-v1',**asdict(self),'scope':'a_b_c_to_affine_z_v1','representation':'real_row_major','measurement':{'includes':['input DMA','GEMM','FP16 temporary writeback','ADD','barriers','output DMA'],'excludes':['common input generation']}}

def fixture(task,seed):
    task.validate();g=np.random.default_rng(seed);B,M,K,N=task.batch_size,task.m,task.k,task.n
    values={'A':g.uniform(-.25,.25,(B,M,K)),'B':g.uniform(-.25,.25,(B,K,N)),'C':g.uniform(-.5,.5,(B,M,N))}
    # Deterministic exact-zero and signed-zero cases are part of the fixed suite.
    values['A'][0]=0;values['B'][1]=0;values['C'][2]=0;values['A'][3]=-0.;values['C'][3]=-0.
    return {'inputs':{k:policy_for(task.profile).store(v) for k,v in values.items()},'math_inputs':values}

def task_reference(data,task):
    v=data['math_inputs'];return v['A']@v['B']+v['C']

def adapter(operator,task,iterations):
    if operator!='affine_gemm_v1' or iterations!=1:raise Unsupported('unknown affine operator or iteration parameter')
    task.validate();model={'name':operator,'matrix_m':task.m,'matrix_k':task.k,'matrix_n':task.n,'batch_size':task.batch_size,'task_contract':task.frozen(),'precision_policy':policy_for(task.profile).frozen()}
    return {'mode':'affine_gemm_test'},dict(PARAMETERS),{'models':[model]},simulate_affine

def workload_check(groups):
    for seq in groups.values():
        c=Counter(i['opcode'] for i in seq)
        if c[0]!=3 or c[1]!=1 or c[3]+c[4]!=1 or c[10]!=1 or c[25]>8 or set(c)-{0,1,3,4,10,25}:raise Invalid('affine frozen workload requires three LOAD, one GEMM, one ADD, one STORE and bounded barriers')

BACKEND={'adapter':adapter,'fixture':fixture,'task_reference':task_reference,'dma_shapes':lambda t:((t.m,t.k),(t.k,t.n),(t.m,t.n),(t.m,t.n)),
         'algorithm_reference':lambda fn,d,t,i:fn(d['inputs'],t),'blocks':lambda a:a['output'],'decode':lambda v,t,p:v,
         'diagnostics':lambda d:('task_inputs.npz',d['math_inputs']),'workload_check':workload_check}
