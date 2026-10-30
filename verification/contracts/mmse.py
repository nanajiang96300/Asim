"""Common external task and measured boundary; independent of candidate algorithm."""
from dataclasses import dataclass,asdict
import numpy as np
from verification.contracts.core import Invalid,Unsupported,digest
from verification.fixtures.rayleigh import real_block

SCOPES={
 'mmse_h_y_to_xhat_v1':{'inputs':['H','Y','RegI','Identity'],'includes':['input DMA','Gram','regularization','H^H Y','algorithm initialization','iterations/factorization','inverse application/output recovery','output DMA'],'excludes':['common channel generation','common real-block encoding/decoding']},
 'prepared_solver_kernel_v1':{'inputs':['A','b','X0','C'],'includes':['kernel input DMA','inverse/solve and apply','output DMA'],'excludes':['Gram','regularization','H^H Y','initialization preparation']}}

@dataclass(frozen=True)
class MMSETask:
    batch_size:int=96
    nr:int=64
    nt:int=16
    nrhs:int=8
    regularization:float=.1
    channel:str='rayleigh'
    scope:str='mmse_h_y_to_xhat_v1'
    profile:str='structure_f64_real_block'
    def validate(self):
        if self.channel!='rayleigh':raise Unsupported('CDL-B not implemented; no fallback')
        if self.scope not in SCOPES:raise Unsupported('unknown measurement scope')
        if self.profile!='structure_f64_real_block':raise Unsupported('hardware precision not implemented')
        if any(type(x) is not int or x<=0 for x in (self.batch_size,self.nr,self.nt,self.nrhs)) or self.batch_size%24 or self.batch_size>384 or self.nr<self.nt:
            raise Invalid('task requires batch multiple24<=384, Nr>=Nt>0, Nrhs>0')
        if not np.isfinite(self.regularization) or self.regularization<=0:raise Invalid('positive finite lambda required')
    def frozen(self):
        self.validate();d=asdict(self);d.update(schema='mmse-task-v1',representation='R(Z)=[[Re,-Im],[Im,Re]]',measurement=SCOPES[self.scope]);return d
    @property
    def identity(self):return digest(self.frozen())


def fixture(task,seed):
    """No candidate-specific A/b/X0 is supplied in the full-scope input map."""
    task.validate()
    if task.scope!='mmse_h_y_to_xhat_v1':raise Unsupported('use prepared-kernel fixture for prepared_solver_kernel_v1')
    rng=np.random.default_rng(seed);B,M,U,R=task.batch_size,task.nr,task.nt,task.nrhs
    H=(rng.standard_normal((B,M,U))+1j*rng.standard_normal((B,M,U)))/np.sqrt(2*M)
    symbols=((2*rng.integers(0,2,(B,U,R))-1)+1j*(2*rng.integers(0,2,(B,U,R))-1))/np.sqrt(2)
    Y=H@symbols+np.sqrt(task.regularization/2)*(rng.standard_normal((B,M,R))+1j*rng.standard_normal((B,M,R)))
    return {'inputs':{'H':np.stack([real_block(h) for h in H]),'Y':np.stack([real_block(y) for y in Y]),'RegI':task.regularization*np.eye(2*U),'Identity':np.eye(2*U)},'H':H,'Y':Y,'symbols':symbols,'task':task.frozen(),'seed':seed}


def task_reference(data,task):
    h,y=data['H'],data['Y'];a=h.conj().transpose(0,2,1)@h+task.regularization*np.eye(task.nt)
    return np.linalg.solve(a,h.conj().transpose(0,2,1)@y)


def decode_checked(block,nt,nrhs,tolerance=1e-10):
    z=np.asarray(block)
    if z.shape!=(2*nt,2*nrhs) or not np.isfinite(z).all():raise Invalid('output real-block shape/finite mismatch')
    a,b=z[:nt,:nrhs],z[nt:,:nrhs]
    scale=max(1.,float(np.max(np.abs(z))))
    if max(np.max(np.abs(z[nt:,nrhs:]-a)),np.max(np.abs(z[:nt,nrhs:]+b)))/scale>tolerance:
        raise Invalid('inconsistent real-block quadrants')
    return a+1j*b


def assert_comparable(left,right):
    if left!=right:raise Invalid('task/input/precision/measurement scope differs; no ranking')
