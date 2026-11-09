"""Fixed-coefficient FP16 heavy-ball candidate. No eig/solve in candidate execution."""
from collections import Counter
import numpy as np
from verification.contracts.core import Invalid,Unsupported
from verification.studies.fp16_primitives import Primitives,block

PARAMETERS=dict(id='fp16_fixed_hb_v1',block_size_complex=1,column_energy_domain=[2**-8,8.],energy_sum_domain=[8.,32.],energy_norm_scale=8.,preconditioner_norm_scale=128.,enclosure_inflation=1.125,lower_safety=.9375,alpha_safety=.9375,initialization='x0=previous=carry=0',bounds='scaled_L4_energy_L2_P_trace',compensated=True,representation='left_complex_columns')

class HeavyBall:
    def __init__(self,H,Y,lam):
        if not 1<=len(H)<=192 or H.ndim!=3 or H.shape[1:]!=(64,16) or Y.shape!=(len(H),64,8) or lam not in (1.,.1,.01):raise Unsupported('fixed HB task domain')
        self.e=e=Primitives();B=len(H)
        self.h=e.const(block(H));self.y=e.const(np.concatenate((Y.real,Y.imag),axis=1));self.l=e.const(np.full((B,1,1),lam))
        self.zero=e.const(np.zeros((B,1,1)));one=e.const(np.ones((B,1,1)))
        eight=e.const(np.full((B,1,1),8.));scaleP=e.const(np.full((B,1,1),128.));inflate=e.const(np.full((B,1,1),1.125));safety=e.const(np.full((B,1,1),.9375));four=e.const(np.full((B,1,1),4.))
        energy=[];p=[];Esum=e.op('COPY',self.zero);E4=e.op('COPY',self.zero);P2=e.op('COPY',self.zero);trace=e.op('COPY',self.zero)
        for j in range(16):
            col=self.h[:,:,j:j+1];v=e.op('GEMM',col.transpose(0,2,1),col)
            if np.any(v<2**-8) or np.any(v>8):raise Unsupported('HB column energy domain before norm powers')
            energy.append(v)
            Esum=e.op('ADD',Esum,v);pj=e.op('DIV',one,e.op('ADD',v,self.l));p.append(pj)
            scaled=e.op('DIV',v,eight);square=e.op('MUL',scaled,scaled);fourth=e.op('MUL',square,square);E4=e.op('ADD',E4,fourth)
            scaled=e.op('DIV',pj,scaleP);P2=e.op('ADD',P2,e.op('MUL',scaled,scaled));trace=e.op('ADD',trace,e.op('MUL',pj,v))
        energies=np.concatenate(energy,axis=1);pd=np.concatenate(p,axis=1)
        if np.any(energies<2**-8) or np.any(energies>8) or np.any(Esum<8) or np.any(Esum>32):raise Unsupported('HB energy domain; no clipping/fallback')
        Eu=e.op('MUL',e.op('MUL',e.op('SQRT',e.op('SQRT',E4)),eight),inflate)
        Pu=e.op('MUL',e.op('MUL',e.op('SQRT',P2),scaleP),inflate)
        self.lo=e.op('MUL',e.op('DIV',self.l,e.op('ADD',Eu,self.l)),safety)
        self.hi=e.op('MUL',e.op('ADD',trace,e.op('MUL',self.l,Pu)),inflate)
        upper_root=e.op('SQRT',self.hi);lower_root=e.op('SQRT',self.lo);den=e.op('ADD',upper_root,lower_root)
        self.alpha=e.op('MUL',e.op('DIV',four,e.op('MUL',den,den)),safety)
        ratio=e.op('DIV',e.op('SUB',upper_root,lower_root),den);self.beta=e.op('MUL',ratio,ratio)
        if np.any(self.lo<=0) or np.any(self.hi<self.lo) or np.any(self.beta<0) or np.any(self.beta>=1) or np.any(self.alpha<=0) or np.any(self.alpha.astype(float)*self.hi>=2*(1+self.beta.astype(float))):raise Unsupported('rounded HB coefficients fail frozen stability inequality')
        # Explicit layout action; a future native mapping must charge row copies/expansion.
        self.p=e.op('COPY',np.repeat(np.concatenate((pd,pd),axis=1),8,axis=2))
        self.x=e.const(np.zeros((B,32,8)));self.previous=e.op('COPY',self.x);self.carry=e.const(np.zeros_like(self.x))
        self.setup=dict(energies=energies,P_diagonal=pd,energy_upper=Eu,preconditioner_upper=Pu,lower=self.lo,upper=self.hi,alpha=self.alpha,beta=self.beta,stored_lambda=self.l)
    def iterate(self,K):
        if type(K) is not int or not 1<=K<=256:raise Invalid('HB K budget 1..256')
        e=self.e
        for k in range(1,K+1):
            residual=e.op('SUB',self.y,e.op('GEMM',self.h,self.x))
            gradient=e.op('SUB',e.op('GEMM',self.h.transpose(0,2,1),residual),e.op('MUL',self.l,self.x))
            gradient=e.op('ADD',gradient,self.zero)
            direction=e.op('MUL',self.p,gradient)
            delta=e.op('ADD',e.op('MUL',self.alpha,direction),e.op('MUL',self.beta,e.op('SUB',self.x,self.previous)))
            update=e.op('SUB',delta,self.carry);value=e.op('ADD',self.x,update)
            self.carry=e.op('SUB',e.op('SUB',value,self.x),update);self.previous,self.x=self.x,value
            before=Counter(e.counts);macs=e.macs
            re=e.op('COPY',self.x[:,:16,:]);im=e.op('COPY',self.x[:,16:,:]);negative=e.op('SUB',e.const(np.zeros_like(im)),im)
            output=e.op('COPY',np.concatenate((np.concatenate((re,negative),axis=2),np.concatenate((im,re),axis=2)),axis=1))
            yield k,output,dict(ops=dict(e.counts),modeled_dense_macs=e.macs)
            e.counts=before;e.macs=macs
