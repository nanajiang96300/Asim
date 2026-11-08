"""Batched real primitive model matching the frozen FP16-storage policy.

No solver, eigendecomposition or complex BLAS in candidate execution.
All persisted operation results, coefficients and factorization states are FP16.
"""
from collections import Counter
import numpy as np
from verification.contracts.core import Invalid,Unsupported

class Primitives:
    def __init__(self,storage='float16'):
        if storage not in ('float16','float64'):raise Unsupported('study storage policy')
        self.storage=storage;self.counts=Counter();self.macs=0
    def put(self,x):
        if np.iscomplexobj(x) or not np.isfinite(x).all():raise Invalid('nonfinite/complex primitive result')
        with np.errstate(over='raise',invalid='raise',under='ignore'):r=np.asarray(x).astype(self.storage)
        if not np.isfinite(r).all():raise Invalid('nonfinite writeback')
        return r
    def const(self,x):self.counts['CONST']+=1;return self.put(np.asarray(x))
    def op(self,code,*args):
        if any(a.dtype!=np.dtype(self.storage) or not np.isfinite(a).all() for a in args):raise Invalid('primitive input storage/finite mismatch')
        self.counts[code]+=1;dtype=np.float32 if self.storage=='float16' else np.float64
        a=[x.astype(dtype) for x in args]
        with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
            if code=='GEMM':
                if a[0].ndim!=3 or a[1].ndim!=3 or a[0].shape[0]!=a[1].shape[0] or a[0].shape[2]!=a[1].shape[1]:raise Invalid('batched GEMM shape')
                z=np.zeros((a[0].shape[0],a[0].shape[1],a[1].shape[2]),dtype=dtype)
                for k in range(a[0].shape[2]):z=np.add(z,np.multiply(a[0][:,:,k:k+1],a[1][:,k:k+1,:],dtype=dtype),dtype=dtype)
                self.macs+=int(np.prod(z.shape))*a[0].shape[2]
            elif code=='COPY':z=a[0].copy()
            elif code=='SQRT':z=np.sqrt(a[0],dtype=dtype)
            elif code in ('ADD','SUB','MUL','DIV'):z={'ADD':np.add,'SUB':np.subtract,'MUL':np.multiply,'DIV':np.divide}[code](*a,dtype=dtype)
            else:raise Unsupported('primitive '+code)
        return self.put(z)


def block(z):
    return np.concatenate((np.concatenate((z.real,-z.imag),axis=2),np.concatenate((z.imag,z.real),axis=2)),axis=1)


def precondition(e,H,lam,complex_block):
    B,_,n=H.shape;U=n//2
    if complex_block not in (1,2,4,8) or U%complex_block:raise Invalid('complex block size')
    P=e.const(np.zeros((B,n,n)));one=e.const(np.ones((B,1,1)))
    for first in range(0,U,complex_block):
        idx=list(range(first,first+complex_block))+list(range(U+first,U+first+complex_block));d=len(idx)
        hh=e.op('COPY',H[:,:,idx]);I=e.const(np.broadcast_to(np.eye(d),(B,d,d)))
        A=e.op('ADD',e.op('GEMM',hh.transpose(0,2,1),hh),e.op('MUL',lam,I))
        if complex_block==1:
            # Complex diagonal Jacobi: explicitly duplicate the real-column energy.
            v=e.op('DIV',one,A[:,:1,:1]);pb=e.op('MUL',I,v)
        else:
            L=e.const(np.zeros_like(A))
            for j in range(d):
                t=e.op('COPY',A[:,j:j+1,j:j+1])
                if j:t=e.op('SUB',t,e.op('GEMM',L[:,j:j+1,:j],L[:,j:j+1,:j].transpose(0,2,1)))
                L[:,j:j+1,j:j+1]=e.op('SQRT',t)
                for i in range(j+1,d):
                    t=e.op('COPY',A[:,i:i+1,j:j+1])
                    if j:t=e.op('SUB',t,e.op('GEMM',L[:,i:i+1,:j],L[:,j:j+1,:j].transpose(0,2,1)))
                    L[:,i:i+1,j:j+1]=e.op('DIV',t,L[:,j:j+1,j:j+1])
            V=e.const(np.zeros_like(A))
            for j in range(d):
                t=e.op('COPY',I[:,j:j+1,:])
                if j:t=e.op('SUB',t,e.op('GEMM',L[:,j:j+1,:j],V[:,:j,:]))
                V[:,j:j+1,:]=e.op('DIV',t,L[:,j:j+1,j:j+1])
            pb=e.op('GEMM',V.transpose(0,2,1),V)
        # Explicit packing/scatter; no arithmetic or precision promotion.
        for j,r in enumerate(idx):P[:,r,idx]=e.op('COPY',pb[:,j,:])
    return P


def run(H,Y,lam,complex_block,kind,steps=256,bounds=(.25,8.),storage='float16',left=False,restart=0,compensated=False,momentum=1.,hermitian=False):
    if kind not in ('explicit_inverse','implicit_inverse','implicit_inverse_add','explicit_solve','implicit_solve'):raise Invalid('candidate residual')
    if type(steps) is not int or not 1<=steps<=256:raise Invalid('K budget')
    if type(left) is not bool or type(compensated) is not bool or type(restart) is not int or restart<0 or not 0<momentum<=1 or (hermitian and (not left or 'inverse' not in kind or compensated)):raise Invalid('study update parameters')
    e=Primitives(storage);hh=e.const(block(H));yy=e.const(block(Y));B,_,n=hh.shape
    eye=e.const(np.broadcast_to(np.eye(n),(B,n,n)));l=e.const(np.full((B,1,1),lam));P=precondition(e,hh,l,complex_block)
    one=e.const(np.ones((B,1,1)));two=e.const(np.full((B,1,1),2.));lo=e.const(np.full((B,1,1),lam*bounds[0]));hi=e.const(np.full((B,1,1),bounds[1]))
    d=e.op('DIV',e.op('ADD',hi,lo),two);c=e.op('DIV',e.op('SUB',hi,lo),two);rho=e.op('DIV',c,d)
    inverse='inverse' in kind
    if left:
        eye=e.op('COPY',eye[:,:,:n//2]);yy=e.op('COPY',yy[:,:,:yy.shape[-1]//2])
    X=e.const(np.zeros((B,n,eye.shape[-1] if inverse else yy.shape[-1])));previous=e.op('COPY',X);carry=e.const(np.zeros_like(X)) if compensated else None
    full_eye=e.const(np.broadcast_to(np.eye(n),(B,n,n)))
    if kind.startswith('explicit'):
        A=e.op('ADD',e.op('GEMM',hh.transpose(0,2,1),hh),e.op('MUL',l,full_eye))
        rhs=eye if inverse else e.op('GEMM',hh.transpose(0,2,1),yy)
    for k in range(steps):
        if k==0 or (restart and k%restart==0):
            rho=e.op('DIV',c,d);alpha=e.op('DIV',one,d);beta=e.const(np.zeros((B,1,1)))
        else:
            den=e.op('SUB',e.op('MUL',two,d),e.op('MUL',c,rho));newrho=e.op('DIV',c,den)
            alpha=e.op('DIV',two,den);beta=e.op('MUL',newrho,rho);rho=newrho
        if momentum!=1.:beta=e.op('MUL',beta,e.const(np.full((B,1,1),momentum)))
        if kind.startswith('explicit'):r=e.op('SUB',rhs,e.op('GEMM',A,X))
        elif inverse:
            hx=e.op('GEMM',hh,X);ax=e.op('GEMM',hh.transpose(0,2,1),hx);lx=e.op('MUL',l,X)
            r=e.op('SUB',eye,e.op('ADD',ax,lx)) if kind=='implicit_inverse_add' else e.op('SUB',e.op('SUB',eye,ax),lx)
        else:
            data_res=e.op('SUB',yy,e.op('GEMM',hh,X));r=e.op('SUB',e.op('GEMM',hh.transpose(0,2,1),data_res),e.op('MUL',l,X))
        direction=e.op('GEMM',P,r);moment=e.op('MUL',beta,e.op('SUB',X,previous))
        delta=e.op('ADD',e.op('MUL',alpha,direction),moment)
        if compensated:
            update=e.op('SUB',delta,carry);value=e.op('ADD',X,update);carry=e.op('SUB',e.op('SUB',value,X),update)
        else:value=e.op('ADD',X,delta)
        if hermitian:
            U=n//2;re=value[:,:U,:];im=value[:,U:,:]
            re=e.op('DIV',e.op('ADD',re,re.transpose(0,2,1)),two);im=e.op('DIV',e.op('SUB',im,im.transpose(0,2,1)),two)
            value=e.op('COPY',np.concatenate((re,im),axis=1))
        previous,X=X,value
        # Charge only one final reconstruction at the selected K, not every diagnostic snapshot.
        before=Counter(e.counts);before_macs=e.macs
        if left:
            U=n//2;re=e.op('COPY',X[:,:U,:]);im=e.op('COPY',X[:,U:,:]);neg=e.op('SUB',e.const(np.zeros_like(im)),im)
            output=e.op('COPY',np.concatenate((np.concatenate((re,neg),axis=2),np.concatenate((im,re),axis=2)),axis=1))
        else:output=X.copy()
        yield k+1,output,dict(ops=dict(e.counts),modeled_dense_macs=e.macs,storage=storage,compensation_storage=storage if compensated else None)
        e.counts=before;e.macs=before_macs
