"""Independent complex Newton–Schulz; no instruction interpreter imports."""
import numpy as np
from verification.contracts.core import Invalid


def simulate_ns(H,Y,regularization,iterations):
    if type(iterations) is not int or not 1<=iterations<=32:raise Invalid('NS iteration budget 1..32')
    B,_,U=H.shape;eye=np.eye(U);outputs=[];initializers=[]
    for h,y in zip(H,Y):
        a=h.conj().T@h+regularization*eye
        # trace(R(A)) is twice trace(A), matching actual real-block scalar sum.
        x=eye/(2*np.trace(a).real);initializers.append(x.copy())
        for _ in range(iterations):x=x@(2*eye-a@x)
        outputs.append(x@(h.conj().T@y))
    return {'output':np.stack(outputs),'X0':np.stack(initializers),'iterations':iterations,'initialization':'I/(2*trace(complex A))'}


def simulate_ns_fp16_model(H,Y,regularization,iterations):
    """Independent ordered real-block model, deliberately not using engine/policy."""
    if type(iterations) is not int or not 1<=iterations<=32:raise Invalid('iteration budget')
    def block(z):return np.concatenate((np.concatenate((z.real,-z.imag),axis=1),np.concatenate((z.imag,z.real),axis=1)),axis=0)
    def half(x):
        with np.errstate(over='raise',invalid='raise',under='ignore'):out=np.asarray(x).astype(np.float16)
        if not np.isfinite(out).all():raise Invalid('reference nonfinite half storage')
        return out
    def op(f,*a):return half(f(*(x.astype(np.float32) for x in a)))
    def dot(a,b):
        acc=np.zeros((a.shape[0],b.shape[1]),np.float32)
        with np.errstate(over='raise',invalid='raise',under='ignore'):
            for j in range(a.shape[1]):
                term=a[:,j].astype(np.float32)[:,None]*b[j,:].astype(np.float32)[None,:]
                acc=(acc+term).astype(np.float32)
        return half(acc)
    outputs=[];blocks=[];initializers=[];n=2*H.shape[2]
    for h,y in zip(H,Y):
        hh,yy=half(block(h)),half(block(y));identity=half(np.eye(n));reg=half(regularization*np.eye(n))
        a=op(np.add,dot(hh.T,hh),reg);rhs=dot(hh.T,yy)
        tau=op(np.multiply,a[0:1,0:1],identity[0:1,0:1])
        for j in range(1,n):tau=op(np.add,tau,a[j:j+1,j:j+1])
        zero=op(np.subtract,identity[0:1,0:1],identity[0:1,0:1]);neg=op(np.subtract,zero,identity[0:1,0:1])
        two=op(np.add,identity,identity);x=op(np.divide,identity,tau);initializers.append(x.copy())
        for _ in range(iterations):
            t=dot(a,x);negative=op(np.multiply,t,neg);res=op(np.add,two,negative);x=dot(x,res)
        result=dot(x,rhs);blocks.append(result)
        u=H.shape[2];r=Y.shape[2];outputs.append(result[:u,:r].astype(np.float64)+1j*result[u:,:r].astype(np.float64))
    return {'output':np.stack(outputs),'output_block':np.stack(blocks),'X0':np.stack(initializers),'iterations':iterations,'initialization':'FP16 instruction-by-instruction I/trace(real A)'}
