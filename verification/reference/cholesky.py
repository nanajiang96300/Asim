"""Independent direct Cholesky references. Never imports instruction execution."""
import numpy as np

def simulate_cholesky(H,Y,regularization,iterations):
    if iterations!=1:raise ValueError('direct algorithm has no iteration count')
    values=[];factors=[]
    for h,y in zip(H,Y):
        a=h.conj().T@h+regularization*np.eye(h.shape[1]);l=np.linalg.cholesky(a)
        values.append(np.linalg.solve(l.conj().T,np.linalg.solve(l,h.conj().T@y)));factors.append(l)
    return {'output':np.stack(values),'evidence':{'factor_L':np.stack(factors)}}

def simulate_cholesky_fp16_model(H,Y,regularization,iterations):
    if iterations!=1:raise ValueError('direct algorithm has no iteration count')
    def block(z):return np.block([[z.real,-z.imag],[z.imag,z.real]])
    def half(x):
        with np.errstate(over='raise',invalid='raise',under='ignore'):a=np.asarray(x).astype(np.float16)
        if not np.isfinite(a).all():raise ValueError('nonfinite independent reference storage')
        return a
    def op(f,*args):
        with np.errstate(divide='raise',over='raise',invalid='raise',under='ignore'):return half(f(*(a.astype(np.float32) for a in args)))
    def dot(a,b):
        c=np.zeros((a.shape[0],b.shape[1]),dtype=np.float32)
        for k in range(a.shape[1]):c=np.add(c,np.multiply(a[:,k:k+1].astype(np.float32),b[k:k+1,:].astype(np.float32),dtype=np.float32),dtype=np.float32)
        return half(c)
    outputs=[];blocks=[];factors=[]
    for h,y in zip(H,Y):
        n=2*h.shape[1];r=2*y.shape[1];hh=half(block(h));yy=half(block(y));eye=half(np.eye(n));reg=half(regularization*np.eye(n))
        a=op(np.add,dot(hh.T,hh),reg);b=dot(hh.T,yy)
        zero=op(np.subtract,eye[:1,:1],eye[:1,:1]);one=op(np.divide,reg[:1,:1],reg[:1,:1]);negative=op(np.subtract,zero,one)
        l=op(np.multiply,eye,zero)
        for j in range(n):
            l[j:j+1,j:j+1]=op(np.sqrt,a[j:j+1,j:j+1])
            for i in range(j+1,n):l[i:i+1,j:j+1]=op(np.divide,a[i:i+1,j:j+1],l[j:j+1,j:j+1])
            if j+1<n:
                t=dot(l[j+1:,j:j+1],l[j+1:,j:j+1].T)
                a[j+1:,j+1:]=op(np.add,a[j+1:,j+1:],op(np.multiply,t,negative))
        z=np.empty((n,r),np.float16)
        for j in range(n):
            z[j:j+1]=op(np.divide,b[j:j+1],l[j:j+1,j:j+1])
            if j+1<n:b[j+1:]=op(np.add,b[j+1:],op(np.multiply,dot(l[j+1:,j:j+1],z[j:j+1]),negative))
        back=op(np.multiply,z,one);x=np.empty((n,r),np.float16)
        for j in range(n-1,-1,-1):
            x[j:j+1]=op(np.divide,back[j:j+1],l[j:j+1,j:j+1])
            if j:back[:j]=op(np.add,back[:j],op(np.multiply,dot(l[j:j+1,:j].T,x[j:j+1]),negative))
        factors.append(l);blocks.append(x);u=h.shape[1];rr=y.shape[1];outputs.append(x[:u,:rr].astype(float)+1j*x[u:,:rr].astype(float))
    return {'output':np.stack(outputs),'output_block':np.stack(blocks),'evidence':{'factor_L_block':np.stack(factors)}}
