"""Independent LDL references; no engine or precision policy imports."""
import numpy as np

def simulate_ldl(H,Y,regularization,iterations):
    if iterations!=1:raise ValueError('direct LDL has no iteration count')
    values=[];factors=[];diagonals=[]
    for h,y in zip(H,Y):
        a=h.conj().T@h+regularization*np.eye(h.shape[1]);n=a.shape[0];l=np.eye(n,dtype=np.complex128);d=np.empty(n,complex)
        for j in range(n):
            d[j]=a[j,j]-sum(l[j,k]*d[k]*l[j,k].conjugate() for k in range(j))
            for i in range(j+1,n):l[i,j]=(a[i,j]-sum(l[i,k]*d[k]*l[j,k].conjugate() for k in range(j)))/d[j]
        z=np.linalg.solve(l,h.conj().T@y);values.append(np.linalg.solve(l.conj().T,z/d[:,None]));factors.append(l);diagonals.append(d)
    return {'output':np.stack(values),'evidence':{'factor_L':np.stack(factors),'factor_D':np.stack(diagonals)}}

def simulate_ldl_fp16_model(H,Y,regularization,iterations):
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
    outputs=[];blocks=[];factors=[];diagonals=[]
    for h,y in zip(H,Y):
        n=2*h.shape[1];r=2*y.shape[1];hh=half(block(h));yy=half(block(y));eye=half(np.eye(n));reg=half(regularization*np.eye(n))
        a=op(np.add,dot(hh.T,hh),reg);b=dot(hh.T,yy)
        zero=op(np.subtract,eye[:1,:1],eye[:1,:1]);one=op(np.divide,reg[:1,:1],reg[:1,:1]);negative=op(np.subtract,zero,one)
        l=op(np.multiply,eye,one);d=np.empty((1,n),np.float16)
        for j in range(n):
            d[:,j:j+1]=op(np.multiply,a[j:j+1,j:j+1],one)
            if j+1<n:
                l[j+1:,j:j+1]=op(np.divide,a[j+1:,j:j+1],d[:,j:j+1])
                weighted=op(np.multiply,l[j+1:,j:j+1],d[:,j:j+1])
                t=dot(weighted,l[j+1:,j:j+1].T)
                a[j+1:,j+1:]=op(np.add,a[j+1:,j+1:],op(np.multiply,t,negative))
        z=np.empty((n,r),np.float16)
        for j in range(n):
            z[j:j+1]=op(np.multiply,b[j:j+1],one)
            if j+1<n:b[j+1:]=op(np.add,b[j+1:],op(np.multiply,dot(l[j+1:,j:j+1],z[j:j+1]),negative))
        back=np.empty((n,r),np.float16)
        for j in range(n):back[j:j+1]=op(np.divide,z[j:j+1],d[:,j:j+1])
        x=np.empty((n,r),np.float16)
        for j in range(n-1,-1,-1):
            x[j:j+1]=op(np.multiply,back[j:j+1],one)
            if j:back[:j]=op(np.add,back[:j],op(np.multiply,dot(l[j:j+1,:j].T,x[j:j+1]),negative))
        factors.append(l);diagonals.append(d);blocks.append(x);u=h.shape[1];rr=y.shape[1];outputs.append(x[:u,:rr].astype(float)+1j*x[u:,:rr].astype(float))
    return {'output':np.stack(outputs),'output_block':np.stack(blocks),'evidence':{'factor_L_block':np.stack(factors),'factor_D_block':np.stack(diagonals)}}
