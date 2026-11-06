"""Independent stationary B2-Richardson; no reconstruction imports."""
import numpy as np

def simulate_bri(H,Y,regularization,iterations):
    if not 1<=iterations<=256:raise ValueError('BRI iteration budget')
    values=[];blocks=[];preconditioners=[]
    def block(z):return np.block([[z.real,-z.imag],[z.imag,z.real]])
    for h,y in zip(H,Y):
        hh=block(h);yy=block(y);n=hh.shape[1];a=hh.T@hh+regularization*np.eye(n);rhs=hh.T@yy;b=np.zeros_like(a)
        for j in range(0,n,2):b[j:j+2,j:j+2]=np.linalg.solve(a[j:j+2,j:j+2],np.eye(2))
        p=.5*b;s=np.eye(n)-p@a;c=p@rhs;x=np.zeros_like(rhs)
        for _ in range(iterations):x=s@x+c
        blocks.append(x);preconditioners.append(b);u=h.shape[1];r=y.shape[1];values.append(x[:u,:r]+1j*x[u:,:r])
    return {'output':np.stack(values),'output_block':np.stack(blocks),'evidence':{'preconditioner_block':np.stack(preconditioners)}}

def simulate_bri_fp16_model(H,Y,regularization,iterations):
    if not 1<=iterations<=256:raise ValueError("BRI iteration budget")
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
    outputs=[];blocks=[];preconditioners=[]
    for h,y in zip(H,Y):
        n=2*h.shape[1];r=2*y.shape[1];hh=half(block(h));yy=half(block(y));eye=half(np.eye(n));reg=half(regularization*np.eye(n))
        a=op(np.add,dot(hh.T,hh),reg);rhs=dot(hh.T,yy)
        zero=op(np.subtract,eye[:1,:1],eye[:1,:1]);one=op(np.divide,reg[:1,:1],reg[:1,:1]);negative=op(np.subtract,zero,one)
        two=op(np.add,one,one);omega=op(np.divide,one,two);b=op(np.multiply,eye,zero)
        for j in range(0,n,2):
            ad=op(np.multiply,a[j:j+1,j:j+1],a[j+1:j+2,j+1:j+2]);bc=op(np.multiply,a[j:j+1,j+1:j+2],a[j+1:j+2,j:j+1]);det=op(np.subtract,ad,bc)
            b[j:j+1,j:j+1]=op(np.divide,a[j+1:j+2,j+1:j+2],det)
            b[j:j+1,j+1:j+2]=op(np.divide,op(np.multiply,a[j:j+1,j+1:j+2],negative),det)
            b[j+1:j+2,j:j+1]=op(np.divide,op(np.multiply,a[j+1:j+2,j:j+1],negative),det)
            b[j+1:j+2,j+1:j+2]=op(np.divide,a[j:j+1,j:j+1],det)
        p=op(np.multiply,b,omega);pa=dot(p,a);s=op(np.add,eye,op(np.multiply,pa,negative));c=dot(p,rhs);x=op(np.multiply,rhs,zero)
        for _ in range(iterations):x=op(np.add,dot(s,x),c)
        x=op(np.multiply,x,one);blocks.append(x);preconditioners.append(b);u=h.shape[1];rr=y.shape[1];outputs.append(x[:u,:rr].astype(float)+1j*x[u:,:rr].astype(float))
    return {'output':np.stack(outputs),'output_block':np.stack(blocks),'evidence':{'preconditioner_block':np.stack(preconditioners)}}
