"""Independent blocked/direct-reduction solvers; no interpreter or numeric-policy imports."""
import numpy as np

def _solve(H,Y,regularization,iterations,variant,modeled):
    if iterations!=1:raise ValueError('direct structured solver needs iterations=1 sentinel')
    def block(z):return np.block([[z.real,-z.imag],[z.imag,z.real]])
    def store(x):
        with np.errstate(over='raise',invalid='raise',under='ignore'):v=np.asarray(x).astype(np.float16 if modeled else np.float64)
        if not np.isfinite(v).all():raise ValueError('nonfinite independent reference')
        return v
    def op(f,*a):
        with np.errstate(divide='raise',over='raise',invalid='raise',under='ignore'):return store(f(*(x.astype(np.float32 if modeled else np.float64) for x in a)))
    def dot(a,b):
        if not modeled:return a@b
        value=np.zeros((a.shape[0],b.shape[1]),np.float32)
        for k in range(a.shape[1]):value=np.add(value,np.multiply(a[:,k:k+1].astype(np.float32),b[k:k+1,:].astype(np.float32),dtype=np.float32),dtype=np.float32)
        return store(value)
    def minus(a,b):return op(np.add,a,op(np.multiply,b,negative))
    ldl=variant=='ldl_block';merge=variant=='merge';outputs=[];blocks=[];factors=[];diagonals=[]
    for h,y in zip(H,Y):
        n=2*h.shape[1];r=2*y.shape[1];bs=4
        if n%bs:raise ValueError('real block size4 required')
        hh,yy=store(block(h)),store(block(y));eye=store(np.eye(n));reg=store(regularization*np.eye(n))
        a=op(np.add,dot(hh.T,hh),reg);b=dot(hh.T,yy)
        zero=op(np.subtract,eye[:1,:1],eye[:1,:1]);one=op(np.divide,reg[:1,:1],reg[:1,:1]);negative=op(np.subtract,zero,one)
        l=op(np.multiply,eye,one if ldl else zero);d=np.empty((1,n),dtype=a.dtype)
        if merge:
            for j in range(n):
                if j:a[j:j+1,j:j+1]=op(np.subtract,a[j:j+1,j:j+1],dot(l[j:j+1,:j],l[j:j+1,:j].T))
                l[j:j+1,j:j+1]=op(np.sqrt,a[j:j+1,j:j+1])
                if j+1<n:
                    if j:a[j+1:,j:j+1]=minus(a[j+1:,j:j+1],dot(l[j+1:,:j],l[j:j+1,:j].T))
                    l[j+1:,j:j+1]=op(np.divide,a[j+1:,j:j+1],l[j:j+1,j:j+1])
        else:
            for start in range(0,n,bs):
                end=start+bs
                for j in range(start,end):
                    if ldl:d[:,j:j+1]=op(np.multiply,a[j:j+1,j:j+1],one)
                    else:l[j:j+1,j:j+1]=op(np.sqrt,a[j:j+1,j:j+1])
                    if j+1<end:
                        l[j+1:end,j:j+1]=op(np.divide,a[j+1:end,j:j+1],d[:,j:j+1] if ldl else l[j:j+1,j:j+1])
                        left=op(np.multiply,l[j+1:end,j:j+1],d[:,j:j+1]) if ldl else l[j+1:end,j:j+1]
                        a[j+1:end,j+1:end]=minus(a[j+1:end,j+1:end],dot(left,l[j+1:end,j:j+1].T))
                if end<n:
                    weighted=np.empty((n-end,bs),dtype=a.dtype)
                    for j in range(start,end):
                        l[end:,j:j+1]=op(np.divide,a[end:,j:j+1],d[:,j:j+1] if ldl else l[j:j+1,j:j+1])
                        if ldl:weighted[:,j-start:j-start+1]=op(np.multiply,l[end:,j:j+1],d[:,j:j+1])
                        if j+1<end:
                            left=weighted[:,j-start:j-start+1] if ldl else l[end:,j:j+1]
                            a[end:,j+1:end]=minus(a[end:,j+1:end],dot(left,l[j+1:end,j:j+1].T))
                    a[end:,end:]=minus(a[end:,end:],dot(weighted if ldl else l[end:,start:end],l[end:,start:end].T))
        z=np.empty((n,r),dtype=a.dtype)
        for j in range(n):
            if merge and j:b[j:j+1]=minus(b[j:j+1],dot(l[j:j+1,:j],z[:j]))
            z[j:j+1]=op(np.multiply,b[j:j+1],one) if ldl else op(np.divide,b[j:j+1],l[j:j+1,j:j+1])
            if not merge and j+1<n:b[j+1:]=minus(b[j+1:],dot(l[j+1:,j:j+1],z[j:j+1]))
        if ldl:
            back=np.empty_like(z)
            for j in range(n):back[j:j+1]=op(np.divide,z[j:j+1],d[:,j:j+1])
        else:back=op(np.multiply,z,one)
        x=np.empty_like(z)
        for j in range(n-1,-1,-1):
            if merge and j+1<n:back[j:j+1]=minus(back[j:j+1],dot(l[j+1:,j:j+1].T,x[j+1:]))
            x[j:j+1]=op(np.multiply,back[j:j+1],one) if ldl else op(np.divide,back[j:j+1],l[j:j+1,j:j+1])
            if not merge and j:back[:j]=minus(back[:j],dot(l[j:j+1,:j].T,x[j:j+1]))
        factors.append(l);blocks.append(x)
        if ldl:diagonals.append(d)
        u=h.shape[1];rr=y.shape[1];outputs.append(x[:u,:rr].astype(float)+1j*x[u:,:rr].astype(float))
    evidence={'factor_L_block':np.stack(factors)}
    if ldl:evidence['factor_D_block']=np.stack(diagonals)
    return {'output':np.stack(outputs),'output_block':np.stack(blocks),'evidence':evidence}

def simulate_cholesky_block(H,Y,regularization,iterations):return _solve(H,Y,regularization,iterations,'chol_block',False)
def simulate_cholesky_block_fp16(H,Y,regularization,iterations):return _solve(H,Y,regularization,iterations,'chol_block',True)
def simulate_ldl_block(H,Y,regularization,iterations):return _solve(H,Y,regularization,iterations,'ldl_block',False)
def simulate_ldl_block_fp16(H,Y,regularization,iterations):return _solve(H,Y,regularization,iterations,'ldl_block',True)
def simulate_merge(H,Y,regularization,iterations):return _solve(H,Y,regularization,iterations,'merge',False)
def simulate_merge_fp16(H,Y,regularization,iterations):return _solve(H,Y,regularization,iterations,'merge',True)
