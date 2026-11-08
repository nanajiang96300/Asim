"""BRI inverse study: explicit P=D^-1, T=P A, X=Y P; independent of replay."""
import numpy as np
from verification.contracts.core import Invalid


def store(z,precision):
    z=np.asarray(z)
    if precision=='fp64':return z.astype(np.complex128)
    if precision=='fp32':return z.astype(np.complex64)
    if precision!='half_storage':raise Invalid('unknown study precision')
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        return z.real.astype(np.float16).astype(np.float32)+1j*z.imag.astype(np.float16).astype(np.float32)


def hermitian_store(z,precision):
    # Explicit alternative algorithm: compute/store upper triangle once, mirror it.
    # This is not silent averaging/repair of an external matrix.
    u=store(np.triu(z),precision);idx=np.arange(u.shape[-1]);u[:,idx,idx]=store(z[:,idx,idx].real,precision)
    return u+np.triu(u,1).conj().transpose(0,2,1)


def system(H,lam,block,precision,hermitian_storage=False):
    if H.ndim!=3 or not np.isfinite(H).all() or not np.isfinite(lam) or lam<=0:raise Invalid('invalid H/lambda')
    n=H.shape[-1]
    if type(block) is not int or block<1 or n%block:raise Invalid('complex block must divide Nt')
    q=lambda z:store(z,precision);hh=q(H);eye=q(np.broadcast_to(np.eye(n),(len(H),n,n)))
    matrix_store=(lambda z:hermitian_store(z,precision)) if hermitian_storage else q
    A=matrix_store(matrix_store(hh.conj().transpose(0,2,1)@hh)+q(lam*eye));P=np.zeros_like(A)
    for j in range(0,n,block):
        b=A[:,j:j+block,j:j+block]
        if np.min(np.linalg.eigvalsh(b.astype(np.complex128)))<=0:raise Invalid('non-SPD modeled block, no repair')
        L=np.linalg.cholesky(b);invL=np.linalg.solve(L,np.broadcast_to(np.eye(block,dtype=b.dtype),b.shape))
        P[:,j:j+block,j:j+block]=matrix_store(invL.conj().transpose(0,2,1)@invL)
    if not np.isfinite(P).all() or np.min(np.linalg.eigvalsh(P.astype(np.complex128)))<=0:raise Invalid('non-SPD/nonfinite stored P')
    # PA is generally not Hermitian: obtain its real positive spectrum by similarity.
    lp=np.linalg.cholesky(P.astype(np.complex128));C=lp.conj().transpose(0,2,1)@A.astype(np.complex128)@lp
    if np.max(np.abs(C-C.conj().transpose(0,2,1)))>1e-5*max(1.,np.max(abs(C))):raise Invalid('modeled A not Hermitian')
    eigenvalues=np.linalg.eigvalsh((C+C.conj().transpose(0,2,1))/2)
    if np.min(eigenvalues)<=0:raise Invalid('non-SPD modeled A')
    return A,P,eigenvalues


def enclosure(eigenvalues):
    return .99*eigenvalues[:,0],1.01*eigenvalues[:,-1]


def check_bounds(eigenvalues,lo,hi):
    lo,hi=np.broadcast_arrays(lo,hi)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo<=0) or np.any(hi<lo) or np.any(lo>eigenvalues[:,0]) or np.any(hi<eigenvalues[:,-1]):
        raise Invalid('bounds must enclose full preconditioned spectrum')


def iterate(A,P,eigenvalues,steps,precision='fp64',method='chebyshev',form='direct',bounds=None,diagnostic_unsafe=False):
    if type(steps) is not int or not 1<=steps<=256:raise Invalid('K budget 1..256')
    if form not in ('direct','scaled','literal_P'):raise Invalid('unknown inverse variable')
    if method not in ('half','optimal','chebyshev'):raise Invalid('unknown Richardson schedule')
    q=lambda z:store(z,precision);eye=q(np.broadcast_to(np.eye(A.shape[-1]),A.shape));X=np.zeros_like(A);previous=X.copy()
    if form=='literal_P':
        T=P;eigs=np.linalg.eigvalsh(P.astype(np.complex128));lo,hi=enclosure(eigs)
    else:
        eigs=eigenvalues;lo,hi=enclosure(eigs);T=q(P@A)
    if bounds is not None:lo,hi=bounds
    if not diagnostic_unsafe:check_bounds(eigs,lo,hi)
    d=(hi+lo)/2;c=(hi-lo)/2;rho=np.divide(c,d);alpha=1/d
    for k in range(steps):
        if method=='half':a=np.full_like(d,.5);beta=np.zeros_like(d)
        elif method=='optimal':a=1/d;beta=np.zeros_like(d)
        elif k==0:a=alpha;beta=np.zeros_like(d)
        else:
            newrho=np.divide(c,2*d-c*rho);a=np.divide(2.,2*d-c*rho);beta=newrho*rho;rho=newrho
        if precision!='fp64':a=a.astype(np.float32);beta=beta.astype(np.float32)
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            if form=='direct':R=q(P@q(eye-q(A@X)))
            else:R=q(eye-q(T@X))
            update=q(q(a[:,None,None]*R)+q(beta[:,None,None]*q(X-previous)))
            value=q(X+update);previous,X=X,value
            output=q(X@P) if form=='scaled' else X.copy()
        yield k+1,output


def root_product(A,P,steps,lo,hi,form='scaled'):
    """Separate two-term polynomial implementation for FP64 cross-checks."""
    n=A.shape[-1];I=np.broadcast_to(np.eye(n),A.shape);Y=np.zeros_like(A,dtype=np.complex128);T=P@A
    for k in range(steps):
        w=1/((hi+lo)/2+(hi-lo)/2*np.cos(np.pi*(2*k+1)/(2*steps)))
        Y+=w[:,None,None]*(I-T@Y)
    return Y@P


def relative(values,reference):
    # Scaling avoids overflow in diagnostic divergence norms.
    scale=np.maximum(1.,np.maximum(np.max(abs(values),axis=(1,2)),np.max(abs(reference),axis=(1,2))))
    return np.linalg.norm((values-reference)/scale[:,None,None],axis=(1,2))/np.maximum(np.linalg.norm(reference/scale[:,None,None],axis=(1,2)),1e-300)
