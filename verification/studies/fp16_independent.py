"""Independent point-Jacobi factored solve; no primitive engine imports."""
import numpy as np
from verification.contracts.core import Invalid


def point_solve(H,Y,lam,K,bounds=(.25,8.),momentum=.9,compensated=True,restart=0,inverse=False):
    def q(x):
        with np.errstate(over='raise',invalid='raise',under='ignore'):v=np.asarray(x).astype(np.float16)
        if not np.isfinite(v).all():raise Invalid('independent nonfinite')
        return v
    def op(f,*args):
        with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):return q(f(*(a.astype(np.float32) for a in args)))
    def dot(a,b):
        z=np.zeros((a.shape[0],b.shape[1]),np.float32)
        for k in range(a.shape[1]):z=np.add(z,np.multiply(a[:,k:k+1].astype(np.float32),b[k:k+1,:].astype(np.float32),dtype=np.float32),dtype=np.float32)
        return q(z)
    results=[]
    for h,y in zip(H,Y):
        hh=q(np.block([[h.real,-h.imag],[h.imag,h.real]]));yy=q(np.concatenate((y.real,y.imag),axis=0));n=hh.shape[1];U=n//2
        one=q([[1.]]);two=q([[2.]]);l=q([[lam]]);zero=q([[0.]]);P=q(np.zeros((n,n)))
        for j in range(U):
            diag=op(np.add,dot(hh[:,j:j+1].T,hh[:,j:j+1]),l);v=op(np.divide,one,diag)
            P[j,j]=v[0,0];P[U+j,U+j]=v[0,0]
        lo=q([[lam*bounds[0]]]);hi=q([[bounds[1]]]);d=op(np.divide,op(np.add,hi,lo),two);c=op(np.divide,op(np.subtract,hi,lo),two);rho=op(np.divide,c,d)
        identity=q(np.eye(n)[:,:U]);X=q(np.zeros((n,U if inverse else yy.shape[1])));prev=X.copy();carry=X.copy()
        for k in range(K):
            if k==0 or (restart and k%restart==0):rho=op(np.divide,c,d);a=op(np.divide,one,d);b=zero
            else:
                den=op(np.subtract,op(np.multiply,two,d),op(np.multiply,c,rho));rr=op(np.divide,c,den);a=op(np.divide,two,den);b=op(np.multiply,rr,rho);rho=rr
            if momentum!=1:b=op(np.multiply,b,q([[momentum]]))
            if inverse:g=op(np.subtract,identity,op(np.add,dot(hh.T,dot(hh,X)),op(np.multiply,l,X)))
            else:
                R=op(np.subtract,yy,dot(hh,X));g=op(np.subtract,dot(hh.T,R),op(np.multiply,l,X))
            v=dot(P,g)
            delta=op(np.add,op(np.multiply,a,v),op(np.multiply,b,op(np.subtract,X,prev)))
            if compensated:
                change=op(np.subtract,delta,carry);new=op(np.add,X,change);carry=op(np.subtract,op(np.subtract,new,X),change)
            else:new=op(np.add,X,delta)
            prev,X=X,new
        re,im=X[:U],X[U:];neg=op(np.subtract,q(np.zeros_like(im)),im)
        results.append(q(np.block([[re,neg],[im,re]])))
    return np.stack(results)
