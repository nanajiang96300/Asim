"""Independent per-sample fixed-HB reference; no primitive/replay imports."""
import numpy as np
from verification.contracts.core import Invalid,Unsupported


def simulate(H,Y,lam,K):
    if H.ndim!=3 or not 1<=len(H)<=192 or H.shape[1:]!=(64,16) or Y.shape!=(len(H),64,8):raise Unsupported('independent HB shape/batch domain')
    if type(K) is not int or not 1<=K<=256 or lam not in (1.,.1,.01):raise Invalid('reference HB budget/domain')
    def q(x):
        with np.errstate(over='raise',invalid='raise',under='ignore'):z=np.asarray(x).astype(np.float16)
        if not np.isfinite(z).all():raise Invalid('reference nonfinite FP16')
        return z
    def op(f,*a):
        with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):return q(f(*(v.astype(np.float32) for v in a)))
    def mm(a,b):
        z=np.zeros((a.shape[0],b.shape[1]),np.float32)
        for i in range(a.shape[1]):z=np.add(z,np.multiply(a[:,i:i+1].astype(np.float32),b[i:i+1,:].astype(np.float32),dtype=np.float32),dtype=np.float32)
        return q(z)
    outputs=[];states=[]
    for h,y in zip(H,Y):
        hh=q(np.block([[h.real,-h.imag],[h.imag,h.real]]));yy=q(np.concatenate((y.real,y.imag),axis=0))
        zero=q([[0.]]);one=q([[1.]]);l=q([[lam]]);eight=q([[8.]]);s128=q([[128.]]);infl=q([[1.125]]);safe=q([[.9375]])
        esum=zero.copy();es4=zero.copy();ps2=zero.copy();tr=zero.copy();energy=[];p=[]
        for j in range(16):
            c=hh[:,j:j+1];en=mm(c.T,c)
            if np.any(en<2**-8) or np.any(en>8):raise Unsupported('independent column energy domain')
            pj=op(np.divide,one,op(np.add,en,l));energy.append(en[0,0]);p.append(pj[0,0]);esum=op(np.add,esum,en)
            sc=op(np.divide,en,eight);sq=op(np.multiply,sc,sc);es4=op(np.add,es4,op(np.multiply,sq,sq))
            sc=op(np.divide,pj,s128);ps2=op(np.add,ps2,op(np.multiply,sc,sc));tr=op(np.add,tr,op(np.multiply,pj,en))
        energy=np.asarray(energy,np.float16);p=np.asarray(p,np.float16)
        if np.any(energy<2**-8) or np.any(energy>8) or not 8<=esum[0,0]<=32:raise Unsupported('independent energy domain')
        eu=op(np.multiply,op(np.multiply,op(np.sqrt,op(np.sqrt,es4)),eight),infl)
        pu=op(np.multiply,op(np.multiply,op(np.sqrt,ps2),s128),infl)
        lo=op(np.multiply,op(np.divide,l,op(np.add,eu,l)),safe);hi=op(np.multiply,op(np.add,tr,op(np.multiply,l,pu)),infl)
        ur=op(np.sqrt,hi);lr=op(np.sqrt,lo);den=op(np.add,ur,lr)
        alpha=op(np.multiply,op(np.divide,q([[4.]]),op(np.multiply,den,den)),safe)
        ratio=op(np.divide,op(np.subtract,ur,lr),den);beta=op(np.multiply,ratio,ratio)
        if not 0<=float(beta[0,0])<1 or not float(alpha[0,0])*float(hi[0,0])<2*(1+float(beta[0,0])):raise Unsupported('independent rounded stability')
        weights=q(np.repeat(np.tile(p,2)[:,None],8,axis=1));X=q(np.zeros((32,8)));old=X.copy();carry=X.copy()
        for _ in range(K):
            r=op(np.subtract,yy,mm(hh,X));g=op(np.subtract,mm(hh.T,r),op(np.multiply,l,X));g=op(np.add,g,zero)
            d=op(np.add,op(np.multiply,alpha,op(np.multiply,weights,g)),op(np.multiply,beta,op(np.subtract,X,old)))
            change=op(np.subtract,d,carry);value=op(np.add,X,change);carry=op(np.subtract,op(np.subtract,value,X),change);old,X=X,value
        real,imag=X[:16],X[16:];neg=op(np.subtract,q(np.zeros_like(imag)),imag);outputs.append(q(np.block([[real,neg],[imag,real]])))
        states.append(dict(energies=energy,P_diagonal=p,energy_upper=eu[0,0],preconditioner_upper=pu[0,0],lower=lo[0,0],upper=hi[0,0],alpha=alpha[0,0],beta=beta[0,0],stored_lambda=l[0,0]))
    setup={k:np.stack([s[k] for s in states]) for k in states[0]}
    return dict(output_block=np.stack(outputs),setup=setup)
