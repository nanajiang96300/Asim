"""Independent spectral-polynomial audit of saved FP64 BRI inverse matrices."""
import argparse,json,math
from pathlib import Path
import numpy as np
from verification.reconstruction.daily import sha


def rel(a,b):
    scale=np.maximum(1.,np.maximum(np.max(abs(a),axis=(1,2)),np.max(abs(b),axis=(1,2))))
    return np.linalg.norm((a/scale[:,None,None])-(b/scale[:,None,None]),axis=(1,2))/np.maximum(np.linalg.norm(b/scale[:,None,None],axis=(1,2)),1e-300)


def audit(root):
    r=json.loads((root/'result.json').read_text());H=[]
    for a in r['channel_artifacts']:
        p=Path(a['path'])
        if sha(p)!=a['sha256']:raise ValueError('channel artifact changed')
        with np.load(p,allow_pickle=False) as n:H.append(n['H'])
    H=np.concatenate(H);N,_,U=H.shape;I=np.broadcast_to(np.eye(U),(N,U,U));records=[]
    with np.load(root/'inverse_snapshots.npz',allow_pickle=False) as saved:
        for lam in r['regularizations']:
            A=H.conj().transpose(0,2,1)@H+lam*I
            for block in (2,4,8):
                P=np.zeros_like(A)
                for j in range(0,U,block):P[:,j:j+block,j:j+block]=np.linalg.solve(A[:,j:j+block,j:j+block],np.broadcast_to(np.eye(block),(N,block,block)))
                L=np.linalg.cholesky(P);C=L.conj().transpose(0,2,1)@A@L;eig,V=np.linalg.eigh(C)
                lo=.99*eig[:,0];hi=1.01*eig[:,-1];d=(hi+lo)/2;c=(hi-lo)/2
                for method in ('fixed_half','stable_stationary','chebyshev_X','chebyshev_Y_recover'):
                    for k in (8,16,32,64,128,256):
                        if method=='fixed_half':pol=(1-.5*eig)**k
                        elif method=='stable_stationary':pol=(1-eig/d[:,None])**k
                        else:
                            t=(d[:,None]-eig)/c[:,None]
                            if np.any(abs(t)>1+1e-12):raise ValueError('invalid Chebyshev spectral interval')
                            ac=k*np.arccosh(d/c);inverse_denominator=2*np.exp(-ac)/(1+np.exp(-2*ac))
                            pol=np.cos(k*np.arccos(np.clip(t,-1,1)))*inverse_denominator[:,None]
                        expected=L@((V*((1-pol)/eig)[:,None,:])@V.conj().transpose(0,2,1))@L.conj().transpose(0,2,1)
                        key=f'l{lam}-b{block}-fp64-{method}-k{k}';actual=saved[key]
                        dev=float(rel(actual,expected).max())
                        if not math.isfinite(dev) or dev>1e-8:raise ValueError('independent polynomial mismatch '+key+': '+str(dev))
                        records.append(dict(snapshot=key,worst_relative_difference=dev,individual_matrices=N))
    comparison=[]
    for lam in r['regularizations']:
        A=H.conj().transpose(0,2,1)@H+lam*I
        for domain in ('complex_block2','contiguous_real_block2'):
            if domain=='complex_block2':a=A
            else:a=np.concatenate((np.concatenate((A.real,-A.imag),axis=2),np.concatenate((A.imag,A.real),axis=2)),axis=1)
            p=np.zeros_like(a)
            for j in range(0,a.shape[-1],2):p[:,j:j+2,j:j+2]=np.linalg.inv(a[:,j:j+2,j:j+2])
            lp=np.linalg.cholesky(p);C=lp.conj().transpose(0,2,1)@a@lp;eigen=np.linalg.eigvalsh(C)
            rho=np.max(abs(1-.5*eigen),axis=1)
            comparison.append(dict(regularization=lam,preconditioner=domain,channels=N,unstable_half_channels=int(sum(rho>=1)),max_spectral_radius=float(rho.max()),scope='same H, FP64 nominal spectrum, omega=0.5; not old instruction replay'))
    rows=list(__import__('csv').DictReader((root/'curves.csv').open()))
    if len(rows)!=108*256 or len({(a['regularization'],a['complex_block'],a['precision'],a['method'],a['K']) for a in rows})!=len(rows):raise ValueError('incomplete curve grid')
    return dict(status='INDEPENDENT_BRI_SPECTRAL_AUDIT_PASSED',preconditioner_comparison=comparison,snapshots=len(records),individual_inverse_matrices=sum(r['individual_matrices'] for r in records),max_relative_difference=max(r['worst_relative_difference'] for r in records),method='separate block solve, Hermitian eigendecomposition, closed-form stationary/Chebyshev polynomial; no iterative solver imported',scope='FP64 algebraic correctness only; not finite-precision or instruction certification',records=records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--result-sha256',required=True);a=ap.parse_args()
    if sha(a.root/'result.json')!=a.result_sha256:raise ValueError('approved result changed')
    out=a.root/'independent_audit.json'
    if out.exists():raise ValueError('refusing overwrite')
    r=audit(a.root)
    if sha(a.root/'result.json')!=a.result_sha256:raise ValueError('result changed during audit')
    r['auditor_sha256']=sha(__file__);out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='records'}))

if __name__=='__main__':main()
