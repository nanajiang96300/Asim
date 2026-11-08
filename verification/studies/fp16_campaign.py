"""Bounded pre-implementation FP16 primitive screening; no Asim or score."""
import argparse,json,math,platform,time
from pathlib import Path
import numpy as np
from verification.studies.fp16_primitives import Primitives,run,block
from verification.studies.bri_math import relative
from verification.contracts.core import Invalid
from verification.reconstruction.daily import sha

FILES=['verification/studies/fp16_campaign.py','verification/studies/fp16_primitives.py','verification/studies/bri_math.py','verification/profiles/numeric.py','verification/contracts/core.py']
def sources(repo):return {p:sha(repo/p) for p in FILES}


def metrics(X,truth,inv,hh,rhs,kind):
    inverse='inverse' in kind;e=Primitives();value=e.op('GEMM',X,rhs) if inverse else X
    u=inv.shape[-1]//2;r=truth.shape[-1]//2
    def quadrant(v):
        a,b=v[:,:u,:v.shape[-1]//2],v[:,u:,:v.shape[-1]//2];n=v.shape[-1]//2
        scale=np.maximum(1.,np.max(abs(v.astype(float)),axis=(1,2)))
        return np.maximum(np.max(abs(v[:,u:,n:].astype(float)-a),axis=(1,2)),np.max(abs(v[:,:u,n:].astype(float)+b),axis=(1,2)))/scale
    q=quadrant(X);task=relative(value.astype(float),truth)
    ie=relative(X.astype(float),inv) if inverse else None
    good=max(task.max(),q.max(),ie.max() if ie is not None else 0)<=.02
    return dict(task_error_max=float(task.max()),task_error_median=float(np.median(task)),inverse_error_max=float(ie.max()) if ie is not None else None,quadrant_error_max=float(q.max()),passed=bool(good))


def execute(c,repo,out):
    if c['sources']!=sources(repo):raise Invalid('frozen FP16 study code changed')
    if c['schema']!='fp16-primitive-study-v1' or c['steps']>256 or c['steps']<1 or len(c['cases'])>12 or c['max_wall_seconds']>3600:raise Invalid('study limits')
    if out.exists():raise Invalid('refusing existing study')
    hs=[];bits=[];noises=[]
    for a in c['inputs']:
        p=repo/a['path']
        if sha(p)!=a['sha256']:raise Invalid('input changed')
        with np.load(p,allow_pickle=False) as z:
            sl=slice(*a['slice']);hs.append(z['H'][sl]);bits.append(z['bits'][sl]);noises.append(z['noise'][sl])
    H=np.concatenate(hs);bits=np.concatenate(bits);noise=np.concatenate(noises);N=H.shape[0]
    if N%24 or not 24<=N<=192 or H.shape[1:]!=(64,16):raise Invalid('batch/domain budget')
    symbols=((2*bits[...,0].astype(float)-1)+1j*(2*bits[...,1].astype(float)-1))/math.sqrt(2)
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();results=[]
    for cid,s in enumerate(c['cases']):
        lam=s['lambda'];Y=H@symbols+math.sqrt(lam)*noise;Ht=H.conj().transpose(0,2,1);A=Ht@H+lam*np.eye(16)
        inv=block(np.linalg.solve(A,np.broadcast_to(np.eye(16),A.shape)));truth=block(np.linalg.solve(A,Ht@Y))
        e=Primitives();hh=e.const(block(H));yy=e.const(block(Y));rhs=e.op('GEMM',hh.transpose(0,2,1),yy)
        rows=[];snap={};first=None;begin=time.perf_counter();counts=None;failure=None
        try:
            for k,X,counts in run(H,Y,lam,s['block'],s['kind'],c['steps'],tuple(s['bounds'])):
                if time.perf_counter()-start>c['max_wall_seconds']:raise Invalid('wall budget exceeded')
                m=metrics(X,truth,inv,hh,rhs,s['kind']);rows.append(dict(K=k,**m))
                if first is None and m['passed']:first=k;snap['first_passing']=X.copy()
                if k in (16,32,64,128,256):snap['K'+str(k)]=X.copy()
                if k%32==0:(out/'live.json').write_text(json.dumps(dict(case=cid,K=k,first_passing=first,latest=m),indent=2)+'\n')
        except (Invalid,FloatingPointError,ValueError,OverflowError) as exc:failure=type(exc).__name__+': '+str(exc)
        np.savez_compressed(out/f'case{cid:02d}.npz',**snap)
        result=dict(case=cid,**s,channels=N,first_passing_K=first,rows=rows,failure=failure,primitive_work=counts,seconds=time.perf_counter()-begin)
        results.append(result);(out/f'case{cid:02d}.json').write_text(json.dumps(result,indent=2)+'\n');(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['case','lambda','block','kind','channels','first_passing_K','failure','seconds']}),flush=True)
    if c['sources']!=sources(repo):raise Invalid('evaluator changed')
    for a in c['inputs']:
        if sha(repo/a['path'])!=a['sha256']:raise Invalid('input changed during study')
    result=dict(status='FP16_PRIMITIVE_STUDY_COMPLETE',role=c['role'],channels=N,cases=results,elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,operator_verified=False,score=None,numeric_policy='FP16 tensors and coefficients, ordered FP32 temporary arithmetic; no FP32 persistent state',bounds_source='fixed frozen trial constants; no candidate eigenvalue calls; not certified for all channels',sources=c['sources'],python=platform.python_version(),numpy=np.__version__)
    (out/'contract.json').write_text(json.dumps(c,indent=2)+'\n');(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',required=True,type=Path);ap.add_argument('--sha256',required=True);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if sha(a.contract)!=a.sha256:raise Invalid('contract SHA')
    c=json.loads(a.contract.read_text());execute(c,Path(__file__).resolve().parents[2],a.output)
    if sha(a.contract)!=a.sha256:raise Invalid('contract changed')

if __name__=='__main__':main()
