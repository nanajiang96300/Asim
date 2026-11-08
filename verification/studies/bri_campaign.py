"""Bounded independent BRI inverse/solve study on real Sionna CDL-B snapshots."""
import argparse,csv,json,math,platform,time
from pathlib import Path
import numpy as np
from verification.studies.bri_math import system,store,iterate,enclosure,relative,root_product
from verification.channels.cdlb import SPEC,generate,provenance
from verification.contracts.core import Invalid
from verification.reconstruction.daily import sha

FILES=['verification/studies/bri_math.py','verification/studies/bri_campaign.py','verification/channels/cdlb.py','verification/contracts/core.py','scripts/aaa/bj_inverse.py']

def sources(repo):return {p:sha(repo/p) for p in FILES}

def guard(c,repo):
    if c['schema']!='bri-principles-v1' or c['sources']!=sources(repo) or c['channel_spec']!=SPEC or c['toolchain']!=provenance():raise Invalid('frozen math/channel/toolchain changed')
    if c['batch']!=96 or c['channel_seeds']!=[48001,48002] or c['complex_blocks']!=[2,4,8] or c['regularizations']!=[1.,.1,.01] or c['steps']!=256:raise Invalid('unapproved study extent')
    if c['max_scenarios']!=108 or c['max_wall_seconds']!=3600:raise Invalid('unapproved study budget')


def measure(X,Atrue,Xtrue,Amodel,Xmodel,rhs,truth,model_rhs,precision,bits):
    good=np.isfinite(X).all(axis=(1,2));n=int(good.sum());r=dict(finite_channels=n,failed_channels=len(X)-n)
    if not n:return {**r,'inverse_error_max':None,'inverse_error_median':None,'inverse_model_error_max':None,'residual_max':None,'task_error_max':None,'ber':None,'passed':False}
    value=X[good].astype(np.complex128);ref=Xtrue[good];I=np.broadcast_to(np.eye(X.shape[-1]),value.shape)
    ie=relative(value,ref);me=relative(value,Xmodel[good]);res=relative(Atrue[good]@value,I)
    xhat=store(store(value,precision)@model_rhs[good],precision);finite=np.isfinite(xhat).all(axis=(1,2))
    if not finite.all():return {**r,'inverse_error_max':float(ie.max()),'inverse_error_median':float(np.median(ie)),'inverse_model_error_max':float(me.max()),'residual_max':float(res.max()),'task_error_max':None,'ber':None,'passed':False}
    te=relative(xhat,truth[good]);rx=np.stack([xhat.real>=0,xhat.imag>=0],axis=-1)
    return dict(**r,inverse_error_max=float(ie.max()),inverse_error_median=float(np.median(ie)),inverse_model_error_max=float(me.max()),residual_max=float(res.max()),task_error_max=float(te.max()),ber=float(np.mean(rx!=bits[good])),passed=n==len(X) and max(ie.max(),te.max())<=.02)


def old_helper_check(H,out):
    from scripts.aaa.bj_inverse import BJacobiConfig,block_jacobi_inverse
    A,P,ev=system(H[:1],.01,2,'fp64');lo,hi=enclosure(ev);rows=[]
    for k in (4,8,16,32):
        cfg=BJacobiConfig(block_size=2,num_layers=k,use_fp16_quant=False,adaptive_omega=False,chebyshev_bmin=float(lo[0]),chebyshev_bmax=float(hi[0]))
        old=block_jacobi_inverse(A[0],cfg)[None];new=list(iterate(A,P,ev,k))[-1][1];root=root_product(A,P,k,lo,hi)
        rows.append(dict(K=k,old_vs_stable_relative=float(relative(old,new)[0]),root_vs_stable_relative=float(relative(root,new)[0]),old_inverse_error=float(relative(old,np.linalg.inv(A))[0]),stable_inverse_error=float(relative(new,np.linalg.inv(A))[0]),scope='one FP64 sample; explicit enclosing bounds; no old FP16 or Asim certification'))
    (out/'old_helper_check.json').write_text(json.dumps(rows,indent=2)+'\n')


def run(c,repo,out):
    guard(c,repo)
    if out.exists():raise Invalid('refusing existing principle study')
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();chunks=[];data_rows=[]
    for seed in c['channel_seeds']:
        d=generate(seed,c['batch']);p=out/f'cdlb-{seed}.npz';np.savez_compressed(p,**d);chunks.append(d['H'])
        data_rows.append(dict(seed=seed,path=str(p),bytes=p.stat().st_size,sha256=sha(p),conversion_max_abs_error=float(d['conversion_max_abs_error'])))
    H=np.concatenate(chunks);N,M,U=H.shape;I=np.broadcast_to(np.eye(U),(N,U,U));rng=np.random.default_rng(49001)
    bits=rng.integers(0,2,(N,U,8,2),dtype=np.uint8);s=((2*bits[...,0].astype(float)-1)+1j*(2*bits[...,1].astype(float)-1))/math.sqrt(2)
    noise=(rng.normal(size=(N,M,8))+1j*rng.normal(size=(N,M,8)))/math.sqrt(2)
    np.savez_compressed(out/'payload.npz',bits=bits,symbols=s,base_noise=noise)
    old_helper_check(H,out);rows=[];spectra=[];saved={};scenarios=0
    for lam in c['regularizations']:
        Atrue=H.conj().transpose(0,2,1)@H+lam*I;Xtrue=np.linalg.solve(Atrue,I);Y=H@s+math.sqrt(lam)*noise;rhs=H.conj().transpose(0,2,1)@Y;truth=np.linalg.solve(Atrue,rhs)
        for block in c['complex_blocks']:
            for precision in ('fp64','fp32','half_storage'):
                A,P,ev=system(H,lam,block,precision);lo,hi=enclosure(ev);Xmodel=np.linalg.inv(A.astype(np.complex128));hq=store(H,precision);yq=store(Y,precision);model_rhs=store(hq.conj().transpose(0,2,1)@yq,precision)
                rho=np.max(abs(1-.5*ev),axis=1)
                spectra.append(dict(regularization=lam,complex_block=block,precision=precision,eigen_min=float(ev.min()),eigen_max=float(ev.max()),unstable_half_channels=int(np.sum(rho>=1)),rho_half_max=float(rho.max()),quantized_system_inverse_error_max=float(relative(Xmodel,Xtrue).max()),bound_source='per-sample exact diagnostic eigendecomposition of Lp^H A Lp, expanded 1%; oracle cost not an NPU implementation'))
                variants=[('fixed_half','half','direct'),('stable_stationary','optimal','direct'),('chebyshev_X','chebyshev','direct'),('chebyshev_Y_recover','chebyshev','scaled')]
                for name,method,form in variants:
                    scenarios+=1
                    if scenarios>c['max_scenarios'] or time.perf_counter()-start>c['max_wall_seconds']:raise Invalid('study resource budget exceeded')
                    for k,X in iterate(A,P,ev,c['steps'],precision,method,form):
                        r=dict(regularization=lam,complex_block=block,precision=precision,method=name,K=k,channels=N,**measure(X,Atrue,Xtrue,A,Xmodel,rhs,truth,model_rhs,precision,bits));rows.append(r)
                        if k in (8,16,32,64,128,256):saved[f'l{lam}-b{block}-{precision}-{name}-k{k}']=X
                    print(json.dumps({k:rows[-1][k] for k in ('regularization','complex_block','precision','method','K','inverse_error_max','task_error_max','failed_channels','passed')}),flush=True)
                (out/'progress.json').write_text(json.dumps(dict(completed_scenarios=scenarios,last=rows[-1]),indent=2)+'\n')
    np.savez_compressed(out/'inverse_snapshots.npz',**saved)
    # Literal chapter error and unsafe old bounds are explicit negative diagnostics.
    negative=[]
    for lam in c['regularizations']:
        A,P,ev=system(H,lam,2,'fp64');ref=np.linalg.inv(A);lo,hi=enclosure(ev)
        for name,form,bounds,unsafe in [('literal_chapter_P','literal_P',None,False),('old_clipped_spectrum_diagnostic','scaled',(np.maximum(ev[:,0],.01),np.minimum(ev[:,-1],2.)),True)]:
            for k,X in iterate(A,P,ev,c['steps'],'fp64','chebyshev',form,bounds,unsafe):
                if k not in (1,4,8,16,32,64,128,256):continue
                good=np.isfinite(X).all(axis=(1,2));negative.append(dict(regularization=lam,diagnostic=name,K=k,finite_channels=int(good.sum()),inverse_error_max=float(relative(X[good],ref[good]).max()) if good.any() else None,wrong_target_or_unsafe=True))
    summaries=[]
    for sp in spectra:
        for method in ('fixed_half','stable_stationary','chebyshev_X','chebyshev_Y_recover'):
            rr=[r for r in rows if (r['regularization'],r['complex_block'],r['precision'],r['method'])==(sp['regularization'],sp['complex_block'],sp['precision'],method)]
            first=next((r['K'] for r in rr if r['passed']),None)
            summaries.append(dict(**sp,method=method,first_passing_K=first,final=rr[-1],scope='fixed 192-H dataset, both worst inverse and task error<=2%; not operator acceptance or independent holdout'))
    result=dict(status='BRI_PRINCIPLE_SIMULATION_COMPLETE',source_commit=c['source_commit'],scenarios=scenarios,channels=N,regularizations=c['regularizations'],summaries=summaries,spectra=spectra,negative_diagnostics=negative,channel_artifacts=data_rows,elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,operator_verified=False,score=None)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    guard(c,repo);plot(rows,out)
    return result


def plot(rows,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['C3','C2','C0','C4'];names=['fixed_half','stable_stationary','chebyshev_X','chebyshev_Y_recover']
    for metric in ['inverse_error_max','task_error_max','residual_max']:
        fig,axs=plt.subplots(3,3,figsize=(14,10))
        for i,precision in enumerate(('fp64','fp32','half_storage')):
            for j,lam in enumerate((1.,.1,.01)):
                ax=axs[i,j]
                for method,color in zip(names,colors):
                    rr=[r for r in rows if r['complex_block']==2 and r['precision']==precision and r['regularization']==lam and r['method']==method];vv=[r for r in rr if r[metric] is not None and r['failed_channels']==0]
                    ax.semilogy([r['K'] for r in vv],[min(100.,max(1e-15,r[metric])) for r in vv],color=color,label=method)
                    bad=[r for r in rr if r[metric] is None or r['failed_channels']]
                    if bad:ax.scatter([bad[0]['K']],[100.],marker='x',color=color,s=35)
                ax.axhline(.02,color='black',ls=':',label='2% comparison');ax.set_ylim(1e-15,200);ax.grid(True,which='both',alpha=.2);ax.set_xlabel('K');ax.set_ylabel(metric);ax.set_title(f'{precision}, Es/N0={-10*math.log10(lam):g}dB')
                if i==0 and j==0:ax.legend(fontsize=6)
        fig.suptitle('BRI, complex block=2, 192 static CDL-B matrices; values>100 clipped; x=first nonfinite sample')
        fig.tight_layout();fig.savefig(out/f'{metric}.png',dpi=150);fig.savefig(out/f'{metric}.pdf');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',required=True,type=Path);ap.add_argument('--sha256',required=True);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if sha(a.contract)!=a.sha256:raise Invalid('approved math contract changed')
    c=json.loads(a.contract.read_text());run(c,Path(__file__).resolve().parents[2],a.output)
    if sha(a.contract)!=a.sha256:raise Invalid('contract changed during study')

if __name__=='__main__':main()
