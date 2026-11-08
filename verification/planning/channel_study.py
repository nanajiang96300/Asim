"""Frozen channel-domain validation and descriptive K/error curves; no score or tuning."""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import fields
from pathlib import Path
import numpy as np
from verification.channels.cdlb import SPEC,generate,normalize,provenance
from verification.communication.qpsk_link import Replayer,frame,group_metrics
from verification.communication.qpsk import measurements
from verification.communication.runner import load_case,check_artifacts
from verification.contracts.core import Invalid
from verification.contracts.mmse import MMSETask,task_reference
from verification.fixtures.rayleigh import real_block
from verification.planning.iterations import assess
from verification.profiles.numeric import policy_for
from verification.reconstruction.daily import load_runtime,sha
from verification.reference.ns import simulate_ns,simulate_ns_fp16_model
from verification.reference.bri import simulate_bri,simulate_bri_fp16_model


def sources(repo):
    paths=[]
    for directory in ('channels','communication','reference'):
        paths+=list((repo/'verification'/directory).glob('*.py'))
    paths += [repo/p for p in ['verification/planning/channel_study.py','verification/planning/iterations.py',
                              'verification/contracts/mmse.py','verification/operators/parameters.py']]
    return {str(p.relative_to(repo)):sha(p) for p in sorted(paths)}


def validate(c):
    keys={'schema','runtime','source_index','evaluator_sources','channel_toolchain','cdlb','channel_seeds','curve_seed','frame_seeds','ks','limits','budget'}
    if set(c)!=keys or c['schema']!='channel-domain-study-v1' or c['cdlb']!=SPEC:raise Invalid('unknown channel study contract/profile')
    if type(c['curve_seed']) is not int or c['curve_seed']<0:raise Invalid('invalid curve seed')
    for name in ('channel_seeds','frame_seeds'):
        v=c[name]
        if not isinstance(v,list) or len(v)!=2 or len(set(v))!=len(v) or any(type(s) is not int or s<0 for s in v):raise Invalid('two distinct channel/frame seeds required')
    if c['curve_seed'] in c['channel_seeds']:raise Invalid('curve and reconstruction holdout overlap')
    from verification.communication.qpsk_link import validate as link_validate
    link_validate(dict(schema='uncoded-qpsk-link-v1',runtime={},source_index={},evaluator_sources={},channel_seeds=c['channel_seeds'],frame_seeds=c['frame_seeds'],limits=c['limits'],budget=dict(max_batch_replays=1,max_output_bytes=1,min_free_bytes=1,max_wall_seconds=1),control_seed=1,control_symbols=1))
    if set(c['ks'])!={'ns','bri'}:raise Invalid('unknown iteration curves')
    for name,maximum in [('ns',32),('bri',256)]:
        ks=c['ks'][name]
        if not isinstance(ks,list) or not ks or ks!=sorted(set(ks)) or any(type(k) is not int or not 1<=k<=maximum for k in ks):raise Invalid('invalid frozen K grid')
    if set(c['budget'])!={'max_wall_seconds','max_batch_replays','max_curve_calls','min_free_bytes','max_output_bytes'} or any(type(v) is not int or v<=0 for v in c['budget'].values()):raise Invalid('resource budget required')


def payload(task,H,hseed,fseed):
    if H.shape!=(task.batch_size,task.nr,task.nt):raise Invalid('external H differs from template tensor shape')
    data=frame(task,hseed,fseed);data['H']=H;data['Y']=H@data['symbols']+data['noise']
    p=policy_for(task.profile)
    data['inputs'].update(H=p.store(np.stack([real_block(h) for h in H])),Y=p.store(np.stack([real_block(y) for y in data['Y']])))
    return data


def guard(c,repo):
    validate(c)
    if c['evaluator_sources']!=sources(repo):raise Invalid('study evaluator changed after freeze')
    if sha(repo/c['runtime']['path'])!=c['runtime']['sha256']:raise Invalid('runtime identity changed')
    runtime=load_runtime(repo/c['runtime']['path'],repo)
    p=repo/c['source_index']['path']
    if sha(p)!=c['source_index']['sha256'] or sha(p.parent/'contract.json')!=c['source_index']['contract_sha256']:raise Invalid('Asim source index changed')
    if provenance()!=c['channel_toolchain']:raise Invalid('CDL generator package/version changed')
    return runtime,json.loads(p.read_text()),json.loads((p.parent/'contract.json').read_text())


def dataset(c,out):
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for seed in [c['curve_seed'],*c['channel_seeds']]:
        start=time.perf_counter();d=generate(seed,96)
        path=out/f'cdlb-{seed}.npz';np.savez_compressed(path,**d)
        rng=np.random.default_rng(seed)
        H=normalize((rng.standard_normal((96,64,16))+1j*rng.standard_normal((96,64,16)))/math.sqrt(128))
        dest=out/f'rayleigh-{seed}.npz';np.savez_compressed(dest,H=H)
        eig=np.linalg.eigvalsh(d['H'].conj().transpose(0,2,1)@d['H'])
        record=dict(seed=seed,split='curve' if seed==c['curve_seed'] else 'reconstruction_holdout',seconds=time.perf_counter()-start,
                    cdlb_path=str(path),cdlb_sha256=sha(path),rayleigh_path=str(dest),rayleigh_sha256=sha(dest),
                    cdlb_gram_min_eigenvalue=float(eig.min()),cdlb_H_condition_max=float(np.linalg.cond(d['H']).max()),
                    conversion_max_abs_error=float(d['conversion_max_abs_error']))
        rows.append(record);(out/'index.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(record),flush=True)
    return rows


def load_H(data,channel,seed):
    rows=json.loads((data/'index.json').read_text());row=next(r for r in rows if r['seed']==seed)
    p=Path(row[channel+'_path'])
    if sha(p)!=row[channel+'_sha256']:raise Invalid('channel fixture changed')
    with np.load(p,allow_pickle=False) as a:return a['H']


def curves(c,task,data,out):
    out.mkdir(parents=True,exist_ok=False);rows=[];calls=0;start=time.perf_counter()
    for channel in ('rayleigh','cdlb'):
        H=load_H(data,channel,c['curve_seed'])
        for lam in (1.,.1,.01):
            t=MMSETask(**{**task,'regularization':lam});d=payload(t,H,c['curve_seed'],c['frame_seeds'][0]);truth=task_reference(d,t)
            eig=np.linalg.eigvalsh(H.conj().transpose(0,2,1)@H+lam*np.eye(t.nt))
            for name,f16,f64 in [('ns',simulate_ns_fp16_model,simulate_ns),('bri',simulate_bri_fp16_model,simulate_bri)]:
                for k in c['ks'][name]:
                    for precision,fn in [('fp16_fp32',f16),('fp64',f64)]:
                        if calls>=c['budget']['max_curve_calls'] or time.perf_counter()-start>c['budget']['max_wall_seconds']:raise Invalid('curve call/time budget exceeded')
                        calls+=1;record=dict(channel=channel,regularization=lam,algorithm=name,K=k,precision=precision,channels=96,
                                           gram_condition_max=float((eig[:,-1]/eig[:,0]).max()))
                        try:
                            errors=assess(fn(H,d['Y'],lam,k)['output'],truth)
                            record.update(status='MEASURED',median_error=float(np.median(errors)),max_error=max(errors),
                                          p95_error=float(np.percentile(errors,95)),passed_2pct=max(errors)<=.02)
                        except (ValueError,Invalid,FloatingPointError,OverflowError) as exc:record.update(status='NUMERIC_REJECTED',reason=str(exc),passed_2pct=False)
                        rows.append(record);(out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(json.dumps(rows[-2:]),flush=True)
    (out/'result.json').write_text(json.dumps(dict(status='DESCRIPTIVE_CURVES_COMPLETE',rows=rows,calls=calls,seconds=time.perf_counter()-start,selection_performed=False),indent=2)+'\n')
    plot_curves(rows,out)


def plot_curves(rows,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=['channel','regularization','algorithm','K','precision','channels','gram_condition_max','status','median_error','max_error','p95_error','passed_2pct','reason']
    with (out/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    for channel in ('rayleigh','cdlb'):
        fig,axs=plt.subplots(2,3,figsize=(14,8),sharey=True)
        for i,name in enumerate(('ns','bri')):
            for j,lam in enumerate((1.,.1,.01)):
                ax=axs[i,j]
                for precision,color in [('fp16_fp32','C0'),('fp64','C1')]:
                    rr=[r for r in rows if r['channel']==channel and r['algorithm']==name and r['regularization']==lam and r['precision']==precision and r['status']=='MEASURED']
                    ax.semilogy([r['K'] for r in rr],[max(r['max_error'],1e-15) for r in rr],'-o',ms=3,color=color,label=precision+' max')
                    ax.semilogy([r['K'] for r in rr],[max(r['median_error'],1e-15) for r in rr],'--',color=color,alpha=.6,label=precision+' median')
                ax.axhline(.02,color='red',ls=':',label='2% numerical threshold');ax.set_title(f'{name.upper()}, Es/N0={-10*math.log10(lam):g} dB')
                ax.set_xlabel('Iterations K');ax.set_ylabel('Relative output error vs FP64 MMSE');ax.grid(True,which='both',alpha=.25)
                if i==0 and j==0:ax.legend(fontsize=7)
        fig.suptitle(f'{channel}: independent algorithm models, 96 fixed channels; no operator timing or K promotion')
        fig.tight_layout();fig.savefig(out/f'{channel}_K_error.png',dpi=160);fig.savefig(out/f'{channel}_K_error.pdf');plt.close(fig)


def replay(c,runtime,index,source_spec,data,out,repo):
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();rows=[];ledger=[];replays=0;cache={}
    expected={(o['operator'],lam) for o in source_spec['algorithms'] for lam in source_spec['noise_variances']}
    if {(s['operator'],s['regularization']) for s in index['cases']}!=expected or len(index['cases'])!=len(expected):raise Invalid('source coverage mismatch')
    count=len(expected)*len(c['channel_seeds'])*len(c['frame_seeds'])*96
    if count>c['budget']['max_batch_replays']:raise Invalid('replay budget exceeded')
    for s in index['cases']:
        if time.perf_counter()-start>c['budget']['max_wall_seconds']:raise Invalid('replay time budget exceeded')
        path=Path(s['result_path']);r=load_case(dict(path=str(path),sha256=s['result_sha256'],role='historical_diagnostic'),runtime,repo)
        r=json.loads(path.read_text());task=MMSETask(**{f.name:r['task'][f.name] for f in fields(MMSETask)})
        declared=MMSETask(**{**source_spec['task'],'regularization':s['regularization']})
        if s['status']!='SOURCE_CHECKED' or task.frozen()!=declared.frozen() or r['operator']!=s['operator'] or r['algorithm'].get('iterations',1)!=s['iterations']:raise Invalid('source task/algorithm mismatch')
        executor=Replayer(path,r,task);streams=[];errors=[];rejected=[]
        for hs in c['channel_seeds']:
            H=load_H(data,'cdlb',hs);cohort=None
            for fs in c['frame_seeds']:
                import shutil
                if shutil.disk_usage(out).free<c['budget']['min_free_bytes'] or time.perf_counter()-start>c['budget']['max_wall_seconds']:raise Invalid('replay disk/time budget exceeded')
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>c['budget']['max_output_bytes']:raise Invalid('replay output budget exceeded')
                key=(task.regularization,hs,fs)
                if key not in cache:
                    d=payload(task,H,hs,fs);target=task_reference(d,task)
                    folder=out/'inputs';folder.mkdir(exist_ok=True);p=folder/f'lambda{task.regularization}-h{hs}-f{fs}.npz'
                    np.savez_compressed(p,**{k:d[k] for k in ('H','Y','bits','symbols','noise')},**{n+'_input':v for n,v in d['inputs'].items()},mmse=target)
                    cache[key]=(d,dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
                d,a=cache[key];replays+=task.batch_size
                try:v=executor.execute(d)
                except (Invalid,ValueError,FloatingPointError,OverflowError) as exc:
                    rejected.append(dict(channel_seed=hs,frame_seed=fs,reason=str(exc),status='REPLAY_REJECTED'));continue
                m=measurements(v['output'],v['target'],d['bits']);errors.extend(v['task_errors'])
                if cohort is None:
                    cohort=dict(channel_seed=hs,channels=96,bits_per_channel=0,symbols_per_channel=0,H_hashes=[hashlib.sha256(h.tobytes()).hexdigest() for h in H],
                                candidate={k:np.zeros_like(vv) for k,vv in m['candidate'].items()},mmse={k:np.zeros_like(vv) for k,vv in m['mmse'].items()},candidate_confusion=np.zeros((4,4),int),mmse_confusion=np.zeros((4,4),int))
                for who in ('candidate','mmse'):
                    for k,vv in m[who].items():cohort[who][k]+=vv
                    cohort[who+'_confusion']+=m[who+'_confusion']
                cohort['bits_per_channel']+=m['bits_per_channel'];cohort['symbols_per_channel']+=m['symbols_per_channel']
                p=out/f'{s["id"]}-h{hs}-f{fs}.npz';np.savez_compressed(p,actual_output_block=v['raw'].astype(np.float16),task_error=np.asarray(v['task_errors']))
                ledger.append(dict(operator=s['operator'],regularization=task.regularization,channel_seed=hs,frame_seed=fs,max_task_error=max(v['task_errors']),precision_bit_equal=v['precision_bit_equal'],input_artifact=a,output_artifact=dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))))
            if cohort is not None:streams.append(cohort)
        g=dict(source=str(path),operator=s['operator'],iterations=s['iterations'],regularization=task.regularization,es_no_db=-10*math.log10(task.regularization),source_finish_cycle=r['asim']['finish_cycle'],input_domain='3gpp_cdl_b_sionna_static_frequency_snapshot',template_task=task.frozen(),rejected_frames=rejected)
        if rejected:g.update(status='REPLAY_REJECTED',max_task_error=max(errors) if errors else None)
        else:
            g.update(group_metrics(streams,c['limits'],len(expected)));g['max_task_error']=max(errors)
            if max(errors)>.02:g['status']='NUMERIC_QUALITY_FAILED'
        rows.append(g);(out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'frame_progress.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps({k:g[k] for k in ('operator','regularization','status','max_task_error')}),flush=True);check_artifacts(path.parent,r)
    if replays!=count:raise Invalid('missing attempted batch coverage')
    guard(c,repo)
    for e in ledger:
        for key in ('input_artifact','output_artifact'):
            p=out/e[key]['path']
            if p.stat().st_size!=e[key]['bytes'] or sha(p)!=e[key]['sha256']:raise Invalid('replay artifact changed')
    passed=all(g['status']=='QPSK_GROUP_CHECKED' for g in rows)
    result=dict(status='CDLB_DOMAIN_CHECKED' if passed else 'CDLB_QUALITY_NOT_VERIFIED',groups=rows,frames=ledger,batch_attempts=replays,
                elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,score=None,
                note='unchanged Asim-accepted matrix templates; explicit external CDL-B H/Y domain; no hardware or complete multi-carrier assertion')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage',choices=['dataset','curves','replay'],required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[2]
    if sha(a.contract)!=a.sha256:raise Invalid('approved contract changed')
    c=json.loads(a.contract.read_text());rt,index,sc=guard(c,repo)
    if a.output.exists() and a.stage=='dataset':raise Invalid('refusing existing study output')
    a.output.mkdir(parents=True,exist_ok=True)
    result=None
    if a.stage=='dataset':dataset(c,a.output/'dataset')
    elif a.stage=='curves':curves(c,sc['task'],a.output/'dataset',a.output/'curves')
    else:result=replay(c,rt,index,sc,a.output/'dataset',a.output/'replay',repo)
    guard(c,repo)
    if sha(a.contract)!=a.sha256:raise Invalid('contract changed during study')
    return 1 if result and result['status']!='CDLB_DOMAIN_CHECKED' else 0


if __name__=='__main__':raise SystemExit(main())
