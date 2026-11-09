"""Bounded pre-implementation HB calibration/frozen holdout. No Asim or score."""
import argparse,hashlib,json,math,platform,time,shutil
from pathlib import Path
import numpy as np
from verification.contracts.core import Invalid,Unsupported
from verification.studies.stable_hb import HeavyBall,PARAMETERS
from verification.reference.stable_hb import simulate as independent
from verification.channels.cdlb import SPEC,generate,provenance
def file_hash(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

FILES=['verification/studies/stable_hb.py','verification/studies/stable_hb_campaign.py','verification/reference/stable_hb.py','verification/studies/fp16_primitives.py','verification/channels/cdlb.py','verification/profiles/numeric.py','verification/contracts/core.py']
def sources(repo):return {n:file_hash(repo/n) for n in FILES}
def dump(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def error(a,b):return np.linalg.norm(a-b,axis=(1,2))/np.maximum(np.linalg.norm(b,axis=(1,2)),1e-30)
def complex_output(block):
    if block.shape[1:]!=(32,16) or not np.isfinite(block).all():raise Invalid('full output shape/finite')
    if not np.array_equal(block[:,16:,8:],block[:,:16,:8]) or not np.array_equal(block[:,:16,8:],-block[:,16:,:8]):raise Invalid('full output quadrants')
    return block[:,:16,:8].astype(float)+1j*block[:,16:,:8].astype(float)

def guard(c,p,h,repo):
    if file_hash(p)!=h:raise Invalid('approved contract changed')
    if c['schema']!='fp16-fixed-hb-study-v1' or c['algorithm']!=PARAMETERS or c['sources']!=sources(repo):raise Invalid('schema/complete source/algorithm mismatch')
    if c['calibration']!=dict(channel_seed=58001,payload_seed=58101,batch=96) or c['holdout']!=[dict(channel_seed=58002,payload_seed=58102,batch=96),dict(channel_seed=58003,payload_seed=58103,batch=96)]:raise Invalid('frozen split/seed differs')
    if c['lambdas']!=[1.,.1,.01] or c['max_K']!=256 or c['calibration_tolerance']!=.01 or c['acceptance_tolerance']!=.02 or c['selection_rule']!='first_suffix_all_le_1pct_until_256':raise Invalid('quality/K rule changed')
    if c['budget']!=dict(max_wall_seconds=1800,max_output_bytes=268435456,min_free_bytes=536870912):raise Invalid('budget differs')
    if c['python']!=platform.python_version() or c['numpy']!=np.__version__ or c['cdlb']!=SPEC:raise Invalid('toolchain/domain changed')

def resource(c,out,start):
    if time.perf_counter()-start>c['budget']['max_wall_seconds'] or shutil.disk_usage(out.parent).free<c['budget']['min_free_bytes']:raise Invalid('resource budget exceeded')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>c['budget']['max_output_bytes']:raise Invalid('output size budget exceeded')

def dataset(c,root,split):
    if provenance()!=c['channel_toolchain']:raise Invalid('CDL source/version changed')
    out=root/(split+'_inputs');out.mkdir(exist_ok=False);entries=[]
    specs=[c['calibration']] if split=='calibration' else c['holdout']
    for s in specs:
        data=generate(s['channel_seed'],96);rng=np.random.default_rng(s['payload_seed']);bits=rng.integers(0,2,(96,16,8,2),dtype=np.uint8);noise=(rng.standard_normal((96,64,8))+1j*rng.standard_normal((96,64,8)))/math.sqrt(2)
        p=out/f'H{s["channel_seed"]}.npz';np.savez_compressed(p,**data,bits=bits,noise=noise);entries.append(dict(**s,path=str(p.resolve()),sha256=file_hash(p)))
    dump(out/'index.json',dict(schema='hb-inputs-v1',split=split,entries=entries));return entries

def load_inputs(c,root,split):
    d=json.loads((root/(split+'_inputs/index.json')).read_text());specs=[c['calibration']] if split=='calibration' else c['holdout']
    if d['schema']!='hb-inputs-v1' or d['split']!=split or len(d['entries'])!=len(specs) or [{k:e[k] for k in s} for e,s in zip(d['entries'],specs)]!=specs:raise Invalid('exact dataset coverage differs')
    arrays=[]
    for e in d['entries']:
        p=Path(e['path'])
        if p.resolve()!=(root/(split+'_inputs')/f'H{e["channel_seed"]}.npz').resolve() or file_hash(p)!=e['sha256']:raise Invalid('dataset changed')
        with np.load(p,allow_pickle=False) as z:arrays.append({k:z[k] for k in ('H','bits','noise')})
    return {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]},d

def spectral_audit(H,setup):
    rows=[]
    for i,h in enumerate(H):
        hh=np.block([[h.real,-h.imag],[h.imag,h.real]]).astype(np.float16).astype(float);p=np.tile(setup['P_diagonal'][i,:,0].astype(float),2);r=np.sqrt(p)
        lam=float(setup['stored_lambda'][i,0,0]);eig=np.linalg.eigvalsh(r[:,None]*(hh.T@hh+lam*np.eye(32))*r[None,:])
        lo,hi,alpha,beta=[float(setup[k][i,0,0]) for k in ('lower','upper','alpha','beta')]
        # Constant-coefficient endpoint Schur condition covers the entire interval.
        if not 0<lo<=eig[0]<=eig[-1]<=hi or not 0<=beta<1 or not 0<alpha*lo<=alpha*hi<2*(1+beta):raise Invalid('independent spectral enclosure/stability failed')
        roots=np.array([np.roots([1,-(1+beta-alpha*t),beta]) for t in eig])
        rows.append(dict(batch=i,eigen_min=float(eig[0]),eigen_max=float(eig[-1]),lower=lo,upper=hi,alpha=alpha,beta=beta,max_mode_radius=float(np.max(abs(roots)))))
    return rows

def truth(H,Y,lam):
    h=H.conj().transpose(0,2,1);return np.linalg.solve(h@H+lam*np.eye(16),h@Y)

def run(c,contract,approved,root,stage,selection_sha=None):
    repo=Path(__file__).resolve().parents[2];guard(c,contract,approved,repo);root.parent.mkdir(parents=True,exist_ok=True);root.mkdir(exist_ok=True);start=time.perf_counter()
    if stage in ('dataset-holdout','holdout') and (not selection_sha or file_hash(root/'selection.json')!=selection_sha):raise Invalid('approved selection SHA missing/changed')
    if stage=='dataset-calibration':
        entries=dataset(c,root,'calibration');guard(c,contract,approved,repo);resource(c,root,start);return dict(status='CALIBRATION_INPUTS_FROZEN',entries=entries)
    if stage=='dataset-holdout':
        sel=selection(c,root,approved);entries=dataset(c,root,'holdout');guard(c,contract,approved,repo);resource(c,root,start)
        if file_hash(root/'selection.json')!=selection_sha:raise Invalid('selection changed during generation')
        return dict(status='HOLDOUT_INPUTS_FROZEN',selection_sha256=selection_sha,entries=entries)
    split='calibration' if stage=='calibrate' else 'holdout';data,index=load_inputs(c,root,split);H,bits,noise=[data[k] for k in ('H','bits','noise')]
    sym=((2*bits[...,0].astype(float)-1)+1j*(2*bits[...,1].astype(float)-1))/math.sqrt(2)
    dest=root/split;dest.mkdir(exist_ok=False);rows=[];selected=selection(c,root,approved) if split=='holdout' else None
    for case,lam in enumerate(c['lambdas']):
        Y=H@sym+math.sqrt(lam)*noise;target=truth(H,Y,lam);solver=HeavyBall(H,Y,lam);spectrum=spectral_audit(H,solver.setup);curves=[];tic=time.perf_counter();K=256 if split=='calibration' else selected['cases'][case]['K']
        for k,actual,counts in solver.iterate(K):
            if k%16==0:resource(c,root,start)
            errs=error(complex_output(actual),target);curves.append(dict(K=k,max_error=float(errs.max()),median_error=float(np.median(errs))))
        final=complex_output(actual);err=error(final,target);probe=None
        if split=='calibration':
            suffix=np.maximum.accumulate(np.array([r['max_error'] for r in curves])[::-1])[::-1];possible=np.flatnonzero(suffix<=.01);probe=int(possible[0]+1) if len(possible) else None
            # Save selected output by one fresh execution of the same fixed model; no additional parameter choice.
            if probe is not None:
                replay=HeavyBall(H,Y,lam)
                for _,chosen,_ in replay.iterate(probe):pass
                np.savez_compressed(dest/f'case{case}-selected.npz',output=chosen,**solver.setup)
        else:
            independent_start=time.perf_counter();ref=independent(H,Y,lam,K);independent_seconds=time.perf_counter()-independent_start
            if actual.dtype!=np.float16 or not np.array_equal(actual.view(np.uint16),ref['output_block'].view(np.uint16)):raise Invalid('independent final FP16 output bits differ')
            for name,v in solver.setup.items():
                if not np.array_equal(v.reshape(len(H),-1).view(np.uint16),ref['setup'][name].reshape(len(H),-1).view(np.uint16)):raise Invalid('independent coefficient/bound state differs: '+name)
        p=dest/f'case{case}.npz';np.savez_compressed(p,output=actual,task_errors=err,**solver.setup)
        row=dict(lambda_=lam,K=K,channels=len(H),first_2pct_K=next((r['K'] for r in curves if r['max_error']<=.02),None),first_1pct_K=next((r['K'] for r in curves if r['max_error']<=.01),None),selected_suffix_K=probe,
                 max_task_error=float(err.max()),failed_channels=int(np.sum(err>.02)),curves=curves,spectrum=spectrum,primitive_work=counts,seconds=time.perf_counter()-tic,
                 precision_bit_equal=True if split=='holdout' else None,independent_reference_seconds=independent_seconds if split=='holdout' else None)
        if split=='holdout':
            cb=np.stack((final.real>=0,final.imag>=0),axis=-1);tb=np.stack((target.real>=0,target.imag>=0),axis=-1)
            row.update(candidate_bit_errors=int(np.count_nonzero(cb!=bits)),mmse_bit_errors=int(np.count_nonzero(tb!=bits)),bits=bits.size)
        rows.append(row);dump(dest/f'case{case}.json',row);dump(dest/'progress.json',rows);print(json.dumps({k:v for k,v in row.items() if k not in ('curves','spectrum','primitive_work')}),flush=True)
    guard(c,contract,approved,repo)
    if split=='holdout' and file_hash(root/'selection.json')!=selection_sha:raise Invalid('selection changed during holdout')
    if index!=json.loads((root/(split+'_inputs/index.json')).read_text()):raise Invalid('input index changed during execution')
    status=('CALIBRATION_SELECTED' if all(r['selected_suffix_K'] is not None for r in rows) else 'CALIBRATION_FAILED') if split=='calibration' else ('FIXED_HB_NUMERIC_HOLDOUT_CHECKED' if all(r['max_task_error']<=.02 for r in rows) else 'FIXED_HB_HOLDOUT_FAILED')
    r=dict(status=status,cases=rows,seconds=time.perf_counter()-start,contract_sha256=approved,input_index_sha256=file_hash(root/(split+'_inputs/index.json')),score=None,asim_run=False,operator_verified=False,pure_model_thinking_seconds=None)
    dump(dest/'result.json',r);return r

def selection(c,root,approved=None):
    path=root/'selection.json';s=json.loads(path.read_text());cal=root/'calibration/result.json'
    result=json.loads(cal.read_text())
    if approved is not None and result['contract_sha256']!=approved:raise Invalid('calibration belongs to another contract')
    if result['status']!='CALIBRATION_SELECTED' or len(result['cases'])!=3 or [r['lambda_'] for r in result['cases']]!=c['lambdas']:raise Invalid('calibration case coverage/status differs')
    expected=[dict(lambda_=lam,K=r['selected_suffix_K']) for lam,r in zip(c['lambdas'],result['cases'])]
    if s['schema']!='fixed-hb-selection-v1' or s['calibration_result_sha256']!=file_hash(cal) or s['cases']!=expected or len(expected)!=3 or any(type(a['K']) is not int for a in expected):raise Invalid('frozen selection/calibration binding mismatch')
    return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--selection-sha256');p.add_argument('--stage',choices=['dataset-calibration','calibrate','dataset-holdout','holdout'],required=True);a=p.parse_args();c=json.loads(a.contract.read_text());r=run(c,a.contract,a.sha256,a.output,a.stage,a.selection_sha256)
    print(json.dumps({k:v for k,v in r.items() if k not in ('cases','entries')}));return 1 if r['status'] in ('CALIBRATION_FAILED','FIXED_HB_HOLDOUT_FAILED') else 0
if __name__=='__main__':raise SystemExit(main())
