"""Independent task/modem check of completed fresh CDL evidence, no main gates imported."""
from pathlib import Path
import json,hashlib,time,math
import numpy as np
from scipy.stats import beta
from verification.reference.stable_hb import simulate
from verification.profiles.numeric import bit_equal
root=Path('/home/nanajiang/Asim');run=root/'results/optimization/bri/inner-0001';start=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
cpath=root/'experiments/optimization/bri/inner-0001/candidate.json';assert sha(cpath)=='0c4cc1f8b4fd5402370d0f2257a6eebfb719b1a2718f6a8c4699fa0443b4119e';c=json.loads(cpath.read_text())
result=json.loads((run/'holdout/result.json').read_text());assert result['status']=='INNER_CDLB_RECONSTRUCTION_CHECKED'
assert result['independent_H']==192 and result['batch_replays']==576 and len(result['frames'])==6
assert [(g['regularization'],g['iterations']) for g in result['groups']]==[(1.,16),(.1,53),(.01,174)]
dataset=json.loads((run/'dataset/index.json').read_text());assert dataset['status']=='DATASET_FROZEN' and dataset['independent_H']==192
for entry in dataset['entries']:assert sha(entry['path'])==entry['sha256']
labels=np.array([[1,1],[1,0],[0,1],[0,0]],np.uint8);points=((2*labels[:,0].astype(float)-1)+1j*(2*labels[:,1].astype(float)-1))/math.sqrt(2)
rows=[];Hsets=[]
for g in result['groups']:
 lam=g['regularization'];K=g['iterations'];errors=[];be_a=be_b=se_a=se_b=0;flagrows=[];Hhashes=[];frame_checks=[]
 frames=[f for f in result['frames'] if f['regularization']==lam]
 assert len(frames)==2 and [f['channel_seed'] for f in frames]==[61001,61002]
 for frame in frames:
  p=Path(frame['path']);assert sha(p)==frame['sha256']
  with np.load(p) as a:d={k:a[k] for k in a.files}
  hs,fs=frame['channel_seed'],frame['frame_seed'];assert fs=={61001:61101,61002:61102}[hs]
  with np.load(run/f'dataset/H{hs}.npz') as a:H=a['H']
  assert H.shape==(96,64,16) and np.array_equal(H,d['H']);rng=np.random.default_rng([fs,hs]);bits=rng.integers(0,2,(96,16,8,2),dtype=np.uint8)
  symbols=((2*bits[...,0].astype(float)-1)+1j*(2*bits[...,1].astype(float)-1))/math.sqrt(2)
  noise=math.sqrt(lam/2)*(rng.standard_normal((96,64,8))+1j*rng.standard_normal((96,64,8)));Y=H@symbols+noise
  assert all(np.array_equal(v,d[k]) for k,v in [('bits',bits),('symbols',symbols),('noise',noise),('Y',Y)])
  ht=H.conj().transpose(0,2,1);truth=np.linalg.solve(ht@H+lam*np.eye(16),ht@Y);assert np.array_equal(truth,d['mmse'])
  raw=d['output'];assert raw.shape==(96,32,16) and np.isfinite(raw).all();out=raw[:,:16,:8].astype(float)+1j*raw[:,16:,:8].astype(float)
  e=np.linalg.norm(out-truth,axis=(1,2))/np.linalg.norm(truth,axis=(1,2));assert np.allclose(e,d['task_errors'],rtol=1e-12,atol=1e-12);errors.extend(e)
  ca=labels[np.argmin(np.abs(out[...,None]-points)**2,axis=-1)];cb=labels[np.argmin(np.abs(truth[...,None]-points)**2,axis=-1)]
  ba=ca!=bits;bb=cb!=bits;sa=np.any(ba,axis=-1);sb=np.any(bb,axis=-1)
  bae=np.count_nonzero(ba,axis=(1,2,3));bbe=np.count_nonzero(bb,axis=(1,2,3));sae=np.count_nonzero(sa,axis=(1,2));sbe=np.count_nonzero(sb,axis=(1,2))
  eva=np.sqrt(np.mean(np.abs(out-symbols)**2,axis=(1,2)));evb=np.sqrt(np.mean(np.abs(truth-symbols)**2,axis=(1,2)))
  flags=np.stack(((bae-bbe)/256>c['limits']['max_ber_increase'],(sae-sbe)/128>c['limits']['max_ser_increase'],eva-evb>c['limits']['max_evm_increase']),axis=-1)
  flagrows.extend(flags);be_a+=int(bae.sum());be_b+=int(bbe.sum());se_a+=int(sae.sum());se_b+=int(sbe.sum());Hhashes.extend(hashlib.sha256(h.tobytes()).hexdigest() for h in H)
  independent=simulate(H[:1],Y[:1],lam,K)['output_block'];assert bit_equal(independent,raw[:1])
  frame_checks.append(dict(channel_seed=hs,frame_seed=fs,channels=96,regenerated_payload_exact=True,fp64_mmse_exact=True,batch0_fresh_independent_fp16_reference_bit_equal=True))
 assert len(set(Hhashes))==192
 if Hsets:assert set(Hhashes)==Hsets[0]
 else:Hsets.append(set(Hhashes))
 assert abs(max(errors)-g['max_task_error'])<1e-12 and max(errors)<=.02
 assert (be_a,be_b,se_a,se_b)==(g['candidate_bit_errors'],g['mmse_bit_errors'],g['candidate_symbol_errors'],g['mmse_symbol_errors'])
 flags=np.asarray(flagrows);counts=[int(v) for v in flags.sum(axis=0)]+[int(np.any(flags,axis=-1).sum())];alpha=.05/(8*3);bounds={}
 for name,count in zip(['ber','ser','evm','joint'],counts):
  lo=0. if count==0 else float(beta.ppf(alpha,count,192-count+1));hi=1. if count==192 else float(beta.ppf(1-alpha,count+1,192-count))
  actual=g['outage_bounds'][name];assert count==actual['count'] and abs(lo-actual['lower'])<1e-10 and abs(hi-actual['upper'])<1e-10
  assert hi<=c['limits']['max_outage_probability'];bounds[name]=dict(count=count,lower=lo,upper=hi)
 rows.append(dict(regularization=lam,K=K,independent_H=192,max_task_error=float(max(errors)),candidate_bit_errors=be_a,mmse_bit_errors=be_b,candidate_symbol_errors=se_a,mmse_symbol_errors=se_b,outage_bounds=bounds,frames=frame_checks))
assert len(rows)==3
summary=dict(status='INDEPENDENT_FRESH_CDL_TASK_MODEM_CHECKED',groups=rows,independent_H=192,reused_H_across_three_SNR=True,batch_tasks=576,candidate_bits_recalculated=147456,mmse_bits_recalculated=147456,elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,scope='small completed frame artifacts; independent RNG/modem/FP64 MMSE and scipy beta-quantile confidence bounds; independent FP16 reference only six batch0 examples; no second complete 576-batch instruction replay')
(run/'independent_cdl_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
