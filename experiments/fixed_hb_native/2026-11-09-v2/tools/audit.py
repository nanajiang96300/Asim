"""Primary-agent deterministic final check, independently recalculated task/modem.

Not a second independent-agent certificate. No candidate/core/reference mutation.
Only the existing independent algorithm reference is used for FP16 bit checks.
"""
from pathlib import Path
import json,hashlib,time,math
import numpy as np
from verification.reference.stable_hb import simulate
from verification.reconstruction.source_views import execute_batch
from verification.reconstruction.daily import load_runtime
repo=Path('/home/nanajiang/Asim');run=repo/'results/fixed_hb_native/2026-11-09-v2';start=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def equal(a,b):return np.array_equal(np.asarray(a,np.float16).view(np.uint16),np.asarray(b,np.float16).view(np.uint16))
def decode(v):
 assert v.shape[1:]==(32,16)
 return v[:,:16,:8].astype(float)+1j*v[:,16:,:8].astype(float)
def target(H,Y,l):
 ht=H.conj().transpose(0,2,1)
 return np.linalg.solve(ht@H+l*np.eye(16),ht@Y)
def errs(a,b):return np.linalg.norm(a-b,axis=(1,2))/np.linalg.norm(b,axis=(1,2))
def first_batch(p):
 # Incremental top-level decode; retain only header and batch0 instructions.
 decoder=json.JSONDecoder();buffer=''
 with p.open() as f:
  while '"header":' not in buffer:buffer+=f.read(65536)
  pos=buffer.index('"header":')+len('"header":')
  while True:
   try:header,end=decoder.raw_decode(buffer,pos);break
   except json.JSONDecodeError:buffer+=f.read(65536)
  while '"instructions":[' not in buffer[end:]:buffer+=f.read(65536)
  pos=buffer.index('"instructions":[',end)+len('"instructions":[');out=[]
  while True:
   buffer=buffer[pos:].lstrip(' ,\n\r\t');pos=0
   while True:
    try:v,end=decoder.raw_decode(buffer);break
    except json.JSONDecodeError:
     more=f.read(65536)
     if not more:raise ValueError('incomplete instruction stream')
     buffer+=more
   if v['batch']!=0:return header,out
   out.append(v);pos=end
c=json.loads((repo/'experiments/fixed_hb_native/2026-11-09-v2/contract.json').read_text())
for name,h in c['sources'].items():assert sha(repo/name)==h
for name,h in c['dependencies'].items():assert sha(Path(name))==h
spec=load_runtime(repo/'orchestrator/codex_runtime.json',repo);templates=[]
for case in c['cases']:
 l=case['regularization'];K=case['iterations'];p=run/f'asim-lambda{l}';r=json.loads((p/'result.json').read_text());a=json.loads((p/'asim.json').read_text())
 assert r['status']=='MODELED_NUMERIC_CHECKED' and r['schedule_status']=='MODEL_RW_CHECKED_CONSERVATIVE' and not r['failed_batches'] and a['returncode']==0 and a['finish_cycle']==r['asim']['finish_cycle']
 with np.load(p/'channel.npz') as z:H=z['H'];Y=z['Y']
 with np.load(p/'references.npz') as z:expected=z['algorithm_block'];mmse=z['task']
 output=np.load(p/'outputs.npy');assert equal(output,expected)
 ref=target(H,Y,l);assert np.array_equal(ref,mmse)
 e=errs(decode(output),ref);assert np.allclose(e,[b['task_error'] for b in r['batches']],rtol=1e-12,atol=1e-12)
 templates.append(dict(regularization=l,K=K,channels=96,cycles=a['finish_cycle'],max_error=float(e.max()),all_saved_reference_bits_equal=True))
# Actual K174 batch0, readonly capture; no independent logger or memory filling.
p=run/'asim-lambda0.01';h,program=first_batch(p/'instructions.jsonl');assert len(program)==591+46*174
with np.load(p/'inputs.npz') as a:inputs={k:a[k] for k in a.files}
with np.load(p/'channel.npz') as a:H=a['H'][:1];Y=a['Y'][:1]
ind=simulate(H,Y,.01,174);setup=ind['setup'];bound=[inputs[t['name'].split('.')[-1]] for t in h['inputs']]
views={}
for key,suffix in {'alpha':'coefficient.alpha_safety','beta':'coefficient.beta','lower':'bound.lower.safety','upper':'bound.upper.inflate','energy_upper':'bound.energy.inflate','preconditioner_upper':'bound.P.inflate'}.items():
 ins=next(i for i in program if i['id'].endswith('.'+suffix));views[key]=ins['semantic']['dst']
pview=dict(next(i for i in program if i['id'].endswith('.allocateP'))['semantic']['dst']);pview.update(rows=32,cols=8);views['expanded_P']=pview
setup_index=next(j for j,i in enumerate(program) if i['id'].endswith('.coefficient.beta'));last_index=next(j for j,i in enumerate(program) if i['id'].endswith('.iteration173.state'))
captures={j:{f'{j}:{k}':v for k,v in views.items()} for j in (setup_index,last_index)}
got=execute_batch(h,program,bound,0,ind['output_block'][0],return_output=True,captures=captures)
assert got['status']=='STRUCTURE_MATCH' and not got['cost_issues'] and equal(got['_output'],ind['output_block'][0])
for j in captures:
 for key in views:
  v=got['_observations'][f'{j}:{key}'];assert v['status']=='CAPTURED'
  expected=np.repeat(np.tile(setup['P_diagonal'][0],2)[:,None],8,axis=1) if key=='expanded_P' else setup[key][0].reshape(1,1)
  assert equal(v['value'],expected),(j,key)
constant_bases={v['base'] for v in views.values()}
assert all(i['opcode']==25 or i['semantic']['dst']['base'] not in constant_bases for i in program if '.iteration' in i['id'])
# Fresh CDL: independently regenerate modem/noise, solve task, nearest-point counts.
result=json.loads((run/'holdout/result.json').read_text());assert result['status']=='FIXED_HB_CDLB_RECONSTRUCTION_CHECKED';groups=[]
labels=np.array([[1,1],[1,0],[0,1],[0,0]],np.uint8);points=((2*labels[:,0].astype(float)-1)+1j*(2*labels[:,1].astype(float)-1))/math.sqrt(2)
for group in result['groups']:
 l=group['regularization'];K=group['iterations'];tot_a=tot_b=0;flags=[];errors=[];hashes=[];frame_records=[]
 for frame in [f for f in result['frames'] if f['regularization']==l]:
  p=Path(frame['path']);assert sha(p)==frame['sha256']
  with np.load(p) as a:d={k:a[k] for k in a.files}
  hs,fs=frame['channel_seed'],frame['frame_seed']
  with np.load(run/f'dataset/H{hs}.npz') as a:H=a['H']
  assert np.array_equal(H,d['H']);rng=np.random.default_rng([fs,hs]);bits=rng.integers(0,2,(96,16,8,2),dtype=np.uint8)
  symbols=((2*bits[...,0].astype(float)-1)+1j*(2*bits[...,1].astype(float)-1))/math.sqrt(2)
  noise=math.sqrt(l/2)*(rng.standard_normal((96,64,8))+1j*rng.standard_normal((96,64,8)));Y=H@symbols+noise
  assert all(np.array_equal(x,d[k]) for k,x in [('bits',bits),('symbols',symbols),('noise',noise),('Y',Y)])
  ref=target(H,Y,l);assert np.array_equal(ref,d['mmse']);output=decode(d['output']);e=errs(output,ref);errors.extend(e)
  assert np.allclose(e,d['task_errors'],rtol=1e-12,atol=1e-12)
  ca=labels[np.argmin(np.abs(output[...,None]-points)**2,axis=-1)];cb=labels[np.argmin(np.abs(ref[...,None]-points)**2,axis=-1)]
  ba=ca!=bits;bb=cb!=bits;sa=np.any(ba,axis=-1);sb=np.any(bb,axis=-1)
  be_a=np.count_nonzero(ba,axis=(1,2,3));be_b=np.count_nonzero(bb,axis=(1,2,3));se_a=np.count_nonzero(sa,axis=(1,2));se_b=np.count_nonzero(sb,axis=(1,2))
  evm_a=np.sqrt(np.mean(np.abs(output-symbols)**2,axis=(1,2)));evm_b=np.sqrt(np.mean(np.abs(ref-symbols)**2,axis=(1,2)))
  flags.extend(np.stack(((be_a-be_b)/256>.01,(se_a-se_b)/128>.02,evm_a-evm_b>.01),axis=-1));tot_a+=int(be_a.sum());tot_b+=int(be_b.sum());hashes.extend(hashlib.sha256(x.tobytes()).hexdigest() for x in H)
  fresh=simulate(H[:1],Y[:1],l,K)['output_block'];assert equal(fresh,d['output'][:1]);frame_records.append(dict(channel_seed=hs,fresh_independent_reference_batch=0,bit_equal=True))
 flags=np.asarray(flags);assert not flags.any();assert len(set(hashes))==192
 upper=1-(.05/(8*3))**(1/192)
 assert upper<.05 and all(v['count']==0 and abs(v['upper']-upper)<1e-12 for v in group['outage_bounds'].values())
 assert (tot_a,tot_b)==(group['candidate_bit_errors'],group['mmse_bit_errors']);assert abs(max(errors)-group['max_task_error'])<1e-12
 groups.append(dict(regularization=l,K=K,channels=192,max_error=float(max(errors)),candidate_bit_errors=tot_a,mmse_bit_errors=tot_b,exceedances=[int(x) for x in flags.sum(axis=0)],joint_confidence_upper=upper,frames=frame_records))
summary=dict(status='PRIMARY_DETERMINISTIC_RECHECK_PASSED',templates=templates,cdlb=groups,actual_K174_batch0_instructions=len(program),readonly_setup_and_last_coefficient_bound_P_captures='BIT_EXACT',runtime_files=len(spec['files']),source_files=len(c['sources']),elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,scope='primary-agent deterministic check; independent task/modem calculation and existing independent algorithm reference, not an additional independent-agent certificate; no full huge-event re-audit')
(run/'final_recheck.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
