"""Independent low-memory review. No Asim/build, no full events/JSON loads."""
from pathlib import Path
import json,hashlib,time,gc
import numpy as np
from verification.reference.stable_hb import simulate
from verification.reconstruction.source_views import execute_batch
from verification.profiles.numeric import bit_equal
root=Path('/home/nanajiang/Asim');run=root/'results/optimization/bri/inner-0001';start=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def first_batch(p):
 decoder=json.JSONDecoder();buf=''
 with p.open() as f:
  while '"header":' not in buf:buf+=f.read(65536)
  pos=buf.index('"header":')+len('"header":')
  while True:
   try:header,end=decoder.raw_decode(buf,pos);break
   except json.JSONDecodeError:
    more=f.read(65536)
    if not more:raise ValueError('truncated header')
    buf+=more
  while '"instructions":[' not in buf[end:]:buf+=f.read(65536)
  pos=buf.index('"instructions":[',end)+len('"instructions":[');instructions=[]
  while True:
   buf=buf[pos:].lstrip(' ,\n\r\t')
   while True:
    try:v,end=decoder.raw_decode(buf);break
    except json.JSONDecodeError:
     more=f.read(65536)
     if not more:raise ValueError('truncated instruction')
     buf+=more
   if v['batch']!=0:return header,instructions
   instructions.append(v);pos=end
cpath=root/'experiments/optimization/bri/inner-0001/candidate.json';assert sha(cpath)=='0c4cc1f8b4fd5402370d0f2257a6eebfb719b1a2718f6a8c4699fa0443b4119e';c=json.loads(cpath.read_text())
for name,h in c['evaluator_sources'].items():assert sha(root/name)==h,name
for name,h in c['dependencies'].items():assert sha(name)==h,name
rows=[]
for lam,K in [(1.,16),(.1,53),(.01,174)]:
 p=run/f'asim-lambda{lam}';path=p/'result.json'
 if not path.exists():continue
 r=json.loads(path.read_text());assert r['status']=='MODELED_NUMERIC_CHECKED' and r['schedule_status']=='MODEL_RW_CHECKED_CONSERVATIVE' and not r['failed_batches']
 assert r['algorithm']['iterations']==K and r['task']['regularization']==lam and r['seed']==57001
 receipt=json.loads((p/'asim.json').read_text());assert receipt['status']=='COMPLETED' and receipt['returncode']==0 and receipt['binary_sha256']==c['binary_sha256'] and receipt['finish_cycle']==r['asim']['finish_cycle']
 required={'inputs.npz','references.npz','channel.npz','outputs.npy','config.json','models.json','dispatch.jsonl','asim.json','schedule.json'}
 for a in r['artifacts']:
  if a['path'] in required:
   q=p/a['path'];assert q.stat().st_size==a['bytes'] and sha(q)==a['sha256']
 assert required<={a['path'] for a in r['artifacts']}
 with np.load(p/'channel.npz') as a:H=a['H'];Y=a['Y']
 with np.load(p/'inputs.npz') as a:inputs={k:a[k] for k in a.files}
 with np.load(p/'references.npz') as a:expected=a['algorithm_block'];task=a['task']
 output=np.load(p/'outputs.npy');assert bit_equal(output,expected)
 old=root/f'results/fixed_hb_native/2026-11-09-v2/asim-lambda{lam}'
 baseline=json.loads((old/'result.json').read_text())
 assert baseline['task']==r['task'] and baseline['algorithm']==r['algorithm'] and baseline['comparison_identity']==r['comparison_identity']
 assert bit_equal(output,np.load(old/'outputs.npy'))
 ht=H.conj().transpose(0,2,1);truth=np.linalg.solve(ht@H+lam*np.eye(16),ht@Y);assert np.array_equal(truth,task)
 value=output[:,:16,:8].astype(float)+1j*output[:,16:,:8].astype(float);errors=np.linalg.norm(value-truth,axis=(1,2))/np.linalg.norm(truth,axis=(1,2))
 assert np.allclose(errors,[b['task_error'] for b in r['batches']],rtol=1e-12,atol=1e-12)
 h,program=first_batch(p/'instructions.jsonl');assert len(program)==591+42*K
 assert not any(i['id'].endswith(('.state','.previous')) for i in program)
 by_name={i['id'].split('.',1)[1]:i for i in program}
 regions={v['name']:v for v in h['regions']}
 previous,x,new=(regions[n]['base'] for n in ['previous','X','new']);initial=(previous,x,new)
 for k in range(K):
  prefix=f'iteration{k}.'
  assert by_name[prefix+'HX']['semantic']['src'][1]['base']==x
  assert by_name[prefix+'lambdaX']['semantic']['src'][0]['base']==x
  assert by_name[prefix+'difference.neg']['semantic']['src'][0]['base']==previous
  assert by_name[prefix+'difference.add']['semantic']['src'][0]['base']==x
  assert by_name[prefix+'new']['semantic']['src'][0]['base']==x and by_name[prefix+'new']['semantic']['dst']['base']==new
  assert by_name[prefix+'difference_new.neg']['semantic']['src'][0]['base']==x
  assert by_name[prefix+'difference_new.add']['semantic']['src'][0]['base']==new
  previous,x,new=x,new,previous
 for name in ['output.re','output.im','output.re_copy','output.negative_im']:
  assert by_name[name]['semantic']['src'][0]['base']==x
 assert x=={0:initial[1],1:initial[2],2:initial[0]}[K%3]
 bound=[inputs[t['name'].split('.')[-1]] for t in h['inputs']]
 independent=simulate(H[:1],Y[:1],lam,K)['output_block'][0]
 last=next(j for j,i in enumerate(program) if i['id'].endswith(f'.iteration{K-1}.carry.add'))
 view=lambda base:dict(base=base,row=0,col=0,rows=32,cols=8,transpose=False,conjugate=False)
 got=execute_batch(h,program,bound,0,independent,return_output=True,captures={last:{'logical_current_x':view(x),'logical_previous':view(previous)}})
 assert got['status']=='STRUCTURE_MATCH' and not got['cost_issues'] and bit_equal(got['_output'],independent)
 assert bit_equal(got['_observations']['logical_current_x']['value'],np.concatenate([independent[:16,:8],independent[16:,:8]],axis=0))
 previous_expected=simulate(H[:1],Y[:1],lam,K-1)['output_block'][0]
 assert bit_equal(got['_observations']['logical_previous']['value'],np.concatenate([previous_expected[:16,:8],previous_expected[16:,:8]],axis=0))
 zero_controls=[]
 for sign in (0.,-0.):
  y=np.zeros((1,64,8),complex);y.real.fill(sign);y.imag.fill(sign);zinputs={k:v.copy() for k,v in inputs.items()};zinputs['Y'].fill(sign);zb=[zinputs[t['name'].split('.')[-1]] for t in h['inputs']]
  ref=simulate(H[:1],y,lam,K)['output_block'][0];z=execute_batch(h,program,zb,0,ref,return_output=True);assert z['status']=='STRUCTURE_MATCH' and not z['cost_issues'] and bit_equal(z['_output'],ref)
  zero_controls.append('negative_zero' if np.signbit(sign) else 'positive_zero')
 rows.append(dict(regularization=lam,K=K,cycles=r['asim']['finish_cycle'],baseline_cycles=baseline['asim']['finish_cycle'],source_instruction_count_batch0=len(program),all_96_saved_bits_equal_to_independent_reference_and_baseline=True,max_task_error=float(errors.max()),batch0_new_independent_reference_bits_equal=True,all_iteration_address_roles_verified=True,final_physical_x=x,final_logical_previous_and_current_captures_bit_equal=True,zero_rhs_controls=zero_controls))
 del program,by_name,output;gc.collect()
result=dict(status='INDEPENDENT_DYNAMIC_CASES_CHECKED',complete_templates=len(rows),cases=rows,evaluator_sources_checked=len(c['evaluator_sources']),dependencies_checked=len(c['dependencies']),elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,scope='small artifacts and stream-only actual batch0 extraction; no full events/whole instruction JSON re-audit; independent FP64 task recalculation; fresh independent FP16 ref on batch0 and zeros')
(run/'independent_dynamic_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
