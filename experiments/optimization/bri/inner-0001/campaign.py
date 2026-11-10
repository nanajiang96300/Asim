"""Single frozen instruction-only BRI candidate, trusted evaluator outside worktree."""
from pathlib import Path
import sys,os,json,time,shutil,gc,hashlib,argparse
import numpy as np
REPO=Path('/home/nanajiang/Asim');sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from optimization_ledger import append,seal,actual_run,export,file_sha
from verification.contracts.mmse import MMSETask
from verification.contracts.core import ScreenPolicy
from verification.reconstruction.runner import evaluate
from verification.reconstruction.stable_hb_campaign import source_map,checks
from verification.communication.qpsk_link import Replayer,group_metrics
from verification.communication.qpsk import measurements
from verification.planning.channel_study import payload
from verification.profiles.numeric import bit_equal
OUT=REPO/'results/optimization/bri/inner-0001';EXP=REPO/'experiments/optimization/bri/inner-0001';LEDGER=REPO/'experiments/optimization/bri/ledger.jsonl';BASE=REPO/'experiments/optimization/bri/baseline-v1.json'

def guard(c,approved):
 if file_sha(EXP/'candidate.json')!=approved:raise ValueError('approved candidate manifest SHA changed')
 for key in ['minimum_speedup','maximum_speedup','max_wall_seconds','min_free_bytes','max_output_bytes']:
  if c[key]!={'minimum_speedup':1.001,'maximum_speedup':4,'max_wall_seconds':5400,'min_free_bytes':536870912,'max_output_bytes':3758096384}[key]:raise ValueError('frozen gate/budget differs')
 if c['channel_seeds']!=[61001,61002] or c['frame_seeds']!=[61101,61102]:raise ValueError('new holdout differs')
 if c['evaluator_sources']!=source_map(REPO):raise ValueError('trusted evaluator/reference changed')
 if file_sha(BASE)!=c['baseline_manifest_sha256']:raise ValueError('baseline manifest changed')
 for p,h in c['dependencies'].items():
  if file_sha(p)!=h:raise ValueError('frozen toolchain/dependency/source changed: '+p)
 if c['limits']!=dict(max_ber_increase=.01,max_ser_increase=.02,max_evm_increase=.01,max_outage_probability=.05,family_alpha=.05,min_independent_channels=192):raise ValueError('quality gates differ')
 return json.loads(BASE.read_text())

def run(c,approved,stage):
 base=guard(c,approved);begin=time.perf_counter()
 def budget(needed=0):
  if time.perf_counter()-begin>c['max_wall_seconds'] or shutil.disk_usage(OUT).free<c['min_free_bytes']+needed:raise ValueError('resource budget exceeded before stage')
  if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>c['max_output_bytes']:raise ValueError('artifact budget exceeded')
 os.environ['LD_LIBRARY_PATH']=c['library_dir'];rows=[]
 if stage=='asim':
  for case in base['cases']:
   budget(int(case['raw_bytes']*1.05));l=case['regularization'];k=case['iterations'];dest=OUT/f'asim-lambda{l}'
   if file_sha(case['result'])!=case['result_sha256']:raise ValueError('sealed baseline result changed')
   br=json.loads(Path(case['result']).read_text());config=json.loads(Path(case['result']).with_name('config.json').read_text())
   r=evaluate(c['operator_id'],MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=l),k,57001,Path(c['binary']),config,Path(c['cwd']),dest,ScreenPolicy(case['cycles'],c['minimum_speedup'],c['maximum_speedup']),case['comparison_identity'])
   row=actual_run(dest/'result.json',base,c,f'inner-0001-lambda{l}');row['source_input_and_config_identity_equal']=r['comparison_identity']==br['comparison_identity']
   # Baseline output is .npy, compare stored FP16 values after Asim eligibility.
   if r['status']=='MODELED_NUMERIC_CHECKED' and not bit_equal(np.load(dest/'outputs.npy'),np.load(Path(case['result']).with_name('outputs.npy'))):raise ValueError('instruction-only candidate differs from baseline bits')
   append(LEDGER,row['run_id']+'-evaluated','run_evaluated',row);rows.append(row);(OUT/'asim-progress.json').write_text(json.dumps(rows,indent=2));export(LEDGER,REPO/'experiments/optimization/bri/comparison.csv');print(json.dumps({k:row[k] for k in ['run_id','status','cycles','baseline_cycles','speedup','max_mmse_error']}),flush=True)
   del r;gc.collect()
   if row['status']!='MODELED_NUMERIC_CHECKED':break
  result=dict(status='ASIM_TEMPLATES_CHECKED' if len(rows)==3 and all(r['status']=='MODELED_NUMERIC_CHECKED' for r in rows) else 'ASIM_GATE_FAILED',cases=rows,elapsed_seconds=time.perf_counter()-begin,score=None)
  guard(c,approved);(OUT/'asim-index.json').write_text(json.dumps(result,indent=2));return result
 source=json.loads((OUT/'asim-index.json').read_text())
 if source['status']!='ASIM_TEMPLATES_CHECKED':raise ValueError('Asim gate required before dataset/replay')
 if stage=='dataset':
  from verification.channels.cdlb import generate,SPEC,provenance
  if SPEC!=c['cdlb'] or provenance()!=c['channel_toolchain']:raise ValueError('channel generator changed')
  dest=OUT/'dataset';dest.mkdir(exist_ok=False);entries=[]
  for seed in c['channel_seeds']:
   budget();p=dest/f'H{seed}.npz';np.savez_compressed(p,**generate(seed,96));entries.append(dict(seed=seed,**seal(p)))
  result=dict(status='DATASET_FROZEN',entries=entries,independent_H=192,elapsed_seconds=time.perf_counter()-begin);guard(c,approved);(dest/'index.json').write_text(json.dumps(result,indent=2));return result
 if stage!='replay':raise ValueError('unknown stage')
 data=json.loads((OUT/'dataset/index.json').read_text());dest=OUT/'holdout';dest.mkdir(exist_ok=False);groups=[];frames=[]
 if data['status']!='DATASET_FROZEN' or [e['seed'] for e in data['entries']]!=c['channel_seeds']:raise ValueError('fresh holdout coverage differs')
 for row in source['cases']:
  budget();p=Path(row['result']);r=checks(p);l=row['regularization'];k=row['iterations'];t=MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=l);executor=Replayer(p,r,t);streams=[];errors=[]
  for e,fs in zip(data['entries'],c['frame_seeds']):
   budget()
   if file_sha(e['path'])!=e['sha256']:raise ValueError('fresh H changed')
   with np.load(e['path']) as a:H=a['H']
   d=payload(t,H,e['seed'],fs);tic=time.perf_counter();v=executor.execute(d);m=measurements(v['output'],v['target'],d['bits']);errors.extend(v['task_errors'])
   streams.append(dict(channel_seed=e['seed'],channels=96,bits_per_channel=m['bits_per_channel'],symbols_per_channel=m['symbols_per_channel'],H_hashes=[hashlib.sha256(h.tobytes()).hexdigest() for h in H],candidate=m['candidate'],mmse=m['mmse'],candidate_confusion=m['candidate_confusion'],mmse_confusion=m['mmse_confusion']))
   f=dest/f'lambda{l}-H{e["seed"]}.npz';np.savez_compressed(f,H=H,Y=d['Y'],bits=d['bits'],symbols=d['symbols'],noise=d['noise'],output=v['raw'].astype(np.float16),mmse=v['target'],task_errors=v['task_errors'])
   frames.append(dict(regularization=l,iterations=k,channel_seed=e['seed'],frame_seed=fs,precision_bit_equal=v['precision_bit_equal'],phase_timings_seconds=v['phase_timings_seconds'],elapsed_seconds=time.perf_counter()-tic,**seal(f)));(dest/'frames.json').write_text(json.dumps(frames,indent=2));print(json.dumps(frames[-1]),flush=True)
  g=group_metrics(streams,c['limits'],3);g.update(regularization=l,iterations=k,max_task_error=max(errors),run_id=row['run_id'],source_finish_cycle=r['asim']['finish_cycle'],template_task=t.frozen(),input_domain=c['cdlb'])
  if max(errors)>.02:g['status']='NUMERIC_QUALITY_FAILED'
  groups.append(g);(dest/'progress.json').write_text(json.dumps(groups,indent=2));del executor;gc.collect()
 result=dict(status='INNER_CDLB_RECONSTRUCTION_CHECKED' if all(g['status']=='QPSK_GROUP_CHECKED' for g in groups) and len(groups)==3 and len(frames)==6 else 'INNER_CDLB_GATE_FAILED',groups=groups,frames=frames,independent_H=192,batch_replays=576,elapsed_seconds=time.perf_counter()-begin,pure_model_thinking_seconds=None,score=None)
 guard(c,approved);(dest/'result.json').write_text(json.dumps(result,indent=2))
 for g in groups:
  record=dict(run_id=g['run_id'],candidate_id=c['candidate_id'],status=g['status'],max_task_error=g['max_task_error'],candidate_ber=g['candidate_ber'],mmse_ber=g['mmse_ber'],candidate_ser=g['candidate_ser'],mmse_ser=g['mmse_ser'],evm_delta=g['mean_candidate_raw_evm']-g['mean_mmse_raw_evm'],evidence=[seal(dest/'result.json')])
  append(LEDGER,g['run_id']+'-communication','communication_evaluated',record)
 export(LEDGER,REPO/'experiments/optimization/bri/comparison.csv');return result

if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--stage',choices=['asim','dataset','replay'],required=True);a.add_argument('--sha256',required=True);v=a.parse_args();c=json.loads((EXP/'candidate.json').read_text())
 guard(c,v.sha256) # Wrong approval SHA rejects before any campaign mutation.
 try:r=run(c,v.sha256,v.stage)
 except Exception as e:
  import uuid
  tag=uuid.uuid4().hex;r=dict(status='CAMPAIGN_FAILED',stage=v.stage,error_type=type(e).__name__,reason=str(e),score=None);f=OUT/(v.stage+'-failure-'+tag+'.json');f.write_text(json.dumps(r,indent=2));append(LEDGER,'inner-0001-failure-'+tag,'campaign_failed',dict(candidate_id=c['candidate_id'],**r,evidence=[seal(f)]));print(json.dumps(r),flush=True);raise
 print(json.dumps({k:v for k,v in r.items() if k not in ('cases','groups','frames','entries')}),flush=True)
 if r['status'] not in ('ASIM_TEMPLATES_CHECKED','DATASET_FROZEN','INNER_CDLB_RECONSTRUCTION_CHECKED'):raise SystemExit(1)
