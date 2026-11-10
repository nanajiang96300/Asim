"""Reconstruct sealed baseline on identical fresh holdout; no new Asim cycles."""
from pathlib import Path
import json,sys,time,gc
import numpy as np
REPO=Path('/home/nanajiang/Asim');sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from verification.communication.qpsk_link import Replayer
from verification.planning.channel_study import payload
from verification.contracts.mmse import MMSETask
from verification.profiles.numeric import bit_equal
from verification.reconstruction.stable_hb_campaign import checks
from optimization_ledger import file_sha,seal,append
exp=REPO/'experiments/optimization/bri/inner-0001';out=REPO/'results/optimization/bri/inner-0001';c=json.loads((exp/'candidate.json').read_text());basepath=exp.parent/'baseline-v1.json';assert file_sha(basepath)==c['baseline_manifest_sha256'];base=json.loads(basepath.read_text());hold=json.loads((out/'holdout/result.json').read_text());assert hold['status']=='INNER_CDLB_RECONSTRUCTION_CHECKED'
begin=time.perf_counter();rows=[]
for case in base['cases']:
 p=Path(case['result']);assert file_sha(p)==case['result_sha256'];r=checks(p);t=MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=case['regularization']);executor=Replayer(p,r,t)
 for hs,fs in zip(c['channel_seeds'],c['frame_seeds']):
  with np.load(out/f'dataset/H{hs}.npz') as a:H=a['H']
  data=payload(t,H,hs,fs);tic=time.perf_counter();v=executor.execute(data);candidate=out/f'holdout/lambda{t.regularization}-H{hs}.npz'
  with np.load(candidate) as a:
   assert bit_equal(v['raw'],a['output']) and np.array_equal(v['target'],a['mmse'])
   assert np.allclose(v['task_errors'],a['task_errors'],rtol=1e-12,atol=1e-12)
  row=dict(regularization=t.regularization,iterations=case['iterations'],channel_seed=hs,frame_seed=fs,channels=96,baseline_and_candidate_bits_equal=True,task_reference_equal=True,phase_timings_seconds=v['phase_timings_seconds'],elapsed_seconds=time.perf_counter()-tic,instructions_executed=v['instructions_executed'],candidate_evidence=seal(candidate));rows.append(row);print(json.dumps(row),flush=True)
 del executor;gc.collect()
result=dict(status='PAIRED_BASELINE_HOLDOUT_BITS_CHECKED',frames=rows,independent_H=192,paired_batch_replays=576,elapsed_seconds=time.perf_counter()-begin,pure_model_thinking_seconds=None,note='same new H/payload for baseline and candidate; sealed baseline real-Asim templates reused, no new cycle measurements; identical bits imply identical BER/SER/EVM on this cohort')
p=out/'paired_baseline.json'
with p.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
append(exp.parent/'ledger.jsonl','inner-0001-paired-baseline','paired_baseline_checked',dict(candidate_id=c['candidate_id'],baseline_id=c['baseline_id'],status=result['status'],paired_batch_replays=576,evidence=[seal(p)]))
print(json.dumps({k:v for k,v in result.items() if k!='frames'}))
