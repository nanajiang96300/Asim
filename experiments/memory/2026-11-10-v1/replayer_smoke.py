from pathlib import Path
import json,time
import numpy as np
from verification.communication.qpsk_link import Replayer
from verification.planning.channel_study import payload
from verification.contracts.mmse import MMSETask
from verification.profiles.numeric import bit_equal
from verification.asim.runner import file_hash
root=Path('/home/nanajiang/Asim');source=root/'results/fixed_hb_native/2026-11-09-v2';p=source/'asim-lambda1.0/result.json';r=json.loads(p.read_text());start=time.perf_counter()
for a in r['artifacts']:
 q=p.parent/a['path'];assert q.stat().st_size==a['bytes'] and file_hash(q)==a['sha256']
t=MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=1.)
with np.load(source/'dataset/H59001.npz') as a:H=a['H']
data=payload(t,H,59001,59101);v=Replayer(p,r,t).execute(data)
with np.load(source/'holdout/lambda1.0-H59001.npz') as a:
 assert bit_equal(v['raw'],a['output']) and np.array_equal(v['target'],a['mmse'])
assert v['precision_bit_equal'] and v['numeric_passed']
result=dict(status='QPSK_REPLAYER_MEMORY_READER_CHECKED',source=str(p),batch_size=96,output_bits_equal=True,task_reference_equal=True,phase_timings_seconds=v['phase_timings_seconds'],elapsed_seconds=time.perf_counter()-start,scope='sealed actual Asim and reused CDL input; no new independent sample or cycle',pure_model_thinking_seconds=None)
(root/'results/memory/2026-11-10-v3/replayer_smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
