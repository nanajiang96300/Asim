import os
from pathlib import Path
from copy import deepcopy
import json,sys,subprocess
import numpy as np
from verification.contracts.mmse import MMSETask
from verification.contracts.core import ScreenPolicy
from verification.reconstruction.runner import evaluate
from verification.reconstruction.binding import validate_actual
from verification.reconstruction.source_views import execute_batch
from verification.profiles.numeric import bit_equal
from verification.reference.stable_hb import simulate
from verification.asim.runner import file_hash
root=Path('/home/nanajiang/Asim');run=Path(os.environ.get('HB_RUN_DIR',str(root/'results/fixed_hb_native/2026-11-09-v2')));dest=run/'controls';dest.mkdir(exist_ok=False)
p=run/'asim-lambda1.0';r=json.loads((p/'result.json').read_text());config=json.loads((p/'config.json').read_text())
record=json.loads((p/'instructions.jsonl').read_text());dispatch=[json.loads(l) for l in (p/'dispatch.jsonl').read_text().splitlines()]
with np.load(p/'inputs.npz',allow_pickle=False) as a:inputs={k:a[k] for k in a.files}
with np.load(p/'references.npz',allow_pickle=False) as a:expected=a['algorithm_block']
task=MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=1.)
h,groups,bound,_=validate_actual(record,dispatch,task,r['algorithm'],inputs,config)
rows=[]
for name in ('missing_semantics','wrong_sign','missing_feedback','wrong_quadrant'):
    program=deepcopy(groups[0])
    suffix={'missing_semantics':'iteration0.HX','wrong_sign':'iteration0.data_residual.add','missing_feedback':'iteration0.state','wrong_quadrant':'output.im'}[name]
    ins=next(i for i in program if i['id'].endswith('.'+suffix))
    if name=='missing_semantics':ins.pop('semantic')
    elif name=='wrong_sign':ins['opcode']=11;ins['semantic']['opcode']=11
    elif name=='missing_feedback':ins['src_addrs'][0]=ins['dest_addr'];ins['semantic']['src'][0]['base']=ins['dest_addr']
    else:ins['semantic']['src'][0]['row']=0
    got=execute_batch(h,program,bound,0,expected[0]);assert got['status']!='STRUCTURE_MATCH',name
    rows.append(dict(control=name,status='EXPECTED_REJECTION',replay=got,actual_asim_fault_binary=False))
with np.load(p/'channel.npz',allow_pickle=False) as a:H=a['H'][:1]
for sign in (0.,-0.):
    Y=np.zeros((1,64,8),complex);Y.real.fill(sign);Y.imag.fill(sign)
    inputs2={k:v.copy() for k,v in inputs.items()};inputs2['Y'].fill(sign)
    _,_,bound2,_=validate_actual(record,dispatch,task,r['algorithm'],inputs2,config)
    exp=simulate(H,Y,1.,16)['output_block'][0]
    got=execute_batch(h,groups[0],bound2,0,exp,return_output=True);actual=got.pop('_output',None)
    assert got['status']=='STRUCTURE_MATCH' and actual is not None and bit_equal(actual,exp)
    rows.append(dict(control='negative_zero_rhs' if np.signbit(sign) else 'zero_rhs',status='BIT_EXACT',replay=got))
# A complete second real simulation, using the identical frozen candidate.
c=json.loads((root/'experiments/fixed_hb_native/2026-11-09-v2/contract.json').read_text())
import os;os.environ['LD_LIBRARY_PATH']=c['library_dir']
repeat=evaluate('bri_stable_hb_fp16_v1',task,16,57001,Path(c['binary']),config,Path(c['cwd']),dest/'repeat')
assert repeat['status']=='MODELED_NUMERIC_CHECKED'
for name in ('instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','outputs.npy'):
    assert file_hash(p/name)==file_hash(dest/'repeat'/name),name
rows.append(dict(control='real_repeat',status='IDENTICAL',cycles=repeat['asim']['finish_cycle'],files=['instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','outputs.npy']))
screen=evaluate('bri_stable_hb_fp16_v1',task,16,57001,Path(c['binary']),config,Path(c['cwd']),dest/'performance_reject',ScreenPolicy(r['asim']['finish_cycle'],1.01,10),r['comparison_identity'])
assert screen['status']=='SKIPPED_NOT_COMPETITIVE' and screen['numeric_status']=='NOT_RUN'
assert all(not (dest/'performance_reject'/n).exists() for n in ('numerics_started.json','references.npz','outputs.npy'))
rows.append(dict(control='real_performance_reject',status=screen['status'],cycles=screen['asim']['finish_cycle'],numeric_status=screen['numeric_status']))
negative=dest/'bad-sha-output';cmd=[sys.executable,'-m','verification.reconstruction.stable_hb_campaign','--contract',str(root/'experiments/fixed_hb_native/2026-11-09-v2/contract.json'),'--sha256','0'*64,'--output',str(negative),'--stage','asim']
proc=subprocess.run(cmd,cwd=Path(c['cwd']).parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
assert proc.returncode!=0 and not negative.exists() and 'approved campaign SHA changed' in proc.stdout
(dest/'bad-sha.log').write_text(proc.stdout);rows.append(dict(control='real_CLI_wrong_SHA',status='PREFLIGHT_REJECTED',output_created=False))
(dest/'result.json').write_text(json.dumps(dict(status='CONTROLS_CHECKED',rows=rows,score=None),indent=2));print(json.dumps(rows),flush=True)
