"""Targeted wrong-slot/missing-view and actual performance-rejection controls."""
from pathlib import Path
import sys,json,subprocess,os
from copy import deepcopy
import numpy as np
ROOT=Path('/home/nanajiang/Asim');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from verification.reconstruction.stream import Reader
from verification.reconstruction.source_views import execute_batch
from verification.reconstruction.runner import evaluate
from verification.contracts.mmse import MMSETask
from verification.contracts.core import ScreenPolicy
from optimization_ledger import seal,file_sha,append
raw=ROOT/'results/optimization/bri/inner-0001';exp=ROOT/'experiments/optimization/bri/inner-0001';rows=[]

def batch0(path):
 with path.open() as f:
  r=Reader(f,65536);r.token('{');header=None
  while True:
   key=r.value();r.token(':')
   if key=='instructions':
    assert header is not None;r.token('[');ops=[]
    while True:
     i=r.value()
     if i['batch']!=0:return header,ops
     ops.append(i);r.token(',')
   value=r.value()
   if key=='header':header=value
   r.token(',')
for l in [1.,.1,.01]:
 p=raw/f'asim-lambda{l}';rr=json.loads((p/'result.json').read_text());a=next(a for a in rr['artifacts'] if a['path']=='instructions.jsonl');assert file_sha(p/'instructions.jsonl')==a['sha256']
 h,program=batch0(p/'instructions.jsonl')
 with np.load(p/'inputs.npz') as a:inputs={k:a[k] for k in a.files}
 with np.load(p/'references.npz') as a:expected=a['algorithm_block'][0]
 bound=[inputs[t['name'].split('.')[-1]] for t in h['inputs']]
 for control in ['wrong_final_slot','missing_semantics']:
  bad=deepcopy(program);suffix='output.re' if control=='wrong_final_slot' else 'iteration0.HX';i=next(i for i in bad if i['id'].endswith('.'+suffix));i['semantic']=deepcopy(i['semantic'])
  if control=='wrong_final_slot':
   state_bases=[r['base'] for r in h['regions'] if r['name'] in ['X','previous','new']];assert len(state_bases)==3
   current=i['semantic']['src'][0]['base'];wrong=next(b for b in state_bases if b!=current);i['src_addrs'][0]=wrong;i['semantic']['src'][0]['base']=wrong
  else:i.pop('semantic')
  got=execute_batch(h,bad,bound,0,expected)
  assert got['status']!='STRUCTURE_MATCH'
  rows.append(dict(regularization=l,control=control,status=got['status'],actual_fault_binary=False))
c=json.loads((exp/'candidate.json').read_text());os.environ['LD_LIBRARY_PATH']=c['library_dir'];p=raw/'asim-lambda1.0';r=json.loads((p/'result.json').read_text());config=json.loads((p/'config.json').read_text());dest=raw/'controls-performance-reject'
rejected=evaluate(c['operator_id'],MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=1.),16,57001,Path(c['binary']),config,Path(c['cwd']),dest,ScreenPolicy(r['asim']['finish_cycle'],1.001,4),r['comparison_identity'])
assert rejected['status']=='SKIPPED_NOT_COMPETITIVE' and rejected['numeric_status']=='NOT_RUN'
assert all(not (dest/n).exists() for n in ['numerics_started.json','references.npz','outputs.npy'])
files=['instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl'];assert all(file_sha(p/n)==file_sha(dest/n) for n in files)
rows.append(dict(control='real_performance_reject_and_repeat',status=rejected['status'],cycles=rejected['asim']['finish_cycle'],numeric_status='NOT_RUN',identical_actual_asim_files=files))
ledger=exp.parent/'ledger.jsonl';before=file_sha(ledger)
cmd=[sys.executable,str(exp/'campaign.py'),'--stage','asim','--sha256','0'*64];test=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
assert test.returncode!=0 and 'approved candidate manifest SHA changed' in test.stdout and file_sha(ledger)==before
(raw/'bad-sha-control.log').write_text(test.stdout);rows.append(dict(control='real_wrong_approval_SHA',status='PREFLIGHT_REJECTED',ledger_unchanged=True))
result=dict(status='INNER_CONTROLS_CHECKED',rows=rows,script=seal(Path(__file__)),score=None,note='six mutations are offline batch0 controls; performance rejection and wrong-SHA are real entry invocations; no faulty-binary claim')
f=raw/'controls.json'
with f.open('x') as s:s.write(json.dumps(result,indent=2)+'\n')
append(ledger,'inner-0001-controls','controls_checked',dict(candidate_id=c['candidate_id'],status=result['status'],evidence=[seal(f)]));print(json.dumps(result))
