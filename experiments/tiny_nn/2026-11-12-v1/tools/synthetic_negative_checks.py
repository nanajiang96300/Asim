from pathlib import Path
import sys,tarfile,io,json,copy
import numpy as np
repo=Path('/home/nanajiang/asim-worktrees/tiny-nn-primitives-v1');sys.path.insert(0,str(repo))
from verification.tasks.tiny_nn import NNTask,captures,workload
from verification.reconstruction.source_views import execute_batch
from verification.reconstruction.runner import compare_stage_captures
from verification.primitives.nn import validate_instruction
raw=Path(__file__).resolve().parent;case='attention-l32-d32-f64-s91001'
with tarfile.open(raw/'suite-v3'/(case+'.tar.gz')) as tf:
 def read(name):return tf.extractfile(case+'/'+name).read()
 r=json.loads(read('instructions.jsonl'));d=dict(np.load(io.BytesIO(read('inputs.npz'))));refs=dict(np.load(io.BytesIO(read('references.npz'))))
task=NNTask();h=r['header'];seq=[i for i in r['instructions'] if i['batch']==31];inputs=[d[t['name'].split('.')[-1]] for t in h['inputs']]
expected_names={k for v in captures(task,seq).values() for k in v}
base=execute_batch(h,seq,inputs,31,refs['algorithm_block'][31],captures=captures(task,seq));assert base['status']=='STRUCTURE_MATCH'
rows=[]
for name,op,key,value in [('wrong_axis',8,'axis',0),('unknown_gelu_formula',22,'formula','erf'),('too_small_softmax_work',8,'compute_size',64)]:
 i=copy.deepcopy(next(x for x in seq if x['opcode']==op))
 if key=='compute_size':i[key]=value
 else:i['semantic'][key]=value
 try:validate_instruction(i,task.profile)
 except ValueError as e:rows.append({'case':name,'status':'REJECTED','reason':str(e),'layer':'instruction semantic/workload validator'})
 else:raise AssertionError(name)
for name,node in [('omit_gelu','activation'),('omit_residual','U')]:
 changed=[copy.deepcopy(i) for i in seq if i['id']!='batch31.'+node]
 try:workload({31:changed},task)
 except ValueError as e:rows.append({'case':name,'status':'REJECTED','reason':str(e),'layer':'frozen actual graph work/order validator'})
 else:raise AssertionError(name)
changed=copy.deepcopy(seq);i=next(i for i in changed if i['id']=='batch31.logits');x=next(i for i in changed if i['id']=='batch31.load.X')['semantic']['dst'];i['semantic']['src'][1]=copy.deepcopy(x);i['src_addrs'][1]=x['base']
workload({31:changed},task)
a=execute_batch(h,changed,inputs,31,refs['algorithm_block'][31],captures=captures(task,changed));checks=compare_stage_captures(a.get('_observations',{}),expected_names,refs,31)
assert not all(c['bit_equal'] for c in checks)
rows.append({'case':'wrong_scaling_operand_X_instead_of_constant','status':'REJECTED','layer':'actual source-view numerical replay + independent stages','first_stage':next(c['name'] for c in sorted(checks,key=lambda c:c['index'] or 0) if not c['bit_equal'])})
(raw/'synthetic-negative-summary.json').write_text(json.dumps({'status':'SIX_SYNTHETIC_MUTATIONS_REJECTED','scope':'post-seal in-memory mutations for validator/replay tests; no new Asim or hardware fault experiment claimed','baseline':'real accepted archive '+case,'batch':31,'checks':rows},indent=2)+'\n')
