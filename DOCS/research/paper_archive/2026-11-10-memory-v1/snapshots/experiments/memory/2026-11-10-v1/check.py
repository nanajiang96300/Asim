"""Offline recheck of sealed actual Asim artifacts; no new cycles invented."""
from pathlib import Path
import json,csv,time,resource,gc,hashlib,sys
import numpy as np
from collections import Counter
from verification.reconstruction.stream import load_operation
from verification.reconstruction.binding import validate_actual
from verification.reconstruction.source_views import execute_batch
from verification.reporting.pipeline import statistics,write_detail
from verification.schedule.runtime import check_runtime
from verification.asim.batch import check_events
from verification.contracts.mmse import MMSETask
from verification.profiles.numeric import bit_equal
repo=Path('/home/nanajiang/Asim');p=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();phases=[]
def mark(name):
 phases.append(dict(phase=name,elapsed_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));print(json.dumps(phases[-1]),flush=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
r=json.loads((p/'result.json').read_text());assert r['status']=='MODELED_NUMERIC_CHECKED'
for a in r['artifacts']:
 q=p/a['path'];assert q.stat().st_size==a['bytes'] and sha(q)==a['sha256']
record=load_operation(p/'instructions.jsonl');dispatch=[json.loads(l) for l in (p/'dispatch.jsonl').open()];config=json.loads((p/'config.json').read_text())
task=MMSETask(**{k:r['task'][k] for k in MMSETask.__dataclass_fields__});finish=r['asim']['finish_cycle']
with np.load(p/'inputs.npz') as a:inputs={k:a[k] for k in a.files}
h,groups,bound,counts=validate_actual(record,dispatch,task,r['algorithm'],inputs,config);mark('load_and_binding')
events={};dma=Counter();mapping={d['batch']:d for d in dispatch}
for i in record['instructions']:
 core=mapping[i['batch']]['core']
 if i['opcode'] in (0,1):dma[core,'Load' if i['opcode']==0 else 'Store']+=len(i['src_addrs'])
 else:events[i['id']]=f'Core{core}_'+('Cube' if i['opcode'] in (3,4) else 'Scalar' if 17<=i['opcode']<=21 else 'Vector')
with (p/'trace.csv').open() as f:check_events(csv.DictReader(f),events,dma,finish,mapping)
del events;mark('trace_coverage')
stats=statistics(p/'trace.csv',record['instructions'],finish,24,dispatch)
assert stats['status']=='TRACE_VALID'
with (out/'pipeline_statistics.json').open('w') as f:json.dump(stats,f,indent=2)
write_detail(out/'instruction_pipeline_detail.csv',stats);del stats;gc.collect();mark('pipeline_statistics')
assert sha(out/'pipeline_statistics.json')==sha(p/'pipeline_statistics.json')
assert sha(out/'instruction_pipeline_detail.csv')==sha(p/'instruction_pipeline_detail.csv')
statistics_hashes={name:sha(out/name) for name in ('pipeline_statistics.json','instruction_pipeline_detail.csv')}
for name in statistics_hashes:(out/name).unlink() # Redundant byte-identical generated diagnostics; originals remain sealed.
limits={'max_bytes':2*1024**3,'max_events':4000000} if r['operator']=='bri_stable_hb_fp16_v1' else None
schedule=check_runtime(record,dispatch,p/'rw_events.jsonl',p/'trace.csv',config,finish,stream_limits=limits)
assert schedule==json.loads((p/'schedule.json').read_text());mark('runtime_schedule')
with np.load(p/'references.npz') as a:expected=a['algorithm_block']
outputs=[]
for b in range(task.batch_size):
 got=execute_batch(h,groups[b],bound,b,expected[b],return_output=True);v=got.pop('_output',None)
 assert got['status']=='STRUCTURE_MATCH' and not got['cost_issues'] and bit_equal(v,expected[b]);outputs.append(v)
assert bit_equal(np.stack(outputs),np.load(p/'outputs.npy'));mark('all_batches_replayed')
result=dict(status='SEALED_ASIM_RECHECK_PASSED',source=str(p),operator=r['operator'],iterations=r['algorithm'].get('iterations'),batch_size=task.batch_size,finish_cycle=finish,phase_measurements=phases,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,within_2gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=2*1024**2,statistics_byte_equal=True,statistics_hashes=statistics_hashes,schedule_exact_equal=True,outputs_bit_equal=True,pure_model_thinking_seconds=None,scope='reuses hash-verified accepted Asim; not a new simulation or new independent algorithm reference run')
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
