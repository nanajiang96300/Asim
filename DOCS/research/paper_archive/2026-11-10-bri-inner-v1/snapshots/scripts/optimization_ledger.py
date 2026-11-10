"""Append-only experiment facts; CSV is a rebuildable view, never a score source."""
import argparse,csv,hashlib,json,fcntl,math
from pathlib import Path

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):
 rows=[];prev=None;seen=set()
 if not Path(path).exists():return rows
 for line in Path(path).read_text().splitlines():
  r=json.loads(line);h=r.pop('record_sha256')
  if r['previous_sha256']!=prev or digest(r)!=h or r['event_id'] in seen:raise ValueError('ledger chain/identity changed')
  r['record_sha256']=h;rows.append(r);prev=h;seen.add(r['event_id'])
 return rows

def append(path,event_id,kind,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);rows=read(path)
  if event_id in {r['event_id'] for r in rows}:raise ValueError('refusing duplicate event_id')
  for e in payload.get('evidence',[]):
   p=Path(e['path'])
   if not p.is_file() or p.stat().st_size!=e['bytes'] or file_sha(p)!=e['sha256']:raise ValueError('evidence changed/missing')
  record=dict(schema='optimization-ledger-v1',event_id=event_id,kind=kind,previous_sha256=rows[-1]['record_sha256'] if rows else None,payload=payload)
  record['record_sha256']=digest(record);f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');f.flush()
  import os;os.fsync(f.fileno())
 return record

def seal(path):
 p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))

def actual_run(result,baseline,candidate,run_id):
 p=Path(result).resolve();r=json.loads(p.read_text());b=next(x for x in baseline['cases'] if x['regularization']==r['task']['regularization'])
 if r['operator']!=candidate['operator_id'] or r['algorithm']!=b['algorithm'] or r['task']!=b['task'] or r['comparison_identity']!=b['comparison_identity']:raise ValueError('candidate/task/algorithm/comparison baseline mismatch')
 if r['asim']['binary_sha256']!=candidate['binary_sha256']:raise ValueError('candidate binary mismatch')
 receipt=json.loads((p.parent/'asim.json').read_text())
 if receipt['binary_sha256']!=r['asim']['binary_sha256'] or receipt['finish_cycle']!=r['asim']['finish_cycle']:raise ValueError('actual Asim receipt mismatch')
 if r['score'] is not None:raise ValueError('this supervised campaign has no promotion score')
 if r['status']=='MODELED_NUMERIC_CHECKED':
  if r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or r['numeric_status']!='EXECUTED' or r['failed_batches'] or not all(x['precision_bit_equal'] and not x['cost_issues'] for x in r['batches']):raise ValueError('incomplete numerical/schedule acceptance')
 for a in r['artifacts']:
  q=p.parent/a['path']
  if q.stat().st_size!=a['bytes'] or file_sha(q)!=a['sha256']:raise ValueError('actual artifact mismatch')
 status=r['status'];cycles=r['asim']['finish_cycle'];eligible=status=='MODELED_NUMERIC_CHECKED'
 return dict(candidate_id=candidate['candidate_id'],baseline_id=baseline['baseline_id'],run_id=run_id,algorithm_id=candidate['algorithm_id'],change_kind=candidate['change_kind'],status=status,regularization=r['task']['regularization'],iterations=r['algorithm']['iterations'],task=r['task'],parameters=r['algorithm'],comparison_identity=r['comparison_identity'],cycles=cycles,baseline_cycles=b['cycles'],speedup=b['cycles']/cycles if eligible else None,max_mmse_error=max((x['task_error'] for x in r.get('batches',[]) if isinstance(x.get('task_error'),(int,float)) and math.isfinite(x['task_error'])),default=None),all_algorithm_bits_equal=(all(x.get('precision_bit_equal',False) for x in r.get('batches',[])) if r.get('numeric_status')=='EXECUTED' else None),phase_timings_seconds=r['phase_timings_seconds'],score=None,communication_status='NOT_RUN',result=str(p),evidence=[seal(p)],pipeline_statistics=seal(p.parent/'pipeline_statistics.json') if (p.parent/'pipeline_statistics.json').exists() else None,candidate_commit=candidate['candidate_commit'],binary_sha256=candidate['binary_sha256'],evaluator_sources=candidate['evaluator_sources'],note='speedup is a local comparable case diagnostic; full quality/reproduction promotion is separate')

def export(ledger,out):
 rows=read(ledger);runs={r['payload']['run_id']:dict(r['payload']) for r in rows if r['kind']=='run_evaluated'}
 for r in rows:
  if r['kind']=='communication_evaluated':
   p=r['payload'];runs[p['run_id']].update(communication_status=p['status'],cdlb_max_mmse_error=p['max_task_error'],candidate_ber=p['candidate_ber'],mmse_ber=p['mmse_ber'],candidate_ser=p['candidate_ser'],mmse_ser=p['mmse_ser'],evm_delta=p['evm_delta'])
 for row in runs.values():
  row.update({k:row['task'][k] for k in ['nr','nt','nrhs','batch_size','profile','channel']})
  row.update({k:row['parameters'].get(k) for k in ['initialization','block_size_complex','coefficient_rule','enclosure_rule','representation']})
 fields=['nr','nt','nrhs','batch_size','profile','channel','initialization','block_size_complex','coefficient_rule','enclosure_rule','representation','candidate_id','baseline_id','run_id','algorithm_id','change_kind','regularization','iterations','cycles','baseline_cycles','speedup','max_mmse_error','all_algorithm_bits_equal','status','communication_status','cdlb_max_mmse_error','candidate_ber','mmse_ber','candidate_ser','mmse_ser','evm_delta','candidate_commit','binary_sha256','comparison_identity','score','result']
 with Path(out).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(runs.values())

if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--ledger',type=Path,required=True);a.add_argument('--csv',type=Path);v=a.parse_args()
 rows=read(v.ledger)
 if v.csv:export(v.ledger,v.csv)
 print(json.dumps(dict(status='LEDGER_CHAIN_CHECKED',records=len(rows))))
