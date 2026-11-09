from pathlib import Path
import json,csv,gc,shutil
from verification.asim.runner import file_hash
repo=Path('/home/nanajiang/Asim');root=repo/'results/fixed_hb_native/2026-11-09-v2';out=repo/'experiments/fixed_hb_native/2026-11-09-v2'
source=json.loads((root/'asim-index.json').read_text());hold=json.loads((root/'holdout/result.json').read_text());reg=json.loads((root/'regression-index.json').read_text())
assert source['status']=='ASIM_TEMPLATES_CHECKED' and hold['status']=='FIXED_HB_CDLB_RECONSTRUCTION_CHECKED' and len(reg)==2 and len(hold['frames'])==6
cases=[];units=[];cores=[]
paths=[Path(e['path']) for e in source['cases']]+[root/'regression'/r['operator']/'result.json' for r in reg]
for p in paths:
 r=json.loads(p.read_text());stats=json.loads((p.parent/'pipeline_statistics.json').read_text())
 assert r['status']=='MODELED_NUMERIC_CHECKED' and stats['status']=='TRACE_VALID'
 for a in r['artifacts']:
  q=p.parent/a['path'];assert q.stat().st_size==a['bytes'] and file_hash(q)==a['sha256']
 row=dict(operator=r['operator'],regularization=r['task']['regularization'],iterations=r['algorithm'].get('iterations',1),cycles=r['asim']['finish_cycle'],max_rayleigh_task_error=max(b['task_error'] for b in r['batches']),source_instruction_count=stats['source_instruction_count'],comparison_identity=r['comparison_identity'],source=str(p.relative_to(repo)),**{k+'_seconds':v for k,v in r['phase_timings_seconds'].items()});cases.append(row)
 for unit,m in stats['pipelines'].items():units.append({**{k:row[k] for k in ('operator','regularization','iterations','cycles')},'unit':unit,**m})
 for core,m in stats['cores'].items():
  for unit,metrics in m['units'].items():cores.append({**{k:row[k] for k in ('operator','regularization','iterations','cycles')},'core':core,'unit':unit,**metrics,'core_busy_union':m['busy_union'],'core_idle_in_makespan':m['idle_in_makespan'],'core_multi_unit_overlap_union':m['multi_unit_overlap_union']})
 del stats;gc.collect()
for name,rows in [('cycles_and_replay.csv',cases),('pipeline_summary.csv',units),('core_pipeline_summary.csv',cores)]:
 with (out/name).open('w',newline='') as f:
  fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
assert len({r['comparison_identity'] for r in cases if r['regularization']==.1})==1
summary=dict(status=hold['status'],cases=cases,cdlb=[{k:v for k,v in g.items() if k not in ('streams',)} for g in hold['groups']],frames=hold['frames'],cdlb_independent_H=192,cdlb_replayed_batches=576,reconstruction_consistency_all_passed=True,all_cdlb_quality_passed=True,score=None,hardware_timing='OWNER_BOARD_VALIDATION_EXTERNAL',precision='FP16 storage/FP32 temporary arithmetic; subnormal preserve; no persistent FP32 tensors',pure_model_thinking_seconds=None)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for src,name in [(root/'controls/result.json','controls.json'),(repo/'results/fixed-hb-native-static-review-20261109.md','static_audit.md'),(repo/'results/fixed-hb-native-dynamic-review-20261109.md','dynamic_audit.md'),(repo/'results/fixed_hb/native-unit-tests-20261109.log','unit-tests.log'),(repo/'results/fixed_hb/native-build-cache.json','build_cache_receipt.json')]:shutil.copy2(src,out/name)
print(json.dumps(dict(status=summary['status'],case_rows=len(cases),pipeline_rows=len(units),core_rows=len(cores))))
