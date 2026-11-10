from pathlib import Path
import json,csv,gc,shutil,subprocess,sys
import numpy as np
REPO=Path('/home/nanajiang/Asim');sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from optimization_ledger import file_sha,seal,read,export
out=REPO/'experiments/optimization/bri/inner-0001';raw=REPO/'results/optimization/bri/inner-0001';base=json.loads((out.parent/'baseline-v1.json').read_text());cand=json.loads((out/'candidate.json').read_text());asim=json.loads((raw/'asim-index.json').read_text());hold=json.loads((raw/'holdout/result.json').read_text());paired=json.loads((raw/'paired_baseline.json').read_text())
assert asim['status']=='ASIM_TEMPLATES_CHECKED' and hold['status']=='INNER_CDLB_RECONSTRUCTION_CHECKED' and paired['status']=='PAIRED_BASELINE_HOLDOUT_BITS_CHECKED';rows=[];pipelines=[];cores=[];parameter_rows=[]
for case,row,g in zip(base['cases'],asim['cases'],hold['groups']):
 assert case['regularization']==row['regularization']==g['regularization'] and case['iterations']==row['iterations']==g['iterations']
 r=json.loads(Path(row['result']).read_text());errors=[]
 for f in [f for f in hold['frames'] if f['regularization']==g['regularization']]:
  with np.load(f['path']) as a:errors.extend(a['task_errors'].tolist())
 record=dict(candidate_id=cand['candidate_id'],baseline_id=base['baseline_id'],algorithm_id=cand['algorithm_id'],snr_db=int(round(-10*np.log10(case['regularization']))),regularization=case['regularization'],iterations=case['iterations'],baseline_cycles=case['cycles'],candidate_cycles=row['cycles'],speedup=case['cycles']/row['cycles'],cycle_reduction_percent=100*(1-row['cycles']/case['cycles']),rayleigh_max_mmse_error=row['max_mmse_error'],cdlb_median_mmse_error=float(np.median(errors)),cdlb_p95_mmse_error=float(np.quantile(errors,.95)),cdlb_max_mmse_error=max(errors),candidate_ber=g['candidate_ber'],mmse_ber=g['mmse_ber'],candidate_ser=g['candidate_ser'],mmse_ser=g['mmse_ser'],candidate_raw_evm=g['mean_candidate_raw_evm'],mmse_raw_evm=g['mean_mmse_raw_evm'],quality_status=g['status'],baseline_and_candidate_bits_equal=True,source=str(Path(row['result']).relative_to(REPO)),candidate_commit=cand['candidate_commit'],binary_sha256=cand['binary_sha256'],comparison_identity=row['comparison_identity'],score=None)
 rows.append(record)
 with np.load(Path(row['result']).with_name('references.npz')) as a:
  for name in ['alpha','beta','lower','upper','energy_upper','preconditioner_upper','stored_lambda']:
   parameter_rows.append(dict(regularization=case['regularization'],iterations=case['iterations'],name=name,min=float(a[name].min()),max=float(a[name].max()),source='sealed independent algorithm reference setup; formula unchanged by instruction-only candidate'))
 for label,path in [('baseline',Path(case['result'])),('candidate',Path(row['result']))]:
  stats=json.loads(path.with_name('pipeline_statistics.json').read_text());assert stats['status']=='TRACE_VALID'
  expected=(591+(46 if label=='baseline' else 42)*case['iterations'])*96;assert stats['source_instruction_count']==expected
  record[label+'_instruction_count']=stats['source_instruction_count']
  for unit,m in stats['pipelines'].items():pipelines.append(dict(implementation=label,regularization=case['regularization'],iterations=case['iterations'],cycles=case['cycles'] if label=='baseline' else row['cycles'],unit=unit,**m))
  for core,metrics in stats['cores'].items():
   for unit,m in metrics['units'].items():cores.append(dict(implementation=label,regularization=case['regularization'],iterations=case['iterations'],core=core,unit=unit,busy_union=metrics['busy_union'],idle_in_makespan=metrics['idle_in_makespan'],multi_unit_overlap_union=metrics['multi_unit_overlap_union'],**m))
  del stats;gc.collect()
for name,data in [('cases.csv',rows),('pipeline_summary.csv',pipelines),('core_pipeline_summary.csv',cores),('coefficient_ranges.csv',parameter_rows)]:
 with (out/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in data for k in r));w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(data)
parameters=dict(task=base['cases'][0]['task'],cases=[dict(regularization=c['regularization'],iterations=c['iterations'],algorithm=c['algorithm']) for c in base['cases']],energy_norm_scale=8,preconditioner_norm_scale=128,enclosure_inflation=9/8,lower_safety=15/16,alpha_safety=15/16,beta_rule='fixed HB from same rounded enclosure; no separate beta damping',compensated=True,precision='FP16 persistent tensors/coefficient/state; FP32 temporary operations, ordered K GEMM; RNE, preserve subnormal, no fusion',template_seed=57001,channel_seeds=cand['channel_seeds'],frame_seeds=cand['frame_seeds'],minimum_speedup=cand['minimum_speedup'],maximum_speedup=cand['maximum_speedup'],limits=cand['limits'],cdlb=cand['cdlb'])
(out/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
summary=dict(status='INNER_LOOP_SOFTWARE_CANDIDATE_CHECKED',candidate_id=cand['candidate_id'],baseline_id=base['baseline_id'],candidate_commit=cand['candidate_commit'],change_kind=cand['change_kind'],cases=rows,all_algorithm_and_baseline_bits_equal=True,fresh_independent_H=192,candidate_fresh_batch_replays=576,paired_baseline_batch_replays=576,score=None,promotion='NOT_PROMOTED_M3_PENDING',default_runtime='UNCHANGED_BASELINE',note='not a global BRI leaderboard; same-input cycle comparison, fresh CDL paired baseline equality, no K/precision/formula change; cached isolated build, not cold reproduction',pure_model_thinking_seconds=None)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for src,name in [(raw/'paired_baseline.json','paired_baseline.json'),(raw/'independent_dynamic_audit.json','independent_dynamic_audit.json'),(REPO/'results/bri-inner-0001-static-audit.md','static_audit.md'),(REPO/'results/bri-inner-0001-dynamic-audit.md','dynamic_audit.md'),(raw/'build_cache.json','build_cache.json'),(raw/'configure.log','configure.log'),(raw/'build.log','build.log')]:shutil.copy2(src,out/name)
patch=subprocess.check_output(['git','diff','d26206b9',cand['candidate_commit'],'--','src/inverse/bri_stable_hb/BRIStableHBModel.cc'],cwd=REPO,text=True);(out/'candidate.patch').write_text(patch)
export(out.parent/'ledger.jsonl',out.parent/'comparison.csv');read(out.parent/'ledger.jsonl')
files=[seal(p) for p in sorted(raw.rglob('*')) if p.is_file()]
(out/'run_manifest.json').write_text(json.dumps(dict(schema='optimization-artifacts-v1',files=files),indent=2)+'\n')
print(json.dumps(dict(status=summary['status'],cases=rows,pipeline_rows=len(pipelines),core_rows=len(cores))))
