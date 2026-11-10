from pathlib import Path
import json,sys,subprocess
REPO=Path('/home/nanajiang/Asim');sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from optimization_ledger import seal,file_sha,append
from verification.reconstruction.stable_hb_campaign import source_map
from verification.channels.cdlb import SPEC,provenance
from verification.contracts.mmse import MMSETask
from verification.operators.catalog import adapter
w=Path('/home/nanajiang/asim-worktrees/bri-inner-0001');out=REPO/'experiments/optimization/bri';exp=out/'inner-0001';src=REPO/'results/fixed_hb_native/2026-11-09-v2'
basepath=out/'baseline-v1.json';assert not basepath.exists();cases=[]
for l,k in [(1.,16),(.1,53),(.01,174)]:
 p=src/f'asim-lambda{l}/result.json';r=json.loads(p.read_text());assert r['status']=='MODELED_NUMERIC_CHECKED' and not r['failed_batches']
 for a in r['artifacts']:
  q=p.parent/a['path'];assert q.stat().st_size==a['bytes'] and file_sha(q)==a['sha256']
 cases.append(dict(regularization=l,iterations=k,cycles=r['asim']['finish_cycle'],task=r['task'],algorithm=r['algorithm'],comparison_identity=r['comparison_identity'],rayleigh_max_mmse_error=max(x['task_error'] for x in r['batches']),result=str(p),result_sha256=file_sha(p),raw_bytes=sum(x['bytes'] for x in r['artifacts']),phase_timings_seconds=r['phase_timings_seconds'],binary_sha256=r['asim']['binary_sha256']))
base=dict(schema='optimization-baseline-v1',family_id='bri',algorithm_id='bri_fixed_hb_fp16_v1',baseline_id='bri-hb-base-v1',candidate_id='bri-hb-base-v1',operator_id='bri_stable_hb_fp16_v1',candidate_commit='bd1e520b74ce63920bffbecb8a55558d617b509e',cases=cases,cdlb=json.loads((src/'holdout/result.json').read_text())['groups'],evidence=[seal(REPO/'experiments/fixed_hb_native/2026-11-09-v2/summary.json'),seal(src/'holdout/result.json'),seal(REPO/'experiments/memory/2026-11-10-v1/summary.json')],validation_epoch='original native fixed-HB with M1 shared evaluator compatibility evidence; different original evaluator version explicitly retained',score=None)
with basepath.open('x') as f:f.write(json.dumps(base,indent=2)+'\n')
append(out/'ledger.jsonl','bri-hb-base-v1-registered','baseline_registered',dict(baseline_id=base['baseline_id'],candidate_id=base['candidate_id'],algorithm_id=base['algorithm_id'],cases=cases,evidence=[seal(basepath)]))
binary=w/'build/bin/Simulator';lib=w/'build/lib/libramulator.so';source=w/'src/inverse/bri_stable_hb/BRIStableHBModel.cc'
paths=[binary,lib,source,w/'configs/ramulator2_configs/HBM2.yaml',exp/'campaign.py',REPO/'scripts/optimization_ledger.py',REPO/'results/bri-inner-0001-static-audit.md']
for case in cases:paths.append(Path(case['result']).with_name('config.json'))
c=dict(schema='inner-loop-candidate-v1',family_id='bri',algorithm_id=base['algorithm_id'],operator_id=base['operator_id'],candidate_id='bri-hb-inner-0001',parent_candidate_id=base['candidate_id'],baseline_id=base['baseline_id'],change_kind='instruction_layout_state_rotation',candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),worktree=str(w),binary=str(binary),binary_sha256=file_sha(binary),cwd=str(w/'build'),library_dir=str(w/'build/lib'),baseline_manifest_sha256=file_sha(basepath),evaluator_sources=source_map(REPO),dependencies={str(p):file_sha(p) for p in paths},minimum_speedup=1.001,maximum_speedup=4,max_wall_seconds=5400,min_free_bytes=536870912,max_output_bytes=3758096384,channel_seeds=[61001,61002],frame_seeds=[61101,61102],cdlb=SPEC,channel_toolchain=provenance(),limits=dict(max_ber_increase=.01,max_ser_increase=.02,max_evm_increase=.01,max_outage_probability=.05,family_alpha=.05,min_independent_channels=192),parameters=[case['algorithm'] for case in cases],task_scope=cases[0]['task']['scope'],modification_scope=['src/inverse/bri_stable_hb/BRIStableHBModel.cc'],hypothesis='rotate previous/x/value physical addresses, eliminate two state copies and two barriers per iteration without arithmetic changes',score=None)
with (exp/'candidate.json').open('x') as f:f.write(json.dumps(c,indent=2)+'\n')
append(out/'ledger.jsonl',c['candidate_id']+'-frozen','candidate_frozen',dict(candidate_id=c['candidate_id'],parent_candidate_id=c['parent_candidate_id'],baseline_id=c['baseline_id'],algorithm_id=c['algorithm_id'],candidate_commit=c['candidate_commit'],binary_sha256=c['binary_sha256'],change_kind=c['change_kind'],parameters=c['parameters'],evidence=[seal(exp/'candidate.json')]))
print(dict(candidate_manifest_sha256=file_sha(exp/'candidate.json'),candidate_commit=c['candidate_commit'],baseline_cycles=[x['cycles'] for x in cases]))
