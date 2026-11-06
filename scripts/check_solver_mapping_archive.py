#!/usr/bin/env python3
"""Check frozen mapping/actual-stage/daily evidence, preserving expected quality rejection."""
import hashlib,json,sys
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.archive_checks import indexed,nested,verify_files,CASE_REQUIRED

def validate_stage_coverage(stages,expected,batch_sizes):
    cases=stages['cases']
    if stages['status']!='STAGE_WITNESSES_PASSED' or len(cases)!=len(expected) or {c['case'] for c in cases}!=expected or not all(c['passed'] and c['batch_size']==batch_sizes[c['case']] for c in cases) or sum(c['batch_size'] for c in cases)!=672:raise ValueError('incomplete stage coverage')


def check(path):
    repo=Path(__file__).resolve().parents[1];m=json.loads(Path(path).read_text());seen=set()
    if m['schema']!='solver-mapping-archive-v1':raise ValueError('unknown archive schema')
    outer=indexed(m['artifacts']);verify_files(repo,outer);seen=set(outer)
    required={str(Path(m['campaign'])/'summary.json'),str(Path(m['stages'])/'summary.json'),m['effort'],m['core_identity'],*{str(Path(p)/'result.json') for p in m['daily_runs']}}
    if not required<=seen:raise ValueError('missing bound primary evidence')
    campaign=repo/m['campaign'];s=json.loads((campaign/'summary.json').read_text())
    names={f'{family}_{case}' for family in ('ldl','bri') for case in ('f64','f16_42','f16_123','shape2')}|{'bri_low_k','ns_regression','chol_regression'}
    if s['status']!='SUPPORTED_MAPPING_GATES_PASSED' or {c['case'] for c in s['cases']}!=names or len(s['cases'])!=11:raise ValueError('incomplete mapping campaign')
    for c in s['cases']:
        root=Path(m['campaign'])/c['case'];r=json.loads((repo/root/'result.json').read_text())
        if str(root/'result.json') not in seen:raise ValueError('unbound result')
        nested(outer,root,r['artifacts'],CASE_REQUIRED)
        if r['status']!=c['expected_status'] or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or r['numeric_status']!='EXECUTED' or r['score'] is not None or len(r['batches'])!=r['task']['batch_size']:raise ValueError('case gate mismatch')
        if c['case']=='bri_low_k':
            if r['status']!='ALGORITHM_QUALITY_FAILED' or not r['structural_reference_passed'] or r['task_reference_passed']:raise ValueError('low K incorrectly accepted')
        elif r['failed_batches']:raise ValueError('positive case failed')
        if r['precision']=='modeled_fp16_fp32_real_block_v1' and not all(b.get('precision_bit_equal') for b in r['batches']):raise ValueError('precision mismatch')
    if len(s['offline_negatives'])!=2 or any(n['status']!='NUMERIC_FAILED' or n['actual_mutant_simulated'] for n in s['offline_negatives']):raise ValueError('incorrect offline negative claim')
    stages=json.loads((repo/m['stages']/'summary.json').read_text())
    expected={name for name in names if name.startswith(('ldl_','bri_')) and name!='bri_low_k'}
    validate_stage_coverage(stages,expected,{c['case']:c['task']['batch_size'] for c in s['cases']})
    if stages['campaign_summary_sha256']!=outer[str(Path(m['campaign'])/'summary.json')]['sha256']:raise ValueError('stages from another campaign')
    for c in stages['cases']:
        root=Path(m['stages'])/c['case']
        if not {str(root/'batches.json'),str(root/'actual_stage_snapshots.npz')}<=seen:raise ValueError('unbound stage evidence')
        batches=json.loads((repo/root/'batches.json').read_text())
        if len(batches)!=c['batch_size'] or {b['batch'] for b in batches}!=set(range(c['batch_size'])) or any(b['status']!='STAGE_WITNESSES_MATCH' for b in batches):raise ValueError('incomplete per-batch stage witnesses')
    for run in m['daily_runs']:
        r=json.loads((repo/run/'result.json').read_text())
        nested(outer,run,r['artifacts'],CASE_REQUIRED|{'evaluation_result.json','runtime_pin.json'})
        if r['status']!='MODELED_NUMERIC_CHECKED' or r['failed_batches']:raise ValueError('daily failed')
    e=json.loads((repo/m['effort']).read_text())
    if e['pure_model_thinking_seconds'] is not None:raise ValueError('unobservable thinking-time claim')
    return dict(status='SOLVER_MAPPING_ARCHIVE_VALID',files_checked=len(seen),new_families=['LDL','BRI'],low_K='QUALITY_REJECTED',score=None)

if __name__=='__main__':print(json.dumps(check(sys.argv[1])))
