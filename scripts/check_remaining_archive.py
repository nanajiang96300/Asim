#!/usr/bin/env python3
"""Check seven-entry mapping expansion, factor witnesses and invalid legacy Merge evidence."""
import json,sys,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.archive_checks import indexed,nested,verify_files,CASE_REQUIRED,PROCESS_REQUIRED
from scripts.remaining_contracts import validate_case,validate_checks,normalize_probe

FAMILIES=('cholesky_block','ldl_block','cholesky_merge')

def check(path):
    repo=Path(__file__).resolve().parents[1];m=json.loads(Path(path).read_text())
    if m['schema']!='remaining-mapping-archive-v1':raise ValueError('unknown archive schema')
    outer=indexed(m['artifacts']);verify_files(repo,outer)
    required={m['campaign']+'/summary.json',m['forensics']+'/summary.json',m['effort'],m['core_identity'],m['daily_acceptance']}
    if not required<=outer.keys():raise ValueError('unbound primary evidence')
    summary=json.loads((repo/m['campaign']/'summary.json').read_text())
    expected={f'{f}_{s}' for f in FAMILIES for s in ('f64','f16_42','f16_123','shape2')}|{p+'_regression' for p in ('ns_corrected_v1','cholesky_explicit_v1','ldl_explicit_v1','bri_explicit_v1')}
    if summary['status']!='REMAINING_MAPPING_GATES_PASSED' or len(summary['cases'])!=16 or {c['case'] for c in summary['cases']}!=expected:raise ValueError('incomplete remaining mapping')
    by_case={}
    for c in summary['cases']:
        root=Path(m['campaign'])/c['case']
        if str(root/'result.json') not in outer:raise ValueError('unbound result')
        r=json.loads((repo/root/'result.json').read_text());validate_case(c['case'],r);by_case[c['case']]=r
        nested(outer,root,r['artifacts'],CASE_REQUIRED)
        if r['status'] not in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') or r['numeric_status']!='EXECUTED' or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or r['failed_batches'] or r['score'] is not None or len(r['batches'])!=r['task']['batch_size'] or {b['batch'] for b in r['batches']}!=set(range(r['task']['batch_size'])):raise ValueError('positive case failed')
        if r['status']=='MODELED_NUMERIC_CHECKED' and not all(b['precision_bit_equal'] for b in r['batches']):raise ValueError('precision mismatch')
    stage_names={f'{f}_{s}' for f in FAMILIES for s in ('f64','f16_42')}
    stages=summary['stage_cases']
    if len(stages)!=6 or {s['case'] for s in stages}!=stage_names or sum(s['batch_size'] for s in stages)!=576 or not all(s['passed'] for s in stages):raise ValueError('incomplete factor stage coverage')
    for s in stages:
        root=Path(m['campaign'])/s['case'];n=by_case[s['case']]['task']['batch_size']
        if s['batch_size']!=n or not {str(root/'factor_snapshots.npz'),str(root/'factor_witnesses.json')}<=outer.keys():raise ValueError('unbound factor witnesses')
        rows=json.loads((repo/root/'factor_witnesses.json').read_text())
        if len(rows)!=n or {b['batch'] for b in rows}!=set(range(n)) or not all(b['passed'] for b in rows):raise ValueError('factor batch missing or failed')
        for b in rows:validate_checks(s['case'],b['checks'])
    negatives=summary['negative_cases']
    if len(negatives)!=2 or {n['kind'] for n in negatives}!={'scalar_width16_cost_admission','dropped_reduction_term'} or any(n['status'] not in ('REJECTED','NUMERIC_FAILED') or n['actual_mutant_simulated'] for n in negatives):raise ValueError('negative gates failed')
    forensic=json.loads((repo/m['forensics']/'summary.json').read_text())
    if forensic['status']!='ANOMALY_REPRODUCED' or len(forensic['legacy_cases'])!=2 or len(forensic['scalar_probes'])!=2:raise ValueError('incomplete forensics')
    if {c['operator'] for c in forensic['legacy_cases']}!={'cholesky_noblock','cholesky_noblock_merge'}:raise ValueError('legacy coverage mismatch')
    for c in forensic['legacy_cases']:
        root=Path(m['forensics'])/c['operator'];required={str(root/p) for p in (*PROCESS_REQUIRED,'asim.json','inputs.npz','task_reference.npz','batches.json','pipeline_statistics.json','instruction_pipeline_detail.csv')}
        if not required<=outer.keys():raise ValueError('unbound legacy artifacts')
        receipt=json.loads((repo/root/'asim.json').read_text());nested(outer,root,receipt['artifacts'],PROCESS_REQUIRED)
        batches=json.loads((repo/root/'batches.json').read_text())
        if receipt['finish_cycle']!=c['cycles'] or receipt['binary_sha256']!=forensic['binary_sha256']:raise ValueError('legacy receipt mismatch')
        stats=json.loads((repo/root/'pipeline_statistics.json').read_text());records=[json.loads(line) for line in (repo/root/'instructions.jsonl').read_text().splitlines()]
        if len(records)!=1:raise ValueError('legacy operation count')
        raw=records[0]['instructions'];scalar=[i for i in raw if 17<=i['opcode']<=21]
        metrics=dict(instruction_count=len(raw),scalar_instructions=len(scalar),scalar_declared_elements=sum(max(1,i['compute_size']) for i in scalar),illegal_scalar_width_count=sum(i['compute_size']>1 for i in scalar),scalar_duration_sum=stats['pipelines']['Scalar']['duration_sum'])
        if any(c[k]!=v for k,v in metrics.items()) or stats['status']!='TRACE_VALID':raise ValueError('legacy raw/stat summary mismatch')
        if receipt['status']!='COMPLETED' or len(batches)!=96 or {r['batch'] for r in batches}!=set(range(96)) or any(r['status']!='SEMANTIC_AMBIGUITY' for r in batches):raise ValueError('legacy ambiguity not reproduced')
    legacy={c['operator']:c for c in forensic['legacy_cases']};a,b=legacy['cholesky_noblock'],legacy['cholesky_noblock_merge']
    if a['scalar_declared_elements']!=b['scalar_declared_elements'] or b['illegal_scalar_width_count']<=0:raise ValueError('underaccounting evidence missing')
    probes=forensic['scalar_probes']
    if {p['width'] for p in probes}!={1,16} or probes[0]['cycles']!=probes[1]['cycles'] or probes[0]['scalar_duration']!=probes[1]['scalar_duration'] or any(p['numeric_certified'] for p in probes):raise ValueError('invalid probe interpretation')
    actual_probe_programs=[]
    for p in probes:
        root=Path(m['forensics'])/('scalar_width'+str(p['width']))
        if str(root/'asim.json') not in outer:raise ValueError('unbound probe receipt')
        receipt=json.loads((repo/root/'asim.json').read_text());nested(outer,root,receipt['artifacts'],PROCESS_REQUIRED)
        if receipt['status']!='COMPLETED' or receipt['finish_cycle']!=p['cycles']:raise ValueError('probe process failed')
        records=[json.loads(line) for line in (repo/root/'instructions.jsonl').read_text().splitlines()]
        if len(records)!=1:raise ValueError('probe operation count')
        raw=next(i for i in records[0]['instructions'] if i['id']=='probe.scalar')
        if raw['opcode']!=19 or raw['compute_size']!=p['width'] or raw['semantic']['dst']['rows']*raw['semantic']['dst']['cols']!=1:raise ValueError('probe source binding differs')
        actual_probe_programs.append(normalize_probe(records[0]))
        with (repo/root/'trace.csv').open() as f:events=[r for r in csv.DictReader(f) if r['name']=='probe.scalar' and r['unit'].endswith('_Scalar')]
        if len(events)!=1 or float(events[0]['end_cycle'])-float(events[0]['start_cycle'])!=p['scalar_duration']:raise ValueError('actual scalar event differs')
    if actual_probe_programs[0]!=actual_probe_programs[1]:raise ValueError('cost probes changed more than compute_size')
    if len(m['daily_runs'])!=4 or len(set(m['daily_runs']))!=4:raise ValueError('four unique daily runs required')
    daily=[]
    spec_path=m['runtime_spec']
    if spec_path not in outer:raise ValueError('unbound runtime specification')
    spec=json.loads((repo/spec_path).read_text());spec_sha=outer[spec_path]['sha256'];binary_sha=outer[spec['binary']]['sha256']
    for run in m['daily_runs']:
        if str(Path(run)/'result.json') not in outer:raise ValueError('unbound daily result')
        r=json.loads((repo/run/'result.json').read_text());nested(outer,run,r['artifacts'],CASE_REQUIRED|{'evaluation_result.json','runtime_pin.json'})
        if r['status']!='MODELED_NUMERIC_CHECKED' or r['failed_batches'] or r['task']['batch_size']!=96:raise ValueError('daily failed')
        if r['numeric_status']!='EXECUTED' or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or len(r['batches'])!=96 or {b['batch'] for b in r['batches']}!=set(range(96)) or not all(b.get('precision_bit_equal') for b in r['batches']):raise ValueError('daily coverage/precision incomplete')
        pin=json.loads((repo/run/'runtime_pin.json').read_text())
        if pin['spec_sha256']!=spec_sha or r['asim']['binary_sha256']!=binary_sha:raise ValueError('daily runtime differs')
        case=r['operator'][:-12]+'_f16_42' if r['operator'].endswith('_explicit_v1') else ''
        validate_case(case,r)
        daily.append((run,r))
    acceptance=json.loads((repo/m['daily_acceptance']).read_text())
    if set(acceptance['merge_repeat_identical'])!={'outputs.npy','rw_events.jsonl','instructions.jsonl'} or not all(acceptance['merge_repeat_identical'].values()):raise ValueError('daily repeat differs')
    merged=[(run,r) for run,r in daily if r['operator']=='cholesky_merge_explicit_v1']
    if len(merged)!=2 or {r['operator'] for _,r in daily}!={'cholesky_block_explicit_v1','ldl_block_explicit_v1','cholesky_merge_explicit_v1'}:raise ValueError('daily operator coverage incomplete')
    (a,ra),(b,rb)=merged
    if ra['baseline_creation']==rb['baseline_creation'] or ra['asim']['finish_cycle']!=rb['asim']['finish_cycle'] or ra['algorithm']!=rb['algorithm'] or ra['comparison_identity']!=rb['comparison_identity']:raise ValueError('baseline repeat identity differs')
    for name in acceptance['merge_repeat_identical']:
        if outer[str(Path(a)/name)]['sha256']!=outer[str(Path(b)/name)]['sha256']:raise ValueError('actual repeat artifact differs')
    effort=json.loads((repo/m['effort']).read_text())
    if effort['pure_model_thinking_seconds'] is not None:raise ValueError('unobservable thinking claim')
    return dict(status='REMAINING_MAPPING_ARCHIVE_VALID',files_checked=len(outer),legacy_merge='REJECTED_UNDEFINED_AND_UNDERPRICED',new_entries=7,score=None)

if __name__=='__main__':print(json.dumps(check(sys.argv[1])))
