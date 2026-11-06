#!/usr/bin/env python3
"""Verify the frozen cross-algorithm evidence, without promoting candidates."""
import hashlib,json,sys
from pathlib import Path

def check(path):
    repo=Path(__file__).resolve().parents[1];m=json.loads(Path(path).read_text());seen=set()
    if m['schema']!='generality-archive-v1':raise ValueError('unknown schema')
    for e in m['artifacts']:
        rel=Path(e['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:raise ValueError('unsafe/duplicate path')
        seen.add(str(rel));p=repo/rel
        if p.stat().st_size!=e['bytes']:raise ValueError('artifact size mismatch')
        sha=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):sha.update(b)
        if sha.hexdigest()!=e['sha256']:raise ValueError('artifact hash mismatch '+str(rel))
    required={m['campaign']+'/summary.json',m['effort'],m['core_identity']}
    if not required<=seen:raise ValueError('unbound primary evidence')
    s=json.loads((repo/m['campaign']/'summary.json').read_text())
    if s['status']!='SUPPORTED_SCOPE_PASSED' or len(s['cases'])!=5 or len(s['negative_cases'])!=4:raise ValueError('campaign incomplete')
    for case in s['cases']:
        root=Path(m['campaign'])/case['case'];r=json.loads((repo/root/'result.json').read_text())
        if str(root/'result.json') not in seen or not {str(root/e['path']) for e in r['artifacts']}<=seen:raise ValueError('incomplete case evidence')
        if r['status'] not in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') or r['failed_batches'] or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or len(r['batches'])!=r['task']['batch_size']:raise ValueError('case failed')
        if r['operator']=='cholesky_explicit_v1' and 'iterations' in r['algorithm']:raise ValueError('direct algorithm was bound as NS')
        if r['status']=='MODELED_NUMERIC_CHECKED' and not all(b['precision_bit_equal'] for b in r['batches']):raise ValueError('precision bit mismatch')
    if any(n['status']=='STRUCTURE_MATCH' or n['actual_mutant_simulated'] for n in s['negative_cases']):raise ValueError('incorrect negative claim')
    for run in m['daily_runs']:
        r=json.loads((repo/run/'result.json').read_text())
        if str(Path(run)/'result.json') not in seen or not {str(Path(run)/e['path']) for e in r['artifacts']}<=seen:raise ValueError('incomplete daily evidence')
        if r['status'] not in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') or r['failed_batches']:raise ValueError('daily failed')
    effort=json.loads((repo/m['effort']).read_text())
    if effort['pure_model_thinking_seconds'] is not None or effort['agent_or_uninstrumented_seconds']<0:raise ValueError('unobservable thinking-time claim')
    identity=json.loads((repo/m['core_identity']).read_text())
    if len(identity)!=4 or any(e['unchanged_from']!='7ea3405e' for e in identity):raise ValueError('core reuse evidence incomplete')
    return dict(status='GENERALITY_SUPPORTED_SCOPE_ARCHIVE_VALID',files_checked=len(seen),algorithm_families=2,pure_thinking_seconds=None,score=None)

if __name__=='__main__':print(json.dumps(check(sys.argv[1])))
