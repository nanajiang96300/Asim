#!/usr/bin/env python3
"""Integrity and expected diagnostic states; never promotes a legacy operator."""
import hashlib,json,sys
from pathlib import Path


def check(path):
    path=Path(path);m=json.loads(path.read_text());repo=Path(__file__).resolve().parents[1]
    if m['schema']!='legacy-semantic-archive-v1':raise ValueError('unknown archive schema')
    seen=set()
    for e in m['artifacts']:
        p=Path(e['path'])
        if p.is_absolute() or '..' in p.parts or str(p) in seen:raise ValueError('unsafe/duplicate artifact')
        seen.add(str(p));p=repo/p
        if p.stat().st_size!=e['bytes']:raise ValueError('size mismatch '+str(p))
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=e['sha256']:raise ValueError('hash mismatch '+str(p))
    required={str(Path(m['campaign'])/'summary.json'),str(Path(m['stages'])/'summary.json')}
    required|={str(Path(run)/'result.json') for run in m['ns_regressions']}
    if not required<=seen:raise ValueError('missing bound summaries or regression results')
    raw=json.loads((repo/m['campaign']/'summary.json').read_text());stages=json.loads((repo/m['stages']/'summary.json').read_text())
    if raw['status']!='CORRECTNESS_BLOCKED' or stages['status']!='DIAGNOSTICS_COMPLETE_OPERATORS_BLOCKED' or any(c['complete_operator_passed'] or c['score'] is not None for c in stages['cases']):raise ValueError('unexpected operator validation claim')
    for root,summary in ((m['campaign'],raw),(m['stages'],stages)):
        if not {str(Path(root)/e['path']) for e in summary['artifacts']}<=seen:raise ValueError('incomplete nested artifact manifest')
    if len(stages['cases'])!=7 or sum(c['coverage_counts']['instructions'] for c in stages['cases'])!=714912:raise ValueError('incomplete instruction coverage')
    by_name={c['operator']:c for c in stages['cases']}
    for name,c in by_name.items():
        if name!='newton_schulz' and c['stage_counts']['regularized_A']!={'STAGE_MATCH':96}:raise ValueError('common Gram stage regression')
    for name,stage,state in [('cholesky_noblock','factor_L','STAGE_MATCH'),('ldl_noblock','factor_D','STAGE_MATCH'),('ldl_noblock','factor_L','STAGE_MISMATCH'),('block_richardson','preconditioner','STAGE_MISMATCH'),('block_richardson','initializer','STAGE_MISMATCH')]:
        if by_name[name]['stage_counts'][stage]!={state:96}:raise ValueError('unexpected stage diagnosis')
    for run in m['ns_regressions']:

        r=json.loads((repo/run/'result.json').read_text())
        if not {str(Path(run)/e['path']) for e in r['artifacts']}<=seen:raise ValueError('incomplete regression artifacts')
        if r['status'] not in ('STRUCTURE_CHECKED','MODELED_NUMERIC_CHECKED') or r['failed_batches'] or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE':raise ValueError('NS regression failed')
    return {'status':'LEGACY_SEMANTIC_ARCHIVE_VALID','files_checked':len(seen),'legacy_operators':'BLOCKED','hardware_timing':'OUT_OF_SCOPE','score':None}

if __name__=='__main__':print(json.dumps(check(sys.argv[1])))
