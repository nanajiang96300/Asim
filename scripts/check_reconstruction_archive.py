#!/usr/bin/env python3
"""Verify a frozen legacy diagnostic archive; does not promote operators."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest',type=Path)
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    manifest=json.loads(args.manifest.read_text());checked=0
    for key in ('artifacts','supporting_artifacts','small_artifacts'):
        for item in manifest[key]:
            p=repo/item['path'];data=p.read_bytes()
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('artifact changed: '+str(p))
            checked+=1
    if manifest.get('schema_version')=='precision-model-campaign-v1':
        gates=json.loads((args.manifest.parent/'gates.json').read_text())
        if gates['status']!='GATES_PASSED' or gates['score'] is not None or len(gates['cases'])!=5 or not all(c['passed'] for c in gates['cases']):raise ValueError('precision model gates incomplete')
        for seed in (42,123,456):
            r=json.loads((Path(manifest['raw_root'])/f'seed{seed}/result.json').read_text())
            if r['status']!='MODELED_NUMERIC_CHECKED' or r['score'] is not None or len(r['batches'])!=96 or not all(b.get('precision_bit_equal') is True for b in r['batches']):raise ValueError('missing exact FP16 bit-pattern evidence')
        if gates['hardware_calibration']!='NOT_DONE' or gates['G4']!='NOT_DONE':raise ValueError('unexpected precision calibration or G4 claim')
        print(json.dumps({'status':'PRECISION_MODEL_ARCHIVE_VALID','files_checked':checked,'bit_equal_batches':288,'hardware_calibration':'NOT_DONE','promotion':'BLOCKED'}))
        return
    if manifest.get('schema_version')=='unified-reconstruction-campaign-v1':
        gates=json.loads((args.manifest.parent/'gates.json').read_text())
        if gates['status']!='GATES_PASSED' or gates['score'] is not None or len(gates['cases'])!=6 or not all(c['passed'] for c in gates['cases']+gates['offline_mutations']):raise ValueError('unified gates incomplete')
        for seed in (42,123,456):
            r=json.loads((Path(manifest['raw_root'])/f'seed{seed}/result.json').read_text())
            if r['status']!='STRUCTURE_CHECKED' or r['score'] is not None or len(r['batches'])!=96 or r['failed_batches']:raise ValueError('missing three-path batch evidence')
        if len(gates['legacy_shared_replay'])!=7 or not all(c['same_as_frozen'] for c in gates['legacy_shared_replay']):raise ValueError('legacy replay changed')
        print(json.dumps({'status':'ARCHIVE_HASHES_AND_STRUCTURE_GATES_VALID','files_checked':checked,'NS_seeds':3,'batches_per_seed':96,'legacy_entries':7,'promotion':'BLOCKED_PENDING_PRECISION_G4_QUALITY'}))
        return
    cases=json.loads((args.manifest.parent/'cases.json').read_text())
    expected={'cholesky_block','cholesky_noblock','cholesky_noblock_merge','ldl_block','ldl_noblock','newton_schulz','block_richardson'}
    if len(cases)!=7 or {c['operator'] for c in cases}!=expected:raise ValueError('incomplete campaign')
    for c in cases:
        if c['asim_status']!='COMPLETED' or sum(c['status_counts'].values())!=96:raise ValueError('missing Asim or batch evidence')
        if c['score'] is not None or c['verified_comparison_eligible']:raise ValueError('unexpected promotion')
    if manifest['score'] is not None or manifest['verified_comparison']['rows']:raise ValueError('diagnostic scores must be null')
    print(json.dumps({'status':'ARCHIVE_HASHES_AND_DIAGNOSTIC_BOUNDARIES_VALID','files_checked':checked,'operators':7,'operator_correctness':manifest['status'],'promotion':'BLOCKED'}))

if __name__=='__main__':main()
