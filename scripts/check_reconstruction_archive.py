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
    cases=json.loads((args.manifest.parent/'cases.json').read_text())
    expected={'cholesky_block','cholesky_noblock','cholesky_noblock_merge','ldl_block','ldl_noblock','newton_schulz','block_richardson'}
    if len(cases)!=7 or {c['operator'] for c in cases}!=expected:raise ValueError('incomplete campaign')
    for c in cases:
        if c['asim_status']!='COMPLETED' or sum(c['status_counts'].values())!=96:raise ValueError('missing Asim or batch evidence')
        if c['score'] is not None or c['verified_comparison_eligible']:raise ValueError('unexpected promotion')
    if manifest['score'] is not None or manifest['verified_comparison']['rows']:raise ValueError('diagnostic scores must be null')
    print(json.dumps({'status':'ARCHIVE_HASHES_AND_DIAGNOSTIC_BOUNDARIES_VALID','files_checked':checked,'operators':7,'operator_correctness':manifest['status'],'promotion':'BLOCKED'}))

if __name__=='__main__':main()
