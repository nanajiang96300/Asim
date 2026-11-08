#!/usr/bin/env python3
"""Actual legacy Merge rejection and Scalar workload latency diagnostics."""
import argparse,csv,json,gc
from collections import Counter,defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from verification.asim.runner import simulate,file_hash
from verification.legacy.replay import external_inputs,execute_batch
from verification.reporting.pipeline import statistics,write_detail
from scripts.archive_checks import indexed,verify_files,PROCESS_REQUIRED


def run(binary,config,cwd,output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);rows=[];probe=[];repo=Path(__file__).resolve().parents[1]
    registry=json.loads((repo/'orchestrator/operator_registry.json').read_text())['operators']
    for name in ('cholesky_noblock','cholesky_noblock_merge'):
        entry=registry[name];models=json.loads((repo/entry['config']).read_text())
        for m in models['models']:m['batch_size']=96
        dest=root/name;receipt=simulate(binary,config,models,entry['mode'],cwd,dest,max_cycles=1000000)
        if receipt['status']!='COMPLETED':raise ValueError('legacy Asim failed')
        verify_files(dest,indexed(receipt['artifacts'],PROCESS_REQUIRED))
        records=[json.loads(line) for line in (dest/'instructions.jsonl').read_text().splitlines()]
        if len(records)!=1:raise ValueError('one legacy operation required')
        record=records[0];h=record['header'];ops=record['instructions'];dispatch=[json.loads(line) for line in (dest/'dispatch.jsonl').read_text().splitlines()]
        stats=statistics(dest/'trace.csv',ops,receipt['finish_cycle'],24,dispatch)
        if stats['status']!='TRACE_VALID':raise ValueError('legacy trace invalid')
        (dest/'pipeline_statistics.json').write_text(json.dumps(stats,indent=2));write_detail(dest/'instruction_pipeline_detail.csv',stats)
        values,oracle=external_inputs(h,42);np.savez(dest/'inputs.npz',**{str(i):v for i,v in enumerate(values)});np.savez(dest/'task_reference.npz',inverse=oracle['inverse'],A=oracle['A'])
        groups=defaultdict(list)
        for i in ops:groups[i['batch']].append(i)
        if set(groups)!=set(range(96)):raise ValueError('missing batch coverage')
        batches=[execute_batch(h,groups[b],values,b,oracle['inverse'][b]) for b in range(96)]
        (dest/'batches.json').write_text(json.dumps(batches,indent=2));scalar=[i for i in ops if 17<=i['opcode']<=21]
        row=dict(operator=name,cycles=receipt['finish_cycle'],instruction_count=len(ops),scalar_instructions=len(scalar),scalar_declared_elements=sum(max(1,i['compute_size']) for i in scalar),illegal_scalar_width_count=sum(i['compute_size']>1 for i in scalar),scalar_duration_sum=stats['pipelines']['Scalar']['duration_sum'],status_counts=dict(Counter(b['status'] for b in batches)),first_bad=batches[0].get('first_bad'),actual_source_sha256=file_hash(repo/entry['source']),scope='legacy logical complex128/precision2 timing; unverified inverse output',score=None)
        rows.append(row);print(name,row['cycles'],row['status_counts'],flush=True)
        del records,record,ops,groups,dispatch,stats,scalar,values,oracle,batches;gc.collect()
    for width in (1,16):
        dest=root/('scalar_width'+str(width));receipt=simulate(binary,config,{'models':[dict(name='scalar_cost_probe',width=width)]},'scalar_cost_probe_test',cwd,dest,max_cycles=10000)
        if receipt['status']!='COMPLETED':raise ValueError('cost probe failed')
        verify_files(dest,indexed(receipt['artifacts'],PROCESS_REQUIRED))
        with (dest/'trace.csv').open() as f:events=[row for row in csv.DictReader(f) if row['name']=='probe.scalar' and row['unit'].endswith('_Scalar')]
        if len(events)!=1:raise ValueError('one actual Scalar event required')
        duration=float(events[0]['end_cycle'])-float(events[0]['start_cycle'])
        recs=[json.loads(line) for line in (dest/'instructions.jsonl').read_text().splitlines()];i=next(i for i in recs[0]['instructions'] if i['id']=='probe.scalar')
        probe.append(dict(width=width,cycles=receipt['finish_cycle'],scalar_duration=duration,actual_compute_size=i['compute_size'],declared_view_elements=i['semantic']['dst']['rows']*i['semantic']['dst']['cols'],workload_status='INVALID_SCALAR_WORKLOAD' if width!=1 else 'ONE_SCALAR_DIAGNOSTIC',numeric_certified=False,score=None))
        print('scalar_width',width,probe[-1],flush=True)
    expected=rows[0]['scalar_declared_elements']==rows[1]['scalar_declared_elements'] and rows[1]['illegal_scalar_width_count']>0 and all(r['status_counts']=={'SEMANTIC_AMBIGUITY':96} for r in rows) and probe[0]['cycles']==probe[1]['cycles'] and probe[0]['scalar_duration']==probe[1]['scalar_duration']
    summary=dict(schema='merge-forensics-v1',status='ANOMALY_REPRODUCED' if expected else 'UNEXPECTED_DIAGNOSTIC',legacy_cases=rows,scalar_probes=probe,score=None,binary_sha256=file_hash(binary),conclusion='legacy Merge has undefined reduction plus scalar fixed-latency under-accounting; no certified speedup or guessed numeric answer')
    (root/'summary.json').write_text(json.dumps(summary,indent=2));return expected

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','config','cwd','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(0 if run(a.binary,json.loads(a.config.read_text()),a.cwd,a.output) else 1)
