"""Source-annotated legacy reconstruction campaign. Never turns diagnostic cycles into a verified ranking."""
import argparse
from collections import Counter,defaultdict
import csv
import gc
import json
from pathlib import Path
import subprocess
import numpy as np
from verification.asim.runner import simulate,file_hash
from verification.legacy.replay import external_inputs,execute_batch
from verification.reporting.pipeline import statistics,write_detail


EXPECTED_ENTRIES={'cholesky_block','cholesky_noblock','cholesky_noblock_merge','ldl_block','ldl_noblock','newton_schulz','block_richardson'}

def require_complete_registry(registry):
    if set(registry)!=EXPECTED_ENTRIES:raise ValueError('campaign requires exactly seven legacy entries')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','config','registry','cwd','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());registry=json.loads(args.registry.read_text())['operators']
    if config['num_cores']!=24 or config['precision']!=2:raise ValueError('legacy campaign requires original24-core precision2 cost profile')
    require_complete_registry(registry)
    source_root=Path(__file__).resolve().parents[2];cases=[]
    for name,entry in sorted(registry.items()):
        models=json.loads((source_root/entry['config']).read_text())
        for m in models['models']:m['batch_size']=96
        path=root/name
        print('START',name,flush=True)
        receipt=simulate(args.binary,config,models,entry['mode'],args.cwd,path,timeout=120,max_cycles=1000000)
        result={'operator':name,'mode':entry['mode'],'source':entry['source'],'config_entry':entry['config'],'model_config':models,
                'asim_status':receipt['status'],'finish_cycle':receipt['finish_cycle'],'numeric_status':'NOT_RUN','correctness':'NOT_VALIDATED','score':None,'verified_comparison_eligible':False}
        try:
            if receipt['status']!='COMPLETED':raise ValueError('Asim did not complete')
            records=[json.loads(s) for s in (path/'instructions.jsonl').read_text().splitlines()]
            if len(records)!=1:raise ValueError('expected one operation')
            record=records[0];header=record['header'];instructions=record['instructions']
            if not header or header.get('schema')!='legacy-source-views-v1':raise ValueError('source semantic migration missing')
            if {c['core'] for c in header.get('capacities',[])}!=set(range(24)) or len(header['capacities'])!=24:raise ValueError('missing24-core capacity evidence')
            groups=defaultdict(list)
            for i in instructions:groups[i['batch']].append(i)
            dispatch=[json.loads(s) for s in (path/'dispatch.jsonl').read_text().splitlines()]
            if set(groups)!=set(range(96)) or len(dispatch)!=96 or set(d['batch'] for d in dispatch)!=set(range(96)):raise ValueError('batch coverage missing')
            counts=Counter(d['core'] for d in dispatch)
            if set(counts)!=set(range(24)) or set(counts.values())!={4}:raise ValueError('24-core placement missing')
            stats=statistics(path/'trace.csv',instructions,receipt['finish_cycle'],24,dispatch)
            (path/'pipeline_statistics.json').write_text(json.dumps(stats,indent=2));write_detail(path/'instruction_pipeline_detail.csv',stats)
            if stats['status']!='TRACE_VALID':raise ValueError('trace invalid')
            result.update(core_tile_counts=dict(counts),instruction_count=len(instructions),semantic_annotated_count=sum(i['semantic'] is not None for i in instructions),
                          semantic_ambiguous_count=sum(bool((i['semantic'] or {}).get('invalid_reason')) for i in instructions),opcode_counts=stats['source_opcode_counts'])
            values,oracle=external_inputs(header,42)
            np.savez(path/'inputs.npz',**{str(n):a for n,a in enumerate(values)})
            np.savez(path/'task_reference.npz',**{k:v for k,v in oracle.items()})
            expected=oracle['xhat'] if name=='block_richardson' else oracle['inverse']
            rows=[execute_batch(header,groups[b],values,b,expected[b]) for b in range(96)]
            result['numeric_status']='EXECUTED';result['batches']=rows;result['status_counts']=dict(Counter(r['status'] for r in rows))
            result['failed_batches']=[r['batch'] for r in rows if r['status']!='STRUCTURE_MATCH']
            result['correctness']='STRUCTURE_FAILED' if result['failed_batches'] else 'STRUCTURE_MATCH'
            result['structural_contract_issues']=[]
            outshape=header['outputs'][0]['shape'];want=[96,header['U'],header['U']]
            if outshape!=want:result['structural_contract_issues'].append({'code':'DECLARED_OUTPUT_SHAPE_MISMATCH','declared':outshape,'task':want})
            if name=='newton_schulz':result['structural_contract_issues'].append({'code':'INPUT_ROLE_MISMATCH','detail':'Model supplies H/RegI/X_init/C; Op consumes first three as A/X0/C'})
            if result['structural_contract_issues']:result['correctness']='CONTRACT_FAILED'
            result['performance_status']='DIAGNOSTIC_ONLY_UNVERIFIED'
        except (ValueError,KeyError,TypeError,OSError) as error:
            result.update(correctness='INFRA_OR_SEMANTIC_GATE_FAILED',reason=str(error))
        (path/'reconstruction.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        cases.append({k:v for k,v in result.items() if k!='batches'})
        print('DONE',name,result['asim_status'],result['finish_cycle'],result['correctness'],result.get('status_counts'),flush=True)
        records=record=header=instructions=groups=dispatch=stats=values=oracle=rows=None
        gc.collect()
    structure_passed=len(cases)==7 and all(c['correctness']=='STRUCTURE_MATCH' for c in cases)
    summary={'schema_version':'legacy-reconstruction-campaign-v1','status':'STRUCTURE_MATCH_ONLY' if structure_passed else 'CORRECTNESS_BLOCKED','all_numerics_passed':structure_passed,
        'score':None,'verified_comparison':{'status':'BLOCKED','rows':[],'reason':'legacy correctness/precision/schedule gates not all passed'},
        'cases':cases,'binary_sha256':file_hash(args.binary),'binary_source_commit':subprocess.check_output(['git','-C',str(args.cwd),'rev-parse','HEAD'],text=True).strip(),
        'runner_commit':subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip(),
        'sources':[{'path':str(p.relative_to(source_root)),'sha256':file_hash(p)} for base in ('src','verification') for p in sorted((source_root/base).rglob('*')) if p.suffix in ('.py','.cc','.h')],
        'artifacts':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(root.rglob('*')) if p.is_file()],
        'limitations':['structure complex128 replay is not hardware FP16','ambiguous source indices are explicit errors, not guessed','diagnostic inverse/BRI output scopes and NS dimensions differ; no unified performance ranking','full G4/quality/reproducibility not certified','independent same-algorithm path not implemented for seven legacy entries; task reference alone cannot certify full three-path validation','capacities follow current Asim SRAM model units, not calibrated physical 910B capacity']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (root/'UNVERIFIED_TIMING_DIAGNOSTICS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['operator','mode','finish_cycle','asim_status','numeric_status','correctness','verified_comparison_eligible']);writer.writeheader()
        writer.writerows({k:c.get(k) for k in writer.fieldnames} for c in cases)
    print('ARCHIVED',root,summary['status'],'; no verified ranking',flush=True)
    return 0 if structure_passed else 1

if __name__=='__main__':raise SystemExit(main())
