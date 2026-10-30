"""Audit every actual legacy instruction and capture executed stage values.

Requires completed, hash-verified Asim evidence. No resumed execution after a fault.
"""
import argparse,csv,gc,json,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from verification.asim.runner import file_hash
from verification.legacy.campaign import EXPECTED_ENTRIES
from verification.legacy.coverage import audit
from verification.legacy.replay import external_inputs,execute_batch
from verification.legacy.stages import capture_plan,targets,compare


def validate_manifest(summary):
    entries={}
    for e in summary['artifacts']:
        p=Path(e['path'])
        if p.is_absolute() or '..' in p.parts or e['path'] in entries:raise ValueError('unsafe or duplicate campaign artifact')
        entries[e['path']]=e
    required={'asim.json','instructions.jsonl','dispatch.jsonl','trace.csv','inputs.npz','task_reference.npz','config.json','models.json','simulator.log','reconstruction.json','pipeline_statistics.json','instruction_pipeline_detail.csv','rw_events.jsonl'}
    for name in EXPECTED_ENTRIES:
        if not {name+'/'+p for p in required}<=entries.keys():raise ValueError('missing bound campaign evidence: '+name)
    return entries


def source_fingerprint():
    root=Path(__file__).resolve().parents[2]
    return [{'path':str(p.relative_to(root)),'sha256':file_hash(p)} for p in sorted((root/'verification').rglob('*.py')) if 'tests' not in p.parts]


def run(campaign,output):
    campaign=Path(campaign).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((campaign/'summary.json').read_text())
    if {c['operator'] for c in summary['cases']}!=EXPECTED_ENTRIES or len(summary['cases'])!=7:raise ValueError('seven-entry evidence required')
    entries=validate_manifest(summary);sources=source_fingerprint();cases=[]
    for case in summary['cases']:
        name=case['operator'];raw=campaign/name;dest=output/name;dest.mkdir()
        receipt=json.loads((raw/'asim.json').read_text())
        if receipt['status']!='COMPLETED' or receipt['returncode']!=0 or receipt['finish_cycle']!=case['finish_cycle'] or receipt['binary_sha256']!=summary['binary_sha256'] or receipt['mode']!=case['mode']:raise ValueError('Asim receipt/campaign mismatch: '+name)
        receipt_names=[e['path'] for e in receipt['artifacts']]
        if len(set(receipt_names))!=len(receipt_names) or not {'instructions.jsonl','dispatch.jsonl','trace.csv','config.json','models.json','simulator.log','rw_events.jsonl'}<=set(receipt_names):raise ValueError('missing/duplicate actual process evidence')
        for e in receipt['artifacts']:
            if entries.get(name+'/'+e['path'])!={**e,'path':name+'/'+e['path']}:raise ValueError('receipt differs from campaign manifest')
        # The frozen campaign manifest binds receipt, input fixture and every raw
        # artifact. Recheck per case to bound memory, including the observer bytes.
        for e in summary['artifacts']:
            if Path(e['path']).parts[0]!=name:continue
            p=campaign/e['path']
            if not p.is_file() or p.stat().st_size!=e['bytes'] or file_hash(p)!=e['sha256']:raise ValueError('campaign artifact changed: '+str(p))
        records=[json.loads(line) for line in (raw/'instructions.jsonl').read_text().splitlines()]
        if len(records)!=1:raise ValueError('one actual operation required')
        header=records[0]['header'];groups=defaultdict(list)
        for i in records[0]['instructions']:groups[i['batch']].append(i)
        if set(groups)!=set(range(96)):raise ValueError('requires batch96')
        inputs,oracle=external_inputs(header,42)
        with np.load(raw/'inputs.npz') as archived:
            if set(archived.files)!={str(i) for i in range(len(inputs))} or any(not np.array_equal(archived[str(i)],v) for i,v in enumerate(inputs)):raise ValueError('input fixture differs from actually archived fixture')
        counts=Counter();issue_groups={};stages=[];primary=Counter();snapshots={}
        wanted=oracle['xhat'] if name=='block_richardson' else oracle['inverse']
        for batch,ins in sorted(groups.items()):
            coverage=audit(header,ins,inputs,batch,wanted[batch].shape);counts.update(coverage['counts'])
            for issue in coverage['issues']:
                key=(issue['code'],issue['site'],issue['reason'])
                if key not in issue_groups:issue_groups[key]={'code':key[0],'site':key[1],'reason':key[2],'count':0,'batches':set(),'first':issue}
                issue_groups[key]['count']+=1;issue_groups[key]['batches'].add(batch)
            plan=capture_plan(header,ins)
            r=execute_batch(header,ins,inputs,batch,wanted[batch],captures=plan)
            observations=r.pop('_observations');primary[r['status']]+=1
            expected=targets(name,oracle['A'][batch],float(oracle['lambda']))
            comparisons=compare(observations,expected)
            stages.append({'batch':batch,'execution':r,'stages':comparisons})
            for key,value in observations.items():
                if value['status']=='CAPTURED':snapshots[f'batch{batch}.{key}']=value['value']
        grouped=[dict(v,batches=sorted(v['batches'])) for v in issue_groups.values()]
        np.savez(dest/'actual_stage_snapshots.npz',**snapshots)
        (dest/'batches.json').write_text(json.dumps(stages,indent=2,allow_nan=False))
        (dest/'coverage_issues.json').write_text(json.dumps(grouped,indent=2))
        stage_counts={k:dict(Counter(b['stages'][k]['status'] for b in stages)) for k in stages[0]['stages']}
        row={'operator':name,'asim_finish_cycle':receipt['finish_cycle'],'primary_status_counts':dict(primary),'coverage_counts':dict(counts),'stage_counts':stage_counts,'complete_operator_passed':False,'score':None,
             'raw_receipt_sha256':file_hash(raw/'asim.json'),'raw_instruction_sha256':file_hash(raw/'instructions.jsonl')}
        (dest/'summary.json').write_text(json.dumps(row,indent=2));cases.append(row)
        print(name,json.dumps(stage_counts),flush=True)
        del records,groups,inputs,oracle,stages,snapshots;gc.collect()
    if source_fingerprint()!=sources:raise ValueError('verification sources changed during diagnostic campaign')
    report={'schema':'legacy-semantic-stage-campaign-v1','status':'DIAGNOSTICS_COMPLETE_OPERATORS_BLOCKED','cases':cases,'campaign':str(campaign),'campaign_summary_sha256':file_hash(campaign/'summary.json'),
            'runner_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'hardware_timing_certification':'OUT_OF_SCOPE_USER_BOARD_VALIDATION','score':None,
            'limits':['logical complex128 replay with legacy precision2 cycle model, not FP16 validation','stage targets diagnose mathematical stages; not full same-algorithm verification','static coverage after first fault is not numeric execution','no continuation, invented zero or implicit reduction after ambiguity'],
            'sources':sources,
            'artifacts':[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(output.rglob('*')) if p.is_file()]}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    r=run(a.campaign,a.output);print(r['status']);return 0 # diagnostic harness succeeded; operators remain blocked

if __name__=='__main__':raise SystemExit(main())
