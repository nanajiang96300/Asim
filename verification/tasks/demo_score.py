"""Task-local cycle ratio demonstration; no research promotion or self-reported scores."""
import json
from pathlib import Path
from verification.asim.runner import file_hash
from verification.contracts.core import Invalid

def checked(path,expected_sha256):
    path=Path(path)
    if file_hash(path)!=expected_sha256:raise Invalid('owner-frozen result SHA mismatch')
    r=json.loads(path.read_text())
    if r['task']['schema']!='affine-task-v1' or r['status']!='MODELED_NUMERIC_CHECKED' or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or r['numeric_status']!='EXECUTED' or r['failed_batches'] or len(r['batches'])!=r['task']['batch_size'] or not all(x['precision_bit_equal'] for x in r['batches']):raise Invalid('demo score requires complete affine numerical/event acceptance')
    required={'asim.json','config.json','models.json','instructions.jsonl','trace.csv','rw_events.jsonl','schedule.json','inputs.npz','references.npz','outputs.npy','numerics_started.json','pipeline_statistics.json','task_inputs.npz','dispatch.jsonl','instruction_pipeline_detail.csv','simulator.log'}
    index={e['path']:e for e in r['artifacts']}
    if len(index)!=len(r['artifacts']) or not required<=set(index):raise Invalid('missing or duplicate score evidence')
    for name,e in index.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise Invalid('unsafe score evidence path')
        f=path.parent/p
        if not f.is_file() or f.stat().st_size!=e['bytes'] or file_hash(f)!=e['sha256']:raise Invalid('score artifact changed: '+name)
    return r

def score(baseline,candidate,baseline_sha256,candidate_sha256):
    a=checked(baseline,baseline_sha256);b=checked(candidate,candidate_sha256)
    for key in ['task','algorithm','precision_policy','seed','comparison_identity']:
        if a[key]!=b[key]:raise Invalid('demo baseline/candidate identity mismatch: '+key)
    return {'schema':'affine-demo-cycle-score-v1','status':'DEMO_SCORE_COMPUTED','score':a['asim']['finish_cycle']/b['asim']['finish_cycle'],'definition':'accepted baseline cycles / accepted candidate cycles','research_score':None,'promotion_allowed':False,'optimization_claimed':False,'baseline_cycle':a['asim']['finish_cycle'],'candidate_cycle':b['asim']['finish_cycle'],'communication_status':'NOT_APPLICABLE_REAL_AFFINE_TASK'}
