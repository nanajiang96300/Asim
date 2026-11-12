from pathlib import Path
import json,tarfile,hashlib,copy
raw=Path(__file__).resolve().parent

def check(row, archive, binary):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=row['archive']['sha256']:raise ValueError('archive SHA mismatch')
    with tarfile.open(archive) as tf:
        m=tf.getmember(row['case']+'/result.json');data=tf.extractfile(m).read()
    if hashlib.sha256(data).hexdigest()!=row['result_sha256']:raise ValueError('result SHA mismatch')
    r=json.loads(data)
    if r['status']!='MODELED_NUMERIC_CHECKED' or r['numeric_status']!='EXECUTED' or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or r['failed_batches']:raise ValueError('actual acceptance rejected')
    if r['task']!=row['task'] or r['seed']!=row['seed'] or r['asim']['binary_sha256']!=binary or r['asim']['finish_cycle']!=row['cycles']:raise ValueError('summary identity drift')
    batches=r['batches']
    if len(batches)!=96 or sorted(b['batch'] for b in batches)!=list(range(96)):raise ValueError('batch coverage')
    if not all(b['precision_bit_equal'] and b['stage_checks'] and all(c['bit_equal'] for c in b['stage_checks']) for b in batches):raise ValueError('stage or final mismatch')
    if max(b['task_error'] for b in batches)!=row['max_task_error']:raise ValueError('metric drift')
    return {'case':row['case'],'status':'ARCHIVED_RESULT_CONSISTENT','batches':len(batches),'cycles':row['cycles']}

if __name__=='__main__':
    s=json.loads((raw/'suite-v3/summary.json').read_text());assert s['status']=='SUITE_CHECKED'
    rows=[check(r,raw/'suite-v3'/r['archive']['path'],s['binary_sha256']) for r in s['cases']]
    bad=copy.deepcopy(s['cases'][0]);bad['cycles']+=1
    try:check(bad,raw/'suite-v3'/bad['archive']['path'],s['binary_sha256'])
    except ValueError as e:negative=str(e)
    else:raise AssertionError('summary drift accepted')
    (raw/'archive-audit.json').write_text(json.dumps({'status':'36_ARCHIVES_AND_RESULTS_CHECKED','scope':'independent read-only publication audit; freeze script still trusts accepted summary writer','checks':rows,'changed_cycle_negative':negative},indent=2)+'\n')
