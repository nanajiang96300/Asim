"""Serial Asim source preparation for a frozen multi-noise QPSK campaign.

Only missing task variants run Asim. No operator/registry/runtime modifications.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from verification.contracts.core import Invalid
from verification.contracts.mmse import MMSETask
from verification.operators.catalog import adapter
from verification.profiles.numeric import policy_for
from verification.reconstruction.daily import load_runtime,sha
from verification.reconstruction.runner import evaluate
from verification.communication.runner import load_case

PRODUCER_FILES=['verification/communication/qpsk_sources.py','verification/communication/runner.py',
                'verification/communication/metrics.py','verification/planning/handoff.py','verification/planning/execution.py']


def guard(c,repo):
    if c['producer_sources']!={p:sha(repo/p) for p in PRODUCER_FILES}:raise Invalid('source producer changed after freeze')
    if sha(repo/c['runtime']['path'])!=c['runtime']['sha256']:raise Invalid('source runtime changed')
    return load_runtime(repo/c['runtime']['path'],repo)


def validate(c):
    required={'schema','runtime','producer_sources','task','noise_variances','algorithms','source_seed','budget'}
    if set(c)!=required or c['schema']!='qpsk-source-campaign-v1':raise Invalid('unknown source preparation contract')
    t=MMSETask(**c['task']);t.validate()
    if t.profile!='modeled_fp16_fp32_real_block_v1' or t.scope!='mmse_h_y_to_xhat_v1':raise Invalid('full FP16 task required')
    if type(c['source_seed']) is not int or c['source_seed']<0:raise Invalid('invalid source seed')
    vals=c['noise_variances']
    if not isinstance(vals,list) or not vals or len(vals)>5 or len(set(vals))!=len(vals):raise Invalid('distinct frozen noise variances required')
    for lam in vals:MMSETask(**{**c['task'],'regularization':lam}).validate()
    ops=c['algorithms']
    if not isinstance(ops,list) or not ops or len(ops)>7 or len({o['operator'] for o in ops})!=len(ops):raise Invalid('distinct operator list required')
    for op in ops:
        if set(op)!={'operator','iterations','source'} or set(op['source'])!={'path','sha256'}:raise Invalid('incomplete frozen operator source')
    b=c['budget']
    if set(b)!={'max_new_asim_runs','max_new_source_bytes','min_free_bytes','max_wall_seconds'} or any(type(v) is not int or v<=0 for v in b.values()):raise Invalid('positive source budgets required')
    missing=len(ops)*sum(lam!=t.regularization for lam in vals)
    if missing>b['max_new_asim_runs']:raise Invalid('new Asim count budget exceeded')
    return t


def run(c,repo,output):
    base=validate(c);repo=Path(repo).resolve();output=Path(output).resolve()
    if output.exists():raise Invalid('refusing existing source campaign')
    rt=guard(c,repo);output.mkdir(parents=True,exist_ok=False)
    if rt.get('library_dir'):os.environ['LD_LIBRARY_PATH']=str(repo/rt['library_dir'])
    start=time.perf_counter();rows=[];used=0;runs=0;resource_stop=False
    for op in c['algorithms']:
        original=repo/op['source']['path']
        if sha(original)!=op['source']['sha256']:raise Invalid('frozen original source changed')
        old=json.loads(original.read_text())
        if old['task']!=base.frozen() or old['operator']!=op['operator']:raise Invalid('source task/operator differs')
        _,params,_,_=adapter(op['operator'],base,op['iterations'])
        if old['algorithm']!=params:raise Invalid('source algorithm parameters differ')
        load_case({**op['source'],'role':'historical_diagnostic'},rt,repo)
        estimate=int(sum(a['bytes'] for a in old['artifacts'])*1.15)
        for lam in c['noise_variances']:
            cid=f'case{len(rows):03d}';task=MMSETask(**{**c['task'],'regularization':lam})
            record=dict(id=cid,operator=op['operator'],iterations=op['iterations'],regularization=lam)
            if lam==base.regularization:
                record.update(status='SOURCE_CHECKED',result_path=str(original),result_sha256=sha(original),reused=True,
                              finish_cycle=old['asim']['finish_cycle'])
            elif resource_stop or used+estimate>c['budget']['max_new_source_bytes'] or shutil.disk_usage(output).free-estimate<c['budget']['min_free_bytes'] or time.perf_counter()-start>c['budget']['max_wall_seconds']:
                resource_stop=True;record.update(status='NOT_RUN_RESOURCE_BUDGET')
            else:
                guard(c,repo);print(json.dumps(dict(phase='ASIM_SOURCE_START',**record)),flush=True)
                config=json.loads((repo/rt['config']).read_text());config['precision']=policy_for(task.profile).bytes
                for key in ('dram_config_path','icnt_config_path'):
                    if config.get(key):config[key]=str((repo/rt['cwd']/config[key]).resolve())
                directory=output/cid
                r=evaluate(op['operator'],task,op['iterations'],c['source_seed'],repo/rt['binary'],config,directory,directory)
                runs+=1;guard(c,repo)
                used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                if r['status']=='MODELED_NUMERIC_CHECKED':
                    load_case(dict(path=str(directory/'result.json'),sha256=sha(directory/'result.json'),role='historical_diagnostic'),rt,repo)
                record.update(status='SOURCE_CHECKED' if r['status']=='MODELED_NUMERIC_CHECKED' else 'SOURCE_GATE_FAILED',
                              source_status=r['status'],result_path=str(directory/'result.json'),result_sha256=sha(directory/'result.json'),
                              reused=False,finish_cycle=r.get('asim',{}).get('finish_cycle'))
            rows.append(record);print(json.dumps(record),flush=True)
            (output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    guard(c,repo)
    result=dict(schema='qpsk-source-result-v1',status='ASIM_SOURCES_CHECKED' if all(r['status']=='SOURCE_CHECKED' for r in rows) else 'SOURCE_SUITE_INCOMPLETE',
                cases=rows,new_asim_runs=runs,new_source_bytes=used,elapsed_seconds=time.perf_counter()-start,
                pure_model_thinking_seconds=None,score=None,runtime=c['runtime'],producer_sources=c['producer_sources'])
    (output/'contract.json').write_text(json.dumps(c,indent=2)+'\n');(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contract',required=True,type=Path);ap.add_argument('--contract-sha256',required=True)
    ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if sha(a.contract)!=a.contract_sha256:raise Invalid('approved source contract changed')
    r=run(json.loads(a.contract.read_text()),Path(__file__).resolve().parents[2],a.output)
    if sha(a.contract)!=a.contract_sha256:raise Invalid('contract changed during source preparation')
    print(json.dumps({k:r[k] for k in ('status','new_asim_runs','new_source_bytes','elapsed_seconds')}))
    return 0 if r['status']=='ASIM_SOURCES_CHECKED' else 1


if __name__=='__main__':raise SystemExit(main())
