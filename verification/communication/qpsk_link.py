"""Complete frozen uncoded QPSK bit-to-bit tests on actual Asim instruction templates.

Fresh H/bits/noise are functional replay inputs. Asim timing is certified once
per unchanged task/algorithm/shape/precision, never invented for each frame.
"""
import argparse
import gc
import hashlib
import json
import math
import platform
import shutil
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from verification.communication.qpsk import modulate,measurements,awgn_controls
from verification.communication.metrics import outage_interval
from verification.communication.runner import load_case,check_artifacts
from verification.contracts.core import Invalid,digest
from verification.contracts.mmse import MMSETask,decode_checked,task_reference
from verification.fixtures.rayleigh import real_block
from verification.operators.catalog import adapter
from verification.profiles.numeric import policy_for,bit_equal
from verification.reconstruction.binding import validate_actual
from verification.reconstruction.daily import load_runtime,sha
from verification.reconstruction.runner import error
from verification.reconstruction.source_views import execute_batch
from verification.reconstruction.stream import load_operation


def sources(repo):
    return {str(p.relative_to(repo)):sha(p) for p in sorted((repo/'verification/communication').glob('*.py'))}


def validate(c):
    keys={'schema','runtime','source_index','evaluator_sources','channel_seeds','frame_seeds','limits','budget','control_seed','control_symbols'}
    if set(c)!=keys or c['schema']!='uncoded-qpsk-link-v1':raise Invalid('unknown/incomplete link contract')
    for key in ('channel_seeds','frame_seeds'):
        vals=c[key]
        if not isinstance(vals,list) or len(vals)<2 or len(vals)>8 or len(vals)!=len(set(vals)) or any(type(v) is not int or v<0 for v in vals):raise Invalid('distinct frozen channel/frame seeds required')
    limits=c['limits']
    if set(limits)!={'max_ber_increase','max_ser_increase','max_evm_increase','max_outage_probability','family_alpha','min_independent_channels'}:
        raise Invalid('incomplete QPSK statistical limits')
    for key in ('max_ber_increase','max_ser_increase','max_evm_increase'):
        if type(limits[key]) not in (int,float) or not math.isfinite(limits[key]) or limits[key]<0:raise Invalid('invalid metric tolerance')
    for key in ('max_outage_probability','family_alpha'):
        if type(limits[key]) not in (int,float) or not 0<limits[key]<1:raise Invalid('invalid quality probability')
    if type(limits['min_independent_channels']) is not int or limits['min_independent_channels']<2:raise Invalid('independent channel minimum required')
    if set(c['budget'])!={'max_batch_replays','max_output_bytes','min_free_bytes','max_wall_seconds'} or any(type(v) is not int or v<=0 for v in c['budget'].values()):raise Invalid('bounded link resources required')


def frame(task,hseed,fseed):
    task.validate();B,M,U,R=task.batch_size,task.nr,task.nt,task.nrhs
    hrng=np.random.default_rng(hseed);rng=np.random.default_rng([fseed,hseed])
    H=(hrng.standard_normal((B,M,U))+1j*hrng.standard_normal((B,M,U)))/math.sqrt(2*M)
    bits=rng.integers(0,2,(B,U,R,2),dtype=np.uint8);symbols=modulate(bits)
    noise=math.sqrt(task.regularization/2)*(rng.standard_normal((B,M,R))+1j*rng.standard_normal((B,M,R)))
    Y=H@symbols+noise;p=policy_for(task.profile)
    inputs=dict(H=np.stack([real_block(h) for h in H]),Y=np.stack([real_block(y) for y in Y]),
                RegI=task.regularization*np.eye(2*U),Identity=np.eye(2*U))
    inputs={k:p.store(v) for k,v in inputs.items()}
    return dict(H=H,Y=Y,bits=bits,symbols=symbols,noise=noise,inputs=inputs)


class Replayer:
    def __init__(self,path,r,task):
        self.r=r;self.task=task;self.path=path
        records=[load_operation(path.parent/'instructions.jsonl')]
        if len(records)!=1:raise Invalid('one actual operation required')
        self.record=records[0];self.dispatch=[json.loads(x) for x in (path.parent/'dispatch.jsonl').read_text().splitlines()]
        self.config=json.loads((path.parent/'config.json').read_text())
        _,params,_,self.reference=adapter(r['operator'],task,r['algorithm'].get('iterations',1))
        if params!=r['algorithm']:raise Invalid('algorithm/reference/source mismatch')

    def execute(self,data):
        t=self.task;p=policy_for(t.profile)
        h,groups,bound,_=validate_actual(self.record,self.dispatch,t,self.r['algorithm'],data['inputs'],self.config)
        reference_started=time.perf_counter()
        alg=self.reference(data['H'],data['Y'],t.regularization,self.r['algorithm'].get('iterations',1))
        expected=alg.get('output_block')
        if expected is None:expected=np.stack([real_block(x) for x in alg['output']])
        target=task_reference(data,t);reference_seconds=time.perf_counter()-reference_started;replay_started=time.perf_counter();actual=[];errors=[];executed=0
        for b in range(t.batch_size):
            r=execute_batch(h,groups[b],bound,b,expected[b],return_output=True);value=r.pop('_output',None)
            if r['status']!='STRUCTURE_MATCH' or value is None or r['executed']!=r['total'] or r['cost_issues'] or not bit_equal(value,expected[b]):
                raise Invalid('actual frame replay/reference failed at batch '+str(b)+': '+str(r))
            z=decode_checked(value,t.nt,t.nrhs,p.quadrant_tolerance)
            errors.append(error(z,target[b]));actual.append(value);executed+=r['executed']
        raw=np.stack(actual)
        output=np.stack([decode_checked(v,t.nt,t.nrhs,p.quadrant_tolerance) for v in raw])
        return dict(raw=raw,output=output,target=target,task_errors=errors,
                    numeric_passed=max(errors)<=p.task_tolerance,precision_bit_equal=True,instructions_executed=executed,
                    phase_timings_seconds=dict(independent_references=reference_seconds,instruction_replay_and_comparison=time.perf_counter()-replay_started))


def group_metrics(streams,limits,group_count):
    """Aggregate repeated frames within H before exact channel-outage inference."""
    rows=[];total_bits=total_symbols=ae=be=ase=bse=0
    confa=np.zeros((4,4),np.int64);confb=confa.copy();perstream=[]
    for cohort in streams:
        n=cohort['channels'];bits=cohort['bits_per_channel'];symbols=cohort['symbols_per_channel']
        a=cohort['candidate'];b=cohort['mmse']
        for i in range(n):
            av=math.sqrt(float(a['squared_error'][i])/symbols);bv=math.sqrt(float(b['squared_error'][i])/symbols)
            rows.append(dict(channel_sha256=cohort['H_hashes'][i],channel_seed=cohort['channel_seed'],batch=i,
                             bits=bits,symbols=symbols,candidate_bit_errors=int(a['bit_errors'][i]),mmse_bit_errors=int(b['bit_errors'][i]),
                             candidate_symbol_errors=int(a['symbol_errors'][i]),mmse_symbol_errors=int(b['symbol_errors'][i]),
                             ber_increase=float((a['bit_errors'][i]-b['bit_errors'][i])/bits),
                             ser_increase=float((a['symbol_errors'][i]-b['symbol_errors'][i])/symbols),
                             evm_increase=av-bv,candidate_raw_evm=av,mmse_raw_evm=bv))
        total_bits+=n*bits;total_symbols+=n*symbols;ae+=int(a['bit_errors'].sum());be+=int(b['bit_errors'].sum());ase+=int(a['symbol_errors'].sum());bse+=int(b['symbol_errors'].sum())
        confa+=cohort['candidate_confusion'];confb+=cohort['mmse_confusion']
    if len({r['channel_sha256'] for r in rows})!=len(rows):raise Invalid('duplicate H cannot inflate sample size')
    alpha=limits['family_alpha']/(8*group_count);bounds={}
    flags=[]
    for name in ('ber','ser','evm'):
        f=[row[name+'_increase']>limits['max_'+name+'_increase'] for row in rows];flags.append(f)
        bounds[name]=outage_interval(sum(f),len(rows),alpha)
    bounds['joint']=outage_interval(sum(any(v) for v in zip(*flags)),len(rows),alpha)
    if np.any(confa.sum(axis=1)==0) or not np.array_equal(confa.sum(axis=1),confb.sum(axis=1)):
        raise Invalid('full QPSK constellation coverage/paired transmitted labels missing')
    upper=all(b['upper']<=limits['max_outage_probability'] for b in bounds.values())
    lower=any(b['lower']>limits['max_outage_probability'] for b in bounds.values())
    status='QUALITY_FAILED' if lower else 'QPSK_GROUP_CHECKED' if upper and len(rows)>=limits['min_independent_channels'] else 'INCONCLUSIVE'
    # Stream-level data stay visible even though the frozen acceptance contract is per H.
    U=streams[0]['candidate']['stream_bit_errors'].shape[1]
    for u in range(U):
        perstream.append(dict(stream=u,candidate_ber=sum(int(s['candidate']['stream_bit_errors'][:,u].sum()) for s in streams)/(total_bits/U),
                              mmse_ber=sum(int(s['mmse']['stream_bit_errors'][:,u].sum()) for s in streams)/(total_bits/U),
                              candidate_ser=sum(int(s['candidate']['stream_symbol_errors'][:,u].sum()) for s in streams)/(total_symbols/U),
                              mmse_ser=sum(int(s['mmse']['stream_symbol_errors'][:,u].sum()) for s in streams)/(total_symbols/U)))
    return dict(status=status,bits=total_bits,symbols=total_symbols,independent_channels=len(rows),
                candidate_bit_errors=ae,mmse_bit_errors=be,candidate_symbol_errors=ase,mmse_symbol_errors=bse,
                candidate_ber=ae/total_bits,mmse_ber=be/total_bits,candidate_ser=ase/total_symbols,mmse_ser=bse/total_symbols,
                mean_candidate_raw_evm=float(np.mean([r['candidate_raw_evm'] for r in rows])),
                mean_mmse_raw_evm=float(np.mean([r['mmse_raw_evm'] for r in rows])),
                candidate_confusion=confa.tolist(),mmse_confusion=confb.tolist(),stream_metrics=perstream,
                outage_bounds=bounds,channels=rows,tail_alpha=alpha)


def run(c,repo,output):
    validate(c);repo=Path(repo).resolve();output=Path(output).resolve()
    if output.exists():raise Invalid('refusing existing link campaign')
    if c['evaluator_sources']!=sources(repo):raise Invalid('link evaluator changed after freeze')
    rp=repo/c['runtime']['path']
    if sha(rp)!=c['runtime']['sha256']:raise Invalid('link runtime changed')
    runtime=load_runtime(rp,repo)
    if runtime['python_version']!=platform.python_version() or runtime['numpy_version']!=np.__version__:raise Invalid('link toolchain mismatch')
    sp=repo/c['source_index']['path']
    if sha(sp)!=c['source_index']['sha256']:raise Invalid('source index changed')
    if sha(sp.parent/'contract.json')!=c['source_index']['contract_sha256']:raise Invalid('source task contract changed')
    index=json.loads(sp.read_text());source_contract=json.loads((sp.parent/'contract.json').read_text())
    if index['runtime']!=c['runtime'] or source_contract['runtime']!=c['runtime']:raise Invalid('source/runtime mismatch')
    if c['channel_seeds']==c['frame_seeds'] or source_contract['source_seed'] in c['channel_seeds']:
        raise Invalid('calibration/source seed cannot be heldout channel seed')
    expected={(o['operator'],lam) for o in source_contract['algorithms'] for lam in source_contract['noise_variances']}
    frozen_iterations={o['operator']:o['iterations'] for o in source_contract['algorithms']}
    if {(r['operator'],r['regularization']) for r in index['cases']}!=expected or len(index['cases'])!=len(expected):raise Invalid('source operator/noise coverage incomplete')
    B=source_contract['task']['batch_size'];count=len(expected)*len(c['channel_seeds'])*len(c['frame_seeds'])*B
    if count>c['budget']['max_batch_replays']:raise Invalid('batch replay budget exceeded')
    if B*len(c['channel_seeds'])<c['limits']['min_independent_channels']:raise Invalid('insufficient declared independent H')
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();rows=[];ledger=[];payloads={};replays=0
    controls=awgn_controls(c['control_seed'],c['control_symbols']);(output/'modem_controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    if controls['status']!='MODEM_CONTROLS_PASSED':raise Invalid('QPSK modem control failed')
    def budget():
        if time.perf_counter()-start>c['budget']['max_wall_seconds'] or shutil.disk_usage(output).free<c['budget']['min_free_bytes']:
            raise Invalid('link wall/disk budget exceeded')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>c['budget']['max_output_bytes']:raise Invalid('link output byte budget exceeded')
    for s in index['cases']:
        budget();group=dict(source=s['result_path'],operator=s['operator'],iterations=s['iterations'],regularization=s['regularization'],
                            es_no_db=-10*math.log10(s['regularization']),eb_no_db=-10*math.log10(2*s['regularization']),
                            expected_receive_snr_db=10*math.log10(source_contract['task']['nt']/source_contract['task']['nr']/s['regularization']))
        if s['status']!='SOURCE_CHECKED':rows.append({**group,'status':'SOURCE_GATE_FAILED'});continue
        path=Path(s['result_path']);item=dict(path=str(path),sha256=s['result_sha256'],role='historical_diagnostic')
        load_case(item,runtime,repo);r=json.loads(path.read_text());task=MMSETask(**{f.name:r['task'][f.name] for f in fields(MMSETask)})
        declared=MMSETask(**{**source_contract['task'],'regularization':s['regularization']})
        if task.frozen()!=declared.frozen() or r['operator']!=s['operator'] or s['iterations']!=frozen_iterations[s['operator']] or r['algorithm'].get('iterations',1)!=s['iterations']:
            raise Invalid('source task/algorithm differs from frozen campaign')
        if r['seed'] in c['channel_seeds']:raise Invalid('original source H is not a new holdout seed')
        executor=Replayer(path,r,task);streams=[];numeric_good=True
        for hs in c['channel_seeds']:
            cohort=None
            for fs in c['frame_seeds']:
                budget();key=(s['regularization'],hs,fs)
                if key not in payloads:
                    data=frame(task,hs,fs);target=task_reference(data,task)
                    fd=output/'inputs';fd.mkdir(exist_ok=True)
                    fpath=fd/f'lambda{task.regularization}-h{hs}-f{fs}.npz'
                    np.savez_compressed(fpath,H=data['H'],Y=data['Y'],bits=data['bits'],symbols=data['symbols'],noise=data['noise'],
                                        **{n+'_input':v for n,v in data['inputs'].items()},mmse=target)
                    payloads[key]=(data,dict(path=str(fpath.relative_to(output)),bytes=fpath.stat().st_size,sha256=sha(fpath)))
                data,input_artifact=payloads[key]
                fresh=executor.execute(data);replays+=task.batch_size;numeric_good&=fresh['numeric_passed']
                measured=measurements(fresh['output'],fresh['target'],data['bits'])
                if cohort is None:
                    cohort=dict(channel_seed=hs,channels=B,bits_per_channel=0,symbols_per_channel=0,
                                H_hashes=[hashlib.sha256(h.tobytes()).hexdigest() for h in data['H']],
                                candidate={k:np.zeros_like(v) for k,v in measured['candidate'].items()},
                                mmse={k:np.zeros_like(v) for k,v in measured['mmse'].items()},
                                candidate_confusion=np.zeros((4,4),np.int64),mmse_confusion=np.zeros((4,4),np.int64))
                for who in ('candidate','mmse'):
                    for k,v in measured[who].items():cohort[who][k]+=v
                    cohort[who+'_confusion']+=measured[who+'_confusion']
                cohort['bits_per_channel']+=measured['bits_per_channel'];cohort['symbols_per_channel']+=measured['symbols_per_channel']
                dest=output/f'{s["id"]}-h{hs}-f{fs}.npz'
                np.savez_compressed(dest,actual_output_block=fresh['raw'].astype(np.float16),task_error=np.asarray(fresh['task_errors']))
                entry=dict(operator=s['operator'],regularization=task.regularization,channel_seed=hs,frame_seed=fs,
                           precision_bit_equal=fresh['precision_bit_equal'],numeric_passed=fresh['numeric_passed'],max_task_error=max(fresh['task_errors']),
                           instructions_executed=fresh['instructions_executed'],input_artifact=input_artifact,
                           transmitted_bit_one_fraction=float(np.mean(data['bits'])),
                           measured_noise_power=float(np.mean(np.abs(data['noise'])**2)),
                           measured_receive_signal_power=float(np.mean(np.abs(data['Y']-data['noise'])**2)),
                           output_artifact=dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest)))
                ledger.append(entry)
                pending=output/'frame_progress.tmp';pending.write_text(json.dumps(ledger,indent=2)+'\n');pending.replace(output/'frame_progress.json')
                print(json.dumps({k:entry[k] for k in ('operator','regularization','channel_seed','frame_seed','max_task_error')}),flush=True)
            streams.append(cohort)
        metric=group_metrics(streams,c['limits'],len(expected))
        if not numeric_good:metric['status']='NUMERIC_QUALITY_FAILED'
        rows.append({**group,**metric, 'source_finish_cycle':r['asim']['finish_cycle']})
        (output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');check_artifacts(path.parent,r)
        del executor;gc.collect()
    if replays!=count:raise Invalid('complete frame/operator replay coverage not achieved')
    if c['evaluator_sources']!=sources(repo) or sha(sp)!=c['source_index']['sha256'] or sha(sp.parent/'contract.json')!=c['source_index']['contract_sha256'] or sha(rp)!=c['runtime']['sha256']:raise Invalid('source/evaluator changed during link test')
    load_runtime(rp,repo)
    for s in index['cases']:
        if s['status']=='SOURCE_CHECKED':
            p=Path(s['result_path'])
            if sha(p)!=s['result_sha256']:raise Invalid('template result changed during link test')
            check_artifacts(p.parent,json.loads(p.read_text()))
    for data,a in payloads.values():
        if sha(output/a['path'])!=a['sha256']:raise Invalid('shared input fixture changed')
    for entry in ledger:
        a=entry['output_artifact'];p=output/a['path']
        if p.stat().st_size!=a['bytes'] or sha(p)!=a['sha256']:raise Invalid('actual replay output artifact changed')
    passed=all(r['status']=='QPSK_GROUP_CHECKED' for r in rows)
    result=dict(schema='uncoded-qpsk-link-result-v1',status='UNCODED_QPSK_SUITE_CHECKED' if passed else 'LINK_QUALITY_NOT_VERIFIED',
                uncoded_qpsk_suite_complete=True,full_communication_certified=False,score=None,groups=rows,frames=ledger,
                batch_replays=replays,bits=sum(r.get('bits',0) for r in rows),symbols=sum(r.get('symbols',0) for r in rows),
                elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,evaluator_sources=c['evaluator_sources'],
                note='fixed existing algorithm configurations; fresh functional H/Y replays, original Asim cycles unchanged')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--contract',required=True,type=Path)
    ap.add_argument('--contract-sha256',required=True);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if a.output.exists():raise Invalid('refusing existing link campaign')
    c=None;start=time.perf_counter()
    try:
        if sha(a.contract)!=a.contract_sha256:raise Invalid('approved link contract changed')
        c=json.loads(a.contract.read_text());r=run(c,Path(__file__).resolve().parents[2],a.output)
        if sha(a.contract)!=a.contract_sha256:raise Invalid('link contract changed during test')
    except (Invalid,ValueError,KeyError,TypeError,OSError,FloatingPointError,IndexError) as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        if (a.output/'result.json').exists():shutil.copyfile(a.output/'result.json',a.output/'evaluation_result.json')
        r=dict(schema='uncoded-qpsk-link-result-v1',status='LINK_RUN_REJECTED',reason=str(exc),
               uncoded_qpsk_suite_complete=False,full_communication_certified=False,score=None,
               elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None)
    a.output.mkdir(parents=True,exist_ok=True)
    if c is not None:(a.output/'contract.json').write_text(json.dumps(c,indent=2)+'\n')
    r['approved_contract_sha256']=a.contract_sha256
    (a.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('status','reason','bits','symbols','batch_replays','elapsed_seconds')}))
    return 0 if r['status']=='UNCODED_QPSK_SUITE_CHECKED' else 1


if __name__=='__main__':raise SystemExit(main())
