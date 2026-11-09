"""Supervised frozen implicit candidate integration; Asim before any replay.

CDL inputs explicitly reuse data-independent accepted instruction templates.
No task header rewriting, no K tuning, no invented per-channel Asim timing.
"""
import argparse,gc,hashlib,json,os,shutil,time,sys
from dataclasses import fields
from pathlib import Path
import numpy as np
from verification.contracts.core import Invalid
from verification.contracts.mmse import MMSETask
from verification.profiles.numeric import bit_equal
from verification.reconstruction.runner import evaluate
from verification.asim.runner import file_hash
from verification.communication.qpsk_link import Replayer,group_metrics
from verification.communication.qpsk import measurements
from verification.planning.channel_study import payload

OP='bri_implicit_fp16_v1'

def source_map(repo):
    paths=[p for p in (repo/'verification').rglob('*.py') if 'tests' not in p.relative_to(repo/'verification').parts]
    paths += [repo/p for p in ('orchestrator/codex_operators.json','src/main.cc','src/semantic/SourceViews.h','src/inverse/bri_implicit/BRIImplicitModel.cc','src/inverse/bri_implicit/BRIImplicitModel.h')]
    return {str(p.relative_to(repo)):file_hash(p) for p in sorted(paths)}


def guard(contract,path,approved):
    repo=Path(__file__).resolve().parents[2]
    if file_hash(path)!=approved:raise Invalid('approved campaign SHA changed')
    if contract['schema']!='implicit-native-campaign-v1' or contract['cases']!=[dict(regularization=l,iterations=k) for l,k in [(1.,13),(.1,33),(.01,76)]]:raise Invalid('frozen implicit parameters changed')
    if contract['channel_seeds']!=[56001,56002] or contract['frame_seeds']!=[56011,56012]:raise Invalid('holdout seeds differ')
    if contract['sources']!=source_map(repo):raise Invalid('complete source closure changed/missing')
    required={contract['binary'],contract['config'],str((Path(contract['library_dir'])/'libramulator.so').resolve()),str((Path(contract['cwd'])/'../configs/ramulator2_configs/HBM2.yaml').resolve())}
    if set(contract['dependencies'])!=required:raise Invalid('required executable/config/library dependency closure differs')
    if contract['python_version']!=sys.version.split()[0] or contract['numpy_version']!=np.__version__:raise Invalid('Python/NumPy version changed')
    for name,h in contract['sources'].items():
        if file_hash(repo/name)!=h:raise Invalid('frozen evaluator/candidate source changed: '+name)
    for name,h in contract['dependencies'].items():
        if file_hash(name)!=h:raise Invalid('frozen binary/config/library changed: '+name)
    if contract['budget']!=dict(max_wall_seconds=5400,min_free_bytes=600000000,max_output_bytes=3500000000,max_batch_replays=576):raise Invalid('resource budget differs')
    if set(contract['limits'])!={'max_ber_increase','max_ser_increase','max_evm_increase','max_outage_probability','family_alpha','min_independent_channels'}:raise Invalid('incomplete quality limits')
    if contract['limits']!=dict(max_ber_increase=.01,max_ser_increase=.02,max_evm_increase=.01,max_outage_probability=.05,family_alpha=.05,min_independent_channels=192):raise Invalid('quality limits changed')
    return repo


def coverage(c,source,datasets):
    entries=source.get('cases',[]);data=datasets.get('entries',[])
    if source.get('status')!='ASIM_TEMPLATES_CHECKED' or len(entries)!=3 or [{k:e.get(k) for k in ('regularization','iterations')} for e in entries]!=c['cases'] or any(e.get('status')!='MODELED_NUMERIC_CHECKED' for e in entries):raise Invalid('exact Asim case coverage required')
    if datasets.get('status')!='DATASET_FROZEN' or datasets.get('channels')!=192 or len(data)!=2 or [e.get('seed') for e in data]!=c['channel_seeds']:raise Invalid('exact fresh CDL dataset coverage required')


def checks(path):
    r=json.loads(path.read_text())
    if r['status']!='MODELED_NUMERIC_CHECKED' or r['schedule_status']!='MODEL_RW_CHECKED_CONSERVATIVE' or r['failed_batches']:raise Invalid('source candidate did not pass')
    required={'asim.json','schedule.json','outputs.npy','inputs.npz','references.npz','channel.npz','instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','config.json','models.json','simulator.log'}
    names=[a['path'] for a in r['artifacts']]
    if len(set(names))!=len(names) or not required<=set(names) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise Invalid('source required artifact coverage unsafe/incomplete')
    for a in r['artifacts']:
        p=path.parent/a['path']
        if p.stat().st_size!=a['bytes'] or file_hash(p)!=a['sha256']:raise Invalid('source artifact changed')
    return r


def run(c,contract,approved,out,stage):
    repo=guard(c,contract,approved);start=time.perf_counter()
    def resource_guard():
        if time.perf_counter()-start>c['budget']['max_wall_seconds'] or shutil.disk_usage(out.parent).free<c['budget']['min_free_bytes']:raise Invalid('wall/disk budget exceeded')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>c['budget']['max_output_bytes']:raise Invalid('output budget exceeded')
    resource_guard()
    if stage=='dataset':
        from verification.channels.cdlb import generate,SPEC,provenance
        out.mkdir(parents=True,exist_ok=True);dest=out/'dataset';dest.mkdir(exist_ok=False);entries=[]
        if SPEC!=c['cdlb'] or provenance()!=c['channel_toolchain']:raise Invalid('CDL model/toolchain differs')
        for seed in c['channel_seeds']:
            resource_guard()
            p=dest/f'H{seed}.npz';np.savez_compressed(p,**generate(seed,96));entries.append(dict(seed=seed,path=str(p.resolve()),sha256=file_hash(p)))
        result=dict(status='DATASET_FROZEN',channels=192,entries=entries,seconds=time.perf_counter()-start)
        guard(c,contract,approved);(dest/'index.json').write_text(json.dumps(result,indent=2));return result
    if stage=='asim':
        out.mkdir(parents=True,exist_ok=True);entries=[]
        config=json.loads(Path(c['config']).read_text());config['precision']=2
        for name in ('dram_config_path','icnt_config_path'):
            if config.get(name):config[name]=str((Path(c['cwd'])/config[name]).resolve())
        for case in c['cases']:
            resource_guard()
            task=MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=case['regularization'])
            dest=out/f'asim-lambda{task.regularization}'
            r=evaluate(OP,task,case['iterations'],57001,Path(c['binary']),config,Path(c['cwd']),dest)
            row=dict(**case,status=r['status'],path=str((dest/'result.json').resolve()),sha256=file_hash(dest/'result.json'),seconds=r['phase_timings_seconds'])
            entries.append(row);(out/'asim-progress.json').write_text(json.dumps(entries,indent=2));print(json.dumps(row),flush=True)
            del r;gc.collect()
            if row['status']!='MODELED_NUMERIC_CHECKED':break
        passed=len(entries)==3 and all(e['status']=='MODELED_NUMERIC_CHECKED' for e in entries)
        result=dict(status='ASIM_TEMPLATES_CHECKED' if passed else 'ASIM_TEMPLATE_GATE_FAILED',cases=entries,seconds=time.perf_counter()-start,score=None)
        guard(c,contract,approved);(out/'asim-index.json').write_text(json.dumps(result,indent=2));return result
    if stage!='replay':raise Invalid('unknown stage')
    source=json.loads((out/'asim-index.json').read_text())
    if source['status']!='ASIM_TEMPLATES_CHECKED':raise Invalid('Asim gate not passed')
    dest=out/'holdout';dest.mkdir(exist_ok=False);datasets=json.loads((out/'dataset/index.json').read_text());coverage(c,source,datasets);rows=[];frames=[]
    for case,entry in zip(c['cases'],source['cases']):
        if case!={k:entry[k] for k in case} or file_hash(entry['path'])!=entry['sha256']:raise Invalid('source case binding changed')
        p=Path(entry['path']);r=checks(p)
        if p.resolve()!=(out/f'asim-lambda{case["regularization"]}'/'result.json').resolve() or r['operator']!=OP or r['task']['regularization']!=case['regularization'] or r['algorithm']['iterations']!=case['iterations'] or r['asim']['binary_sha256']!=c['dependencies'][c['binary']]:raise Invalid('actual source task/binary/case mismatch')
        t=MMSETask(profile='modeled_fp16_fp32_real_block_v1',regularization=case['regularization'])
        from verification.operators.catalog import adapter
        _,expected,_,_=adapter(OP,t,case['iterations'])
        receipt=json.loads((p.parent/'asim.json').read_text())
        if r['task']!=t.frozen() or r['algorithm']!=expected or r['seed']!=57001 or receipt['status']!='COMPLETED' or receipt['returncode']!=0 or receipt['binary_sha256']!=c['dependencies'][c['binary']] or receipt['finish_cycle']!=r['asim']['finish_cycle']:raise Invalid('source frozen task/algorithm/actual receipt mismatch')
        _,_,expected_models,_=adapter(OP,t,case['iterations'])
        expected_config=json.loads(Path(c['config']).read_text());expected_config['precision']=2
        for key in ('dram_config_path','icnt_config_path'):
            if expected_config.get(key):expected_config[key]=str((Path(c['cwd'])/expected_config[key]).resolve())
        if json.loads((p.parent/'models.json').read_text())!=expected_models or json.loads((p.parent/'config.json').read_text())!=expected_config:raise Invalid('actual source model/config differs from frozen request')
        executor=Replayer(p,r,t);streams=[];errors=[]
        for data_entry,fs in zip(datasets['entries'],c['frame_seeds']):
            resource_guard()
            if file_hash(data_entry['path'])!=data_entry['sha256']:raise Invalid('CDL dataset changed')
            with np.load(data_entry['path'],allow_pickle=False) as a:H=a['H']
            d=payload(t,H,data_entry['seed'],fs);tic=time.perf_counter();v=executor.execute(d);wall=time.perf_counter()-tic
            m=measurements(v['output'],v['target'],d['bits']);errors.extend(v['task_errors'])
            streams.append(dict(channel_seed=data_entry['seed'],channels=96,bits_per_channel=m['bits_per_channel'],symbols_per_channel=m['symbols_per_channel'],
                H_hashes=[hashlib.sha256(h.tobytes()).hexdigest() for h in H],candidate=m['candidate'],mmse=m['mmse'],candidate_confusion=m['candidate_confusion'],mmse_confusion=m['mmse_confusion']))
            f=dest/f'lambda{t.regularization}-H{data_entry["seed"]}.npz'
            np.savez_compressed(f,H=d['H'],Y=d['Y'],bits=d['bits'],symbols=d['symbols'],noise=d['noise'],output=v['raw'].astype(np.float16),mmse=v['target'],task_errors=v['task_errors'])
            frames.append(dict(**case,channel_seed=data_entry['seed'],frame_seed=fs,channels=96,precision_bit_equal=v['precision_bit_equal'],max_task_error=max(v['task_errors']),instruction_count=v['instructions_executed'],reference_and_replay_seconds=wall,phase_timings_seconds=v['phase_timings_seconds'],path=str(f.resolve()),sha256=file_hash(f)))
            (dest/'frame-progress.json').write_text(json.dumps(frames,indent=2));print(json.dumps(frames[-1]),flush=True)
        quality=group_metrics(streams,c['limits'],3);quality.update(**case,max_task_error=max(errors),template_task=t.frozen(),input_domain=c['cdlb'],source_result=str(p),source_finish_cycle=r['asim']['finish_cycle'])
        if max(errors)>.02:quality['status']='NUMERIC_QUALITY_FAILED'
        rows.append(quality);(dest/'progress.json').write_text(json.dumps(rows,indent=2));del executor;gc.collect()
    batch_replays=sum(f['channels'] for f in frames)
    if len(rows)!=3 or len(frames)!=6 or batch_replays!=c['budget']['max_batch_replays']:raise Invalid('actual completed replay coverage differs')
    passed=all(r['status']=='QPSK_GROUP_CHECKED' for r in rows)
    result=dict(status='IMPLICIT_CDLB_RECONSTRUCTION_CHECKED' if passed else 'IMPLICIT_CDLB_GATE_FAILED',groups=rows,frames=frames,independent_H=192,batch_replays=batch_replays,
                elapsed_seconds=time.perf_counter()-start,pure_model_thinking_seconds=None,score=None,note='data-independent real Asim templates; CDL replay does not create new cycles')
    guard(c,contract,approved);(dest/'result.json').write_text(json.dumps(result,indent=2));return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--contract',required=True,type=Path);ap.add_argument('--sha256',required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--stage',choices=['dataset','asim','replay'],required=True);a=ap.parse_args()
    c=json.loads(a.contract.read_text());guard(c,a.contract,a.sha256)
    os.environ['LD_LIBRARY_PATH']=c['library_dir']
    try:r=run(c,a.contract,a.sha256,a.output,a.stage)
    except (Invalid,ValueError,OSError,FloatingPointError) as exc:
        r=dict(status='CAMPAIGN_REJECTED',stage=a.stage,reason=str(exc),score=None)
        if a.output.is_dir():(a.output/(a.stage+'-failure.json')).write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ('groups','frames','cases','entries')},ensure_ascii=False))
    return 0 if r['status'] in ('DATASET_FROZEN','ASIM_TEMPLATES_CHECKED','IMPLICIT_CDLB_RECONSTRUCTION_CHECKED') else 1
if __name__=='__main__':raise SystemExit(main())
