"""Real Asim-first primitive/small-model suite, bounded archives, no promotion."""
import argparse,csv,hashlib,json,os,shutil,sys,tarfile,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from verification.tasks.tiny_nn import NNTask,backend
from verification.reconstruction.runner import evaluate
from verification.asim.runner import file_hash
from verification.reconstruction.stream import load_operation

def archive_case(case):
    files={str(p.relative_to(case)):dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in case.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in case.rglob('*')):raise ValueError('unexpected symlink')
    archive=case.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=1) as tf:tf.add(case,arcname=case.name)
    checked=set()
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            if not member.isfile():continue
            name=str(Path(member.name).relative_to(case.name));f=tf.extractfile(member);h=hashlib.file_digest(f,'sha256').hexdigest()
            if member.size!=files[name]['bytes'] or h!=files[name]['sha256']:raise ValueError('archive copy mismatch')
            checked.add(name)
    if checked!=set(files):raise ValueError('archive coverage mismatch')
    receipt={'path':archive.name,'bytes':archive.stat().st_size,'sha256':file_hash(archive),'uncompressed_files':files,'restore':'extract into a new empty directory; do not overwrite prior runs'}
    shutil.rmtree(case);return receipt

def check_nonlinear_cost(path,config):
    rec=load_operation(path/'instructions.jsonl');stats=json.loads((path/'pipeline_statistics.json').read_text());byid={i['id']:i for i in rec['instructions']};checks=[]
    for row in stats['instruction_pipeline_detail']:
        i=byid.get(row['name'])
        if not i or i['opcode'] not in (8,22):continue
        c=config['core_config']['core_'+str(row['core'])];width=c['vector_process_bit']//8
        def tree(n):return 1 if n<=width else (n+width-1)//width+tree((n+width-1)//width)
        it=(i['compute_size']+width-1)//width
        expected=it*c['gelu_latency'] if i['opcode']==22 else i['m']*(2*tree(i['compute_size'])*c['add_tree_latency']+it*(c['add_latency']+c['exp_latency']+c['div_latency']))
        if row['event_count']!=1 or row['min_duration']!=expected or row['max_duration']!=expected:raise ValueError('actual NN Vector duration mismatch: '+i['id'])
        checks.append({'id':i['id'],'opcode':i['opcode'],'rows':i['m'],'cols':i['n'],'cycles':expected})
    if not checks:raise ValueError('missing NN compute events')
    return checks

ACTIVE_OUT=None

def main():
    global ACTIVE_OUT
    ACTIVE_OUT=None
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--binary',type=Path,default=ROOT/'build/bin/Simulator');a.add_argument('--output',type=Path,required=True);a.add_argument('--runtime',type=Path);a.add_argument('--kind',choices=['all','softmax','gelu','mlp','attention'],default='all');a.add_argument('--quick',action='store_true');a.add_argument('--keep-raw',action='store_true');v=a.parse_args()
    out=v.output.resolve();out.mkdir(parents=True,exist_ok=False);ACTIVE_OUT=out;binary=v.binary.resolve();config=json.loads((ROOT/'configs/ascend_910b_nn_v1.json').read_text());config['precision']=2
    config['dram_config_path']=str(ROOT/'configs/ramulator2_configs/HBM2.yaml');config['icnt_config_path']=str(ROOT/'configs/booksim2_configs/fly_c1_m2.icnt')
    os.environ['LD_LIBRARY_PATH']=str(binary.parent.parent/'lib');os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    runtime_spec=None
    if v.runtime:
        from verification.reconstruction.daily import load_runtime
        runtime_spec=load_runtime(v.runtime,ROOT);binary=(ROOT/runtime_spec['binary']).resolve();config=json.loads((ROOT/runtime_spec['config']).read_text())
        os.environ['LD_LIBRARY_PATH']=str(ROOT/runtime_spec['library_dir'])
        if runtime_spec['python_version']!=sys.version.split()[0] or runtime_spec['numpy_version']!=np.__version__:raise ValueError('NN runtime Python/NumPy version mismatch')
        if 'modeled_fp16_fp32_nn_v1' not in runtime_spec['profiles'] or 'nn_graph_v1' not in runtime_spec['operators']:raise ValueError('runtime scope does not accept NN')
    kinds=['softmax','gelu','mlp','attention'] if v.kind=='all' else [v.kind];shapes=[(32,32,64)] if v.quick else [(16,32,64),(32,32,64),(32,64,128)];seeds=[91001] if v.quick else [91001,91002,91003]
    summary={'python_version':sys.version.split()[0],'numpy_version':np.__version__,'schema':'nn-acceptance-v1','status':'RUNNING','binary_sha256':file_hash(binary),'score':None,'promotion_allowed':False,'cases':[]};start=time.perf_counter()
    for kind in kinds:
        for l,d,f in shapes:
            for seed in seeds:
                t=NNTask(kind=kind,length=l,hidden=d,ffn=f);name=f'{kind}-l{l}-d{d}-f{f}-s{seed}';case=out/name
                if shutil.disk_usage(out).free<150*1024**2:raise ValueError('insufficient free disk before NN case')
                r=evaluate('nn_graph_v1',t,1,seed,binary,config,ROOT/runtime_spec['cwd'] if runtime_spec else ROOT/'build',case,task_backend=backend(t));cost=check_nonlinear_cost(case,config) if r.get('schedule_status')=='MODEL_RW_CHECKED_CONSERVATIVE' else []
                row={'case':name,'kind':kind,'task':t.frozen(),'seed':seed,'status':r['status'],'reason':r.get('reason'),'schedule':r.get('schedule_status'),'cycles':r.get('asim',{}).get('finish_cycle'),'failed_batches':r.get('failed_batches'),'all_fp16_bits_equal':bool(r.get('batches')) and all(b.get('precision_bit_equal') for b in r.get('batches',[])),'all_stage_bits_equal':bool(r.get('batches')) and all(b.get('stage_checks') and all(c['bit_equal'] for c in b['stage_checks']) for b in r.get('batches',[])),'max_task_error':max((b.get('task_error',0) for b in r.get('batches',[])),default=None),'phase_seconds':r['phase_timings_seconds'],'result_sha256':file_hash(case/'result.json'),'nonlinear_cost_checks':cost}
                passed=r['status']=='MODELED_NUMERIC_CHECKED' and row['all_fp16_bits_equal'] and row['all_stage_bits_equal'];row['archive']=None if v.keep_raw else archive_case(case);summary['cases'].append(row);summary['wall_seconds']=time.perf_counter()-start;summary['status']='RUNNING' if passed else 'FAILED';(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:x for k,x in row.items() if k not in ['archive','nonlinear_cost_checks','task']}),flush=True)
                if not passed:return 1
    if file_hash(binary)!=summary['binary_sha256']:raise ValueError('binary changed during acceptance')
    if runtime_spec:
        from verification.reconstruction.daily import load_runtime
        if load_runtime(v.runtime,ROOT)['_loaded_sha256']!=runtime_spec['_loaded_sha256']:raise ValueError('runtime changed during suite')
        summary['runtime_spec_sha256']=runtime_spec['_loaded_sha256']
    summary['status']='QUICK_CHECKED' if v.quick else 'SUITE_CHECKED';(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['status']);return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        if ACTIVE_OUT is not None:
            summary_path=ACTIVE_OUT/'summary.json';summary=json.loads(summary_path.read_text()) if summary_path.exists() else {'schema':'nn-acceptance-v1','cases':[]}
            summary.update(status='FAILED',reason=str(e),score=None,promotion_allowed=False);summary_path.write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({'status':'FAILED','reason':str(e)}));raise SystemExit(1)
