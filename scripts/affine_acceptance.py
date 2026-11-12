"""Trusted nine-case affine acceptance and paired task-local score (no promotion)."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from verification.tasks.affine import AffineTask,BACKEND
from verification.tasks.demo_score import score
from verification.reconstruction.runner import evaluate
from verification.contracts.core import ScreenPolicy
from verification.asim.runner import file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['binary','config','cwd','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--single',action='store_true',help='one representative case; never labels nine-case acceptance')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=a.binary.resolve();cwd=a.cwd.resolve();config=json.loads(a.config.read_text());config['precision']=2
    os.environ['LD_LIBRARY_PATH']=str(binary.parent.parent/'lib');os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    begin=time.perf_counter();rows=[];frozen=file_hash(binary)
    summary={'schema':'affine-acceptance-v1','status':'RUNNING','binary_sha256':frozen,'evaluator_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'cases':rows,'research_score':None,'promotion_allowed':False,'communication_status':'NOT_APPLICABLE_REAL_AFFINE_TASK'}
    specs=[(32,32,32)] if a.single else [(16,16,16),(32,32,32),(32,16,48)]
    seeds=[42] if a.single else [42,123,20261111]
    for shape in specs:
        for seed in seeds:
            task=AffineTask(m=shape[0],k=shape[1],n=shape[2]);name='x'.join(map(str,shape))+'-s'+str(seed);path=out/name
            r=evaluate('affine_gemm_v1',task,1,seed,binary,config,cwd,path,task_backend=BACKEND)
            row={'name':name,'shape':shape,'seed':seed,'status':r['status'],'reason':r.get('reason'),'schedule':r.get('schedule_status'),'cycles':r.get('asim',{}).get('finish_cycle'),'failed_batches':r.get('failed_batches'),'algorithm_bits_equal':bool(r.get('batches')) and all(b.get('precision_bit_equal') for b in r.get('batches',[])),'max_task_error':max((b.get('task_error',0) for b in r.get('batches',[])),default=None),'phase_timings_seconds':r['phase_timings_seconds'],'result_sha256':file_hash(path/'result.json')};rows.append(row)
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(row),flush=True)
            if r['status']!='MODELED_NUMERIC_CHECKED' or not row['algorithm_bits_equal']:summary['status']='FAILED';(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return 1
            if shape==(32,32,32) and seed==42:baseline=(path,r,task,row['result_sha256'])
    base,r,task,base_sha=baseline
    repeat=out/'paired-repeat'
    c=evaluate('affine_gemm_v1',task,1,42,binary,config,cwd,repeat,policy=ScreenPolicy(r['asim']['finish_cycle'],1.,10.),baseline_identity=r['comparison_identity'],task_backend=BACKEND)
    if file_hash(binary)!=frozen:raise ValueError('binary changed during acceptance')
    summary['paired_repeat_status']=c['status'];summary['demo_score']=score(base/'result.json',repeat/'result.json',base_sha,file_hash(repeat/'result.json'))
    summary['status']='SINGLE_CASE_CHECKED' if a.single else 'NINE_CASES_CHECKED';summary['wall_seconds']=time.perf_counter()-begin
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['demo_score']));return 0
if __name__=='__main__':raise SystemExit(main())
