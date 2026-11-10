"""Portable real-Asim -> model schedule -> actual reconstruction -> references."""
from pathlib import Path
import sys,os,json,argparse,shutil,time,hashlib,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from verification.contracts.mmse import MMSETask
from verification.reconstruction.runner import evaluate
CASES={'cholesky':('cholesky_explicit_v1',1),'ldl':('ldl_explicit_v1',1),'cholesky-block':('cholesky_block_explicit_v1',1),'ldl-block':('ldl_block_explicit_v1',1),'ns':('ns_corrected_v1',16),'bri':('bri_explicit_v1',192),'bri-hb':('bri_stable_hb_fp16_v1',53)}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--operator',choices=['all',*CASES],default='all');a.add_argument('--output',type=Path,required=True);a.add_argument('--seed',type=int,default=42);a.add_argument('--keep-raw',action='store_true');v=a.parse_args()
 out=v.output.resolve()
 if out.exists():raise ValueError('output must be a new directory')
 out.mkdir(parents=True);(out/'.asim-test-owned').write_text('portable-six-smoke-v1')
 binary=ROOT/'build/bin/Simulator';config=json.loads((ROOT/'configs/ascend_910b_quiet.json').read_text());config['precision']=2
 config['dram_config_path']=str(ROOT/'configs/ramulator2_configs/HBM2.yaml');config['icnt_config_path']=str(ROOT/'configs/booksim2_configs/fly_c1_m2.icnt')
 os.environ['LD_LIBRARY_PATH']=str(ROOT/'build/lib');os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
 selected=list(CASES)[:6] if v.operator=='all' else [v.operator];rows=[];begin=time.perf_counter()
 for name in selected:
  if shutil.disk_usage(out).free<800*1024**2:raise ValueError('at least800MiB free required before a case')
  op,k=CASES[name];case=out/name
  r=evaluate(op,MMSETask(profile='modeled_fp16_fp32_real_block_v1'),k,v.seed,binary,config,ROOT/'build',case)
  artifacts=[dict(path=e['path'],bytes=e['bytes'],sha256=e['sha256']) for e in r['artifacts']]
  row=dict(family=name,operator=op,iterations=k,status=r['status'],numeric_status=r['numeric_status'],schedule_status=r.get('schedule_status'),cycles=r.get('asim',{}).get('finish_cycle'),reason=r.get('reason'),failed_batches=r.get('failed_batches'),max_task_error=max((b['task_error'] for b in r.get('batches',[]) if 'task_error' in b),default=None),algorithm_bits_equal=bool(r.get('batches')) and all(b.get('precision_bit_equal',False) for b in r.get('batches',[])),batch_size=96,nr=64,nt=16,nrhs=8,regularization=.1,seed=v.seed,phase_timings_seconds=r['phase_timings_seconds'],comparison_identity=r.get('comparison_identity'),artifacts_before_cleanup=artifacts,raw_retained=v.keep_raw,score=None)
  rows.append(row);print(json.dumps({k:x for k,x in row.items() if k!='artifacts_before_cleanup'}),flush=True)
  passed=r['status']=='MODELED_NUMERIC_CHECKED' and not r['failed_batches'] and row['algorithm_bits_equal'] and row['schedule_status']=='MODEL_RW_CHECKED_CONSERVATIVE'
  # Only remove directories created by this invocation, after hashes were saved.
  if not v.keep_raw:
   assert case.parent==out and (out/'.asim-test-owned').read_text()=='portable-six-smoke-v1';shutil.rmtree(case)
  summary=dict(status='SIX_CASES_CHECKED' if len(rows)==6 and all(x['status']=='MODELED_NUMERIC_CHECKED' for x in rows) else 'SINGLE_CASE_CHECKED' if len(rows)==1 and passed else 'IN_PROGRESS_OR_FAILED',cases=rows,binary_sha256=sha(binary),numpy_version=np.__version__,elapsed_seconds=time.perf_counter()-begin,scope='one Rayleigh seed, lambda.1,96batch; not all CDL/communication/hardware certification',raw_artifact_notice='hashes describe generated files before optional cleanup, not files claimed present')
  (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');del r;gc.collect()
  if not passed:return 1
 return 0
if __name__=='__main__':raise SystemExit(main())
