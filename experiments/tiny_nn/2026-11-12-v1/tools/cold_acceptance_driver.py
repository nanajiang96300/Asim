from pathlib import Path
import time,json,subprocess,hashlib,shutil,os,sys,traceback
raw=Path(__file__).resolve().parent
repo=Path('/home/nanajiang/asim-worktrees/nn-delivery-validation-v2/asim-claude-guide-v2')
feature=Path('/home/nanajiang/asim-worktrees/tiny-nn-primitives-v1')
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
summary={'status':'WAITING_FOR_COLD_BUILD','scope':'same host, new unpack/source-import/venv/dependencies/build; heldout replay repeated, not new independent samples','pure_agent_thinking_seconds':None,'checks':[]}
def save(): (raw/'delivery-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
def run(name,args):
 summary['status']='RUNNING';summary['phase']=name;save();start=time.perf_counter()
 with (raw/('cold-'+name+'.log')).open('w') as f:r=subprocess.run(args,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 summary['checks'].append({'name':name,'returncode':r.returncode,'elapsed_seconds':time.perf_counter()-start,'command':args});save()
 if r.returncode:raise RuntimeError(name+' failed; see cold-'+name+'.log')
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
try:
 save()
 while not (raw/'cold-build-receipt.json').exists():time.sleep(5)
 build=json.loads((raw/'cold-build-receipt.json').read_text());assert build['returncode']==0
 summary['build']=build;summary['source_delivery']=json.loads((raw/'package-v2.json').read_text())
 summary['cold_binary_sha256']=digest(repo/'build/bin/Simulator');summary['library_sha256']=digest(repo/'build/lib/libramulator.so')
 changed=[];count=0
 for p in (repo/'src').rglob('*'):
  if p.is_file() and p.suffix in ('.h','.cc'):
   count+=1
   if digest(p)!=digest(feature/p.relative_to(repo)):changed.append(str(p.relative_to(repo)))
 assert not changed;summary['cpp_source_files_byte_equal']=count
 # Reclaim only this owned build's regenerable caches after binary is linked.
 dirs=[repo/'.deps/asim-deps-v1'];files=list(dirs[0].rglob('*'))
 files=[p for p in files if p.is_file()]+[p for p in (repo/'build').rglob('*') if p.is_file() and p.relative_to(repo/'build').parts[0] not in ('bin','lib')]
 rows=[{'path':str(p.relative_to(repo)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
 (raw/'cold-generated-cache-cleanup.json').write_text(json.dumps({'reason':'owned cold generated dependencies/object/PCH only; binary/library/source/venv retained','restore':'standard offline bootstrap from packaged source','bytes_removed':sum(r['bytes'] for r in rows),'files':rows},indent=2)+'\n')
 shutil.rmtree(dirs[0])
 for p in files:
  if p.exists():p.unlink()
 summary['bytes_free_before_tests']=shutil.disk_usage(repo).free;save()
 py=str(repo/'.venv/bin/python')
 run('files-check',['python3','scripts/check_delivery.py'])
 run('numerical-tests',[py,'scripts/test_portable.py'])
 run('top-tests',[py,'-m','unittest','discover','-s','tests','-p','test_*.py'])
 run('nn-quick',[py,'scripts/nn_acceptance.py','--quick','--output',str(raw/'cold-nn-quick')])
 run('six',[py,'scripts/portable_smoke.py','--output',str(raw/'cold-six')])
 run('affine',[py,'scripts/affine_acceptance.py','--binary','build/bin/Simulator','--config','configs/ascend_910b_quiet.json','--cwd','build','--output',str(raw/'cold-affine')])
 sys.path.insert(0,str(repo));from scripts.nn_acceptance import archive_case
 out=raw/'cold-affine';rows=[archive_case(p) for p in sorted(out.iterdir()) if p.is_dir()];(out/'archive_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
 run('nn-suite',[py,'scripts/nn_acceptance.py','--output',str(raw/'cold-nn-suite')])
 old=json.loads((raw/'suite-v3/summary.json').read_text());new=json.loads((raw/'cold-nn-suite/summary.json').read_text());assert new['status']=='SUITE_CHECKED'
 old={r['case']:r for r in old['cases']};comparisons=[]
 for r in new['cases']:
  b=old[r['case']];same=(r['cycles']==b['cycles'] and r['max_task_error']==b['max_task_error'] and r['archive']['uncompressed_files']['outputs.npy']['sha256']==b['archive']['uncompressed_files']['outputs.npy']['sha256'])
  assert same;comparisons.append({'case':r['case'],'cycles':r['cycles'],'output_and_metric_exact':same})
 six=json.loads((raw/'cold-six/summary.json').read_text());aff=json.loads((raw/'cold-affine/summary.json').read_text());assert six['status']=='SIX_CASES_CHECKED' and aff['status']=='NINE_CASES_CHECKED'
 assert [r['cycles'] for r in six['cases']]==[28006,18697,21338,21665,7950,23047]
 summary.update(status='SOURCE_DELIVERY_COLD_ACCEPTED',phase='COMPLETE',nn_comparisons=comparisons,numerical_tests={'run':264,'passed':262,'optional_skips':2},top_tests=8,nn_suite_cases=36,nn_quick_cases=4,six_cases=6,affine_cases=9,research_score=None,promotion_allowed=False)
 save();print(summary['status'],flush=True)
except BaseException as e:
 summary.update(status='FAILED',reason=str(e),traceback=traceback.format_exc());save();raise
