"""Install portable templates; optionally clean owned build artifacts and seal manifest."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');v=a.parse_args()
r=ROOT/'expert_delivery/asim-expert-v1';assert r.is_dir()
t=ROOT/'packaging/templates'
for n in ['README.md','AGENTS.md','CLAUDE.md','THIRD_PARTY_NOTICES.md']:shutil.copy2(t/n,r/n)
for n in ['PYTHON_TO_OPERATOR.md','RECONSTRUCTION.md','OPERATORS.md']:shutil.copy2(t/n,r/'docs'/n)
for n in ['bootstrap.sh','smoke.py','check_package.py','algorithm_example.py']:shutil.copy2(t/n,r/'scripts'/n)
(r/'scripts/bootstrap.sh').chmod(0o755)
for n in ['archive_checks.py','remaining_contracts.py','audit_solver_stages.py','check_solver_mapping_archive.py']:shutil.copy2(ROOT/'scripts'/n,r/'scripts'/n)
(r/'requirements.txt').write_text('numpy==2.5.0\n')
(r/'.gitignore').write_text('build/\n.venv/\nwork/\nvalidation-work/\n__pycache__/\n*.pyc\n')
p=r/'verification/tests/test_channel_study.py';s=p.read_text();needle='    def test_real_cdl_reproducible_and_not_kronecker_surrogate(self):'
if 'optional CDL dependencies not bundled' not in s:s=s.replace(needle,'    @unittest.skipUnless(__import__("importlib.util",fromlist=["find_spec"]).find_spec("torch") and __import__("importlib.util",fromlist=["find_spec"]).find_spec("sionna"), "optional CDL dependencies not bundled")\n'+needle);p.write_text(s)
p=r/'configs/timeloop_configs/systolic_ws_8x8/example_AlexNet_layer1_outputs'
if p.exists():shutil.rmtree(p)
# Referenced research context is a portable protocol, not the server's result archive.
p=r/'DOCS/research';p.mkdir(exist_ok=True)
(p/'ADDING_OPERATOR.md').write_text('# 新增算子\n\n读取README.md、docs/PYTHON_TO_OPERATOR.md与docs/RECONSTRUCTION.md。按任务契约→独立精度筛选→候选实现→冻结→Asim→实际重建/同算法/任务参考→质量验收推进。保留基线、参数、源码/工具版本和失败证据。研究结果不随部署包提供。\n')
p=p/'paper_archive';p.mkdir(exist_ok=True);(p/'README.md').write_text('# 论文证据归档\n\n此包不含服务器研究结果。新实验应在experiments/<campaign>/<version>/记录可追溯摘要，results/<campaign>/<version>/保存原始证据；禁止覆盖失败或旧报告。\n')
meta=json.loads((r/'SOURCE_PROVENANCE.json').read_text());meta['packaging_changes']=['portable source-vendored CMake and ONNX_API export definition','Ramulator dependency setup only','standalone source-contract test fixture','explicit optional CDL dependency test skip','remove old timeloop demonstration outputs'];(r/'SOURCE_PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')
if v.seal:
 for n in ['build','.venv','validation-work','relocation-build']:
  p=r/n
  if p.exists():shutil.rmtree(p)
 for p in sorted(r.rglob('__pycache__'),reverse=True):shutil.rmtree(p)
 for p in r.rglob('*.pyc'):p.unlink()
 files=[]
 for p in sorted(r.rglob('*')):
  if p.is_symlink():raise ValueError('symlink forbidden: '+str(p))
  if p.is_file() and p.name!='PACKING_MANIFEST.json':
   with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
   files.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=d))
 (r/'PACKING_MANIFEST.json').write_text(json.dumps({'schema':'asim-source-package-v1','files':files},indent=2)+'\n')
 print('sealed',len(files),'files',sum(f['bytes'] for f in files),'bytes')
