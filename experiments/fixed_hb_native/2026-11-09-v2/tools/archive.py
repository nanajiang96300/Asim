from pathlib import Path
import json,hashlib,shutil,subprocess
repo=Path('/home/nanajiang/Asim')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
for campaign in ['fixed_hb/2026-11-09-v1','fixed_hb_native/2026-11-09-v1','fixed_hb_native/2026-11-09-v2']:
 root=repo/'results'/campaign;out=repo/'experiments'/campaign
 rows=[dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()]
 (out/'run_manifest.json').write_text(json.dumps(dict(schema='research-raw-artifact-manifest-v1',root=str(root.relative_to(repo)),files=rows),indent=2)+'\n')
print('raw manifests sealed')
archive=repo/'DOCS/research/paper_archive/2026-11-09-fixed-hb-v1';archive.mkdir(exist_ok=False)
paths=['experiments/fixed_hb/2026-11-09-v1/'+n for n in ['README.md','summary.json','contract.json','selection.json','math_review.md','old_failure_regression.json','iterations_error.csv','iterations_error.png','iterations_error.pdf','run_manifest.json']]
paths+=['experiments/fixed_hb_native/2026-11-09-v2/'+n for n in ['README.md','summary.json','contract.json','controls.json','cycles_and_replay.csv','pipeline_summary.csv','core_pipeline_summary.csv','dynamic_audit.md','static_audit.md','run_manifest.json','build_cache_receipt.json']]
paths+=['DOCS/research/F16_STABLE_HB_DESIGN.md','DOCS/operators/bri_stable_hb_fp16_v1.md','experiments/fixed_hb_native/2026-11-09-v1/README.md','experiments/fixed_hb_native/2026-11-09-v1/interruption.json','experiments/storage_cleanup/2026-11-09-lossless-v2/README.md','experiments/storage_cleanup/2026-11-09-lossless-v2/execution.json','orchestrator/codex_runtime_stable_hb_fp16_v1.json']
rows=[]
for name in paths:
 p=repo/name;q=archive/'snapshots'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 rows.append(dict(source=name,snapshot=str(q.relative_to(repo)),bytes=q.stat().st_size,sha256=sha(q)))
manifest=dict(schema='paper-archive-v1',archive_id=archive.name,capture_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),status='FIXED_HB_CDLB_RECONSTRUCTION_CHECKED',snapshots=rows)
(archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(archive/'README.md').write_text('''# 固定HB：FP16隐式MMSE的新原生重建验收

状态FIXED_HB_CDLB_RECONSTRUCTION_CHECKED。校准→冻结K16/53/174→新算法留出→真实Asim→实际指令重建→另192新CDL-B与完整未编码QPSK，均为既定软件数值模型；无硬件认证或性能晋级。

[原生报告](../../../../experiments/fixed_hb_native/2026-11-09-v2/README.md)记录周期、流水、重建时间与门禁；[算法研究](../../../../experiments/fixed_hb/2026-11-09-v1/README.md)含K/误差曲线、保守谱界、旧失败回归。新方案不覆盖旧β-only失败；首次原生v1中断也保留，不当作通过。

[manifest](manifest.json)固定快照与原件SHA。大型制品保持results/fixed_hb/与results/fixed_hb_native/的独立版本目录；旧制品空间回收采用可恢复gzip台账，不能静默修改旧manifest。

下一步先优化通用重建器大trace内存，再研究不使用留出调参的更紧谱界/更低指令成本；所有优化重新冻结和重验。
''')
cat=repo/'DOCS/research/paper_archive/catalog.json';d=json.loads(cat.read_text());assert archive.name not in [x['id'] for x in d['archives']]
d['archives'].append(dict(id=archive.name,status=manifest['status'],keywords=['BRI','fixed-HB','FP16','implicit-MMSE','CDL-B','reconstruction','K-calibration','stability','lossless-retention'],readme=str((archive/'README.md').relative_to(repo)),manifest=str((archive/'manifest.json').relative_to(repo)),manifest_sha256=sha(archive/'manifest.json'),supersedes=None));cat.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
with (repo/'DOCS/research/paper_archive/README.md').open('a') as f:f.write('\n2026-11-09：[固定HB原生验收与谱界/选K](2026-11-09-fixed-hb-v1/README.md)：新CDL-B三SNR、192独立H、576batch逐FP16位一致与QPSK门禁；旧β-only失败保留，无硬件/性能晋级。\n')
print(dict(snapshots=len(rows),archive=archive.name))
