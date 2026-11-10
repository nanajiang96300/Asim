from pathlib import Path
import json,hashlib,shutil,subprocess,sys
ROOT=Path('/home/nanajiang/Asim');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from optimization_ledger import file_sha,seal,read,export
from verification.reconstruction.daily import load_runtime
exp=ROOT/'experiments/optimization/bri/inner-0001';family=exp.parent;rows=read(family/'ledger.jsonl');assert len(rows)==11
snapshot=exp/'ledger_snapshot.jsonl';assert not snapshot.exists();shutil.copy2(family/'ledger.jsonl',snapshot)
rebuilt=exp/'comparison-check.tmp';export(family/'ledger.jsonl',rebuilt);assert file_sha(rebuilt)==file_sha(family/'comparison.csv');rebuilt.unlink()
current=load_runtime(ROOT/'orchestrator/codex_runtime.json',ROOT);assert len(current['files'])==91
for e in json.loads((exp/'run_manifest.json').read_text())['files']:
 p=Path(e['path']);assert p.stat().st_size==e['bytes'] and file_sha(p)==e['sha256']
for case in json.loads((family/'baseline-v1.json').read_text())['cases']:
 assert file_sha(case['result'])==case['result_sha256']
for ver in ['v3','v4']:
 for e in json.loads((ROOT/f'experiments/storage_cleanup/2026-11-10-lossless-{ver}/execution.json').read_text())['entries']:
  assert file_sha(ROOT/e['archive'])==e['archive_sha256'] and e['copy_verified'] and e['uncompressed_removed']
status=dict(status='INNER_0001_FINAL_EVIDENCE_CHECKED',ledger_events=len(rows),ledger_head=rows[-1]['record_sha256'],comparison_csv_rebuilt_identical=True,default_baseline_runtime_files=91,default_baseline_unchanged=True,raw_artifacts=len(json.loads((exp/'run_manifest.json').read_text())['files']),candidate_commit='88acdaed5e2f565f76a1a9f91f789ff07a440c67',score=None,promotion='NOT_PROMOTED_M3_PENDING')
(exp/'final_check.json').write_text(json.dumps(status,indent=2)+'\n')
archive=ROOT/'DOCS/research/paper_archive/2026-11-10-bri-inner-v1';archive.mkdir(exist_ok=False)
paths=[exp/name for name in ['README.md','candidate.json','candidate.patch','summary.json','parameters.json','cases.csv','coefficient_ranges.csv','pipeline_summary.csv','core_pipeline_summary.csv','paired_baseline.json','controls.json','static_audit.md','dynamic_audit.md','independent_dynamic_audit.json','independent_cdl_audit.json','run_manifest.json','ledger_snapshot.jsonl','final_check.json','cycle_instruction_comparison.png','cycle_instruction_comparison.pdf']]
paths += [family/'baseline-v1.json',family/'comparison.csv',ROOT/'DOCS/research/OPTIMIZATION_TRACKING.md',ROOT/'DOCS/operators/bri_inner_0001.md',ROOT/'scripts/optimization_ledger.py']
snapshots=[]
for p in paths:
 q=archive/'snapshots'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);snapshots.append(dict(source=str(p.relative_to(ROOT)),snapshot=str(q.relative_to(ROOT)),bytes=q.stat().st_size,sha256=file_sha(q)))
m=dict(schema='paper-archive-v1',archive_id=archive.name,capture_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),status='INNER_LOOP_SOFTWARE_CANDIDATE_CHECKED',snapshots=snapshots)
(archive/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(archive/'README.md').write_text('''# BRI内循环首轮：状态槽轮换与不可变基线

2026-11-10。状态INNER_LOOP_SOFTWARE_CANDIDATE_CHECKED。同Model/mode，独立candidate分支88acdaed；主工作区/默认runtime保持原基线，M3未晋级，score=null。

[完整报告](../../../../experiments/optimization/bri/inner-0001/README.md)记录三档真实周期18409→17488、47121→44128、141017→131248，周期减少5.003%/6.352%/6.928%；公式/K/精度不变，Rayleigh、新192-H CDL三SNR576batch与原基线配对重放均位一致，QPSK门禁通过。不是全部历史BRI/全算子统一排行榜或硬件认证。

[manifest](manifest.json)固定基线、候选、追加台账快照、CSV、参数、精度分布、流水、审计、控制和图。原始大制品仍在results/optimization/bri/inner-0001/；同一算法的后续优化以candidate_id+commit记录，不每轮复制算子类。下一轮独立预声明假设；改谱界/参数必须新校准/留出，当前H610只能回归。
''')
p=ROOT/'DOCS/research/paper_archive/catalog.json';c=json.loads(p.read_text());assert archive.name not in [e['id'] for e in c['archives']]
c['archives'].append(dict(id=archive.name,status=m['status'],keywords=['BRI','inner-loop','baseline','candidate-lineage','state-rotation','cycles','FP16','CDL-B','QPSK'],readme=str((archive/'README.md').relative_to(ROOT)),manifest=str((archive/'manifest.json').relative_to(ROOT)),manifest_sha256=file_sha(archive/'manifest.json'),supersedes=None));p.write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n')
with (ROOT/'DOCS/research/paper_archive/README.md').open('a') as f:f.write('\n2026-11-10：[BRI内循环首轮与基线/候选台账](2026-11-10-bri-inner-v1/README.md)：三档周期减少5.00–6.93%、精度位一致、新CDL/QPSK和基线配对重放；默认基线保留，未M3晋级。\n')
for a in c['archives']:
 p=ROOT/a['manifest'];assert file_sha(p)==a['manifest_sha256']
 for e in json.loads(p.read_text())['snapshots']:
  q=ROOT/e['snapshot'];assert q.stat().st_size==e['bytes'] and file_sha(q)==e['sha256']
print(dict(**status,paper_archives=len(c['archives']),new_snapshots=len(snapshots)))
