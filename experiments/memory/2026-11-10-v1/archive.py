from pathlib import Path
import json,hashlib,shutil,subprocess
repo=Path('/home/nanajiang/Asim');dest=repo/'DOCS/research/paper_archive/2026-11-10-memory-v1';dest.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=['experiments/memory/2026-11-10-v1/'+n for n in ['README.md','summary.json','cases.csv','daily_ns.json','daily_hb.json','fresh_daily_comparison.json','replayer_smoke.json','unit-tests.log','run_manifest.json','redundant-statistics-retirement.json','check.py','replayer_smoke.py']]
paths+=['DOCS/research/M1_MEMORY_DESIGN.md','orchestrator/codex_runtime_memory_v1.json','verification/reconstruction/stream.py','verification/schedule/runtime.py','verification/reporting/pipeline.py']
rows=[]
for name in paths:
 p=repo/name;q=dest/'snapshots'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);rows.append(dict(source=name,snapshot=str(q.relative_to(repo)),bytes=q.stat().st_size,sha256=sha(q)))
m=dict(schema='paper-archive-v1',archive_id=dest.name,capture_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),status='M1_MEMORY_ACCEPTED_BOUNDED_SCOPE',snapshots=rows)
(dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(dest/'README.md').write_text('''# M1通用重建内存优化（2026-11-10）

状态M1_MEMORY_ACCEPTED_BOUNDED_SCOPE。九入口封存真实Asim制品全量回归，HB K174/batch96完整事件检查与重建峰值1.88GiB，流水JSON/逐指令CSV字节一致、schedule报告相等、完整输出逐FP16位一致。249测试，新Asim日常NS/HB及QPSK Replayer接入通过。

[完整报告](../../../../experiments/memory/2026-11-10-v1/README.md)包含三轮内存测量、受限范围、脚本、清理台账、新91文件runtime和后续计划；[manifest](manifest.json)固定本次快照。旧档案不覆盖，原始结果仍在results/memory/独立版本目录。

这是软件验证器资源优化，不改变算子/K/精度/质量阈值，不增加独立留出样本，不升级旧CDL失败，也不是硬件时序、任意规模常数内存或性能晋级。下一步M2在新校准集研究更紧谱界/复用/流水。
''')
p=repo/'DOCS/research/paper_archive/catalog.json';c=json.loads(p.read_text());assert dest.name not in [r['id'] for r in c['archives']]
c['archives'].append(dict(id=dest.name,status=m['status'],keywords=['memory','streaming','reconstruction','actual-instructions','FP16','HB','pipeline','runtime'],readme=str((dest/'README.md').relative_to(repo)),manifest=str((dest/'manifest.json').relative_to(repo)),manifest_sha256=sha(dest/'manifest.json'),supersedes=None));p.write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n')
with (repo/'DOCS/research/paper_archive/README.md').open('a') as f:f.write('\n2026-11-10：[M1通用重建内存优化](2026-11-10-memory-v1/README.md)：K174/batch96峰值1.88GiB、九入口精确回归、新Asim冒烟与QPSK接入，249测试；旧失败与超预算试验保留。\n')
print(dict(archive=dest.name,snapshots=len(rows)))
