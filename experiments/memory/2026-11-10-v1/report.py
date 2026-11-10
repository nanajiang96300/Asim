from pathlib import Path
import json,csv,shutil,hashlib,subprocess
repo=Path('/home/nanajiang/Asim');out=repo/'experiments/memory/2026-11-10-v1';raw=repo/'results/memory/2026-11-10-v3'
hi=json.loads((raw/'hb174/result.json').read_text());regs=json.loads((raw/'regression-index.json').read_text());assert hi['within_2gib'] and len(regs)==8
rows=[hi,*regs]
assert len({r['operator'] for r in rows})==9 and all(r['outputs_bit_equal'] and r['schedule_exact_equal'] and r['statistics_byte_equal'] and r['within_2gib'] for r in rows)
summary=dict(status='M1_MEMORY_ACCEPTED_BOUNDED_SCOPE',cases=rows,attempts=[json.loads((repo/f'results/memory/2026-11-10-v{n}/hb174/result.json').read_text()) for n in (1,2)],test_count=249,binary_unchanged=True,scope='software evaluator; nine sealed actual Asim cases, full high-K replay; fresh daily cases separately sealed',pure_model_thinking_seconds=None)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fields=['operator','iterations','batch_size','finish_cycle','peak_rss_kib','within_2gib','statistics_byte_equal','schedule_exact_equal','outputs_bit_equal']
with (out/'cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
for name in ('daily_ns','daily_hb'):
 p=raw/name/'result.json';r=json.loads(p.read_text());assert r['status']=='MODELED_NUMERIC_CHECKED' and not r['failed_batches'];shutil.copy2(p,out/(name+'.json'))
shutil.copy2(repo/'results/memory-tests-v4.log',out/'unit-tests.log')
shutil.copy2(repo/'results/memory/redundant-statistics-retirement.json',out/'redundant-statistics-retirement.json')
logs=['memory-hb174-v1.log','memory-hb174-v1-run.log','memory-hb174-v2-run.log','memory-hb174-v3-run.log','memory-regression-v3.log','memory-daily-ns-v3.log','memory-daily-hb-v3.log']
for name in logs:shutil.copy2(repo/'results'/name,out/name)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
paths=[p for p in (repo/'results/memory').rglob('*') if p.is_file()]+[repo/'results'/n for n in logs]
(out/'run_manifest.json').write_text(json.dumps(dict(schema='memory-campaign-artifacts-v1',files=[dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]),indent=2)+'\n')
print(dict(status=summary['status'],cases=len(rows),high_peak_gib=hi['peak_rss_kib']/1024**2))
