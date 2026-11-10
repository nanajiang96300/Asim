"""Exact-byte, readonly dedup of selected immutable repeated diagnostic files."""
from pathlib import Path
import hashlib,json,os,stat
R=Path('/home/nanajiang/Asim');O=R/'experiments/distribution/2026-11-10-v1';names=['instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','formula.json','pipeline_statistics.json','instruction_pipeline_detail.csv']
pairs=[('results/memory/2026-11-10-v3/daily_hb','results/fixed_hb_native/2026-11-09-v2/daily_hb'),('results/memory/2026-11-10-v3/daily_ns','results/fixed_hb_native/2026-11-09-v2/daily_ns'),('results/memory/2026-11-10-v3/daily_ns_parser_v2','results/fixed_hb_native/2026-11-09-v2/daily_ns'),('results/optimization/bri/inner-0001/controls-performance-reject','results/optimization/bri/inner-0001/asim-lambda1.0'),('results/fixed_hb_native/2026-11-09-v2/controls/repeat','results/fixed_hb_native/2026-11-09-v2/asim-lambda1.0'),('results/fixed_hb_native/2026-11-09-v2/controls/performance_reject','results/fixed_hb_native/2026-11-09-v2/asim-lambda1.0')]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
import sys
if sys.argv[1]=='plan':
 rows=[]
 for a,b in pairs:
  for name in names:
   p=R/a/name;q=R/b/name
   if not p.is_file() or not q.is_file() or p.is_symlink() or q.is_symlink() or p.stat().st_size<1024**2 or p.stat().st_nlink!=1 or p.stat().st_size!=q.stat().st_size:continue
   h=sha(p)
   if h==sha(q):rows.append(dict(remove_copy=str(p.relative_to(R)),retain=str(q.relative_to(R)),bytes=p.stat().st_size,sha256=h))
 f=O/'dedup_plan.json'
 with f.open('x') as s:s.write(json.dumps(dict(script_sha256=sha(Path(__file__)),entries=rows),indent=2)+'\n')
 print(dict(plan_sha256=sha(f),files=len(rows),bytes=sum(r['bytes'] for r in rows)))
elif sys.argv[1]=='execute':
 f=O/'dedup_plan.json';assert sha(f)==sys.argv[2];c=json.loads(f.read_text());assert c['script_sha256']==sha(Path(__file__));ledger=O/'dedup_execution.json';assert not ledger.exists();rows=[]
 for r in c['entries']:
  p=R/r['remove_copy'];q=R/r['retain'];assert p.stat().st_nlink==1 and p.stat().st_size==q.stat().st_size==r['bytes'] and sha(p)==sha(q)==r['sha256']
  rows.append(dict(**r,linked=False));ledger.write_text(json.dumps(rows,indent=2));tmp=p.with_name(p.name+'.dedup');assert not tmp.exists();os.link(q,tmp);os.replace(tmp,p);q.chmod(stat.S_IMODE(q.stat().st_mode)&~0o222);assert os.path.samefile(p,q);rows[-1]['linked']=True;ledger.write_text(json.dumps(rows,indent=2))
 print(dict(status='READONLY_IDENTICAL_DEDUP_CHECKED',files=len(rows),saved_bytes=sum(r['bytes'] for r in rows)))
