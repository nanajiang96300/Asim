"""Deduplicate immutable identical v1/v2 native diagnostics; preserve both paths."""
from pathlib import Path
import json,hashlib,os,stat,shutil,sys
ROOT=Path('/home/nanajiang/Asim');BASE=ROOT/'results/fixed_hb_native';OUT=ROOT/'experiments/storage_cleanup/2026-11-10-native-dedup-v1'
NAMES={'instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','pipeline_statistics.json','instruction_pipeline_detail.csv'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def save(p,v):
 t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
if sys.argv[1]=='plan':
 assert not (OUT/'plan.json').exists();rows=[]
 for case in ('asim-lambda1.0','asim-lambda0.1','asim-lambda0.01'):
  for name in sorted(NAMES):
   a=BASE/'2026-11-09-v1'/case/name;b=BASE/'2026-11-09-v2'/case/name
   if not a.is_file() or not b.is_file() or a.is_symlink() or b.is_symlink() or a.stat().st_nlink!=1 or b.stat().st_nlink!=1 or a.stat().st_size<8*1024**2 or a.stat().st_size!=b.stat().st_size:continue
   h=sha(a)
   if h==sha(b):rows.append(dict(old=str(a.relative_to(ROOT)),keep=str(b.relative_to(ROOT)),bytes=a.stat().st_size,sha256=h))
 save(OUT/'plan.json',dict(schema='immutable-native-dedup-v1',script_sha256=sha(Path(__file__)),entries=rows,scope='v1 interrupted and v2 completed actual native diagnostics only; exact bytes and both paths preserved; shared files become readonly'))
 print(dict(plan_sha256=sha(OUT/'plan.json'),files=len(rows),logical_duplicate_bytes=sum(r['bytes'] for r in rows)))
elif sys.argv[1]=='execute':
 assert sha(OUT/'plan.json')==sys.argv[2] and not (OUT/'execution.json').exists();plan=json.loads((OUT/'plan.json').read_text());assert plan['script_sha256']==sha(Path(__file__))
 data=dict(status='IN_PROGRESS',plan_sha256=sys.argv[2],free_before=shutil.disk_usage(ROOT).free,entries=[]);save(OUT/'execution.json',data)
 for e in plan['entries']:
  a=ROOT/e['old'];b=ROOT/e['keep'];assert not a.is_symlink() and not b.is_symlink() and a.stat().st_nlink==1 and b.stat().st_nlink==1 and a.stat().st_size==b.stat().st_size==e['bytes'] and sha(a)==sha(b)==e['sha256']
  row={**e,'verified':True,'linked':False};data['entries'].append(row);save(OUT/'execution.json',data)
  temp=a.with_name(a.name+'.dedup');assert not temp.exists();os.link(b,temp);os.replace(temp,a);b.chmod(stat.S_IMODE(b.stat().st_mode)&~0o222)
  assert os.path.samefile(a,b) and sha(a)==e['sha256'];row['linked']=True;row['readonly']=True;save(OUT/'execution.json',data)
 data.update(status='IMMUTABLE_DEDUP_CHECKED',free_after=shutil.disk_usage(ROOT).free,duplicate_bytes_removed=sum(r['bytes'] for r in data['entries']));save(OUT/'execution.json',data);print({k:v for k,v in data.items() if k!='entries'})
else:raise ValueError('unknown action')
