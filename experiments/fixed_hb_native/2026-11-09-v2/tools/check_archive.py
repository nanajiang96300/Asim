from pathlib import Path
import json,hashlib,os,stat
from verification.reconstruction.daily import load_runtime
repo=Path('/home/nanajiang/Asim')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
cat=json.loads((repo/'DOCS/research/paper_archive/catalog.json').read_text());count=0
for entry in cat['archives']:
 p=repo/entry['manifest'];assert sha(p)==entry['manifest_sha256'];m=json.loads(p.read_text())
 for e in m['snapshots']:
  p=repo/e['snapshot'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'];count+=1
raw=[]
for campaign in ['fixed_hb/2026-11-09-v1','fixed_hb_native/2026-11-09-v1','fixed_hb_native/2026-11-09-v2']:
 p=repo/'experiments'/campaign/'run_manifest.json';m=json.loads(p.read_text());actual={str(p.relative_to(repo)) for p in (repo/m['root']).rglob('*') if p.is_file()};assert actual=={e['path'] for e in m['files']}
 for e in m['files']:assert (repo/e['path']).stat().st_size==e['bytes']
 raw.append(dict(campaign=campaign,files=len(m['files']),logical_bytes=sum(e['bytes'] for e in m['files'])))
dedup=json.loads((repo/'experiments/storage_cleanup/2026-11-10-native-dedup-v1/execution.json').read_text())
assert dedup['status']=='IMMUTABLE_DEDUP_CHECKED'
for e in dedup['entries']:
 a=repo/e['old'];b=repo/e['keep'];assert os.path.samefile(a,b) and not (stat.S_IMODE(a.stat().st_mode)&0o222) and sha(a)==e['sha256']
spec=load_runtime(repo/'orchestrator/codex_runtime.json',repo);assert (repo/'orchestrator/codex_runtime.json').read_bytes()==(repo/'orchestrator/codex_runtime_stable_hb_fp16_v1.json').read_bytes()
r=dict(status='FINAL_ARCHIVE_AND_RUNTIME_CHECKED',archives=len(cat['archives']),snapshot_hashes_checked=count,new_raw_manifests=raw,runtime_files=len(spec['files']),immutable_pairs=len(dedup['entries']),raw_hash_scope='SHA-256 streamed at manifest generation; exact coverage/current sizes checked here; duplicate pairs and all paper/runtime hashes reread',pure_model_thinking_seconds=None)
print(json.dumps(r));(repo/'experiments/fixed_hb_native/2026-11-09-v2/final_archive_check.json').write_text(json.dumps(r,indent=2)+'\n')
