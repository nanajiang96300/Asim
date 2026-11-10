from pathlib import Path
import json,sys
from verification.reconstruction.stable_hb_campaign import source_map
from verification.asim.runner import file_hash
repo=Path('/home/nanajiang/Asim');raw=repo/'results/memory/2026-11-10-v3';high=json.loads((raw/'hb174/result.json').read_text());reg=json.loads((raw/'regression-index.json').read_text())
assert high['within_2gib'] and len(reg)==8 and all(r['within_2gib'] for r in reg)
old=json.loads((repo/'orchestrator/codex_runtime_stable_hb_fp16_v1.json').read_text());paths=set(source_map(repo))
paths.update(e['path'] for e in old['files'] if not e['path'].startswith('verification/'))
paths.update(['src/inverse/bri_implicit/BRIImplicitModel.cc','src/inverse/bri_implicit/BRIImplicitModel.h'])
new={k:v for k,v in old.items() if k!='files'}
new['acceptance_note']=old['acceptance_note']+' M1 evaluator memory revision: sealed real-Asim artifacts rechecked for nine entries, full HB K174/batch96 <=2GiB, identical pipeline JSON/CSV, schedule and FP16 outputs. Binary unchanged; source/runtime software identity is new. Fresh daily smoke checks recorded separately.'
new['files']=[dict(path=p,bytes=(repo/p).stat().st_size,sha256=file_hash(repo/p)) for p in sorted(paths)]
text=json.dumps(new,indent=2)+'\n';dest=repo/'orchestrator/codex_runtime_memory_v1.json';assert not dest.exists();dest.write_text(text);(repo/'orchestrator/codex_runtime.json').write_text(text)
print(dict(files=len(new['files']),runtime_sha256=file_hash(dest),binary_source_commit=new['binary_source_commit']))
