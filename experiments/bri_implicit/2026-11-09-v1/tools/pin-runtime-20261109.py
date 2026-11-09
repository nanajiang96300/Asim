from pathlib import Path
import json,shutil,sys,subprocess,numpy as np
from verification.asim.runner import file_hash
from verification.reconstruction.implicit_campaign import source_map
from verification.schedule.runtime import HEADER
repo=Path('/home/nanajiang/Asim');run=repo/'results/bri_implicit/2026-11-09-v1';c=json.loads((repo/'experiments/bri_implicit/2026-11-09-v1/contract.json').read_text())
reg=json.loads((run/'regression-index.json').read_text());assert len(reg)==7 and all(r['status']=='MODELED_NUMERIC_CHECKED' for r in reg)
rt=run/'runtime';(rt/'bin').mkdir(parents=True);(rt/'lib').mkdir();(rt/'runtime').mkdir()
shutil.copy2(c['binary'],rt/'bin/Simulator');shutil.copy2(Path(c['library_dir'])/'libramulator.so',rt/'lib/libramulator.so')
config=json.loads(Path(c['config']).read_text())
for key,name in [('dram_config_path','HBM2.yaml'),('icnt_config_path','fly_c1_m2.icnt')]:
 if config.get(key):
  source=(Path(c['cwd'])/config[key]).resolve();shutil.copy2(source,rt/'runtime'/name);config[key]='runtime/'+name
(rt/'runtime_config.json').write_text(json.dumps(config,indent=2))
rel=lambda p:str(p.relative_to(repo))
deps=[rt/'lib/libramulator.so',rt/'runtime/HBM2.yaml',rt/'runtime/fly_c1_m2.icnt']
paths=[repo/p for p in source_map(repo)]+[rt/'bin/Simulator',rt/'runtime_config.json',*deps]
spec=dict(schema='supervised-reconstruction-runtime-v1',binary=rel(rt/'bin/Simulator'),config=rel(rt/'runtime_config.json'),cwd=rel(rt),library_dir=rel(rt/'lib'),dependencies=[rel(p) for p in deps],binary_source_commit=c['binary_source_commit'],runtime_contract=HEADER,
 operators=[r['operator'] for r in reg]+['bri_implicit_fp16_v1'],profiles=['structure_f64_real_block','modeled_fp16_fp32_real_block_v1'],python_version=sys.version.split()[0],numpy_version=np.__version__,
 acceptance_note='Daily task is Rayleigh MMSE. Implicit FP16 replay passed, but fresh CDL-B 20dB quality failed; experimental, not promoted. Prior seven F64 scope inherited, this run regressed their F16 profiles.',
 experimental_operators=['bri_implicit_fp16_v1'],files=[dict(path=rel(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(set(paths))])
text=json.dumps(spec,indent=2)+'\n'
(repo/'orchestrator/codex_runtime_implicit_fp16_v1.json').write_text(text)
(repo/'orchestrator/codex_runtime.json').write_text(text)
print(dict(files=len(spec['files']),binary_sha256=file_hash(rt/'bin/Simulator'),source_commit=c['source_commit'],runtime_sha256=file_hash(repo/'orchestrator/codex_runtime.json')))
