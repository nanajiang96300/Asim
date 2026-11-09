from pathlib import Path
import json,shutil,sys,numpy as np
from verification.asim.runner import file_hash
from verification.reconstruction.stable_hb_campaign import source_map
from verification.schedule.runtime import HEADER
repo=Path('/home/nanajiang/Asim');run=repo/'results/fixed_hb_native/2026-11-09-v2';c=json.loads((repo/'experiments/fixed_hb_native/2026-11-09-v2/contract.json').read_text())
assert json.loads((run/'holdout/result.json').read_text())['status']=='FIXED_HB_CDLB_RECONSTRUCTION_CHECKED'
reg=json.loads((run/'regression-index.json').read_text());assert len(reg)==2 and all(r['status']=='MODELED_NUMERIC_CHECKED' for r in reg)
assert source_map(repo)==c['sources']
rt=run/'runtime';(rt/'bin').mkdir(parents=True);(rt/'lib').mkdir();(rt/'runtime').mkdir()
shutil.copy2(c['binary'],rt/'bin/Simulator');shutil.copy2(Path(c['library_dir'])/'libramulator.so',rt/'lib/libramulator.so')
assert file_hash(rt/'bin/Simulator')==c['dependencies'][c['binary']]
assert file_hash(rt/'lib/libramulator.so')==c['dependencies'][str(Path(c['library_dir'])/'libramulator.so')]
config=json.loads(Path(c['config']).read_text())
for key,name in [('dram_config_path','HBM2.yaml'),('icnt_config_path','fly_c1_m2.icnt')]:
 if config.get(key):
  source=(Path(c['cwd'])/config[key]).resolve();shutil.copy2(source,rt/'runtime'/name);config[key]='runtime/'+name
(rt/'runtime_config.json').write_text(json.dumps(config,indent=2))
rel=lambda p:str(p.relative_to(repo));deps=[rt/'lib/libramulator.so',rt/'runtime/HBM2.yaml',rt/'runtime/fly_c1_m2.icnt']
# Full current Python closure and both new and legacy implicit C++ provenance.
paths=[repo/p for p in source_map(repo)]+[repo/'src/inverse/bri_implicit/BRIImplicitModel.cc',repo/'src/inverse/bri_implicit/BRIImplicitModel.h',rt/'bin/Simulator',rt/'runtime_config.json',*deps]
old=json.loads((repo/'orchestrator/codex_runtime_implicit_fp16_v1.json').read_text())
spec=dict(schema='supervised-reconstruction-runtime-v1',binary=rel(rt/'bin/Simulator'),config=rel(rt/'runtime_config.json'),cwd=rel(rt),library_dir=rel(rt/'lib'),dependencies=[rel(p) for p in deps],binary_source_commit=c['binary_source_commit'],runtime_contract=HEADER,operators=old['operators']+['bri_stable_hb_fp16_v1'],profiles=old['profiles'],python_version=sys.version.split()[0],numpy_version=np.__version__,acceptance_note='Daily task is Rayleigh. Stable-HB fixed K16/53/174 passed software modeled CDL-B 192-H at three SNR and uncoded QPSK; no hardware certification or performance promotion. Prior implicit v1 remains CDL-quality failed; old seven F64/F16 scope inherited, NS and implicit-v1 F16 rechecked with new binary.',experimental_operators=['bri_implicit_fp16_v1','bri_stable_hb_fp16_v1'],files=[dict(path=rel(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(set(paths))])
text=json.dumps(spec,indent=2)+'\n';(repo/'orchestrator/codex_runtime_stable_hb_fp16_v1.json').write_text(text);(repo/'orchestrator/codex_runtime.json').write_text(text)
print(dict(files=len(spec['files']),binary_sha256=file_hash(rt/'bin/Simulator'),runtime_sha256=file_hash(repo/'orchestrator/codex_runtime.json')))
