from pathlib import Path
import json,sys,subprocess,numpy as np
from verification.reconstruction.implicit_campaign import source_map
from verification.asim.runner import file_hash
from verification.channels.cdlb import SPEC,provenance
repo=Path('/home/nanajiang/Asim');w=Path('/home/nanajiang/asim-worktrees/bri-implicit-fp16-20261109')
binary=w/'build/bin/Simulator';cfg=repo/'configs/ascend_910b_quiet.json';cwd=w/'build';lib=cwd/'lib'
assert binary.is_file()
deps=[binary,cfg,lib/'libramulator.so',w/'configs/ramulator2_configs/HBM2.yaml']
c=dict(schema='implicit-native-campaign-v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),binary_source_commit=subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD'],text=True).strip(),
       binary=str(binary),config=str(cfg),cwd=str(cwd),library_dir=str(lib),dependencies={str(p):file_hash(p) for p in deps},sources=source_map(repo),python_version=sys.version.split()[0],numpy_version=np.__version__,
       cases=[dict(regularization=l,iterations=k) for l,k in [(1.,13),(.1,33),(.01,76)]],channel_seeds=[56001,56002],frame_seeds=[56011,56012],cdlb=SPEC,channel_toolchain=provenance(),
       limits=dict(max_ber_increase=.01,max_ser_increase=.02,max_evm_increase=.01,max_outage_probability=.05,family_alpha=.05,min_independent_channels=192),
       budget=dict(max_wall_seconds=5400,min_free_bytes=600000000,max_output_bytes=3500000000,max_batch_replays=576))
p=repo/'experiments/bri_implicit/2026-11-09-v1/contract.json'
with p.open('x') as f:f.write(json.dumps(c,indent=2)+'\n')
print(file_hash(p))
