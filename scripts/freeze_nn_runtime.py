"""Publish a new NN runtime only from completed suite and same-binary legacy regression."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from verification.asim.runner import file_hash
from verification.schedule.runtime import HEADER

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--suite',type=Path,required=True);a.add_argument('--regression',type=Path,required=True);a.add_argument('--binary',type=Path,default=ROOT/'build/bin/Simulator');a.add_argument('--runtime-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args()
    suite=json.loads(v.suite.read_text());reg=json.loads(v.regression.read_text());identity=file_hash(v.binary)
    if suite['status']!='SUITE_CHECKED' or len(suite['cases'])!=36 or {r['kind'] for r in suite['cases']}!={'softmax','gelu','mlp','attention'} or reg['status']!='SIX_CASES_CHECKED' or len(reg['cases'])!=6:raise ValueError('complete NN suite and six regressions required')
    if suite['binary_sha256']!=identity or reg['binary_sha256']!=identity or any(r['status']!='MODELED_NUMERIC_CHECKED' or not r['all_stage_bits_equal'] or not r['all_fp16_bits_equal'] for r in suite['cases']):raise ValueError('acceptance/binary identity mismatch')
    for row in suite['cases']:
        e=row['archive'];p=v.suite.parent/e['path']
        if not p.is_file() or p.stat().st_size!=e['bytes'] or file_hash(p)!=e['sha256']:raise ValueError('NN archive changed')
    runtime=v.runtime_dir.resolve();out=v.output.resolve()
    if not runtime.is_relative_to(ROOT) or not out.is_relative_to(ROOT) or runtime.exists() or out.exists():raise ValueError('new runtime/spec paths must belong to this checkout')
    runtime.mkdir(parents=True);(runtime/'bin').mkdir();(runtime/'lib').mkdir();(runtime/'runtime').mkdir()
    shutil.copy2(v.binary,runtime/'bin/Simulator');shutil.copy2(v.binary.parent.parent/'lib/libramulator.so',runtime/'lib/libramulator.so')
    shutil.copy2(ROOT/'configs/ramulator2_configs/HBM2.yaml',runtime/'runtime/HBM2.yaml');shutil.copy2(ROOT/'configs/booksim2_configs/fly_c1_m2.icnt',runtime/'runtime/fly_c1_m2.icnt')
    c=json.loads((ROOT/'configs/ascend_910b_nn_v1.json').read_text());c['dram_config_path']='runtime/HBM2.yaml';c['icnt_config_path']='runtime/fly_c1_m2.icnt';(runtime/'runtime_config.json').write_text(json.dumps(c,indent=2)+'\n')
    def rel(p):return str(p.relative_to(ROOT))
    selected=set(p for prefix in ['src','verification'] for p in (ROOT/prefix).rglob('*') if p.is_file() and p.suffix in ['.py','.h','.cc'])
    selected.update(ROOT/p for p in ['scripts/nn_acceptance.py','scripts/freeze_nn_runtime.py','orchestrator/codex_operators.json','configs/ascend_910b_nn_v1.json','dependencies/lock.json','CMakeLists.txt'])
    selected.update(p for p in runtime.rglob('*') if p.is_file())
    files=[dict(path=rel(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(selected)]
    spec={'schema':'supervised-reconstruction-runtime-v1','binary':rel(runtime/'bin/Simulator'),'config':rel(runtime/'runtime_config.json'),'cwd':rel(runtime),'library_dir':rel(runtime/'lib'),'dependencies':[rel(runtime/p) for p in ['lib/libramulator.so','runtime/HBM2.yaml','runtime/fly_c1_m2.icnt']],'binary_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'runtime_contract':HEADER,'operators':['nn_graph_v1'],'profiles':['modeled_fp16_fp32_nn_v1'],'python_version':sys.version.split()[0],'numpy_version':np.__version__,'files':files,'scope':'four frozen trusted graph families; no wireless/training/hardware or performance promotion','acceptance_evidence':[{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in [v.suite,v.regression]]}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({'status':'NEW_NN_RUNTIME_FROZEN','files':len(files),'spec_sha256':file_hash(out)}))
if __name__=='__main__':main()
