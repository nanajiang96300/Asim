from pathlib import Path
import subprocess,sys,json,os
repo=Path('/home/nanajiang/Asim');root=repo/'results/memory/2026-11-10-v3';root.mkdir(parents=True,exist_ok=True);rows=[]
old=repo/'results/bri_implicit/2026-11-09-v1'
paths=[old/'regression'/op for op in ['ns_corrected_v1','cholesky_explicit_v1','ldl_explicit_v1','bri_explicit_v1','cholesky_block_explicit_v1','ldl_block_explicit_v1','cholesky_merge_explicit_v1']]+[repo/'results/fixed_hb_native/2026-11-09-v2/regression/bri_implicit_fp16_v1']
for p in paths:
 dest=root/p.name
 with (root/(p.name+'.log')).open('x') as f:r=subprocess.run([sys.executable,'experiments/memory/2026-11-10-v1/check.py',str(p),str(dest)],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,p
 row=json.loads((dest/'result.json').read_text());assert row['within_2gib'];rows.append(row);(root/'regression-index.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(operator=row['operator'],peak_rss_kib=row['peak_rss_kib'])),flush=True)
