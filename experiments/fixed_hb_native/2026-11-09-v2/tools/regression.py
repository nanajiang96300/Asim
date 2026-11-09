from pathlib import Path
import json,gc,os
from verification.reconstruction.runner import evaluate
from verification.contracts.mmse import MMSETask
root=Path('/home/nanajiang/Asim');run=root/'results/fixed_hb_native/2026-11-09-v2';c=json.loads((root/'experiments/fixed_hb_native/2026-11-09-v2/contract.json').read_text());config=json.loads((run/'asim-lambda0.1/config.json').read_text());os.environ['LD_LIBRARY_PATH']=c['library_dir'];rows=[]
for op,k in [('ns_corrected_v1',16),('bri_implicit_fp16_v1',33)]:
 r=evaluate(op,MMSETask(profile='modeled_fp16_fp32_real_block_v1'),k,57001,Path(c['binary']),config,Path(c['cwd']),run/'regression'/op)
 row=dict(operator=op,iterations=k,status=r['status'],cycles=r.get('asim',{}).get('finish_cycle'),comparison_identity=r.get('comparison_identity'),max_task_error=max((b.get('task_error',float('inf')) for b in r.get('batches',[])),default=None),phase_timings_seconds=r['phase_timings_seconds'])
 rows.append(row);(run/'regression-index.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 if row['status']!='MODELED_NUMERIC_CHECKED':raise SystemExit(1)
 del r;gc.collect()
