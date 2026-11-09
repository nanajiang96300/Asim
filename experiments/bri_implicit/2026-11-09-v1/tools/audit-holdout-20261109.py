"""Independent FP64 task and nearest-point bit audit; no reconstruction imports."""
from pathlib import Path
import numpy as np,json,hashlib
root=Path('/home/nanajiang/Asim/results/bri_implicit/2026-11-09-v1');r=json.loads((root/'holdout/result.json').read_text());rows=[]
labels=np.array([[1,1],[1,0],[0,1],[0,0]],np.uint8);points=((2*labels[:,0].astype(float)-1)+1j*(2*labels[:,1].astype(float)-1))/np.sqrt(2)
for g in r['groups']:
 aerr=berr=0;worst=0.;bad=0;hashes=[];bits=0;target_diff=0.
 for frame in [f for f in r['frames'] if f['regularization']==g['regularization']]:
  p=Path(frame['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==frame['sha256']
  with np.load(p,allow_pickle=False) as z:
   H,Y,tx,raw=z['H'],z['Y'],z['bits'],z['output'];X=raw[:,:16,:8].astype(float)+1j*raw[:,16:,:8].astype(float)
   assert np.array_equal(raw[:,16:,8:],raw[:,:16,:8]) and np.array_equal(raw[:,:16,8:],-raw[:,16:,:8])
   sym=((2*tx[...,0].astype(float)-1)+1j*(2*tx[...,1].astype(float)-1))/np.sqrt(2)
   assert np.allclose(Y,H@sym+z['noise'],rtol=1e-13,atol=1e-13)
   truth=[]
   for h,y in zip(H,Y):truth.append(np.linalg.solve(h.conj().T@h+g['regularization']*np.eye(16),h.conj().T@y))
   truth=np.stack(truth);target_diff=max(target_diff,float(np.max(np.abs(truth-z['mmse']))))
   assert np.allclose(truth,z['mmse'],rtol=1e-11,atol=1e-11)
   errors=np.linalg.norm(X-truth,axis=(1,2))/np.linalg.norm(truth,axis=(1,2));assert np.allclose(errors,z['task_errors'],rtol=1e-10,atol=1e-12)
   nearest=lambda a:labels[np.argmin(np.abs(a[...,None]-points)**2,axis=-1)]
   aerr+=int(np.count_nonzero(nearest(X)!=tx));berr+=int(np.count_nonzero(nearest(truth)!=tx));bits+=tx.size
   worst=max(worst,float(errors.max()));bad+=int(np.sum(errors>.02));hashes.extend(hashlib.sha256(h.tobytes()).hexdigest() for h in H)
 assert len(hashes)==len(set(hashes))==192 and bits==49152 and aerr==g['candidate_bit_errors'] and berr==g['mmse_bit_errors'] and abs(worst-g['max_task_error'])<1e-10
 rows.append(dict(regularization=g['regularization'],candidate_bit_errors=aerr,mmse_bit_errors=berr,bits=bits,max_task_error=worst,failed_channels_2pct=bad,independent_channels=192,max_target_absolute_difference=target_diff))
result=dict(status='INDEPENDENT_REPORT_CHECKED_WITH_EXPECTED_QUALITY_FAILURE',groups=rows,bits_audited=sum(x['bits'] for x in rows),all_quality_passed=False,score=None)
(root/'independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
