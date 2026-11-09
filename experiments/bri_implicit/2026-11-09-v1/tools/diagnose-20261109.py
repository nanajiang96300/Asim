from pathlib import Path
import numpy as np,json
root=Path('/home/nanajiang/Asim/results/bri_implicit/2026-11-09-v1');p=root/'holdout/lambda0.01-H56002.npz'
with np.load(p,allow_pickle=False) as a:H=a['H'];err=a['task_errors'];Y=a['Y']
q=lambda z:np.asarray(z).astype(np.float16)
op=lambda f,*a:q(f(*(x.astype(np.float32) for x in a)))
n=32;u=16;l=q([[.01]]);one=q([[1.]]);two=q([[2.]]);c=op(np.divide,op(np.subtract,q([[8.]]),q([[.01/4]])),two);d=op(np.divide,op(np.add,q([[8.]]),q([[.01/4]])),two);rho=op(np.divide,c,d)
coeff=[]
for k in range(76):
 if k==0:alpha=op(np.divide,one,d);beta=q([[0.]])
 else:
  den=op(np.subtract,op(np.multiply,two,d),op(np.multiply,c,rho));rr=op(np.divide,c,den);alpha=op(np.divide,two,den);beta=op(np.multiply,rr,rho);rho=rr
 beta=op(np.multiply,beta,q([[.9]]));coeff.append([float(alpha[0,0]),float(beta[0,0])])
rows=[]
for i,h in enumerate(H):
 hh=q(np.block([[h.real,-h.imag],[h.imag,h.real]]));P=[]
 for j in range(u):
  energy=np.float32(0)
  for k in range(128):energy=np.add(energy,np.multiply(np.float32(hh[k,j]),np.float32(hh[k,j]),dtype=np.float32),dtype=np.float32)
  diag=op(np.add,q([[energy]]),l);P.append(float(op(np.divide,one,diag)[0,0]))
 pp=np.sqrt(np.tile(P,2));A=hh.astype(float).T@hh.astype(float)+float(l[0,0])*np.eye(n)
 eig=np.linalg.eigvalsh(pp[:,None]*A*pp[None,:]);t=float(eig[-1]);poly0=1.;poly1=1.
 for alpha,beta in coeff:
  new=(1+beta-alpha*t)*poly1-beta*poly0;poly0,poly1=poly1,new
 alpha,beta=coeff[-1];roots=np.roots([1,-(1+beta-alpha*t),beta])
 rows.append(dict(batch=i,task_error=float(err[i]),eigen_min=float(eig[0]),eigen_max=t,inside_nominal_bounds=bool(eig[0]>=.0025 and t<=8),last_step_stability_upper=2*(1+beta)/alpha,last_step_mode_radius=float(max(abs(roots))),scalar_error_polynomial_at_max_eigenvalue=float(poly1)))
worst=max(rows,key=lambda r:r['task_error'])
result=dict(status='DIAGNOSTIC_ONLY_NO_RETUNING',source=str(p),failed_channels=int(np.sum(err>.02)),channels=96,worst=worst,coefficient_last=coeff[-1],rows=rows,
 note='FP64 spectrum of system with FP16 H/lambda/P, independent scalar polynomial; not a new accepted algorithm or hardware proof. Last-step roots are diagnostic, finite-varying recurrence evaluated separately.')
(root/'stability_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
