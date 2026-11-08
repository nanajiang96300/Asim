"""Independent FP16 holdout audit; explicitly separates inverse and MMSE goals."""
import argparse,json,math
from pathlib import Path
import numpy as np
from verification.studies.fp16_independent import point_solve
from verification.studies.fp16_campaign import sources
from verification.reconstruction.daily import sha


def q(x):return np.asarray(x).astype(np.float16)
def rb(z):return np.concatenate((np.concatenate((z.real,-z.imag),axis=2),np.concatenate((z.imag,z.real),axis=2)),axis=1)
def dot(a,b):
    z=np.zeros((len(a),a.shape[1],b.shape[2]),np.float32)
    for k in range(a.shape[2]):z=np.add(z,np.multiply(a[:,:,k:k+1].astype(np.float32),b[:,k:k+1,:].astype(np.float32),dtype=np.float32),dtype=np.float32)
    return q(z)
def errs(a,b):return np.linalg.norm(a-b,axis=(1,2))/np.maximum(np.linalg.norm(b,axis=(1,2)),1e-300)


def validate(contract,result,out):
    if out.exists():raise ValueError('refusing existing holdout audit')
    c=json.loads(contract.read_text());r=json.loads(result.read_text());repo=Path(__file__).resolve().parents[2]
    if c['sources']!=sources(repo):raise ValueError('evaluator version changed')
    if sha(repo/c['frozen_selection']['path'])!=c['frozen_selection']['sha256']:raise ValueError('selection changed')
    if sha(repo/c['independent_reference']['path'])!=c['independent_reference']['sha256']:raise ValueError('independent reference version')
    if c['cases']!=json.loads((repo/c['frozen_selection']['path']).read_text())['cases']:raise ValueError('holdout goal/K mismatch')
    H=[];bits=[];noise=[]
    for a in c['inputs']:
        if sha(repo/a['path'])!=a['sha256']:raise ValueError('input changed')
        with np.load(repo/a['path'],allow_pickle=False) as z:
            sl=slice(*a['slice']);H.append(z['H'][sl]);bits.append(z['bits'][sl]);noise.append(z['noise'][sl])
    H=np.concatenate(H);bits=np.concatenate(bits);noise=np.concatenate(noise);symbols=((2*bits[...,0].astype(float)-1)+1j*(2*bits[...,1].astype(float)-1))/math.sqrt(2)
    rows=[]
    for i,(s,case) in enumerate(zip(c['cases'],r['cases'])):
        if case['failure'] or not case['rows'] or case['rows'][-1]['K']!=s['evaluation_K']:raise ValueError('incomplete frozen-K evaluation')
        if s['block']!=1 or not s['left'] or not s['compensated'] or s['momentum']!=.9:raise ValueError('unsupported independent reference scope')
        lam=s['lambda'];Y=H@symbols+math.sqrt(lam)*noise;inverse='inverse' in s['kind']
        expected=point_solve(H,Y,lam,s['evaluation_K'],tuple(s['bounds']),s['momentum'],s['compensated'],s.get('restart',0),inverse)
        with np.load(result.parent/f'case{i:02d}.npz',allow_pickle=False) as a:actual=a['final']
        if actual.dtype!=np.float16 or not np.array_equal(actual.view(np.uint16),expected.view(np.uint16)):raise ValueError('independent FP16 output bits differ')
        Ht=H.conj().transpose(0,2,1);A=Ht@H+lam*np.eye(16);truth=np.linalg.solve(A,Ht@Y)
        x=actual[:,:16,:actual.shape[-1]//2].astype(float)+1j*actual[:,16:,:actual.shape[-1]//2].astype(float)
        if inverse:
            inv=np.linalg.solve(A,np.broadcast_to(np.eye(16),A.shape));error=errs(x,inv)
            applied=dot(actual,dot(q(rb(H)).transpose(0,2,1),q(rb(Y))));output=applied[:,:16,:8].astype(float)+1j*applied[:,16:,:8].astype(float)
        else:error=errs(x,truth);output=x
        task=errs(applied.astype(float),rb(truth)) if inverse else errs(output,truth);claimed=case['rows'][-1]['inverse_error_max' if inverse else 'task_error_max']
        if abs(float(error.max())-claimed)>1e-10:raise ValueError('independent metric mismatch')
        if inverse and abs(float(task.max())-case['rows'][-1]['task_error_max'])>1e-10:raise ValueError('independent full-block application metric mismatch')
        candidate_bits=np.stack((output.real>=0,output.imag>=0),axis=-1);mmse_bits=np.stack((truth.real>=0,truth.imag>=0),axis=-1)
        rows.append(dict(goal=s['goal'],regularization=lam,K=s['evaluation_K'],channels=len(H),precision_bit_equal=True,max_goal_error=float(error.max()),goal_passed=bool(error.max()<=.02),inverse_then_apply_task_error_max=float(task.max()) if inverse else None,
                         candidate_bit_errors=int(np.count_nonzero(candidate_bits!=bits)),mmse_bit_errors=int(np.count_nonzero(mmse_bits!=bits)),bits=bits.size,communication_scope='BER diagnostic only; no formal communication confidence gate'))
    if len(rows)!=len(c['cases']):raise ValueError('missing goal coverage')
    all_good=all(a['goal_passed'] for a in rows)
    out.mkdir(parents=True,exist_ok=False);v=dict(status='FP16_PRIMITIVE_HOLDOUT_CHECKED' if all_good else 'FP16_HOLDOUT_GOAL_FAILED',rows=rows,all_frozen_goal_gates_passed=all_good,operator_verified=False,asim_run=False,score=None,
        inverse_then_apply_mmse_verified=all(a['inverse_then_apply_task_error_max'] is None or a['inverse_then_apply_task_error_max']<=.02 for a in rows),auditor_sha256=sha(__file__),contract_sha256=sha(contract),result_sha256=sha(result),note='FP16 tensors and coefficients; frozen FP32 temporary arithmetic model, extra compensation state is FP16; no new FP32 storage or Asim modification')
    (out/'result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v));return all_good


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',required=True,type=Path);ap.add_argument('--contract-sha256',required=True);ap.add_argument('--result',required=True,type=Path);ap.add_argument('--result-sha256',required=True);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if sha(a.contract)!=a.contract_sha256 or sha(a.result)!=a.result_sha256:raise ValueError('approved audit inputs changed')
    ok=validate(a.contract,a.result,a.output)
    if sha(a.contract)!=a.contract_sha256 or sha(a.result)!=a.result_sha256:raise ValueError('audit inputs changed')
    return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
