"""Reference-only quantization diagnostics; does not implement an FP16 candidate."""
import argparse,json,platform,subprocess
from pathlib import Path
import numpy as np
from verification.contracts.core import Invalid
from verification.reconstruction.daily import sha
from verification.studies.bri_math import system


def half(z):
    z=np.asarray(z)
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        x=z.real.astype(np.float16).astype(np.float64)+1j*z.imag.astype(np.float16).astype(np.float64)
    if not np.isfinite(x).all():raise Invalid('FP16 representation overflow')
    return x


def error(a,b):
    return np.linalg.norm(a-b,axis=(1,2))/np.maximum(np.linalg.norm(b,axis=(1,2)),1e-300)


def diagnose(H,Y,lam):
    if H.ndim!=3 or Y.ndim!=3 or H.shape[:2]!=Y.shape[:2] or not np.isfinite(H).all() or not np.isfinite(Y).all() or not np.isfinite(lam) or lam<=0:raise Invalid('invalid quantization diagnostic input')
    N,M,U=H.shape;I=np.broadcast_to(np.eye(U),(N,U,U));Ht=H.conj().transpose(0,2,1);A=Ht@H+lam*I;inv=np.linalg.solve(A,I);truth=np.linalg.solve(A,Ht@Y)
    hq,yq=half(H),half(Y);lambdaq=float(np.float16(lam));hqT=hq.conj().transpose(0,2,1)
    A_input=hqT@hq+lambdaq*I
    input_inverse=np.linalg.solve(A_input,I);input_output=np.linalg.solve(A_input,hqT@yq)
    A_stored,_,_=system(H,lam,2,'half_storage',hermitian_storage=True)
    gram_inverse=np.linalg.solve(A_stored.astype(np.complex128),I)
    return dict(regularization=lam,channels=N,reference_only=True,candidate_verified=False,
        output_inverse_fp16_storage_error_max=float(error(half(inv),inv).max()),
        input_fp16_then_fp64_oracle_inverse_error_max=float(error(input_inverse,inv).max()),
        input_fp16_then_fp64_oracle_task_error_max=float(error(input_output,truth).max()),
        stored_gram_fp16_then_fp64_oracle_inverse_error_max=float(error(gram_inverse,inv).max()),
        note='FP64 solve used only to diagnose information loss; all oracle paths are forbidden as candidate implementations; no cycles or acceptance')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True,type=Path);ap.add_argument('--study-sha256',required=True);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if a.output.exists():raise Invalid('refusing existing diagnostic output')
    if sha(a.study)!=a.study_sha256:raise Invalid('approved original result changed')
    r=json.loads(a.study.read_text());H=[];inputs=[]
    for v in r['channel_artifacts']:
        p=Path(v['path'])
        if sha(p)!=v['sha256']:raise Invalid('original channel changed')
        with np.load(p,allow_pickle=False) as n:H.append(n['H'])
        inputs.append(v)
    H=np.concatenate(H)
    if H.shape!=(192,64,16):raise Invalid('bounded original dataset required')
    payload=a.study.parent/'payload.npz'
    with np.load(payload,allow_pickle=False) as n:s=n['symbols'];noise=n['base_noise']
    rows=[diagnose(H,H@s+np.sqrt(lam)*noise,lam) for lam in r['regularizations']]
    repo=Path(__file__).resolve().parents[2]
    files=['verification/studies/fp16_constraints.py','verification/studies/bri_math.py','verification/contracts/core.py','verification/reconstruction/daily.py']
    result=dict(schema='fp16-information-diagnostic-v1',status='REFERENCE_DIAGNOSTIC_COMPLETE',candidate_verified=False,score=None,source_result=dict(path=str(a.study),sha256=a.study_sha256),inputs=inputs,payload=dict(path=str(payload),sha256=sha(payload)),sources={f:sha(repo/f) for f in files},python=platform.python_version(),numpy=np.__version__,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),rows=rows)
    if sha(a.study)!=a.study_sha256:raise Invalid('original result changed during diagnostic')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows))

if __name__=='__main__':main()
