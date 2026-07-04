"""Newton-Schulz v3 — per-operator verification (core primitives only)."""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uobs_dag_executor import prim_gemm, prim_matrix_sub, _cplx_fp16

def fp(x):
    rp=x.real.astype(np.float16).astype(np.float64)
    ip=x.imag.astype(np.float16).astype(np.float64)
    return rp+1j*ip

def verify(formula_path):
    with open(formula_path) as f: data = json.load(f)
    meta = data.get("_metadata", {})
    N = meta.get("matrix_dim", 32)
    K = meta.get("layers", 8)
    
    np.random.seed(42)
    A_mat = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2*N)
    A = A_mat.conj().T @ A_mat + 0.1 * np.eye(N)
    A_ref = fp(np.linalg.inv(fp(A)))
    
    # NS iteration: X_{k+1} = X_k @ (2I - A @ X_k)
    X = np.eye(N, dtype=np.complex128) / 10.0
    I2 = 2.0 * np.eye(N, dtype=np.complex128)
    for k in range(K):
        T = prim_gemm(A, X)
        R = prim_matrix_sub(I2, T)
        X = prim_gemm(X, R)
    
    err = np.linalg.norm(fp(X) - A_ref) / max(np.linalg.norm(A_ref), 1e-15)
    return {"error": float(err), "status": "PASS" if err < 0.10 else "FAIL", "steps": len(data["steps"])}

if __name__ == "__main__":
    r = verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json")
    print(f"Newton-Schulz: err={r['error']:.4e} {r['status']}")
