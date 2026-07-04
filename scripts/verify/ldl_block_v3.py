"""LDL Block v3 — per-operator verification."""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uobs_dag_executor import prim_gemm, prim_diag_add, prim_ldl_decompose, _cplx_fp16

def fp(x):
    rp=x.real.astype(np.float16).astype(np.float64)
    ip=x.imag.astype(np.float16).astype(np.float64)
    return rp+1j*ip

def verify(formula_path):
    with open(formula_path) as f: data = json.load(f)
    meta = data.get("_metadata", {})
    U = meta.get("matrix_dim", 16)
    M = 64
    
    np.random.seed(42)
    H = (np.random.randn(M, U) + 1j * np.random.randn(M, U)) / np.sqrt(2)
    A_mat = H.conj().T @ H + 0.1 * np.eye(U)
    A_ref = fp(np.linalg.inv(fp(A_mat)))
    
    G = prim_gemm(H.conj().T, H)
    A_reg = prim_diag_add(G, 0.1)
    Y = prim_ldl_decompose(A_reg)
    Ainv = prim_gemm(Y.conj().T, Y)
    
    err = np.linalg.norm(fp(Ainv) - A_ref) / max(np.linalg.norm(A_ref), 1e-15)
    return {"error": float(err), "status": "PASS" if err < 0.10 else "FAIL", "steps": len(data["steps"])}

if __name__ == "__main__":
    r = verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json")
    print(f"LDL Block: err={r['error']:.4e} {r['status']}")
