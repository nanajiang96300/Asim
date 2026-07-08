"""LDL Block v3 — algorithm equivalence verification."""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify._base import fp16, load_dag, verify_via_dag, compute_error, run_multi_seed
from uobs_dag_executor import prim_gemm, prim_diag_add, prim_ldl_decompose

THRESHOLD = 0.10
# Threshold: 0.10 — LDL Block has higher FP16 error than Cholesky due to D-factor ops

def verify(formula_path, seed=42):
    dag, data = load_dag(formula_path)
    meta = data.get("_metadata", {})
    U = meta.get("matrix_dim", 16); M = 64
    np.random.seed(seed)
    H = (np.random.randn(M, U) + 1j * np.random.randn(M, U)) / np.sqrt(2)
    
    A_dag = verify_via_dag(dag, H)
    A_reg = prim_diag_add(prim_gemm(H.conj().T, H), 0.1)
    Y = prim_ldl_decompose(A_reg)
    A_ref = fp16(prim_gemm(Y.conj().T, Y))
    
    err_dag = compute_error(A_dag, A_ref)
    return {"error": err_dag, "status": "PASS" if err_dag < THRESHOLD else "FAIL",
            "steps": len(data["steps"]), "seed": seed, "note": "algorithm-equivalence"}

if __name__ == "__main__":
    r = verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json")
    max_e, _ = run_multi_seed(lambda seed: verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json", seed))
    print(f"LDL Block: err={r['error']:.4e} max_err={max_e:.4e} {r['status']}")
