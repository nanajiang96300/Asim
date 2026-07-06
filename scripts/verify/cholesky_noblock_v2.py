"""Cholesky NoBlock v2 — connects C++ formula_steps.json to DAG verification."""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify._base import fp16, load_dag, verify_via_dag, compute_error, run_multi_seed
from uobs_dag_executor import prim_gemm, prim_diag_add, prim_cholesky, prim_trsm

# Threshold: 0.01 — FP16 Cholesky has ~0.1% numerical error on well-conditioned matrices
THRESHOLD = 0.01

def verify(formula_path, seed=42):
    dag, data = load_dag(formula_path)
    meta = data.get("_metadata", {})
    U = meta.get("matrix_dim", 16); M = 64
    np.random.seed(seed)
    H = (np.random.randn(M, U) + 1j * np.random.randn(M, U)) / np.sqrt(2)
    
    # Path A: DAG execution (from C++ FormulaLogger)
    A_dag = verify_via_dag(dag, H)
    
    # Path B: Python reference (independent implementation)
    A_reg = prim_diag_add(prim_gemm(H.conj().T, H), 0.1)
    Y = prim_trsm(prim_cholesky(A_reg))
    A_ref = fp16(prim_gemm(Y.conj().T, Y))
    
    err_dag = compute_error(A_dag, A_ref)
    return {"error": err_dag, "status": "PASS" if err_dag < THRESHOLD else "FAIL",
            "steps": len(data["steps"]), "seed": seed}

if __name__ == "__main__":
    r = verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json")
    max_e, _ = run_multi_seed(lambda seed: verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json", s))
    print(f"Cholesky NoBlock: err={r['error']:.4e} max_err={max_e:.4e} {r['status']}")
