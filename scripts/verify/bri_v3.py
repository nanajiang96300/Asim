"""BRI v3 — DAG self-consistency check.

BRI hardware computes X_hat = Y_{L-1} @ H @ Yin (MMSE estimate).
BRI DAG records a simplified chain: Y_{l+1} = Y_l @ (2I - B @ Y_l), then
BRI_FINAL emits GEMM(Y_{L-1}, Y_{L-1}) as Ainv.

This script performs a DAG self-consistency check: it replays the same
primitives in Python and compares against DAG output. This verifies the
DAG executor correctly replays the formula steps, not mathematical
correctness of the BRI algorithm vs A^{-1}.

C2 fix: was comparing algebraically different quantities (DAG Y@Y vs B^{-1}).
Now compares DAG replay vs same-primitive Python reference (self-consistency)."""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify._base import fp16, load_dag, compute_error, run_multi_seed
from uobs_dag_executor import prim_gemm, prim_diag_add, prim_bri_precond, prim_matrix_sub, prim_matrix_add

# Threshold: 0.01 — self-consistency check should be near-perfect
# (DAG replay uses same primitives as reference, FP16 error only)
THRESHOLD = 0.01

def verify(formula_path, seed=42):
    dag, data = load_dag(formula_path)
    meta = data.get("_metadata", {})
    U = meta.get("matrix_dim", 16); L = meta.get("layers", 8); M = 64
    np.random.seed(seed)
    H = (np.random.randn(M, U) + 1j * np.random.randn(M, U)) / np.sqrt(2)

    # Path A: DAG execution from C++ formula_steps.json
    result = dag.execute({"H": H}, {"lambda": 0.1})
    A_dag = result.get("Ainv")

    # Path B: Same-primitive Python reference (self-consistency check)
    # Replays the exact same primitives the DAG uses
    G = prim_gemm(H.conj().T, H)
    A_reg = prim_diag_add(G, 0.1)
    Bmat = prim_bri_precond(A_reg)
    Y = Bmat.copy()
    I = np.eye(U, dtype=np.complex128)
    for l in range(L):
        BY = prim_gemm(Bmat, Y)
        R = prim_matrix_sub(I, BY)
        Y = prim_matrix_add(Y, R)
    A_ref = fp16(prim_gemm(Y, Y))  # matches BRI_FINAL: GEMM(Y_{L-1}, Y_{L-1})

    if A_dag is None:
        return {"error": float('inf'), "status": "FAIL",
                "steps": len(data["steps"]), "seed": seed,
                "note": "DAG chain incomplete — no Ainv output"}
    err_dag = compute_error(A_dag, A_ref)
    return {"error": err_dag, "status": "PASS" if err_dag < THRESHOLD else "FAIL",
            "steps": len(data["steps"]), "seed": seed,
            "note": "DAG self-consistency (not A^{-1} correctness)"}

if __name__ == "__main__":
    r = verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json")
    max_e, _ = run_multi_seed(lambda seed: verify(sys.argv[1] if len(sys.argv) > 1 else "/tmp/formula.json", seed))
    print(f"BRI: err={r['error']:.4e} max_err={max_e:.4e} {r['status']}")
