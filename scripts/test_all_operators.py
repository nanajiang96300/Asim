#!/usr/bin/env python3
"""Full operator verification test — generates synthetic formula_steps.json
for all 8 operator variants, runs per-operator verify scripts, collects errors.

Usage: .venv/bin/python scripts/test_all_operators.py
"""

import json, numpy as np, sys, os, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify._base import fp16, run_multi_seed

# ── Test matrix configurations ─────────────────────────────────────────────
OPERATORS = {
    # (operator_name, verify_module, U, M, K_or_L, block_size, method)
    "Cholesky NoBlock v2": {
        "algo": "cholesky_noblock_v2", "U": 16, "M": 64, "K": 0, "B": 1,
        "steps": [
            ("GRAM", "GEMM", ["H^H", "H"], "G", [[64,16],[16,64]], [16,16]),
            ("REG", "DIAG_ADD", ["G", "lambda*I"], "A", [[16,16],[16,16]], [16,16]),
            ("POTRF", "CHOLESKY", ["A"], "L", [[16,16]], [16,16]),
            ("FWD_SOLVE", "TRSM", ["L"], "Y", [[16,16]], [16,16]),
            ("BWD_ASSEMBLE", "GEMM", ["Y^H", "Y"], "Ainv", [[16,16],[16,16]], [16,16]),
        ],
        "initial": {"H": (64, 16)},
        "threshold": 0.01,
    },
    "Cholesky NoBlock Merge": {
        "algo": "cholesky_noblock_merge", "U": 16, "M": 64, "K": 0, "B": 1,
        "steps": [
            ("GRAM", "GEMM", ["H^H", "H"], "G", [[64,16],[16,64]], [16,16]),
            ("REG", "DIAG_ADD", ["G", "lambda*I"], "A", [[16,16],[16,16]], [16,16]),
            ("POTRF", "CHOLESKY", ["A"], "L", [[16,16]], [16,16]),
            ("FWD_SOLVE", "TRSM", ["L"], "Y", [[16,16]], [16,16]),
            ("BWD_ASSEMBLE", "GEMM", ["Y^H", "Y"], "Ainv", [[16,16],[16,16]], [16,16]),
        ],
        "initial": {"H": (64, 16)},
        "threshold": 0.01,
    },
    "Cholesky Block v3": {
        "algo": "cholesky_block_v3", "U": 16, "M": 64, "K": 0, "B": 2,
        "steps": [
            ("GRAM", "GEMM", ["H^H", "H"], "G", [[64,16],[16,64]], [16,16]),
            ("REG", "DIAG_ADD", ["G", "lambda*I"], "A", [[16,16],[16,16]], [16,16]),
            ("POTRF", "CHOLESKY", ["A"], "L", [[16,16]], [16,16]),
            ("FWD_SOLVE", "TRSM", ["L"], "Y", [[16,16]], [16,16]),
            ("BWD_ASSEMBLE", "GEMM", ["Y^H", "Y"], "Ainv", [[16,16],[16,16]], [16,16]),
        ],
        "initial": {"H": (64, 16)},
        "threshold": 0.01,
    },
    "LDL NoBlock v2": {
        "algo": "ldl_noblock_v2", "U": 16, "M": 64, "K": 0, "B": 1,
        "steps": [
            ("GRAM", "GEMM", ["H^H", "H"], "G", [[64,16],[16,64]], [16,16]),
            ("REG", "DIAG_ADD", ["G", "lambda*I"], "A", [[16,16],[16,16]], [16,16]),
            ("DECOMPOSE", "LDL_DECOMPOSE", ["A"], "Y", [[16,16]], [16,16]),
            ("BWD_ASSEMBLE", "GEMM", ["Y^H", "Y"], "Ainv", [[16,16],[16,16]], [16,16]),
        ],
        "initial": {"H": (64, 16)},
        "threshold": 0.10,
    },
    "LDL Block v3": {
        "algo": "ldl_block_v3", "U": 16, "M": 64, "K": 0, "B": 2,
        "steps": [
            ("GRAM", "GEMM", ["H^H", "H"], "G", [[64,16],[16,64]], [16,16]),
            ("REG", "DIAG_ADD", ["G", "lambda*I"], "A", [[16,16],[16,16]], [16,16]),
            ("DECOMPOSE", "LDL_DECOMPOSE", ["A"], "Y", [[16,16]], [16,16]),
            ("BWD_ASSEMBLE", "GEMM", ["Y^H", "Y"], "Ainv", [[16,16],[16,16]], [16,16]),
        ],
        "initial": {"H": (64, 16)},
        "threshold": 0.10,
    },
    "Newton-Schulz v3": {
        "algo": "newton_schulz_v3", "U": 32, "M": 32, "K": 8, "B": 0,
        "steps": None,  # built dynamically below
        "initial": {"A": (32, 32), "X": (32, 32)},
        "threshold": 0.10,
        "ns_iterations": 8,
    },
    "Block-Richardson v3": {
        "algo": "block_richardson_v3", "U": 16, "M": 64, "K": 8, "B": 2,
        "steps": None,  # built dynamically below
        "initial": {"H": (64, 16)},
        "threshold": 0.01,  # self-consistency check
        "bri_iterations": 8,
    },
}


def build_ns_steps(N, K):
    """Build Newton-Schulz iteration steps."""
    steps = []
    for k in range(K):
        x_in = "X" if k == 0 else f"X_{k-1}"
        steps.append((f"T_{k}", "GEMM", ["A", x_in], f"T_{k}", [[N,N],[N,N]], [N,N]))
        steps.append((f"R_{k}", "MATRIX_SUB", ["2I", f"T_{k}"], f"R_{k}", [[N,N],[N,N]], [N,N]))
        steps.append((f"X_{k}", "GEMM", [x_in, f"R_{k}"], f"X_{k}", [[N,N],[N,N]], [N,N]))
    x_final = f"X_{K-1}"
    steps.append(("BWD_ASSEMBLE", "GEMM", [x_final, x_final], "Ainv", [[N,N],[N,N]], [N,N]))
    return steps


def build_bri_steps(U, L):
    """Build Block-Richardson iteration steps."""
    steps = [
        ("GRAM", "GEMM", ["H^H", "H"], "G", [[64,U],[U,64]], [U,U]),
        ("REG", "DIAG_ADD", ["G", "lambda*I"], "A", [[U,U],[U,U]], [U,U]),
        ("PRECOND", "BRI_PRECOND", ["A"], "B", [[U,U]], [U,U]),
    ]
    for l in range(L):
        y_in = "I" if l == 0 else f"Y_{l-1}"
        steps.append((f"BY_{l}", "GEMM", ["B", y_in], f"BY_{l}", [[U,U],[U,U]], [U,U]))
        steps.append((f"R_{l}", "MATRIX_SUB", ["I", f"BY_{l}"], f"R_{l}", [[U,U],[U,U]], [U,U]))
        steps.append((f"Y_{l}", "MATRIX_ADD", [y_in, f"R_{l}"], f"Y_{l}", [[U,U],[U,U]], [U,U]))
    y_final = f"Y_{L-1}"
    steps.append(("FINAL", "GEMM", [y_final, y_final], "Ainv", [[U,U],[U,U]], [U,U]))
    return steps


def make_formula_json(op_name, op_cfg, seed):
    """Generate synthetic formula_steps.json matching operator's C++ DAG chain."""
    np.random.seed(seed)
    U = op_cfg["U"]; M = op_cfg["M"]; K = op_cfg.get("K", 0); B = op_cfg.get("B", 1)

    if op_name == "Newton-Schulz v3":
        steps = build_ns_steps(U, op_cfg.get("ns_iterations", 8))
    elif op_name == "Block-Richardson v3":
        steps = build_bri_steps(U, op_cfg.get("bri_iterations", 8))
    else:
        steps = op_cfg["steps"]

    step_dicts = []
    for i, s in enumerate(steps):
        step_dicts.append({
            "step_id": f"{op_cfg['algo']}_{s[0]}",
            "op_type": s[1],
            "input_names": s[2],
            "output_name": s[3],
            "input_shapes": s[4] if len(s) > 4 else [[U,U]],
            "output_shape": s[5] if len(s) > 5 else [U,U],
            "batch": 0,
            "relation_id": f"{op_cfg['algo']}_{s[0]}",
        })

    return {
        "_metadata": {
            "algorithm": op_cfg["algo"],
            "block_size": B,
            "layers": K,
            "matrix_dim": U,
        },
        "steps": step_dicts,
    }


def compute_true_inverse(init, op_name, op_cfg):
    """Compute ground truth A^{-1} via numpy/scipy (high-precision reference)."""
    U = op_cfg["U"]
    if "Cholesky" in op_name or "LDL" in op_name:
        H = init["H"]
        A = H.conj().T @ H + 0.1 * np.eye(U)
        return np.linalg.inv(A)
    elif "Newton-Schulz" in op_name:
        A = init["A"]
        return np.linalg.inv(A)  # A is already the Gram matrix
    elif "Block-Richardson" in op_name:
        H = init["H"]
        A = H.conj().T @ H + 0.1 * np.eye(U)
        return np.linalg.inv(A)
    return None


def verify_operator(op_name, op_cfg, seed):
    """Run full DAG verification for one operator."""
    from uobs_dag_executor import FormulaDAG, prim_gemm, prim_diag_add, prim_cholesky, prim_trsm
    from uobs_dag_executor import prim_ldl_decompose, prim_bri_precond, prim_matrix_sub, prim_matrix_add
    from verify._base import fp16, compute_error

    U = op_cfg["U"]; M = op_cfg["M"]; K = op_cfg.get("K", 0)

    # Generate synthetic formula JSON
    data = make_formula_json(op_name, op_cfg, seed)
    steps = data["steps"]
    dag = FormulaDAG(steps)

    # Build initial tensors
    init = {}
    for name, shape in op_cfg.get("initial", {}).items():
        if name == "A":
            # Generate positive definite Gram-like matrix for NS
            H_raw = (np.random.randn(*shape) + 1j * np.random.randn(*shape)) / np.sqrt(2*max(shape))
            init[name] = H_raw.conj().T @ H_raw + 0.1 * np.eye(shape[0])
        elif name == "X":
            init[name] = np.eye(shape[0], dtype=np.complex128) * 0.1
        else:
            init[name] = (np.random.randn(*shape) + 1j * np.random.randn(*shape)) / np.sqrt(2)

    # Path A: DAG execution
    result = dag.execute(init, {"lambda": 0.1})
    A_dag = result.get("Ainv")

    # Path B: Python reference (matching each operator's algorithm)
    if "Cholesky" in op_name:
        H = init["H"]
        A_reg = prim_diag_add(prim_gemm(H.conj().T, H), 0.1)
        Y = prim_trsm(prim_cholesky(A_reg))
        A_ref = fp16(prim_gemm(Y.conj().T, Y))
    elif "LDL" in op_name:
        H = init["H"]
        A_reg = prim_diag_add(prim_gemm(H.conj().T, H), 0.1)
        Y = prim_ldl_decompose(A_reg)
        A_ref = fp16(prim_gemm(Y.conj().T, Y))
    elif "Newton-Schulz" in op_name:
        # A is already positive definite (generated as H^H@H + lambda*I in initial tensor)
        A = init["A"]
        X = np.eye(U, dtype=np.complex128) * 0.1  # matches initial X seeding
        for k in range(op_cfg.get("ns_iterations", 8)):
            T = prim_gemm(A, X)
            R = prim_matrix_sub(2.0 * np.eye(U, dtype=np.complex128), T)
            X = prim_gemm(X, R)
        A_ref = fp16(prim_gemm(X, X))  # final GEMM matches BWD_ASSEMBLE
    elif "Block-Richardson" in op_name:
        H = init["H"]
        A_reg = prim_diag_add(prim_gemm(H.conj().T, H), 0.1)
        Bmat = prim_bri_precond(A_reg)
        # BRI DAG starts Y_0 from I (identity), not Bmat.
        # MATRIX_ADD("I", R_0) → Y_0 = I + (I - B@I) = 2I - B
        Y = np.eye(U, dtype=np.complex128)
        I = np.eye(U, dtype=np.complex128)
        for l in range(op_cfg.get("bri_iterations", 8)):
            BY = prim_gemm(Bmat, Y)
            R = prim_matrix_sub(I, BY)
            Y = prim_matrix_add(Y, R)
        A_ref = fp16(prim_gemm(Y, Y))  # self-consistency: matches BRI_FINAL
    else:
        raise ValueError(f"Unknown operator: {op_name}")

    if A_dag is None:
        return float('inf'), float('inf')

    # Self-consistency error: DAG vs same-primitive reference
    err_self = compute_error(A_dag, A_ref)

    # True error: DAG output vs numpy.linalg.inv (high-precision ground truth)
    A_true = compute_true_inverse(init, op_name, op_cfg)
    if A_true is not None:
        err_true = compute_error(fp16(A_dag), A_true)
    else:
        err_true = float('inf')

    return err_self, err_true


def main():
    print("=" * 90)
    print("  Full Operator DAG Verification — All 7 Variants (5 seeds × 2 sizes)")
    print("=" * 90)
    print()

    seeds = (42, 123, 456, 789, 1024)
    sizes = [(16, 64), (32, 128)]  # (U, M) pairs
    results = []

    for op_name, op_cfg in OPERATORS.items():
        for U, M in sizes:
            # Clone config with this size
            cfg = dict(op_cfg)
            cfg["U"] = U; cfg["M"] = M
            # Update initial tensor shapes
            cfg["initial"] = {}
            for name, shape in op_cfg["initial"].items():
                cfg["initial"][name] = (M, U) if name == "H" else (U, U)

            errors_self = []
            errors_true = []
            for seed in seeds:
                es, et = verify_operator(op_name, cfg, seed)
                errors_self.append(es)
                errors_true.append(et)

            max_self = max(errors_self)
            max_true = max(errors_true)
            threshold = op_cfg["threshold"]
            status = "PASS" if max_self < threshold else "FAIL"
            results.append((f"{op_name} (U={U})", errors_self, errors_true, max_self, max_true, threshold, status))

    # Print table
    print(f"  {'Operator':<35s} {'Self-Max':>10s} {'True-Max':>10s} {'Thresh':>8s} {'Status':>6s}")
    print("  " + "-" * 75)
    for op_name, errs_self, errs_true, max_s, max_t, thresh, status in results:
        status_str = "✅ PASS" if status == "PASS" else "❌ FAIL"
        print(f"  {op_name:<35s} {max_s:>10.4e} {max_t:>10.4e} {thresh:>8.4f} {status_str:>6s}")

    print()
    passed = sum(1 for _, _, _, _, _, _, s in results if s == "PASS")
    failed = sum(1 for _, _, _, _, _, _, s in results if s == "FAIL")
    print(f"  Summary: {passed}/{len(results)} PASS, {failed}/{len(results)} FAIL")

    # Method summary
    print()
    print("  Method Summary (U=16, U=32 combined):")
    print(f"  {'Method':<20s} {'Avg Self-Err':>13s} {'Avg True-Err':>13s} {'Pass/Fail':>10s}")
    print("  " + "-" * 60)
    for method, ops in [
        ("Direct (Cholesky)", ["Cholesky NoBlock v2", "Cholesky NoBlock Merge", "Cholesky Block v3"]),
        ("Direct (LDL)", ["LDL NoBlock v2", "LDL Block v3"]),
        ("Iterative", ["Newton-Schulz v3", "Block-Richardson v3"]),
    ]:
        method_results = [r for r in results if any(o in r[0] for o in ops)]
        avg_self = np.mean([r[3] for r in method_results])
        avg_true = np.mean([r[4] for r in method_results if not np.isinf(r[4])])
        all_pass = all(r[6] == "PASS" for r in method_results)
        print(f"  {method:<20s} {avg_self:>13.4e} {avg_true:>13.4e} {'✅' if all_pass else '❌':>10s}")

    # Per-seed detail for largest errors
    print()
    print("  Per-Seed Detail (U=32, worst-case):")
    print(f"  {'Operator':<35s} {'S42':>8s} {'S123':>8s} {'S456':>8s} {'S789':>8s} {'S1024':>8s}")
    print("  " + "-" * 75)
    for op_name, errs_self, errs_true, _, _, _, _ in results:
        if "U=32" in op_name:
            print(f"  {op_name:<35s} {errs_self[0]:>8.4f} {errs_self[1]:>8.4f} {errs_self[2]:>8.4f} {errs_self[3]:>8.4f} {errs_self[4]:>8.4f}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
