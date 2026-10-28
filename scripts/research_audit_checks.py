#!/usr/bin/env python3
"""Bounded diagnostics of legacy verifier defects; success is NOT operator validation."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.algo.ldl_noblock import ldl_noblock_inverse
from scripts.uobs_dag_executor import prim_gemm, prim_ldl_decompose
from scripts.uobs_scorer import AlgorithmIdentity, compute_score, validate_trace_formula_consistency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("refusing to overwrite an existing diagnostic")
    checks = []
    algo = AlgorithmIdentity("newton_schulz")
    trace = {"finish_cycle": 100, "cube_util": 1.8, "avg_parallelism": 0.9}
    score, _, accepted = compute_score(algo, trace, float("nan"), 16)
    checks.append({"id": "legacy_nan_error", "kind": "legacy_function_negative_probe",
                   "input": {"rel_error": "NaN", "trace": trace},
                   "observed": {"accepted": accepted, "score": score},
                   "defect_reproduced": accepted})
    zero = dict(trace, finish_cycle=0)
    score, _, accepted = compute_score(algo, zero, 0.0, 16)
    checks.append({"id": "legacy_zero_cycle", "kind": "legacy_function_negative_probe",
                   "input": {"rel_error": 0.0, "trace": zero},
                   "observed": {"accepted": accepted, "score": score},
                   "defect_reproduced": accepted and score > 100})
    with tempfile.TemporaryDirectory(prefix="trace-negative-", dir=args.output.parent) as directory:
        formula = Path(directory) / "formula.json"
        formula.write_text(json.dumps({"steps": [{"op_type": "GEMM"}]}))
        accepted, reason = validate_trace_formula_consistency(str(formula), str(Path(directory) / "absent.csv"))
    checks.append({"id": "legacy_missing_trace_helper", "kind": "legacy_function_negative_probe",
                   "input": {"formula_gemm_count": 1, "trace_exists": False},
                   "observed": {"accepted": accepted, "reason": reason}, "defect_reproduced": accepted,
                   "limitation": "helper only; top-level scorer may fail earlier while reading trace stats"})

    L = np.array([[1, 0], [0.5 + 0.25j, 1]], dtype=np.complex128)
    D = np.diag([2.0, 5.0])
    A = L @ D @ L.conj().T
    ref = np.linalg.solve(A, np.eye(2))
    y_dag = prim_ldl_decompose(A)
    dag = prim_gemm(y_dag.conj().T, y_dag)
    py = ldl_noblock_inverse(A)
    dag_error = float(np.linalg.norm(dag - ref) / np.linalg.norm(ref))
    py_error = float(np.linalg.norm(py - ref) / np.linalg.norm(ref))
    checks.append({"id": "ldl_scaling_direction", "kind": "actual_python_dag_comparison",
                   "input": {"L_real": L.real.tolist(), "L_imag": L.imag.tolist(), "D": [2.0, 5.0]},
                   "observed": {"dag_relative_error": dag_error, "row_scaled_python_relative_error": py_error,
                                "dag_storage_dtype": str(y_dag.dtype), "reference_dtype": str(ref.dtype)},
                   "defect_reproduced": dag_error > 0.01 and py_error < 1e-12,
                   "limitation": "not C++ instruction replay or full-domain quality validation"})

    a = np.diag([2.0, 4.0])
    x0 = np.diag([0.5, 0.25])
    x = x0.copy()
    for _ in range(2):
        x = x @ (2 * np.eye(2) - a @ x)
    squared = x @ x
    checks.append({"id": "ns_final_square_witness", "kind": "explicit_mathematical_witness",
                   "input": {"A": a.tolist(), "X0": x0.tolist(), "C": (2 * np.eye(2)).tolist(), "K": 2},
                   "observed": {"X": x.tolist(), "X_squared": squared.tolist(),
                                "X_residual": float(np.linalg.norm(a @ x - np.eye(2))),
                                "X_squared_residual": float(np.linalg.norm(a @ squared - np.eye(2)))},
                   "witness_confirmed": bool(np.allclose(a @ x, np.eye(2)) and not np.allclose(a @ squared, np.eye(2))),
                   "limitation": "assumes explicit subtraction and valid inputs; current C++ binding/SUB semantics remain unsupported"})
    expected = all(c.get("defect_reproduced", c.get("witness_confirmed", False)) for c in checks)
    names = ["scripts/research_audit_checks.py", "scripts/uobs_scorer.py", "scripts/uobs_dag_executor.py",
             "scripts/algo/ldl_noblock.py", "scripts/reference_inverse_registry.py"]
    result = {"schema_version": "research-diagnostic-v1", "status": "AUDIT_EXPECTATIONS_CONFIRMED" if expected else "UNEXPECTED_RESULT",
              "operator_validation_status": "NOT_PERFORMED", "candidate_score": None,
              "audit_base_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "command": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/research_audit_checks.py --output " + str(args.output),
              "python": sys.version, "numpy": np.__version__, "threads": {k: os.getenv(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
              "sources": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in names], "checks": checks}
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "operator_validation_status": "NOT_PERFORMED", "output": str(args.output)}))
    return 0 if expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
