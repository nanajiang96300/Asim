#!/usr/bin/env python3
"""Negative control: inject errors into formula_steps.json, verify detection."""
import sys, os, json, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import RayleighChannel
from algo import cholesky_noblock_inverse
from uobs_dag_executor import FormulaDAG

def fp16c(x):
    r = x.real.astype(np.float16).astype(np.float64)
    i = x.imag.astype(np.float16).astype(np.float64)
    return r + 1j * i

def test_with_injection(injection_type, H, lam):
    """Test if error injection is detected via increased cross-error."""
    # Generate fresh formula_steps.json from C++ simulator
    formula_path = f"/tmp/neg_ctrl_{injection_type}.json"
    subprocess.run([
        "./build/bin/Simulator",
        "--config", "configs/ascend_910b_quiet.json",
        "--models_list", "example/cholesky_noblock_v2_test.json",
        "--mode", "cholesky_noblock_v2_test",
        "--log_level", "info"
    ], env={**os.environ, "ONNXIM_FORMULA_JSON": formula_path, "ONNXIM_MAX_CORE_CYCLES": "100000"},
       capture_output=True)

    if not os.path.exists(formula_path):
        return float('inf'), "Simulator did not produce formula JSON"

    with open(formula_path) as f:
        data = json.load(f)

    # Inject error
    if injection_type == 'delete_step':
        data['steps'] = [s for s in data['steps'] if 'BWD' not in s['step_id']]
    elif injection_type == 'swap_optype':
        for s in data['steps']:
            if s['op_type'] == 'CHOLESKY' and s['batch'] == 0:
                s['op_type'] = 'DIAG_ADD'
    elif injection_type == 'wrong_shape':
        for s in data['steps']:
            if s['step_id'].startswith('CHOL_NB_GRAM') and s['batch'] == 0:
                s['output_name'] = 'G_wrong'

    U = H.shape[1]
    A = H.conj().T @ H + lam * np.eye(U)
    A_py = fp16c(cholesky_noblock_inverse(A.copy()))

    steps_b0 = [s for s in data['steps'] if s['batch'] == 0]
    try:
        dag = FormulaDAG(steps_b0)
        result = dag.execute({"H": H}, {"lambda": lam})
        A_dag = result.get("Ainv")
        if A_dag is None:
            return float('inf'), "DAG produced no Ainv"
        err_cross = np.linalg.norm(fp16c(A_py) - fp16c(A_dag)) / max(np.linalg.norm(fp16c(A_py)), 1e-15)
        return err_cross, "DAG completed"
    except Exception as e:
        return float('inf'), f"DAG crashed: {str(e)[:80]}"

if __name__ == '__main__':
    print("Negative Control Tests")
    print("=" * 50)

    ch = RayleighChannel()
    H = ch.generate(1, 64, 16, seed=42)[0]
    lam = 0.1

    # Compute baseline error
    A = H.conj().T @ H + lam * np.eye(16)
    A_py = fp16c(cholesky_noblock_inverse(A.copy()))
    A_ref = fp16c(np.linalg.inv(fp16c(A)))
    baseline_err = np.linalg.norm(A_py - A_ref) / max(np.linalg.norm(A_ref), 1e-15)
    threshold = baseline_err * 100
    print(f"  Baseline Py vs Ref: {baseline_err:.4e}")
    print(f"  Detection threshold: {threshold:.4e} (100x baseline)")
    print()

    all_detected = True
    for injection in ['delete_step', 'swap_optype', 'wrong_shape']:
        err, msg = test_with_injection(injection, H, lam)
        detected = (err > threshold) or (err == float('inf'))
        status = "DETECTED" if detected else "MISSED"
        if not detected:
            all_detected = False
        print(f"  {injection:<20} cross_err={err:.6e} -> {status}  ({msg})")

    print()
    print(f"  Overall: {'ALL DETECTED' if all_detected else 'SOME MISSED — verification not sensitive enough'}")
    sys.exit(0 if all_detected else 1)
