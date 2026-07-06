"""Base utilities for per-operator verification scripts."""
import json, numpy as np, sys, os

def fp16(x):
    """Quantize to FP16 (real/imag separately)."""
    r = x.real.astype(np.float16).astype(np.float64)
    i = x.imag.astype(np.float16).astype(np.float64)
    return r + 1j * i

def load_dag(formula_path):
    """Load FormulaDAG from formula_steps.json."""
    from uobs_dag_executor import FormulaDAG
    with open(formula_path) as f:
        data = json.load(f)
    steps = [s for s in data['steps'] if s['batch'] == 0]
    return FormulaDAG(steps), data

def verify_via_dag(dag, H, lam=0.1):
    """Execute DAG with given H, return Ainv from DAG output."""
    result = dag.execute({"H": H}, {"lambda": lam})
    return result.get("Ainv")

def compute_error(A_dag, A_ref):
    """Relative Frobenius error between DAG output and reference."""
    if A_dag is None:
        return float('inf')
    return float(np.linalg.norm(fp16(A_dag) - A_ref) / max(np.linalg.norm(A_ref), 1e-15))

def run_multi_seed(verify_fn, seeds=(42, 123, 456)):
    """Run verification across multiple seeds, return max error."""
    errors = []
    for seed in seeds:
        r = verify_fn(seed=seed)
        errors.append(r['error'])
    return max(errors), errors
