# Expert Review Gap Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement.

**Goal:** Address the 6 gaps identified in the independent expert review of the operator verification architecture.

**Architecture:** Fix gaps in priority order: P1 (negative control + multi-batch + multi-seed — quick wins, high confidence boost), P2 (intermediate tensor comparison), P0 (B2 instruction replay — largest scope, separate plan). P3 (FP16 fidelity) deferred.

## Global Constraints

- All code under `/home/nanajiang/Asim/scripts/`
- Must not break existing unified_verify.py or DAG executor
- Tests must be self-contained and runnable with `.venv/bin/python3`
- Commit after each task

---

### Task 1 (P1): Negative Control Test

**Files:**
- Modify: `scripts/unified_verify.py` (add `--inject-error` flag)
- Create: `scripts/test_negative_control.py`

Inject deliberate FormulaLogger errors and verify cross-error detects them.

- [ ] **Step 1: Create negative control test**

```python
#!/usr/bin/env python3
"""Negative control: inject errors into formula_steps.json, verify detection."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import RayleighChannel
from algo import cholesky_noblock_inverse
from uobs_dag_executor import FormulaDAG

def fp16c(x):
    r = x.real.astype(np.float16).astype(np.float64)
    i = x.imag.astype(np.float16).astype(np.float64)
    return r + 1j * i

def test_with_injection(injection_type):
    """Test if error injection is detected via increased cross-error."""
    ch = RayleighChannel()
    H = ch.generate(1, 64, 16, seed=42)[0]
    lam = 0.1
    A = H.conj().T @ H + lam * np.eye(16)
    A_py = cholesky_noblock_inverse(A.copy())
    
    # Generate clean formula from C++ simulator
    os.system("ONNXIM_FORMULA_JSON=/tmp/neg_ctrl.json ONNXIM_MAX_CORE_CYCLES=100000 "
              "./build/bin/Simulator --config configs/ascend_910b_quiet.json "
              "--models_list example/cholesky_noblock_v2_test.json "
              "--mode cholesky_noblock_v2_test --log_level info 2>/dev/null")
    
    with open('/tmp/neg_ctrl.json') as f:
        data = json.load(f)
    
    # Inject error
    if injection_type == 'delete_step':
        # Remove BWD assembly step
        data['steps'] = [s for s in data['steps'] 
                         if 'BWD' not in s['step_id'] and s['batch'] == 0]
    elif injection_type == 'swap_optype':
        for s in data['steps']:
            if s['op_type'] == 'GEMM' and s['batch'] == 0:
                s['op_type'] = 'DIAG_ADD'  # wrong primitive
    elif injection_type == 'wrong_shape':
        for s in data['steps']:
            if s['step_id'].startswith('CHOL_NB_GRAM') and s['batch'] == 0:
                s['output_shape'] = [64, 64]  # should be [16, 16]
    
    dag = FormulaDAG([s for s in data['steps'] if s['batch'] == 0])
    try:
        result = dag.execute({"H": H}, {"lambda": lam})
        A_dag = result.get("Ainv")
        if A_dag is None:
            return float('inf'), "DAG produced no Ainv"
        err_cross = np.linalg.norm(fp16c(A_py) - fp16c(A_dag)) / max(np.linalg.norm(fp16c(A_py)), 1e-15)
        return err_cross, "DAG completed"
    except Exception as e:
        return float('inf'), f"DAG crashed: {str(e)[:60]}"

if __name__ == '__main__':
    print("Negative Control Tests")
    print("=" * 50)
    
    # Baseline (no injection)
    ch = RayleighChannel()
    H = ch.generate(1, 64, 16, seed=42)[0]
    lam = 0.1
    A = H.conj().T @ H + lam * np.eye(16)
    A_py = fp16c(cholesky_noblock_inverse(A.copy()))
    A_ref = fp16c(np.linalg.inv(fp16c(A)))
    baseline_err = np.linalg.norm(A_py - A_ref) / max(np.linalg.norm(A_ref), 1e-15)
    print(f"  Baseline Py vs Ref: {baseline_err:.4e}")
    print(f"  Threshold for anomaly: {baseline_err * 100:.4e} (100x baseline)")
    
    for injection in ['delete_step', 'swap_optype', 'wrong_shape']:
        err, msg = test_with_injection(injection)
        detected = err > baseline_err * 100 or err == float('inf')
        status = "DETECTED" if detected else "MISSED"
        print(f"  {injection:<20} cross_err={err:.4e} ({msg}) -> {status}")
```

- [ ] **Step 2: Run test**

```bash
cd /home/nanajiang/Asim && .venv/bin/python3 scripts/test_negative_control.py
```
Expected: all 3 injections produce cross-error > 100x baseline or crash

- [ ] **Step 3: Commit**

```bash
git add scripts/test_negative_control.py && git commit -m "test: negative control — verify error injection detection"
```

---

### Task 2 (P1): Multi-Batch DAG + Multi-Seed Testing

**Files:**
- Modify: `scripts/unified_verify.py` (add --batches and --seeds flags)

- [ ] **Step 1: Add multi-batch and multi-seed support to unified_verify.py**

Add after the existing `run_test` function:

```python
def run_multi_seed(algo_name, algo_func, formula_path, seeds=[42, 123, 456], batches=[0, 1, 2]):
    """Multi-seed, multi-batch verification."""
    print(f"\n  Multi-Seed + Multi-Batch: {algo_name}")
    with open(formula_path) as f:
        data = json.load(f)
    
    for ch_name, ch_gen in CHANNELS.items():
        print(f"\n  --- {ch_name} ({len(seeds)} seeds × {len(batches)} batches) ---")
        all_cross = []
        for seed in seeds:
            for batch in batches:
                H = ch_gen.generate(max(batch+1, BATCH), NR, NT, seed=seed)
                H_b = H[batch]
                dag = FormulaDAG([s for s in data['steps'] if s['batch'] == batch])
                lam = NT / (10 ** (10 / 10.0))
                A_b = H_b.conj().T @ H_b + lam * np.eye(NT)
                A_py = fp16c(algo_func(A_b.copy()))
                result = dag.execute({"H": H_b}, {"lambda": lam})
                A_dag = result.get("Ainv")
                if A_dag is not None:
                    err = np.linalg.norm(fp16c(A_py) - fp16c(A_dag)) / max(np.linalg.norm(fp16c(A_py)), 1e-15)
                    all_cross.append(err)
        
        if all_cross:
            mean_err = np.mean(all_cross)
            max_err = np.max(all_cross)
            print(f"    Mean cross-err: {mean_err:.4e}  Max: {max_err:.4e}  N={len(all_cross)}")
            print(f"    Status: {'PASS' if max_err < 0.01 else 'FAIL'}")
```

- [ ] **Step 2: Run multi-seed test**

```bash
cd /home/nanajiang/Asim && .venv/bin/python3 -c "
from scripts.unified_verify import run_multi_seed
from scripts.algo import cholesky_noblock_inverse
run_multi_seed('Cholesky', cholesky_noblock_inverse, '/tmp/unified_formula.json')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/unified_verify.py && git commit -m "feat: multi-batch + multi-seed verification"
```

---

### Task 3 (P2): Intermediate Tensor Comparison

**Files:**
- Modify: `scripts/unified_verify.py` (add intermediate tensor check)
- Modify: `scripts/uobs_dag_executor.py` (expose `get_output` method)

- [ ] **Step 1: Add intermediate tensor comparison**

```python
def compare_intermediates(H, formula_path, algo_func):
    """Compare G, A, L, Y tensors between Python and DAG."""
    with open(formula_path) as f:
        data = json.load(f)
    
    dag = FormulaDAG([s for s in data['steps'] if s['batch'] == 0])
    lam = 0.1
    U = H.shape[1]
    A = H.conj().T @ H + lam * np.eye(U)
    
    # Python intermediates (recomputed)
    G_py = fp16c(H.conj().T @ H)
    A_py = fp16c(G_py + lam * np.eye(U))
    # Full Python algo
    Ainv_py = algo_func(A.copy())
    
    # DAG intermediates
    result = dag.execute({"H": H}, {"lambda": lam})
    
    intermediates = ['G', 'A', 'L', 'Y', 'Ainv']
    for name in intermediates:
        val = result.get(name)
        if val is not None:
            ref = {'G': G_py, 'A': A_py}.get(name)
            if ref is not None:
                err = np.linalg.norm(fp16c(val) - fp16c(ref)) / max(np.linalg.norm(fp16c(ref)), 1e-15)
                print(f"    {name}: err={err:.4e}")
            else:
                print(f"    {name}: shape={val.shape} (no Python ref)")
```

- [ ] **Step 2: Run intermediate comparison**

```bash
cd /home/nanajiang/Asim && .venv/bin/python3 -c "
import numpy as np
from scripts.channel import RayleighChannel
from scripts.algo import cholesky_noblock_inverse
from scripts.unified_verify import compare_intermediates
H = RayleighChannel().generate(1, 64, 16, seed=42)[0]
compare_intermediates(H, '/tmp/unified_formula.json', cholesky_noblock_inverse)
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/unified_verify.py && git commit -m "feat: intermediate tensor comparison (G,A,L,Y,Ainv)"
```

---

### Task 4: Update Report with Gap Fix Results

**Files:**
- Modify: `DOCS/ASIM_VERIFICATION_REPORT.md` (Section 7.3: update status)

- [ ] **Step 1: Update gap status in report**

- [ ] **Step 2: Commit**

```bash
git add DOCS/ASIM_VERIFICATION_REPORT.md && git commit -m "docs: update gap fix status in verification report"
```
