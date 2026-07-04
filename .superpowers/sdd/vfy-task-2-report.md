# vfy-task-2 Report: Enable DAG Path in Reference Inverse Registry

## Changes Applied

**File**: `/home/nanajiang/Asim/scripts/reference_inverse_registry.py`

### 1. Added imports (`json`, `os`)

Added `import json` and `import os` to the top-level imports for use by the new DAG path functions.

### 2. Added `_compute_via_dag()` function

New function that loads a formula_steps.json, builds a `FormulaDAG`, and executes it with `{"A": a_mat}` as the initial tensor. Returns the last valid 2D matrix from the execution results (expected to be `Ainv`). Gracefully returns `None` on any failure (missing file, missing leaf inputs, shape mismatch, etc.), allowing the caller to fall back to the registered per-algorithm function.

### 3. Re-enabled DAG-first dispatch in `compute_reference_inverse()`

Replaced the disabled comment block (lines 187-206) with:

```python
    # Try DAG path first if formula JSON is available
    if formula_json_path and os.path.exists(formula_json_path):
        result = _compute_via_dag(formula_json_path, a_mat, cfg)
        if result is not None:
            return result
```

The DAG path is tried first. If it succeeds (returns a valid result), that result is used. Otherwise, execution falls through to the registered per-algorithm function.

### 4. Registered `cholesky_noblock_v2` algorithm

Added `@register("cholesky_noblock_v2")` delegating to `cholesky_formula_inverse()`, so that when the DAG path fails (because the formula JSON needs `H` as leaf input rather than `A`), the fallback correctly handles the `cholesky_noblock_v2` algorithm name.

## Test Results

All 7 registered algorithms pass the reference inverse test:

```
  block_richardson     err=1.99e-03 PASS
  cholesky_block       err=9.04e-03 PASS
  cholesky_noblock     err=9.04e-03 PASS
  cholesky_noblock_v2  err=9.04e-03 PASS
  ldl_block            err=1.95e-03 PASS
  ldl_noblock          err=3.88e-03 PASS
  newton_schulz        err=7.68e-08 PASS
```

The DAG path is attempted first. For existing formula JSONs (which have `H` as leaf input), `_compute_via_dag` returns `None` and the registered function fallback is used. The DAG path will succeed when a formula JSON has `A` as a leaf input (e.g., for algorithms that start from the Gram matrix directly).

## Commit

```
feat: enable DAG path in reference inverse registry
```

Files committed:
- `/home/nanajiang/Asim/scripts/reference_inverse_registry.py` (changes)
