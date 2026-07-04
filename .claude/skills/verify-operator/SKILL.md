# Operator Numerical Verification (/verify-operator)

Standardized DAG-based numerical verification for operators.

## Usage

```
/verify-operator <operator_name>
```

## Process

### Phase 1: Check FormulaLogger DAG Chain
- Verify `set_algorithm()` is called
- Verify `emit_step()` calls form a complete chain: H→G→A→[decomp]→Ainv
- Check that output names are consistent (each step's output = next step's input)
- For missing steps, add emit_step declarations

### Phase 2: Generate formula_steps.json
- Build and run C++ simulator with `ONNXIM_FORMULA_JSON` env var
- Verify formula_steps.json is produced and non-empty

### Phase 3: Run DAG Verification
```bash
.venv/bin/python3 -c "
from scripts.unified_verify import run_test
from scripts.algo import <algo_func>
run_test('<op_name>', <algo_func>, '<formula_path>')
"
```

### Phase 4: Check Results
- Cross-error (Py vs DAG) must be < 0.01 for PASS
- If FAIL: check FormulaLogger declarations for missing steps
- Run `scripts/trace_audit.py` to verify GEMM coverage is 100%

### Phase 5: Pipeline Integration
- Update `orchestrator/pipeline.json` if operator has verified mode
- Record result in `DOCS/operators/<op>.md` verification section

## Per-Operator DAG Chain Requirements

### Cholesky NoBlock / Block
```
GRAM: GEMM   H^H, H → G
REG:  DIAG_ADD G, λI → A
POTRF_j: CHOLESKY A → L  (per column)
FWD_SOLVE: TRSM L → Y
BWD: GEMM  Y^H, Y → Ainv
```

### LDL NoBlock / Block
```
GRAM: GEMM   H^H, H → G
REG:  DIAG_ADD G, λI → A
D_UPDATE_j: DIAG_INV A → D  (per column)
L_UPDATE_i_j: TRSM A, D → L  (per off-diagonal)
FWD_SOLVE: TRSM L → Y
SQRT_SCALE: SCALE D → Y (sqrt(Dinv) weighting)
BWD: GEMM  Y^H, Y → Ainv
```

### Newton-Schulz
```
GEMM_T_k: GEMM A, X → T  (K times)
RESIDUAL_k: MATRIX_SUB 2I, T → R
GEMM_X_k: GEMM X, R → X
FINAL: GEMM X, X → Ainv
```

### Block-Richardson
```
GRAM: GEMM H^H, H → A
REG: DIAG_ADD A, λI → A
PRECOND: MATRIX_INV_2x2 A → B  (block-diagonal)
BY_l: GEMM B, Y → BY  (L times)
RESIDUAL_l: MATRIX_SUB I, BY → R
UPDATE_l: MATRIX_ADD Y, R → Y
FINAL: GEMM Y, H → W, GEMM W, Y_in → X_hat
```
