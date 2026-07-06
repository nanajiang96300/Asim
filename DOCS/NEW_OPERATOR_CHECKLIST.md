# New Operator Checklist

> Unified checklist for operator development — merges content from OPERATOR_DEVELOPMENT_STANDARD_V3.md, DAG_PRIMITIVES_SPEC.md, per-operator-verification-design.md, and verify-operator/SKILL.md.

Use this checklist when developing a NEW operator or making significant changes to an EXISTING one. Check off each item before considering the operator complete.

---

## Phase 1: Design & Planning

- [ ] **Mathematical formula documented** — formula derivation in `DOCS/operators/<NN>_<name>.md` with all symbols defined
- [ ] **Python reference implementation** — standalone Python script that computes the correct result using numpy/scipy, verified against analytical solution
- [ ] **SPAD layout designed** — memory region map showing address ranges for all SPAD/ACCUM buffers (H, Reg, G, A, L, Y, Tmp, D, Dinv, X, etc.)
- [ ] **Primitive selection justified** — which DAG primitives are needed and why (see [DAG_PRIMITIVES_SPEC.md](DAG_PRIMITIVES_SPEC.md)):
  - [ ] Prefer core primitive composition (GEMM, DIAG_ADD, TRSM, MATRIX_SUB, MATRIX_ADD, SCALE)
  - [ ] Only add algorithm primitives (CHOLESKY, LDL_DECOMPOSE, DIAG_INV) if core composition is insufficient
  - [ ] Only add operator-specific primitives (BRI_PRECOND, MATRIX_INV_2x2, SQRT_SCALE) as last resort
- [ ] **Verification threshold estimated** — documented rationale for expected numerical error (derived from FP16 precision analysis or empirical measurement)

---

## Phase 2: C++ Implementation

### File Structure
- [ ] Files follow naming convention: `src/inverse/<algorithm>/<Algorithm><Variant>BaselineOp.{h,cc}`
- [ ] Header includes proper guards, Operation base class inheritance
- [ ] Source file added to `src/CMakeLists.txt` (if not auto-discovered by GLOB_RECURSE)

### SPAD & Addressing
- [ ] `SPAD_BASE` (0x10000000) for SPAD regions, `ACCUM_SPAD_BASE` (0x20000000) for accumulator output
- [ ] All SPAD regions initialized before use: `ADD(dest=X, src={aReg, aReg})` with valid SPAD address
- [ ] MOVIN `base_addr = 0` (not double-added with operand offset)
- [ ] MOVOUT `src_from_accum = true`, `last_inst = true`
- [ ] `make_address()` used for DRAM address calculation

### SCALAR Patterns
- [ ] **Identity synthesis**: Use `SCALAR_DIV(Reg, Reg)` to create 1.0 (NOT hardcoded constants)
- [ ] **Schur complement**: `SCALAR_MUL` → `SCALAR_MUL` (D-factor) → `SCALAR_SUB`
- [ ] **Negation**: `SCALAR_SUB(Reg, val)` where Reg produces 0
- [ ] **Column scaling**: Iterate over column elements applying `SCALAR_MUL` with sqrt(Dinv)
- [ ] All SCALAR instructions use valid SPAD addresses as src/dest

### Instruction Structure
- [ ] `initialize_instructions()` follows the standard phase structure:
  1. MOVIN (load data from DRAM)
  2. GRAM + REG (Gram matrix + regularization)
  3. Decomposition (algorithm-specific)
  4. Forward/Backward solve (if applicable)
  5. Final GEMM_PRELOAD to ACCUM
  6. MOVOUT (store result to DRAM)
- [ ] Barrier placement correct: LOAD (type=1) → REG2DECOMP (type=3) → COL_j (type=4) → FB_c (type=5) → PRE_MOVOUT (type=6)
- [ ] All instructions have unique `.id` strings
- [ ] `tile->batch` used for per-batch DRAM offset calculation

### FormulaLogger Integration
- [ ] `FormulaLogger::instance().set_algorithm("<algorithm>_<variant>", <block_size>, <layers>, <matrix_dim>)` called at start
- [ ] Every mathematical phase has a corresponding `emit_step()` call
- [ ] DAG chain forms a complete path: initial inputs → ... → `"Ainv"`
- [ ] Step naming is consistent: output name of step N matches an input name of step N+1
- [ ] Per-batch steps use `tile->batch` for batch dimension
- [ ] Relation IDs map to actual instruction IDs

---

## Phase 3: Verification Script

- [ ] `scripts/verify/<op_name>.py` created
- [ ] Script imports from `verify/_base.py` utilities: `fp16`, `load_dag`, `compute_error`, `run_multi_seed`
- [ ] Script implements `verify(formula_path, seed=42)` → `{"error", "status", "steps", "seed"}`
- [ ] **Dual-path verification implemented**:
  - **Path A (DAG)**: Load steps from `formula_steps.json`, execute via `FormulaDAG`, get `Ainv`
  - **Path B (Reference)**: Independent Python implementation using DAG primitives library
- [ ] Error reported as `||fp16(A_dag) - A_ref||_F / max(||A_ref||_F, 1e-15)`
- [ ] `THRESHOLD` constant documented with rationale in comment
- [ ] Script supports multi-seed via `run_multi_seed()` (seeds 42, 123, 456)
- [ ] `__main__` block runs both single-seed and multi-seed, reports max error
- [ ] Script handles `A_dag is None` gracefully (reports FAIL if DAG incomplete)

---

## Phase 4: Build & Test

- [ ] `cmake --build build --target Simulator -j$(nproc)` passes with 0 errors
- [ ] `cmake --build build --target Simulator_test -j$(nproc)` passes
- [ ] `./build/bin/Simulator_test` passes all existing GTest tests
- [ ] Simulator runs without abort for a test workload (100K cycle limit)
- [ ] `ONNXIM_TRACE_CSV=results/trace.csv` produces valid trace output
- [ ] `ONNXIM_FORMULA_JSON=/tmp/formula.json` produces valid formula output

---

## Phase 5: Audit & Verification

### Formula-Code Audit
- [ ] `/audit-operator <name>` passes with 0 CRITICAL findings
- [ ] All SCALAR opcodes match mathematical intent (MUL for multiply, SUB for subtract, DIV for invert, SQRT for sqrt)
- [ ] Loop bounds match block decomposition structure
- [ ] No off-by-one errors in triangular solve iteration ranges
- [ ] GEMM dimensions consistent (M×K @ K×N → M×N)

### Numerical Verification
- [ ] `/verify-operator <name>` returns PASS
- [ ] DAG error < THRESHOLD for all 3 seeds (42, 123, 456)
- [ ] Multi-seed max error < THRESHOLD
- [ ] No `A_dag is None` fallback (DAG chain is complete)

---

## Phase 6: Documentation

- [ ] `DOCS/operators/<NN>_<name>.md` created with:
  - [ ] Operator overview and mathematical formula
  - [ ] SPAD layout diagram
  - [ ] Instruction-to-formula mapping table
  - [ ] FormulaLogger DAG chain diagram
  - [ ] Verification results table (error per seed, threshold, status)
  - [ ] DAG primitives used (with reference to DAG_PRIMITIVES_SPEC.md)
- [ ] `DOCS/operators/README.md` updated with new operator entry
- [ ] `orchestrator/operator_registry.json` updated with new operator metadata
- [ ] Benchmark results archived to `results/<operator>/`

---

## Phase 7: Pipeline Integration

- [ ] Operator passes phase 1 (`design_doc`) — design document exists
- [ ] Operator passes phase 2 (`math_derivation`) — formula derivation in doc
- [ ] Operator passes phase 3 (`code_v3_standard`) — v3 standard compliance
- [ ] Operator passes phase 4 (`compile`) — cmake build success
- [ ] Operator passes phase 5 (`runtime`) — simulator runs without abort
- [ ] Operator passes phase 6 (`audit_review`) — /audit-operator passes
- [ ] Operator passes phase 7 (`benchmark`) — benchmark results archived
- [ ] Operator passes phase 8 (`ext_numeric_verify`) — DAG numerical verification passes
- [ ] CI gate (`scripts/ci_gate.sh`) passes for this operator

---

## Quick Reference: DAG Primitives

| Primitive | Signature | Use Case |
|-----------|-----------|----------|
| `GEMM` | `(A, B) → C = A @ B` | Matrix multiply, Gram, backward assembly |
| `DIAG_ADD` | `(A, λ) → A + λI` | Regularization |
| `TRSM` | `(L) → Y = L^{-1}` | Forward substitution (1-input) |
| `MATRIX_SUB` | `(A, B) → A - B` | Residual computation |
| `MATRIX_ADD` | `(A, B) → A + B` | Iterative update |
| `SCALE` | `(A, α) → α·A` | Scalar multiplication |
| `CHOLESKY` | `(A) → L` | Cholesky decomposition |
| `LDL_DECOMPOSE` | `(A) → Y` | Full LDL: L·D·L^H + forward solve + sqrt(Dinv) |
| `DIAG_INV` | `(D) → D^{-1}` | Diagonal/block-diagonal inversion |
| `BRI_PRECOND` | `(A) → B^{-1}` | Block-diagonal preconditioner |
| `SQRT_SCALE` | `(Y, Dinv) → Y·sqrt(Dinv)` | Column scaling by sqrt of diagonal inverse |

## Quick Reference: Common DAG Chains

```
Cholesky NoBlock/Block:
  GRAM(GEMM: H^H,H→G) → REG(DIAG_ADD: G,lambda*I→A) → POTRF(CHOLESKY: A→L)
  → FWD_SOLVE(TRSM: L→Y) → BWD_ASSEMBLE(GEMM: Y^H,Y→Ainv)

LDL NoBlock/Block:
  GRAM(GEMM: H^H,H→G) → REG(DIAG_ADD: G,lambda*I→A) → LDL_DECOMPOSE(LDL: A→Y)
  → BWD_ASSEMBLE(GEMM: Y^H,Y→Ainv)

Newton-Schulz (K iterations):
  Initial: A, X_init, 2I
  For k=0..K-1:
    GEMM(A, X_k→T_k) → MATRIX_SUB(2I, T_k→R_k) → GEMM(X_k, R_k→X_{k+1})
  BWD_ASSEMBLE(GEMM: X_{K-1}, X_{K-1}→Ainv)

Block-Richardson (L iterations):
  GRAM(GEMM: H^H,H→G) → REG(DIAG_ADD: G,lambda*I→A) → BRI_PRECOND(A→B)
  For l=0..L-1:
    GEMM(A, X_l→T_l) → MATRIX_SUB(2I, T_l→R_l) → MATRIX_ADD(X_l, R_l→X_{l+1})
```

---

## Related Documents

- [Operator Development Standard v3](OPERATOR_DEVELOPMENT_STANDARD_V3.md) — C++ coding standards
- [DAG Primitives Specification](DAG_PRIMITIVES_SPEC.md) — primitive hierarchy and anti-coupling rules
- [Per-Operator Verification Design](specs/2026-07-04-per-operator-verification-design.md) — verification architecture
- [Expert Review Fix Plan](specs/2026-07-04-expert-review-fixes.md) — known issues and fixes
- [Verification Report](ASIM_VERIFICATION_REPORT.md) — current verification status
- [verify-operator Skill](../.claude/skills/verify-operator/SKILL.md) — verification skill usage
- [audit-operator Skill](../.claude/skills/audit-operator/SKILL.md) — audit skill usage
