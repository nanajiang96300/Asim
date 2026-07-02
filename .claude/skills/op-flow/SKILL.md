# Operator Development Pipeline (/op-flow)

Enforces the standardized operator development workflow. Never skip this.

## Usage

```
/op-flow <operator_name> <action>
```

- `<operator_name>`: e.g. `cholesky_noblock_v2`, `cholesky_noblock_merge`
- `<action>`: `new` (new operator) | `optimize` (optimization from baseline)

## Pipeline Phases

Read `orchestrator/pipeline.json` for phase definitions. Execute each phase in order.

### Phase 1: Design Document Check
- **Gate**: `DOCS/operators/<operator_name>.md` must exist
- **If missing**: Create the document using the template from `01_cholesky_noblock_v2.md`:
  - Section 1: File inventory (Python/C++ paths)
  - Section 2: Mathematical formulas (complete derivation)
  - Section 3: SPAD memory layout table
  - Section 4: Phase-by-phase instruction mapping (formula → opcode → src → dest)
  - Section 5: FormulaLogger coverage table
  - Section 6: Verification data

### Phase 2: Mathematical Derivation Check
- **Gate**: Document must contain `## 2. 公式推导` or `## 2. Formula Derivation`
- **For `new` action**: Must include full algorithm derivation from A = H^H H + λI to A^{-1}
- **For `optimize` action**: Must include:
  - Baseline formula
  - Optimization equivalence proof (associativity, vectorization, etc.)
  - Instruction count comparison (baseline vs optimized)
  - Expected speedup analysis

### Phase 3: Code v3.0 Standard Check
- Verify against `DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md`:
  - `base_addr = 0` in MOVIN
  - `FormulaLogger::set_algorithm()` called
  - SPAD regions initialized before SCALAR read
  - `SCALAR_DIV(Reg,Reg)` for unity synthesis
  - `PIPE_BARRIER` at column/phase boundaries
  - `MOVOUT` has `src_from_accum=true, last_inst=true`

### Phase 4: Build Verification
- Run: `CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build build --target Simulator -j$(nproc)`
- **Gate**: Must show `[100%] Built target Simulator`

### Phase 5: Runtime Smoke Test
- Run simulator with 100000 cycle limit
- **Gate**: Must show `finish at <N>` (not `abort` or deadlock)

### Phase 6: Audit Review
- Launch `/audit-operator <operator_name>` as sub-agent
- Feed the operator doc (`DOCS/operators/<op>.md`) and code to the sub-agent
- **Gate**: All checklist items must PASS
- If audit finds issues, fix them and re-audit before proceeding

### Phase 7: Benchmark & Archive
- Run: `bash scripts/run_benchmark_suite.sh <op> <mode> configs/ascend_910b_quiet.json example/<config>.json`
- **Gate**: `results/<operator>/run_NNN/summary.json` must exist
- Record result in operator doc's verification section

### Phase 8: Final Report
Output a summary table:

| Phase | Gate | Status |
|-------|------|--------|
| 1. Design Doc | file exists | ✅/❌ |
| 2. Math Derivation | formula section present | ✅/❌ |
| 3. v3.0 Standard | all checks passed | ✅/❌ |
| 4. Build | cmake success | ✅/❌ |
| 5. Runtime | no deadlock | ✅/❌ |
| 6. Audit | sub-agent PASS | ✅/❌ |
| 7. Benchmark | summary.json exists | ✅/❌ |

## Extending the Pipeline

To add a new verification phase:
1. Append a JSON entry to `orchestrator/pipeline.json["phases"]`
2. Set `"required": true` once the phase is stable
3. Add the phase handler to this Skill document

Example: adding numerical SE verification:
```json
{
  "id": "ext_numeric_verify",
  "name": "Numerical SE Verification",
  "required": false,
  "check": "test -f results/${OP_NAME}/run_001/se_verify.json",
  "action": "python scripts/verify_baseline.py --operator ${OP_NAME}"
}
```
