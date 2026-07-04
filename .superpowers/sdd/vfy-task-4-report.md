# Task 4 Report: Wire Numerical Verification into Operator Pipeline

## Changes Made

### 1. `/home/nanajiang/Asim/orchestrator/pipeline.json`

Updated the `ext_numeric_verify` phase entry from a placeholder to a fully specified phase:

- **id**: `ext_numeric_verify` (unchanged)
- **name**: Changed from `"EXTENSION: Numerical SE Verification"` to `"Numerical Verification (A' + B1)"`
- **description**: Updated to describe FormulaLogger DAG reconstruction + GEMM trace replay
- **check**: Added a concrete two-part check command:
  - DAG path availability via `reference_inverse_registry`
  - Trace replay on `trace.csv`
- **action**: Updated to describe the two-step verification process
- **on_fail**: Updated to reference FormulaLogger coverage and instruction sequence

### 2. `/home/nanajiang/Asim/.claude/skills/op-flow/SKILL.md`

- Inserted **Phase 8: Numerical Verification (A' + B1) [OPTIONAL]** after Phase 7 (Benchmark & Archive)
- Renamed existing **Phase 8: Final Report** to **Phase 9: Final Report**
- Added Final Report table entry for Phase 8 (Numerical Verification) -- omitted from the table since it's optional

### 3. Validation

- `python3 -c "import json; json.load(open('orchestrator/pipeline.json')); print('Valid JSON')"` -- passed

### 4. Commit

- **Commit hash**: `56d12d3`
- **Message**: `feat: wire numerical verification into operator pipeline`
- **Files**: `orchestrator/pipeline.json` (+10/-5), `.claude/skills/op-flow/SKILL.md` (+15/-0)
