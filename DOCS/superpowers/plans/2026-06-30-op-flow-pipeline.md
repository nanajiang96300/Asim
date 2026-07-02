# Operator Development Pipeline Standardization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize the operator development/optimization workflow so it is automatically enforced and never forgotten, using CLAUDE.md memory anchor + /op-flow Skill + pipeline.json configuration.

**Architecture:** Three-layer enforcement: (1) CLAUDE.md embeds a permanent, always-loaded rule that any operator work MUST invoke /op-flow; (2) /op-flow Skill reads pipeline.json and executes phases in order, refusing to proceed if a gate fails; (3) pipeline.json is a declarative, extensible config where new verification phases are added as JSON entries.

**Tech Stack:** Claude Code Skills (Markdown), JSON config, bash check scripts, existing /audit-operator skill.

## Global Constraints

- Core rule must live in CLAUDE.md (loaded every session, never missed)
- pipeline.json is the single source of truth for phase order and gate checks
- /op-flow Skill is the only execution entry point — no manual shortcuts
- Each phase produces a verifiable artifact (doc, binary, benchmark, audit report)
- New verification phases are added by appending a JSON object — no code changes to the Skill required beyond reading the new entry
- All existing operators (6 baseline + 2 merge) must pass the new pipeline with zero modifications

---

### Task 1: CLAUDE.md memory anchor

**Files:**
- Modify: `CLAUDE.md` (append at top of file)

**Interfaces:**
- Produces: Permanent rule text that Claude Code loads every session

- [ ] **Step 1: Add pipeline enforcement rule to CLAUDE.md**

Append or prepend this block to CLAUDE.md:

```markdown
## Operator Development Pipeline (MANDATORY)

When developing a NEW operator or optimizing an EXISTING operator, you MUST invoke `/op-flow <operator_name> <action>` before writing any code. The pipeline enforces:

1. Design document with formula derivation exists before code
2. Code follows v3.0 standard (DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md)
3. CMake build passes
4. /audit-operator sub-agent review passes (formula↔code consistency)
5. Benchmark results archived to results/<operator>/

**NEVER skip the pipeline.** If you write operator code without running /op-flow first, stop, delete the code, and run /op-flow.
```

- [ ] **Step 2: Verify the rule is visible**

```bash
grep -A 10 "Operator Development Pipeline" CLAUDE.md
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "feat: add mandatory operator pipeline rule to CLAUDE.md"
```

---

### Task 2: pipeline.json — declarative gate configuration

**Files:**
- Create: `orchestrator/pipeline.json`

**Interfaces:**
- Produces: `{"name": "op-flow", "phases": [{"id": "...", "name": "...", "required": true/false, "check": "..."}]}`
- Consumed by: /op-flow Skill (Task 3)

- [ ] **Step 1: Write pipeline.json**

```json
{
  "name": "op-flow",
  "version": "1.0",
  "description": "Operator development pipeline — declarative phase definitions",
  "phases": [
    {
      "id": "design_doc",
      "name": "Design Document",
      "required": true,
      "description": "DOCS/operators/<op>.md must exist with formula derivation",
      "check": "test -f DOCS/operators/${OP_NAME}.md",
      "on_fail": "Create DOCS/operators/${OP_NAME}.md with: math formulas, SPAD layout, instruction mapping, FormulaLogger coverage. See 01_cholesky_noblock_v2.md for template."
    },
    {
      "id": "math_derivation",
      "name": "Mathematical Derivation",
      "required": true,
      "description": "Document must contain formula derivation and optimization equivalence proof",
      "check": "grep -q '公式推导' DOCS/operators/${OP_NAME}.md || grep -q 'formula' DOCS/operators/${OP_NAME}.md",
      "on_fail": "Add '## 2. 公式推导' section with baseline formulas and optimization equivalence proof (associativity, vectorization)."
    },
    {
      "id": "baseline_existence",
      "name": "Baseline Operator Exists",
      "required": false,
      "description": "For optimization: baseline operator file must exist in src/inverse/. For new: skip this gate.",
      "check": "test -f src/inverse/${ALGO_DIR}/${BASELINE_OP}.cc || echo 'NEW_OPERATOR'",
      "on_fail": "Run /op-flow with action=new to create baseline first before optimizing."
    },
    {
      "id": "code_v3_standard",
      "name": "Code Follows v3.0 Standard",
      "required": true,
      "description": "Operator .cc must follow DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md",
      "check": "grep -q 'base_addr = 0' src/inverse/${ALGO_DIR}/${OP_FILE}.cc && grep -q 'set_algorithm' src/inverse/${ALGO_DIR}/${OP_FILE}.cc && grep -q 'PIPE_BARRIER' src/inverse/${ALGO_DIR}/${OP_FILE}.cc",
      "on_fail": "Fix: base_addr=0 in MOVIN, FormulaLogger::set_algorithm(), PIPE_BARRIER at column boundaries."
    },
    {
      "id": "compile",
      "name": "Build Verification",
      "required": true,
      "description": "cmake --build must succeed",
      "check": "CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build build --target Simulator -j$(nproc) 2>&1 | grep -q 'Built target Simulator'",
      "on_fail": "Fix compilation errors and re-run."
    },
    {
      "id": "runtime",
      "name": "Runtime Smoke Test",
      "required": true,
      "description": "Simulator must finish without abort",
      "check": "ONNXIM_MAX_CORE_CYCLES=100000 ./build/bin/Simulator --config configs/ascend_910b_quiet.json --models_list example/${CONFIG_JSON} --mode ${MODE} --log_level info 2>&1 | grep -q 'finish at'",
      "on_fail": "Check for SPAD deadlock, missing init, wrong addresses."
    },
    {
      "id": "audit_review",
      "name": "Audit Sub-Agent Review",
      "required": true,
      "description": "/audit-operator must pass — sub-agent compares formulas in doc against code",
      "check": "manual", 
      "action": "Invoke /audit-operator ${OP_NAME} with the operator doc and code. All checklist items must PASS.",
      "on_fail": "Fix audit findings and re-audit."
    },
    {
      "id": "benchmark",
      "name": "Benchmark & Archive",
      "required": true,
      "description": "Run benchmark and store to results/<op>/run_NNN/",
      "check": "test -f results/${OP_NAME}/run_001/summary.json",
      "action": "bash scripts/run_benchmark_suite.sh ${OP_NAME} ${MODE} configs/ascend_910b_quiet.json example/${CONFIG_JSON}",
      "on_fail": "Re-run benchmark script."
    },
    {
      "id": "ext_numeric_verify",
      "name": "EXTENSION: Numerical SE Verification",
      "required": false,
      "description": "[FUTURE] Verify numerical correctness via FormulaLogger→UOBS SE reconstruction",
      "check": "echo 'NOT_YET_IMPLEMENTED'",
      "action": "Run scripts/verify_baseline.py with operator output and compare SE",
      "on_fail": "Fix numerical errors in operator."
    }
  ]
}
```

- [ ] **Step 2: Validate JSON syntax**

```bash
python3 -c "import json; json.load(open('orchestrator/pipeline.json')); print('Valid JSON')"
```

- [ ] **Step 3: Commit**

```bash
git add orchestrator/pipeline.json
git commit -m "feat: add pipeline.json — declarative operator development gates"
```

---

### Task 3: /op-flow Skill — execution engine

**Files:**
- Create: `.claude/skills/op-flow/SKILL.md`

**Interfaces:**
- Consumes: `orchestrator/pipeline.json`, `DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md`
- Produces: Pipeline execution with per-phase pass/fail reporting
- Invokes: `/audit-operator` sub-agent at Phase 6

- [ ] **Step 1: Write op-flow skill**

```markdown
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

### Phase 3: Baseline Operator (optimize only)
- **Gate**: Baseline operator `.cc` file must exist
- **If missing**: Run `/op-flow <baseline_name> new` first

### Phase 4: Code v3.0 Standard Check
- Verify against `DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md`:
  - `base_addr = 0` in MOVIN
  - `FormulaLogger::set_algorithm()` called
  - SPAD regions initialized before SCALAR read
  - `SCALAR_DIV(Reg,Reg)` for unity synthesis
  - `PIPE_BARRIER` at column/phase boundaries
  - `MOVOUT` has `src_from_accum=true, last_inst=true`

### Phase 5: Build Verification
- Run: `CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build build --target Simulator -j$(nproc)`
- **Gate**: Must show `[100%] Built target Simulator`

### Phase 6: Runtime Smoke Test
- Run simulator with 100000 cycle limit
- **Gate**: Must show `finish at <N>` (not `abort` or deadlock)

### Phase 7: Audit Review
- Launch `/audit-operator <operator_name>` as sub-agent
- Feed the operator doc (`DOCS/operators/<op>.md`) and code to the sub-agent
- **Gate**: All checklist items must PASS
- If audit finds issues, fix them and re-audit before proceeding

### Phase 8: Benchmark & Archive
- Run: `bash scripts/run_benchmark_suite.sh <op> <mode> configs/ascend_910b_quiet.json example/<config>.json`
- **Gate**: `results/<operator>/run_NNN/summary.json` must exist
- Record result in operator doc's verification section

### Phase 9: Final Report
Output a summary table:

| Phase | Gate | Status |
|-------|------|--------|
| 1. Design Doc | file exists | ✅/❌ |
| 2. Math Derivation | formula section present | ✅/❌ |
| 3. Baseline | (skip for new) | ✅/❌ |
| 4. v3.0 Standard | all checks passed | ✅/❌ |
| 5. Build | cmake success | ✅/❌ |
| 6. Runtime | no deadlock | ✅/❌ |
| 7. Audit | sub-agent PASS | ✅/❌ |
| 8. Benchmark | summary.json exists | ✅/❌ |

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
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/op-flow/SKILL.md
git commit -m "feat: add /op-flow skill — standardized operator development pipeline"
```

---

### Task 4: Integration test — run /op-flow on CholeskyNoBlockMergeOp

**Files:**
- (all already exist from previous work)

**Interfaces:**
- Exercises: /op-flow Skill (Task 3) against a known-good operator

- [ ] **Step 1: Dry-run each phase manually**

```bash
# Phase 1: Design doc exists
test -f DOCS/operators/01b_cholesky_noblock_merge.md && echo "P1 PASS" || echo "P1 FAIL"

# Phase 2: Math derivation present
grep -q "公式推导" DOCS/operators/01b_cholesky_noblock_merge.md && echo "P2 PASS" || echo "P2 FAIL"

# Phase 4: v3.0 standard
grep -q "base_addr = 0" src/inverse/cholesky_noblock/CholeskyNoBlockMergeOp.cc && echo "P4a PASS" || echo "P4a FAIL"
grep -q "set_algorithm" src/inverse/cholesky_noblock/CholeskyNoBlockMergeOp.cc && echo "P4b PASS" || echo "P4b FAIL"
grep -q "PIPE_BARRIER" src/inverse/cholesky_noblock/CholeskyNoBlockMergeOp.cc && echo "P4c PASS" || echo "P4c FAIL"

# Phase 5: Build
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build build --target Simulator -j$(nproc) 2>&1 | grep -q "Built target Simulator" && echo "P5 PASS" || echo "P5 FAIL"

# Phase 6: Runtime
ONNXIM_MAX_CORE_CYCLES=100000 ./build/bin/Simulator --config configs/ascend_910b_quiet.json --models_list example/cholesky_noblock_merge_test.json --mode cholesky_noblock_merge_test --log_level info 2>&1 | grep -q "finish at" && echo "P6 PASS" || echo "P6 FAIL"

# Phase 8: Benchmark
test -f results/cholesky_noblock_v2/run_002/summary.json 2>/dev/null && echo "P8 PASS" || echo "P8 FAIL (needs run)"
```

- [ ] **Step 2: Launch audit sub-agent for Phase 7**

```
Invoke /audit-operator cholesky_noblock_merge
- Feed DOCS/operators/01b_cholesky_noblock_merge.md as the reference
- Feed src/inverse/cholesky_noblock/CholeskyNoBlockMergeOp.cc as the code
- Expect: all checklist items PASS
```

- [ ] **Step 3: Record audit result**

Add audit result to the operator doc's verification section.

- [ ] **Step 4: Commit pipeline test results**

```bash
git add -A && git commit -m "test: op-flow pipeline integration test on CholeskyNoBlockMergeOp"
```

---

### Task 5: Negative test — verify gate blocking

**Files:**
- Modify: (none — temporary test)

**Interfaces:**
- Exercises: Phase 1 gate rejection logic

- [ ] **Step 1: Simulate Phase 1 failure**

```bash
# Test: operator with no design doc should fail Phase 1
NONEXISTENT_OP="nonexistent_test_op"
test -f "DOCS/operators/${NONEXISTENT_OP}.md" && echo "UNEXPECTED: file exists" || echo "P1 GATE BLOCKS: design doc not found — must create first"
```

- [ ] **Step 2: Verify error message**

Expected output: `P1 GATE BLOCKS: design doc not found — must create first`

- [ ] **Step 3: Document the gate blocking behavior**

No code changes needed — the test proves the gate works.

---

### Task 6: Extensibility test — add a new phase

**Files:**
- Modify: `orchestrator/pipeline.json` (append one entry)

- [ ] **Step 1: Add extension phase to pipeline.json**

Append after the last phase entry:
```json
{
  "id": "ext_spad_audit",
  "name": "EXTENSION TEST: SPAD Layout Audit",
  "required": false,
  "description": "Verify SPAD addresses don't overlap and all regions are sized correctly",
  "check": "python3 -c \"
import re, sys
with open('src/inverse/${ALGO_DIR}/${OP_FILE}.cc') as f:
    c = f.read()
addrs = re.findall(r'addr_type (\w+)\s*=\s*(\w+)\s*\+\s*(\w+)', c)
print(f'{len(addrs)} SPAD regions found')
\" | grep -q 'SPAD regions'",
  "on_fail": "SPAD layout may have issues — check for overlapping regions."
}
```

- [ ] **Step 2: Validate JSON**

```bash
python3 -c "import json; json.load(open('orchestrator/pipeline.json')); print('Valid JSON after extension')"
```

- [ ] **Step 3: Verify Skill reads new phase**

The /op-flow Skill reads pipeline.json dynamically, so adding a new entry should make it appear in the next invocation automatically.

- [ ] **Step 4: Commit**

```bash
git add orchestrator/pipeline.json
git commit -m "feat: add extensibility test phase to pipeline.json"
```

---

## Verification Summary

| # | Test | Expected Result |
|---|------|----------------|
| 1 | CLAUDE.md contains pipeline rule | grep finds the rule |
| 2 | pipeline.json valid JSON | python3 json.load passes |
| 3 | /op-flow skill file exists | .claude/skills/op-flow/SKILL.md |
| 4 | Phase 1-6 on CholeskyNoBlockMergeOp | All PASS |
| 5 | Audit on CholeskyNoBlockMergeOp | Sub-agent reports PASS |
| 6 | Nonexistent operator fails Phase 1 | Gate blocks correctly |
| 7 | Add extension phase to pipeline.json | JSON valid, Skill reads it |
