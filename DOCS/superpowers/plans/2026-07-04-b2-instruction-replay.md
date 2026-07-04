# B2: Instruction-Level Trace Audit — Implementation Plan

> **Goal:** Verify that C++ SCALAR instructions in trace.csv match FormulaLogger declarations in type and count.

**Architecture:** Parse trace.csv to extract SCALAR_MUL/DIV/SUB/SQRT instructions grouped by instruction ID. Cross-reference with formula_steps.json emit_step declarations. Flag mismatches: wrong opcode count, undeclared instructions, declared-but-missing instructions.

## B2 Design Decision

Full numerical SCALAR replay is infeasible because Asim uses base addresses (not element addresses) for SCALAR ops. Data routing through SPAD cannot be tracked from trace alone.

Instead, B2 performs an **instruction-level audit**: verifies that every SCALAR opcode emitted in trace.csv has a corresponding FormulaLogger declaration, and vice versa. This closes the "FormulaLogger doesn't match instructions" gap without requiring full numerical replay.

## Files

- Create: `scripts/trace_audit.py` — main audit tool
- Modify: `scripts/unified_verify.py` — add `--b2-audit` flag

## Task 1: Create trace audit tool

Create `scripts/trace_audit.py` that:
1. Reads trace.csv → counts SCALAR_* instructions per instruction ID prefix
2. Reads formula_steps.json → counts expected operations per relation_id
3. Reports mismatches

## Task 2: Wire into verification pipeline

Add B2 audit to unified_verify.py and pipeline.json.

## Task 3: Run on CholeskyNoBlockOp and report

Execute audit, fix any detected mismatches, update report.
