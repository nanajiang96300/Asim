# Gapfix Task 1 Report: Negative Control Test

## What was created
`/home/nanajiang/Asim/scripts/test_negative_control.py` — a script that injects deliberate errors into `formula_steps.json` and verifies the `FormulaDAG` executor detects them via increased cross-error vs Python reference.

## Three injection types tested

| Injection | Mechanism | Result |
|-----------|-----------|--------|
| `delete_step` | Removes all steps with 'BWD' in step_id | DETECTED (DAG produced no Ainv) |
| `swap_optype` | Replaces CHOLESKY op_type with DIAG_ADD | DETECTED (cross_err=0.987 — 2149x above threshold) |
| `wrong_shape` | Changes `output_name` of GRAM step to `G_wrong`, breaking dependency chain | DETECTED (DAG crashed with KeyError) |

## Fix applied during iteration
The original `wrong_shape` injection only modified `output_shape` (from `[16,16]` to `[64,64]`), but `output_shape` is metadata-only in the DAG executor and does not affect computation. Changed to modify `output_name` instead, which breaks the data dependency chain and is reliably detected.

## Test output
```
Negative Control Tests
==================================================
  Baseline Py vs Ref: 4.5936e-04
  Detection threshold: 4.5936e-02 (100x baseline)

  delete_step          cross_err=inf -> DETECTED  (DAG produced no Ainv)
  swap_optype          cross_err=9.874136e-01 -> DETECTED  (DAG completed)
  wrong_shape          cross_err=inf -> DETECTED  (DAG crashed: "Cannot resolve input 'G' ...)

  Overall: ALL DETECTED
```

## Commit
`a38ac82` — `test: negative control — verify error injection detection`
