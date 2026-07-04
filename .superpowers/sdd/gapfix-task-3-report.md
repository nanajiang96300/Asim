# Gap Fix Task 3: Intermediate Tensor Comparison

## What was done

Added `compare_intermediates()` function to `/home/nanajiang/Asim/scripts/unified_verify.py` and wired it into `__main__` after the multi-seed test.

## Changes

**`scripts/unified_verify.py`** (+59 lines, commit `c7253e1`)

- **`compare_intermediates()`** function (added before `if __name__ == "__main__"`): Runs the DAG executor on a single H matrix and compares intermediate tensors G, A, L, Y, Ainv against direct Python computation. G and A are compared against direct expressions; Ainv is compared against `cholesky_noblock_inverse()`. Prints a table with tensor name, DAG shape, relative error, and PASS/FAIL status.

- **Wiring in `__main__`** (after `run_multi_seed`): Calls `compare_intermediates()` for Rayleigh and CDL-B channels with the generated formula JSON.

## Test results

| Channel  | G        | A        | Ainv     | L / Y        |
|----------|----------|----------|----------|--------------|
| Rayleigh | 2.18e-04 | 2.14e-04 | 1.01e-03 | (no direct ref) |
| CDL-B    | 1.45e-04 | 1.44e-04 | 2.24e-03 | (no direct ref) |

All errors well below 0.01 threshold. Both channels PASS.

L and Y have no direct Python reference (they are internal to the Cholesky algorithm), so they are displayed with shape but no error metric.

## Files

- `/home/nanajiang/Asim/scripts/unified_verify.py` — modified
- `/home/nanajiang/Asim/.superpowers/sdd/gapfix-task-3-report.md` — this report
