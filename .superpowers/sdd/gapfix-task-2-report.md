# Task 2: Multi-Batch + Multi-Seed Verification

## Summary

Added `run_multi_seed()` function to `/home/nanajiang/Asim/scripts/unified_verify.py` and wired it into `__main__`.

## Changes

- **File modified**: `scripts/unified_verify.py` (46 insertions)
- **Added** `run_multi_seed(algo_name, algo_func, formula_path, seeds, batches)` — tests 3 seeds x 3 batches per channel, computes cross-error (Python algo vs DAG replay) for each combination, reports mean/min/max error and PASS/FAIL status per channel.
- **Added** call at end of `__main__` after the existing single-batch summary.

## Test Results

```
Multi-Seed + Multi-Batch Stability Test

  Rayleigh (3 seeds x 3 batches):
    Samples: 9  Mean: 9.49e-04  Min: 7.86e-04  Max: 1.12e-03  PASS

  CDL-B (3 seeds x 3 batches):
    Samples: 9  Mean: 3.60e-03  Min: 2.24e-03  Max: 5.90e-03  PASS

  CDL-B_Harsh (3 seeds x 3 batches):
    Samples: 9  Mean: 2.51e-02  Min: 6.77e-03  Max: 6.14e-02  FAIL
```

Rayleigh and CDL-B pass the 0.01 threshold consistently across all seeds and batches. CDL-B_Harsh fails as expected (very narrow angular spread, low SNR).

## Commit

`37c435d` — `feat: multi-batch + multi-seed verification`
