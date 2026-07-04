# Task 3: GEMM-Level Trace Replay Engine

## Files Changed

- **Created**: `scripts/trace_replay.py` (GEMM-level trace replay engine, 300+ lines)
- **Fixed**: `src/main.cc` (added missing model creation handlers for `cholesky_noblock_v2_test` and `cholesky_noblock_merge_test`)

## What Was Built

`scripts/trace_replay.py` is a Python module that replays MOVIN/GEMM/MOVOUT data flow from a TraceLogger CSV (`ONNXIM_TRACE_CSV`) with optional formula JSON (`ONNXIM_FORMULA_JSON`) integration.

### Capabilities

1. **Trace CSV parsing** -- reads `name,unit,start_cycle,end_cycle` events, classifies into MOVIN/GEMM/MOVOUT/VECTOR/SCALAR/WAIT
2. **Per-core breakdown** -- splits event counts by core ID
3. **Formula JSON integration** -- loads algorithm metadata, step shapes, and `relation_id` links to resolve instruction IDs to mathematical operations
4. **Numerical GEMM replay** -- for each formula GEMM step matched to trace events:
   - Looks up input shapes from formula JSON
   - Generates/reuses random complex matrices keyed by logical names (e.g., "H", "Y^H")
   - Computes `A @ B` with FP16 quantization (`float16 → float64`)
   - Stores results in SPAD (0x10000000) or ACCUM (0x20000000) by output name convention
5. **State export** -- final SPAD and ACCUM contents accessible as numpy arrays keyed by hex address

### Key Classes

- `SpadMemory` -- logical SPAD/ACCUM address space modeling
- `TraceReplayer` -- main replay engine with `replay()`, `get_event_summary()`, `get_final_state()`
- `replay_trace()` -- convenience entry point

### API

```python
from trace_replay import TraceReplayer, replay_trace

# With formula JSON (full numerical replay)
result = replay_trace("trace.csv", "formula.json")
print(result["summary"])        # event counts, per-core breakdown
print(result["num_gemm_ops"])   # 192 for cholesky_noblock 96 batches
print(result["final_state"])    # SPAD and ACCUM matrices

# Without formula JSON (statistics only)
result = replay_trace("trace.csv")
```

## Test Results

### Test 1: matmul_test (256x32x256)

```
$ python3 scripts/trace_replay.py /tmp/test_matmul_trace.csv
Total events:           5632
Unit counts:            {'MOVIN': 512, 'GEMM': 512, 'MOVOUT': 4096, 'VECTOR': 0, ...}
GEMM instruction IDs:   {'GEMM': 512}
GEMM operations replayed: 512
```

### Test 2: cholesky_noblock_v2_test (64x16, 96 batches)

Without formula JSON:
```
Total events:           243741
Unit counts:            {'SCALAR': 235008, 'MOVIN': 3840, 'GEMM': 192, 'MOVOUT': 768, ...}
Per-core:               24 cores with ~8 GEMM each
GEMM instruction IDs:   {'CHOL_NB_GRAM': 96, 'CHOL_NB_BWD_GEMM': 96}
GEMM operations replayed: 192
```

With formula JSON:
```
Algorithm:              cholesky_noblock_v2
Formula steps:          13344
Formula GEMM steps:     192
GEMM operations replayed: 192
SPAD entries:  5
ACCUM entries: 1
```
Final ACCUM holds Ainv (16x16 complex matrix from backward assembly `Y^H @ Y`).

## Bug Fix: Missing Model Creation Handlers in main.cc

`src/main.cc` had mode checks for `cholesky_noblock_v2_test` and `cholesky_noblock_merge_test` at lines 84-86/81-83, but the model creation loop at lines 126-190 had no handlers for these modes, causing them to fall through to the generic ONNX path which crashed with "Error opening file: ./models/...onnx".

Added handlers using `CholeskyNoBlockBaselineModel` and `CholeskyNoBlockMergeModel`.

## Commit

```
8e5c38d feat: add GEMM-level trace replay engine (B1)
```

## Verification

```bash
# Generate trace
ONNXIM_TRACE_CSV=/tmp/t.csv ONNXIM_FORMULA_JSON=/tmp/f.json \
  ./build/bin/Simulator --config configs/ascend_910b_quiet.json \
  --models_list example/cholesky_noblock_v2_test.json \
  --mode cholesky_noblock_v2_test --log_level info

# Replay
.python scripts/trace_replay.py /tmp/t.csv /tmp/f.json
```
