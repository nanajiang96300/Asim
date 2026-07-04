# vfy-task-1 Report: DAG Executor Shape Propagation Fix

## Bug

In `FormulaDAG.execute()`, initial tensors were registered for **all** batches with the same tensor shape, regardless of whether the tensor actually appeared as an input at each batch. This caused multi-batch formula steps to produce wrong output shapes — tensors from one batch leaked into another batch's registry with incorrect shapes.

## Fix Applied

**File**: `/home/nanajiang/Asim/scripts/uobs_dag_executor.py`

Changed the initial tensor registration block (lines 190-195) from registering each tensor for every batch in the DAG, to only registering it for batches where the tensor name actually appears in a node's `input_names`. Also moved the `all_batches` computation to just before the final output collection where it is still needed.

**Before**:
```python
all_batches = sorted({n.batch for n in self.nodes})
for name, tensor in initial_tensors.items():
    for b in all_batches:
        registry[(b, name)] = np.asarray(tensor, dtype=np.complex128)
```

**After**:
```python
for name, tensor in initial_tensors.items():
    for node in self.nodes:
        if name in node.input_names:
            registry[(node.batch, name)] = np.asarray(tensor, dtype=np.complex128)
```

## Test Created

**File**: `/home/nanajiang/Asim/scripts/test_dag_executor.py`

A self-contained test that builds a 2-step DAG (GRAM GEMM + DIAG_ADD regularization) and verifies that the intermediate matrix G has the correct shape `(4, 4)`.

## Test Results

The test passed successfully:
```
PASS: G shape = (4, 4)
```

## Commit

```
git commit -m "fix: DAG executor shape propagation for multi-batch formula steps"
```

Files committed:
- `/home/nanajiang/Asim/scripts/uobs_dag_executor.py` (fix)
- `/home/nanajiang/Asim/scripts/test_dag_executor.py` (test)
