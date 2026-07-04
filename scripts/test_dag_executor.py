#!/usr/bin/env python3
import sys, numpy as np
sys.path.insert(0, '/home/nanajiang/Asim')
from scripts.uobs_dag_executor import FormulaDAG

steps = [
    {"step_id": "GRAM", "op_type": "GEMM", "input_names": ["H^H", "H"], "output_name": "G",
     "input_shapes": [[8, 4], [4, 8]], "output_shape": [4, 4], "batch": 0, "relation_id": "GRAM"},
    {"step_id": "REG", "op_type": "DIAG_ADD", "input_names": ["G", "lambda*I"], "output_name": "A",
     "input_shapes": [[4, 4], [4, 4]], "output_shape": [4, 4], "batch": 0, "relation_id": "REG"},
]

dag = FormulaDAG(steps)
H = np.random.randn(8, 4) + 1j * np.random.randn(8, 4)
result = dag.execute({"H": H}, {"lambda": 0.1})
G = result.get("G")
assert G is not None, "G not found"
assert G.shape == (4, 4), f"Expected (4,4), got {G.shape}"
print(f"PASS: G shape = {G.shape}")
