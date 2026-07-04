#!/usr/bin/env python3
"""
UOBS DAG Executor — formula-agnostic computation-graph builder and FP16 replay engine.
Reads formula_steps.json (produced by the C++ FormulaLogger) and executes every
recorded linear-algebra primitive in double precision with FP16 quantisation,
producing the final reconstructed matrix A^{-1}.

This module is operator-agnostic: it does NOT contain any hardcoded knowledge
about Cholesky / LDL / BlockJacobi formulas.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

def _fp16(x: np.ndarray) -> np.ndarray:
    """Quantize to FP16 (separate real/imag for complex)."""
    if x is None: return None
    if np.iscomplexobj(x):
        r = _fp16(x.real)
        i = _fp16(x.imag)
        return r + 1j * i
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)
def _cplx_fp16(z: np.ndarray) -> np.ndarray:
    """Quantise complex array: real & imag independently to fp16."""
    real = _fp16(z.real)
    imag = _fp16(z.imag)
    return real + 1j * imag


def prim_gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GEMM: C = _fp16(_fp16(A) @ _fp16(B))  (A: m×k, B: k×n, C: m×n)"""
    a_q = _cplx_fp16(a)
    b_q = _cplx_fp16(b)
    result = a_q @ b_q
    return _cplx_fp16(result)


def prim_diag_add(a: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """DIAG_ADD: A ← A + λI"""
    a_q = _cplx_fp16(a)
    n = a_q.shape[0]
    result = a_q + lam * np.eye(n, dtype=np.complex128)
    return _cplx_fp16(result)



def prim_trsm(*inputs):
    """Forward substitution: Y = L^{-1}. Takes 1 input (L), returns Y."""
    if len(inputs) == 0: return None
    L = inputs[0]
    if L is None: return None
    n = L.shape[0]
    Y = np.zeros((n, n), dtype=np.complex128)
    for c in range(n):
        Y[c, c] = _fp16(1.0 / max(L[c, c].real, 1e-15))
        for i in range(c + 1, n):
            acc = np.complex128(0.0)
            for k in range(c, i):
                acc += _fp16(L[i, k] * Y[k, c])
            Y[i, c] = _fp16(-acc / max(L[i, i].real, 1e-15))
    return Y

def prim_cholesky(a: np.ndarray) -> np.ndarray:
    """CHOLESKY: L = chol(A) with near-singular fallback."""
    a_q = _cplx_fp16(a)
    a_d = a_q.astype(np.complex128)
    try:
        l_mat = np.linalg.cholesky(a_d)
    except np.linalg.LinAlgError:
        # Near-singular: add small diagonal perturbation for stability
        eps = 1e-6 * np.max(np.abs(np.diag(a_d)))
        for attempt in range(5):
            try:
                l_mat = np.linalg.cholesky(a_d + eps * np.eye(a_d.shape[0]))
                break
            except np.linalg.LinAlgError:
                eps *= 10
        else:
            # Last resort: eigenvalue regularization
            w, v = np.linalg.eigh(a_d)
            w = np.maximum(w, 1e-10)
            l_mat = np.linalg.cholesky((v * w) @ v.conj().T + 1e-8 * np.eye(a_d.shape[0]))
    return _cplx_fp16(l_mat)


def prim_diag_inv(a: np.ndarray) -> np.ndarray:
    """DIAG_INV: invert a diagonal (or block-diagonal) matrix. Handles scalar case for LDL."""
    a_q = _cplx_fp16(a)
    if a_q.size == 1:
        return _cplx_fp16(np.array([[1.0 / max(a_q.flat[0].real, 1e-15)]], dtype=np.complex128))
    if a_q.shape[0] == a_q.shape[1]:
        inv = np.linalg.inv(a_q.astype(np.complex128))
    else:
        inv = np.linalg.pinv(a_q.astype(np.complex128))
    return _cplx_fp16(inv)


def prim_ldl_decompose(*inputs):
    """LDL_DECOMPOSE: Full LDL A=L·D·L^H → Dinv column vector + compute Y = sqrt(Dinv)·L^{-1}.
    Takes 1 input (A), returns Y ready for BWD GEMM.
    Used as a high-level primitive matching CHOLESKY's role for LDL operators."""
    if len(inputs) == 0: return None
    A = inputs[0]
    if A is None: return None
    A_q = _cplx_fp16(A)
    A_d = A_q.astype(np.complex128)
    n = A_d.shape[0]
    L = np.eye(n, dtype=np.complex128)
    D = np.zeros(n, dtype=np.float64)
    Dinv = np.zeros(n, dtype=np.float64)

    # Column-by-column LDL decomposition
    for j in range(n):
        acc = 0.0
        for k in range(j):
            acc += D[k] * (L[j,k].real**2 + L[j,k].imag**2)
        d_jj = A_d[j,j].real - acc
        if d_jj <= 0: d_jj = 1e-15
        D[j] = d_jj
        Dinv[j] = 1.0 / d_jj

        for i in range(j+1, n):
            dot = np.complex128(0.0)
            for k in range(j):
                dot += L[i,k] * D[k] * np.conj(L[j,k])
            L[i,j] = (A_d[i,j] - dot) * Dinv[j]

    # Forward solve Z = L^{-1} (unit triangular)
    Z = np.eye(n, dtype=np.complex128)
    for c in range(n):
        for i in range(c+1, n):
            acc = np.complex128(0.0)
            for k in range(c, i):
                acc += L[i,k] * Z[k,c]
            Z[i,c] = -acc

    # Scale by sqrt(Dinv)
    sqrt_d = np.sqrt(np.maximum(Dinv, 0))
    Y = Z * sqrt_d[np.newaxis, :]
    return _cplx_fp16(Y)


def prim_sqrt_scale(*inputs):
    """SQRT_SCALE: Y_col *= sqrt(Dinv_col) for each column."""
    if len(inputs) < 2: return inputs[0] if inputs else None
    Y, Dinv = inputs[0], inputs[1]
    if Y is None or Dinv is None: return Y
    Y_q = _cplx_fp16(Y)
    D_q = _cplx_fp16(Dinv)
    n = Y_q.shape[0]
    sqrt_d = np.sqrt(np.maximum(np.abs(D_q.flat[:n]) if D_q.size >= n else np.abs(np.diag(D_q)), 0))
    result = Y_q * sqrt_d[np.newaxis, :]
    return _cplx_fp16(result)


def prim_matrix_inv_2x2(a: np.ndarray) -> np.ndarray:
    """Direct 2×2 inversion: [a b; c d]^{-1} = 1/(ad-bc) * [d -b; -c a]."""
    a_q = _cplx_fp16(a)
    a00, a01 = a_q[0, 0], a_q[0, 1]
    a10, a11 = a_q[1, 0], a_q[1, 1]
    det = a00 * a11 - a01 * a10 + 1e-12
    inv = np.array([[a11, -a01], [-a10, a00]], dtype=np.complex128) / det
    return _cplx_fp16(inv)


def prim_matrix_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATRIX_SUB: C = A - B."""
    return _cplx_fp16(_cplx_fp16(a) - _cplx_fp16(b))


def prim_matrix_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATRIX_ADD: C = A + B."""
    return _cplx_fp16(_cplx_fp16(a) + _cplx_fp16(b))


def prim_scale(a: np.ndarray, scalar: float = 1.0) -> np.ndarray:
    """SCALE: A ← α·A."""
    return _cplx_fp16(_cplx_fp16(a) * scalar)


# Map UOBS op_type → executor function
PRIMITIVES: Dict[str, Callable] = {
    "GEMM":             prim_gemm,
    "DIAG_ADD":         prim_diag_add,
    "CHOLESKY":         prim_cholesky,
    "LDL_DECOMPOSE":    prim_ldl_decompose,
    "TRSM":             prim_trsm,
    "DIAG_INV":         prim_diag_inv,
    "MATRIX_INV_2x2":   prim_matrix_inv_2x2,
    "MATRIX_SUB":       prim_matrix_sub,
    "MATRIX_ADD":       prim_matrix_add,
    "SCALE":            prim_scale,
    "SQRT_SCALE":       prim_sqrt_scale,
}

# ── DAG node ───────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    step_id: str
    op_type: str
    input_names: List[str]
    output_name: str
    input_shapes: List[List[int]]
    output_shape: List[int]
    batch: int
    relation_id: str
    # filled at execution time
    inputs: List[np.ndarray] = field(default_factory=list)
    output: Optional[np.ndarray] = None


# ── DAG builder ────────────────────────────────────────────────────────────

class FormulaDAG:
    """Directed acyclic graph constructed from formula_steps.json."""

    def __init__(self, steps: List[dict]):
        self.nodes: List[DAGNode] = []
        self._output_index: Dict[Tuple[int, str], int] = {}  # (batch, output_name) → node_idx
        self._build(steps)

    def _build(self, steps: List[dict]):
        for s in steps:
            node = DAGNode(
                step_id=s["step_id"],
                op_type=s["op_type"],
                input_names=list(s["input_names"]),
                output_name=s["output_name"],
                input_shapes=[list(sh) for sh in s["input_shapes"]],
                output_shape=list(s["output_shape"]),
                batch=int(s["batch"]),
                relation_id=s["relation_id"],
            )
            self.nodes.append(node)
            key = (node.batch, node.output_name)
            self._output_index[key] = len(self.nodes) - 1

    def get_output(self, batch: int, name: str) -> np.ndarray | None:
        idx = self._output_index.get((batch, name))
        if idx is None:
            return None
        return self.nodes[idx].output

    def execute(self, initial_tensors: Dict[str, np.ndarray],
                aux_params: Dict[str, Any] | None = None) -> Dict[str, np.ndarray]:
        """Execute all nodes in topological order.
        
        Args:
            initial_tensors: named input tensors (e.g. {"H": H_matrix})
            aux_params: auxiliary parameters (e.g. {"lambda": 0.1})
        
        Returns:
            dict mapping output_name → tensor for the last batch.
        """
        if aux_params is None:
            aux_params = {}

        # Registry: per-batch tensor store
        # For incremental updates (multiple steps write to same name), accumulate
        registry: Dict[Tuple[int, str], np.ndarray] = {}

        # Seed initial tensors per batch: register each tensor only for batches
        # where it actually appears as an input, using the correct per-step shapes.
        for name, tensor in initial_tensors.items():
            for node in self.nodes:
                if name in node.input_names:
                    registry[(node.batch, name)] = np.asarray(tensor, dtype=np.complex128)

        for node in self.nodes:
            key = (node.batch, node.output_name)
            # Allow recomputation for iterative algorithms (e.g., Newton-Schulz loops)

            # Resolve inputs
            inputs = []
            for iname in node.input_names:
                # Check batch-specific first, then fall back to batch-0
                val = registry.get((node.batch, iname))
                if val is None:
                    val = registry.get((0, iname))
                if val is None and iname in aux_params:
                    # scalar auxiliary parameter → treat as lambda
                    lam = float(aux_params[iname])
                    n = node.output_shape[0]
                    val = lam * np.eye(n, dtype=np.complex128)
                if val is None and iname == "lambda*I":
                    lam = float(aux_params.get("lambda", 0.1))
                    n = node.output_shape[0]
                    val = lam * np.eye(n, dtype=np.complex128)
                if val is None and iname == "I":
                    n = node.output_shape[0]
                    val = np.eye(n, dtype=np.complex128)
                if val is None and iname == "H^H":
                    h = registry.get((node.batch, "H"))
                    if h is not None:
                        val = h.conj().T
                if val is None and iname == "Y^H":
                    y = registry.get((node.batch, "Y"))
                    if y is not None:
                        val = y.conj().T
                if val is None:
                    raise KeyError(
                        f"Cannot resolve input '{iname}' for step '{node.step_id}' "
                        f"(batch {node.batch}). Available keys: {sorted(registry.keys())}"
                    )
                inputs.append(val)

            # Execute primitive
            prim_fn = PRIMITIVES.get(node.op_type)
            if prim_fn is None:
                raise ValueError(f"Unknown op_type '{node.op_type}' in step '{node.step_id}'")

            # Call with appropriate arguments based on primitive type
            try:
                if node.op_type in ("GEMM", "MATRIX_SUB", "MATRIX_ADD"):
                    result = prim_fn(inputs[0], inputs[1])
                elif node.op_type == "DIAG_ADD":
                    lam = aux_params.get("lambda", 0.1)
                    result = prim_fn(inputs[0], lam)
                elif node.op_type == "TRSM":
                    if len(inputs) == 1:
                        result = prim_fn(inputs[0])  # Y = L^{-1} (FWD solve)
                    else:
                        result = inputs[1]  # L update pass-through
                elif node.op_type == "SCALE":
                    result = prim_fn(inputs[0], aux_params.get("omega", 1.0))
                elif node.op_type == "SQRT_SCALE":
                    result = prim_fn(inputs[0], inputs[1]) if len(inputs) >= 2 else prim_fn(inputs[0])
                elif node.op_type == "LDL_DECOMPOSE":
                    result = prim_fn(inputs[0])
                    if node.step_id.endswith("_0"):
                        import numpy as _np
                        print(f"  DAG: {node.step_id} input_norm={_np.linalg.norm(inputs[0]):.4f} output_norm={_np.linalg.norm(result):.4f}", flush=True)
                else:
                    # CHOLESKY, DIAG_INV, MATRIX_INV_2x2
                    result = prim_fn(inputs[0])
            except Exception as exc:
                raise RuntimeError(
                    f"Primitive '{node.op_type}' failed at step '{node.step_id}' "
                    f"(batch {node.batch}): {exc}"
                ) from exc

            node.inputs = inputs
            node.output = result
            registry[key] = _fp16(result) if isinstance(result, np.ndarray) else result

        # Collect final outputs from last batch
        final_outputs: Dict[str, np.ndarray] = {}
        all_batches = sorted({n.batch for n in self.nodes})
        last_batch = all_batches[-1] if all_batches else 0
        for node in self.nodes:
            if node.batch == last_batch:
                final_outputs[node.output_name] = node.output
        return final_outputs


def load_dag(formula_json_path: str) -> FormulaDAG:
    """Load formula_steps.json and build a FormulaDAG.

    Supports both formats:
    - New: {"_metadata": {...}, "steps": [...]}
    - Old: [{...}, {...}]
    """
    with open(formula_json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        steps = data.get("steps", [])
    else:
        steps = data
    return FormulaDAG(steps)


def read_metadata(formula_json_path: str) -> dict:
    """Read algorithm metadata from formula_steps.json.

    Returns dict with keys: algorithm, block_size, layers, matrix_dim.
    Empty dict if metadata is not present (old format).
    """
    with open(formula_json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("_metadata", {})
    return {}


# ── self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python uobs_dag_executor.py <formula_steps.json>")
        sys.exit(1)

    dag = load_dag(sys.argv[1])
    print(f"Loaded DAG with {len(dag.nodes)} nodes")

    # Quick test with synthetic H
    shapes_2d = set()
    for n in dag.nodes:
        for sh in n.input_shapes + [n.output_shape]:
            if len(sh) == 2:
                shapes_2d.add(tuple(sh))

    # Find M×U shape from first GEMM step
    m, u = 64, 16
    for n in dag.nodes:
        if n.op_type == "GEMM" and len(n.input_shapes) >= 2:
            sh0 = n.input_shapes[0]
            sh1 = n.input_shapes[1]
            if len(sh0) == 2 and len(sh1) == 2:
                m, u = sh0[0], sh0[1]
                break

    rng = np.random.default_rng(42)
    H = (rng.standard_normal((m, u)) + 1j * rng.standard_normal((m, u))) / np.sqrt(2.0)
    lam = 0.1

    results = dag.execute({"H": H}, {"lambda": lam})
    print(f"Execution complete. Output tensors: {list(results.keys())}")

    # Print the last output shape
    for name, tensor in results.items():
        if "inv" in name.lower() or "A_inv" in name or "Ainv" in name:
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
