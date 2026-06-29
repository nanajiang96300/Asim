#!/usr/bin/env python3
"""
Unified Reference Inverse Registry — operator-agnostic SE verification.

Replaces the hardcoded if-elif chain in compute_reference_inverse() with
a registry that maps algorithm names to reference inverse functions. New
algorithms can be added by registering their function here without
modifying uobs_scorer.py.

Usage:
    from scripts.reference_inverse_registry import compute_reference_inverse
    a_inv = compute_reference_inverse(a_mat, algo_identity, formula_json_path=None)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ldl_quality import EvalConfig
from scripts.reconstruct_formula_se_compare import (
    FormulaModelMeta,
    block_richardson_formula_inverse,
    cholesky_formula_inverse,
    ldl_inverse,
    ldl_noblock_formula_inverse,
)
from scripts.evaluate_ns_se import newton_schulz_inverse
from scripts.uobs_dag_executor import FormulaDAG, load_dag, read_metadata


# ── Algorithm identity (compatible with uobs_scorer.AlgorithmIdentity) ─────
class AlgorithmIdentity:
    """Minimal identity struct for reference inverse dispatch."""
    name: str
    block_size: int
    layers: int
    has_adaptive_omega: bool

    def __init__(self, name: str, block_size: int = 2, layers: int = 16,
                 has_adaptive_omega: bool = False):
        self.name = name
        self.block_size = block_size
        self.layers = layers
        self.has_adaptive_omega = has_adaptive_omega


# ── Reference function signature ──────────────────────────────────────────
#   fn(a_mat: np.ndarray, cfg: EvalConfig, **algo_params) -> np.ndarray
ReferenceInverseFn = Callable[..., np.ndarray]


# ── Registry ──────────────────────────────────────────────────────────────
_registry: Dict[str, ReferenceInverseFn] = {}


def register(name: str):
    """Decorator to register a reference inverse function for an algorithm."""
    def decorator(fn: ReferenceInverseFn):
        _registry[name] = fn
        return fn
    return decorator


# ── Built-in reference implementations ────────────────────────────────────

def _make_cfg(nt: int, block_size: int = 2, nr: int = 64) -> EvalConfig:
    return EvalConfig(
        nr=nr, nt=nt, n_sc=1, batch=1, trials=1, block_size=block_size,
        snr_db_list=[10], channel_model="rayleigh", pilot_len=16,
        pilot_snr_db=10.0, num_format="fp16", reciprocal_mode="exact",
        trunc_mantissa_bits=8, modulation="64qam", mac_chunk=4,
        seed=42, out_dir="/tmp")


@register("cholesky_block")
def _chol_block(a_mat: np.ndarray, cfg: EvalConfig, **kwargs) -> np.ndarray:
    return cholesky_formula_inverse(a_mat, cfg)


@register("cholesky_noblock")
def _chol_noblock(a_mat: np.ndarray, cfg: EvalConfig, **kwargs) -> np.ndarray:
    return cholesky_formula_inverse(a_mat, cfg)


@register("ldl_block")
def _ldl_block(a_mat: np.ndarray, cfg: EvalConfig, **kwargs) -> np.ndarray:
    bs = kwargs.get("block_size", 2)
    a_inv, _ = ldl_inverse(a_mat, cfg, block_size=bs)
    return a_inv


@register("ldl_noblock")
def _ldl_noblock(a_mat: np.ndarray, cfg: EvalConfig, **kwargs) -> np.ndarray:
    return ldl_noblock_formula_inverse(a_mat, cfg)


@register("block_richardson")
def _bj(a_mat: np.ndarray, cfg: EvalConfig, **kwargs) -> np.ndarray:
    meta = FormulaModelMeta(
        name="block_richardson", rows=[], by_event={},
        inferred_layers=kwargs.get("layers", 16),
        inferred_block_size=kwargs.get("block_size", 2),
        adaptive_bounds=kwargs.get("has_adaptive_omega", False),
        use_iter_weight=kwargs.get("has_adaptive_omega", False))
    return block_richardson_formula_inverse(a_mat, meta, cfg)


@register("newton_schulz")
def _ns(a_mat: np.ndarray, cfg: EvalConfig, **kwargs) -> np.ndarray:
    iters = max(kwargs.get("layers", 10), 15)
    return newton_schulz_inverse(a_mat, iters=iters, dtype=np.complex64)


# ── Unified dispatch ──────────────────────────────────────────────────────

def compute_reference_inverse(
    a_mat: np.ndarray,
    algo: Any,       # AlgorithmIdentity (from uobs_scorer or this module)
    formula_json_path: Optional[str] = None,
) -> np.ndarray:
    """Compute the reference inverse for an identified algorithm.

    Args:
        a_mat: Input matrix A (H^H·H + λI) in complex128.
        algo: AlgorithmIdentity with name, block_size, layers, etc.
        formula_json_path: Optional path to formula_steps.json. If provided
            and the DAG executor can successfully reconstruct the inverse,
            it is used for a truly operator-agnostic verification.
            Otherwise the registered per-algorithm function is used.

    Returns:
        Reconstructed A^{-1} as a numpy array.

    Raises:
        ValueError: if the algorithm is not registered.
    """
    name = getattr(algo, 'name', str(algo))
    block_size = getattr(algo, 'block_size', 2)
    cfg = _make_cfg(a_mat.shape[0], block_size,
                    nr=a_mat.shape[0] * 4)  # nr = 4 * nt per MIMO convention

    # If formula JSON is available, read metadata to double-check algorithm identity
    if formula_json_path is not None:
        try:
            meta = read_metadata(formula_json_path)
            if meta:
                meta_name = meta.get("algorithm", "")
                if meta_name and meta_name != name:
                    # Metadata overrides pattern-based identification
                    name = meta_name
                    block_size = meta.get("block_size", block_size) or block_size
        except Exception:
            pass

    # NOTE: DAG-based reconstruction is disabled for now.
    # The DAG executor (FormulaDAG) cannot currently handle multi-batch
    # formula steps with correct tensor shape propagation.  It produces
    # wrong-shaped outputs that pass the `is not None` check but are
    # numerically invalid (e.g. norm ~300 instead of ~0.1).
    #
    # TODO: Re-enable DAG path once the following are fixed:
    #   1. Per-batch tensor isolation (H should be (M,U) not (U,U))
    #   2. Intermediate tensor name resolution across all sub-steps
    #   3. Output shape validation (must match expected matrix dimension)
    #
    # When re-enabled, the code is:
    #   if formula_json_path is not None:
    #       try:
    #           dag = load_dag(formula_json_path)
    #           a_inv = _execute_dag_inverse(dag, a_mat, name)
    #           if a_inv is not None and a_inv.shape == (a_mat.shape[0], a_mat.shape[0]):
    #               return a_inv
    #       except Exception:
    #           pass

    # Use registered per-algorithm function
    fn = _registry.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown algorithm: '{name}'. Registered algorithms: "
            f"{sorted(_registry.keys())}. "
            f"To add a new algorithm, use @register('{name}')."
        )

    kwargs = {
        "block_size": getattr(algo, 'block_size', 2),
        "layers": getattr(algo, 'layers', 16),
        "has_adaptive_omega": getattr(algo, 'has_adaptive_omega', False),
    }
    return fn(a_mat, cfg, **kwargs)


def _execute_dag_inverse(dag: FormulaDAG, a_mat: np.ndarray,
                         algo_name: str) -> Optional[np.ndarray]:
    """Try to reconstruct the inverse using the DAG executor.

    Returns None if reconstruction fails (missing intermediates, shape
    mismatches, etc.), signalling the caller to fall back to the registered
    per-algorithm function.
    """
    n = a_mat.shape[0]
    # The DAG needs initial tensors for all leaf inputs.
    # Common inputs for communication operators:
    #   "H" — channel matrix (M×U)
    #   "lambda*I" — regularization term
    #   "I" — identity matrix
    initial: Dict[str, np.ndarray] = {
        "H": a_mat,              # H^H·H will be computed from H
        "I": np.eye(n, dtype=np.complex128),
        "lambda*I": np.eye(n, dtype=np.complex128) * 0.1,  # placeholder
    }
    aux = {"lambda": 0.1}

    results = dag.execute(initial, aux)

    # Look for the output inverse matrix
    for key in results:
        if any(pat in key.lower() for pat in ['inv', 'a_inv', 'ainv', 'x_']):
            return np.asarray(results[key], dtype=np.complex64)

    # Fallback: return the last produced tensor
    if results:
        return np.asarray(list(results.values())[-1], dtype=np.complex64)

    return None


# ── Registry introspection ────────────────────────────────────────────────

def list_algorithms() -> list:
    """Return sorted list of registered algorithm names."""
    return sorted(_registry.keys())


def is_registered(name: str) -> bool:
    """Check if an algorithm name is registered."""
    return name in _registry


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Reference Inverse Registry")
    print(f"Registered algorithms: {list_algorithms()}")
    print()

    # Quick test with a 4x4 random matrix
    rng = np.random.default_rng(123)
    n = 4
    h = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    a = h.conj().T @ h + np.eye(n)
    a_ref = np.linalg.inv(a)

    for name in list_algorithms():
        algo = AlgorithmIdentity(name=name,
                                 block_size=2 if "noblock" not in name else 1,
                                 layers=8 if name != "newton_schulz" else 10)
        try:
            a_inv = compute_reference_inverse(a, algo)
            err = np.linalg.norm(np.asarray(a_inv, dtype=np.complex128) - a_ref)
            err /= max(np.linalg.norm(a_ref), 1e-12)
            status = "✅" if err < 0.01 else "❌"
            print(f"  {name:20s} err={err:.2e} {status}")
        except Exception as e:
            print(f"  {name:20s} FAIL: {e}")
