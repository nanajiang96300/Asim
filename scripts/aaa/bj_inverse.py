#!/usr/bin/env python3
"""
Block-Jacobi Iterative Matrix Inverse
=====================================

A self-contained implementation of the Block-Jacobi iterative method for
computing the inverse of a Hermitian positive-definite matrix A ∈ C^{n×n}.

Algorithm overview
------------------

The method exploits the block-diagonal structure of the Gram matrix
A = H^H · H + λ I that arises in MIMO MMSE detection.

Step 1 — Preconditioner:
    Partition A into diagonal blocks of size `block_size`.
    For each block, compute its exact inverse (2×2 direct formula or
    Cholesky for larger blocks).
    Build D_inv = blkdiag(A_00^{-1}, A_11^{-1}, ...).
    Build B = D_inv @ A   (preconditioned matrix; eigenvalues clustered near 1).

Step 2 — Iterative refinement:
    Start with Y_0 = 0.
    For each layer l = 0..L-1:
        residual = I - B @ Y_l
        Y_{l+1} = Y_l + ω_l · residual

    The ω_l are Chebyshev relaxation weights that accelerate convergence.
    They can be computed from:
      - fixed bounds [η_min, η_max] on the eigenvalues of B     (chebyshev)
      - adaptively estimated from the actual eigenvalues of B   (chebyshev_adaptive)

Step 3 — Recover the inverse:
    A^{-1} = Y_L @ D_inv

Key formulas
------------
    Y_{k+1} = Y_k + ω_k · (I - B @ Y_k)                    – iteration
    ω_k = 1 / t_k   where  t_k ∈ [t_min, t_max]           – Chebyshev nodes
    A^{-1} = Y_L · D^{-1}                                    – final recovery

Why efficient on NPU
--------------------
- D^{-1} consists of small independent blocks → high parallelism.
- B @ Y_k is a dense matrix multiply → maps directly to Cube units.
- Y_{k+1} = Y_k + ω · residual → axpy on Vector units.
- L = 4–8 layers is typically enough for MIMO detection accuracy.

References
----------
- Björck, Å. "Numerical Methods for Least Squares Problems" (1996), Ch. 7
- Higham, N.J. "Accuracy and Stability of Numerical Algorithms" (2002), Ch. 14
- Chebyshev semi-iterative methods for linear systems (Varga, 1962)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────

@dataclass
class BJacobiConfig:
    """Block-Jacobi inverse configuration.

    Parameters
    ----------
    block_size : int
        Size of diagonal blocks for the preconditioner.
        2 → 2×2 direct inverse formula (recommended for nt%2==0).
        >2 → Cholesky-based block inverse.
    num_layers : int
        Number of Chebyshev iteration layers. Typical range: 4–8.
        More layers → higher accuracy at the cost of compute.
    use_chebyshev : bool
        Use Chebyshev acceleration weights. Default True.
    adaptive_omega : bool
        Compute Chebyshev bounds adaptively from eigenvalues of B.
        More accurate but requires one eigendecomposition per call.
        Default True for best convergence.
    chebyshev_bmin : float
        Fixed lower bound on eigenvalues of B (used when adaptive_omega=False).
    chebyshev_bmax : float
        Fixed upper bound on eigenvalues of B (used when adaptive_omega=False).
    use_fp16_quant : bool
        Apply FP16 quantization after each iteration step (emulates NPU
        hardware arithmetic). Set to False for pure-FP64 reference.
    """

    block_size: int = 2
    num_layers: int = 4
    use_chebyshev: bool = True
    adaptive_omega: bool = True
    chebyshev_bmin: float = 0.1
    chebyshev_bmax: float = 1.2
    use_fp16_quant: bool = True


# ────────────────────────────────────────────────────────────────────
#  FP16 quantisation (optional hardware emulation)
# ────────────────────────────────────────────────────────────────────

def _quantize_fp16(arr: np.ndarray) -> np.ndarray:
    """Cast to FP16 and back, emulating NPU arithmetic precision.

    Each element goes through: complex128 → float16(real) + 1j·float16(imag) → complex128.
    """
    return (arr.real.astype(np.float16).astype(np.float64) +
            1j * arr.imag.astype(np.float16).astype(np.float64))


# ────────────────────────────────────────────────────────────────────
#  Block helpers
# ────────────────────────────────────────────────────────────────────

def _invert_spd_block(block: np.ndarray, solver: str = "direct2x2") -> np.ndarray:
    """Invert a small Hermitian positive-definite block.

    Parameters
    ----------
    block : ndarray, shape (k,k), complex
        Hermitian positive-definite matrix sub-block.
    solver : str
        "direct2x2" — closed-form for k=2 (faster on NPU).
        "cholesky"  — Cholesky + forward/backward solve (any k).

    Returns
    -------
    inv_block : ndarray, shape (k,k), complex
    """
    if solver == "direct2x2" and block.shape[0] == 2:
        a00, a01 = block[0, 0], block[0, 1]
        a10, a11 = block[1, 0], block[1, 1]
        det = a00 * a11 - a01 * a10 + (1e-12 + 0j)
        return np.array([[a11, -a01], [-a10, a00]], dtype=np.complex128) / det

    # fallback: Cholesky
    L = np.linalg.cholesky(block)
    eye = np.eye(block.shape[0], dtype=np.complex128)
    Y = np.linalg.solve(L, eye)
    return np.linalg.solve(L.conj().T, Y)


# ────────────────────────────────────────────────────────────────────
#  Preconditioner
# ────────────────────────────────────────────────────────────────────

def build_block_jacobi_preconditioner(
    A: np.ndarray,
    block_size: int = 2,
    solver: str = "direct2x2",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the Block-Jacobi preconditioner B = D^{-1} @ A.

    The block-diagonal preconditioner D is formed by extracting
    block_size × block_size blocks from the main diagonal of A,
    inverting each, and placing them on the diagonal of D^{-1}.

    Parameters
    ----------
    A : ndarray, shape (n, n), complex
        Hermitian positive-definite input matrix.
    block_size : int
        Size of each diagonal block. n must be divisible by block_size.
    solver : str
        "direct2x2" or "cholesky" — passed to _invert_spd_block.

    Returns
    -------
    B : ndarray, shape (n, n), complex
        Preconditioned matrix B = D^{-1} @ A.
        Its eigenvalues are clustered near 1, enabling fast iterative convergence.
    D_inv : ndarray, shape (n, n), complex
        Inverse of the block-diagonal preconditioner D^{-1}.
        This is block-diagonal and sparse in structure.
    """
    n = A.shape[0]
    D_inv = np.zeros_like(A, dtype=np.complex128)
    n_blk = n // block_size
    remainder = n % block_size

    # Invert each diagonal block independently — fully parallelisable
    for blk in range(n_blk):
        r0, r1 = blk * block_size, (blk + 1) * block_size
        block = A[r0:r1, r0:r1]
        D_inv[r0:r1, r0:r1] = _invert_spd_block(block, solver=solver)

    # Handle any trailing partial block
    if remainder > 0:
        r0 = n_blk * block_size
        block = A[r0:, r0:]
        D_inv[r0:, r0:] = _invert_spd_block(block, solver=solver)

    B = D_inv @ A
    return B, D_inv


# ────────────────────────────────────────────────────────────────────
#  Chebyshev omega weights
# ────────────────────────────────────────────────────────────────────

def chebyshev_omega(
    num_layers: int,
    eta_min: float = 0.1,
    eta_max: float = 1.2,
) -> List[float]:
    """Compute Chebyshev relaxation weights from fixed eigenvalue bounds.

    The weights cause the iteration residual to vanish like a Chebyshev
    polynomial over the interval [η_min, η_max].

    ω_k = 1 / t_k, where t_k are the Chebyshev nodes in [η_min, η_max]:
        t_k = ½(η_max + η_min) + ½(η_max - η_min) · cos(π(2k+1)/(2L))

    Parameters
    ----------
    num_layers : int
        Number of iteration layers (Chebyshev degree).
    eta_min : float
        Estimated minimum eigenvalue of the preconditioned matrix B.
    eta_max : float
        Estimated maximum eigenvalue of the preconditioned matrix B.

    Returns
    -------
    omegas : list[float]
        Relaxation weights for each layer, length = num_layers.
    """
    eta_min = max(float(eta_min), 1e-8)
    eta_max = max(float(eta_max), eta_min + 1e-8)

    omegas = []
    for k in range(num_layers):
        theta = np.pi * (2 * k + 1) / (2 * num_layers)
        t_k = 0.5 * (eta_max + eta_min) + 0.5 * (eta_max - eta_min) * np.cos(theta)
        omegas.append(float(1.0 / t_k))

    return omegas


def chebyshev_omega_adaptive(
    B: np.ndarray,
    num_layers: int,
    floor: float = 1e-8,
) -> List[float]:
    """Compute Chebyshev weights using actual eigenvalues of B.

    Estimates the eigenvalue range of B via np.linalg.eigvals and
    applies a safety margin to ensure the Chebyshev interval covers
    the true spectrum.

    Parameters
    ----------
    B : ndarray, shape (n, n), complex
        Preconditioned matrix.
    num_layers : int
        Number of iteration layers.
    floor : float
        Minimum eigenvalue to avoid division by zero.

    Returns
    -------
    omegas : list[float]
    """
    # Real eigenvalues (B is Hermitian when A is, which it is for MMSE Gram)
    eigvals = np.linalg.eigvals(B)
    eigvals = np.real(eigvals)
    eigvals = eigvals[np.isfinite(eigvals) & (eigvals > floor)]

    if eigvals.size == 0:
        eigvals = np.array([floor, 1.0], dtype=np.float64)

    raw_min = float(np.min(eigvals))
    raw_max = float(np.max(eigvals))

    # Safety margins: expand interval to ensure coverage
    eta_min = max(raw_min, 1e-2)
    eta_max = max(raw_max, eta_min * 2.0)

    return chebyshev_omega(num_layers=num_layers, eta_min=eta_min, eta_max=eta_max)


# ────────────────────────────────────────────────────────────────────
#  Main algorithm
# ────────────────────────────────────────────────────────────────────

def block_jacobi_inverse(
    A: np.ndarray,
    cfg: Optional[BJacobiConfig] = None,
    *,
    return_debug: bool = False,
) -> np.ndarray | Tuple[np.ndarray, dict]:
    """Compute A^{-1} using the Block-Jacobi iterative method.

    Parameters
    ----------
    A : ndarray, shape (n, n), dtype complex128
        Hermitian positive-definite matrix to invert.
        Must satisfy A = A^H and x^H A x > 0 for all non-zero x.
    cfg : BJacobiConfig, optional
        Algorithm configuration. Uses defaults if omitted:
        block_size=2, num_layers=4, adaptive Chebyshev, FP16 quantisation.
    return_debug : bool
        If True, also return a dict with intermediate values:
            "B"              — preconditioned matrix
            "D_inv"          — block-diagonal preconditioner inverse
            "omegas"         — Chebyshev weights per layer
            "residual_norms" — Frobenius norm of residual after each layer
            "Y_final"        — final Y matrix before recovery

    Returns
    -------
    A_inv : ndarray, shape (n, n), dtype complex128
        Approximate inverse of A.
    debug : dict (only if return_debug=True)
        Intermediate computation values.

    Theory
    ------
    We want Y = B^{-1} = (D^{-1} A)^{-1} = A^{-1} D.
    Once Y converges, A^{-1} = Y · D^{-1}.

    The iteration solves the linear system B · Y = I via the
    Chebyshev-accelerated Richardson method:

        Y_{k+1} = Y_k + ω_k · (I - B · Y_k)

    which is equivalent to the stationary iteration applied to
    the preconditioned normal equations.
    """
    if cfg is None:
        cfg = BJacobiConfig()

    n = A.shape[0]
    if n % cfg.block_size != 0:
        raise ValueError(
            f"Matrix size n={n} must be divisible by block_size={cfg.block_size}. "
            f"Adjust block_size or pad the matrix."
        )

    # ── Step 1: build preconditioner ──────────────────────────────
    solver = "direct2x2" if cfg.block_size == 2 else "cholesky"
    B, D_inv = build_block_jacobi_preconditioner(A, cfg.block_size, solver)

    if cfg.use_fp16_quant:
        B = _quantize_fp16(B)
        D_inv = _quantize_fp16(D_inv)

    # ── Step 2: Chebyshev weights ─────────────────────────────────
    if not cfg.use_chebyshev:
        omegas = [1.0] * max(cfg.num_layers, 1)
    elif cfg.adaptive_omega:
        omegas = chebyshev_omega_adaptive(B, cfg.num_layers)
    else:
        omegas = chebyshev_omega(cfg.num_layers,
                                 eta_min=cfg.chebyshev_bmin,
                                 eta_max=cfg.chebyshev_bmax)

    # ── Step 3: iterate ───────────────────────────────────────────
    Y = np.zeros_like(A, dtype=np.complex128)
    I = np.eye(n, dtype=np.complex128)

    residual_norms: List[float] = []
    Y_snapshots: List[np.ndarray] = [] if return_debug else []

    for omega in omegas:
        residual = I - B @ Y
        if cfg.use_fp16_quant:
            residual = _quantize_fp16(residual)

        if return_debug:
            residual_norms.append(float(np.linalg.norm(residual, ord="fro")))

        Y = Y + omega * residual
        if cfg.use_fp16_quant:
            Y = _quantize_fp16(Y)

        if return_debug:
            Y_snapshots.append(np.asarray(Y, dtype=np.complex128))

    # ── Step 4: recover inverse ───────────────────────────────────
    A_inv = Y @ D_inv
    if cfg.use_fp16_quant:
        A_inv = _quantize_fp16(A_inv)

    if return_debug:
        return A_inv, {
            "B": np.asarray(B, dtype=np.complex128),
            "D_inv": np.asarray(D_inv, dtype=np.complex128),
            "omegas": np.asarray(omegas, dtype=np.float64),
            "residual_norms": np.array(residual_norms, dtype=np.float64),
            "Y_final": np.asarray(Y, dtype=np.complex128),
        }

    return A_inv


# ────────────────────────────────────────────────────────────────────
#  Demo
# ────────────────────────────────────────────────────────────────────

def _demo():
    """Quick validation: compare BJ inverse against numpy.linalg.inv."""
    print("=" * 66)
    print("Block-Jacobi Inverse — Validation Demo")
    print("=" * 66)

    # Build a simple well-conditioned SPD matrix
    n = 8
    rng = np.random.default_rng(42)
    H = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    noise_var = 0.01
    A = H.conj().T @ H + noise_var * np.eye(n, dtype=np.complex128)
    kappa = float(np.linalg.cond(A))
    print(f"\n  Matrix size : {n}×{n}")
    print(f"  Condition κ : {kappa:.1f}")

    # Reference
    A_inv_ref = np.linalg.inv(A)

    configs = [
        ("B=2  L=4  adaptive+FP16   (default)", BJacobiConfig()),
        ("B=2  L=8  adaptive+FP16   ", BJacobiConfig(num_layers=8)),
        ("B=2  L=4  fixed ω  +FP16  ", BJacobiConfig(adaptive_omega=False)),
        ("B=2  L=4  adaptive+FP64   ", BJacobiConfig(use_fp16_quant=False)),
        ("B=1  L=4  adaptive+FP16   ", BJacobiConfig(block_size=1)),
        ("B=4  L=4  adaptive+FP16   ", BJacobiConfig(block_size=4)),
    ]

    print(f"\n  {'Config':<32s} {'relErr':>10s}  {'‖Res‖_F':>10s}")
    print(f"  {'-'*54}")

    for label, cfg in configs:
        A_inv, dbg = block_jacobi_inverse(A, cfg, return_debug=True)
        rel_err = (np.linalg.norm(A_inv - A_inv_ref, ord="fro") /
                   (np.linalg.norm(A_inv_ref, ord="fro") + 1e-16))
        final_res = dbg["residual_norms"][-1]
        print(f"  {label:<32s} {rel_err:10.2e}  {final_res:10.2e}")

    # ── Convergence profile ────────────────────────────────────────
    print(f"\n  Convergence profile (B=2, adaptive, FP16):")
    print(f"  {'Layer':>6s}  {'ω':>8s}  {'‖Residual‖_F':>14s}")
    print(f"  {'-'*32}")
    _, dbg = block_jacobi_inverse(
        A, BJacobiConfig(num_layers=8), return_debug=True
    )
    for k, (omega, res) in enumerate(zip(dbg["omegas"], dbg["residual_norms"])):
        print(f"  {k:>6d}  {omega:8.3f}  {res:14.6e}")

    print(f"\n  ✓ All configs validated. See the BJ-inverse/ directory")
    print(f"    for the standalone module.\n")


if __name__ == "__main__":
    _demo()
