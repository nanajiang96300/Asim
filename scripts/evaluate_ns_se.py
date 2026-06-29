#!/usr/bin/env python3
"""
Newton-Schulz SE evaluation and comparison with Block-Jacobi.
Computes SE-vs-SNR curves for NS at varying iterations (K=1..10),
compares against BJ baseline (from project data).
"""
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_ldl_quality import EvalConfig, estimate_se, quantize_complex
from scripts.reconstruct_formula_se_compare import (
    block_richardson_formula_inverse, FormulaModelMeta, build_block_richardson_preconditioner,
    chebyshev_omega_adaptive,
)

np.random.seed(42)


def newton_schulz_inverse(a_mat, iters=5, dtype=np.complex64):
    """Newton-Schulz inverse: X_0 = I/||A||_2, X_{k+1} = X_k(2I - A X_k).

    Uses X_0 = I / spectral_norm(A) as the initial guess, which guarantees
    convergence for any Hermitian positive-definite matrix (including MIMO
    Gram matrices H^H·H + λI with moderate condition numbers).

    The previously used X_0 = α·A^H (α = 1/(||A||_1·||A||_∞)) fails for
    MIMO channel matrices because ||A||_1·||A||_∞ can be O(10^4-10^6),
    making α too small, causing the iteration to diverge.
    """
    n = a_mat.shape[0]
    spec_norm = max(np.linalg.norm(a_mat, 2), 1e-12)
    a = a_mat.astype(dtype)
    x = (np.eye(n, dtype=dtype) / spec_norm)
    i2 = (2.0 * np.eye(n, dtype=dtype))
    for k in range(iters):
        x = x @ (i2 - a @ x)
    return x


def ns_se(h, noise_var, iters, ecfg):
    """Compute SE for Newton-Schulz inverse at given iterations."""
    nt = h.shape[1]
    a = quantize_complex(h.conj().T @ h + noise_var * np.eye(nt), ecfg)
    a_np = a  # already quantized
    a_inv = newton_schulz_inverse(a_np, iters=iters, dtype=np.complex64)
    a_inv = quantize_complex(a_inv, ecfg)
    w = quantize_complex(a_inv @ h.conj().T, ecfg)
    return estimate_se(w, h, noise_var, "64qam")


def bj_se_ref(h, noise_var, B, L, ecfg):
    """Compute SE for Block-Jacobi (reference)."""
    nt = h.shape[1]
    a = quantize_complex(h.conj().T @ h + noise_var * np.eye(nt), ecfg)
    bm, bi = build_block_richardson_preconditioner(a, blk=B,
        precond_solver="direct2x2" if B == 2 else "cholesky")
    y = np.zeros_like(a, dtype=complex)
    I = np.eye(nt, dtype=complex)
    for w in chebyshev_omega_adaptive(bm, L, nt=nt):
        y = quantize_complex(y + w * (I - bm @ y), ecfg)
    ai = quantize_complex(y @ bi, ecfg)
    return estimate_se(quantize_complex(ai @ h.conj().T, ecfg), h, noise_var, "64qam")


def ec(nr, nt):
    return EvalConfig(nr=nr, nt=nt, n_sc=1, batch=1, trials=1, block_size=2,
        snr_db_list=[0.0], channel_model="rayleigh", pilot_len=nt, pilot_snr_db=None,
        num_format="float16", reciprocal_mode="lut", trunc_mantissa_bits=10,
        modulation="64qam", mac_chunk=1, seed=42, out_dir="/tmp")


def main():
    dims = [(64, 16, 16, [2, 4, 8, 16]),
            (128, 32, 32, [2, 4, 8, 16, 32]),
            (256, 64, 64, [2, 4, 8, 16, 32, 64])]

    snr_list = [0, 5, 10, 15, 20, 25, 30]
    ns_iters_list = [1, 2, 3, 4, 5, 6, 8, 10]
    trials = 10  # more trials for stable SE

    t0 = time.time()

    for nr, nt, U, Bs in dims:
        cap = U * 6
        print(f"\n{'=' * 70}")
        print(f"U={U} ({nr}x{nt}), capacity={cap}, trials={trials}")
        print(f"{'=' * 70}")

        for snr_db in snr_list:
            noise_var = 10 ** (-snr_db / 10)
            results_ns = {k: [] for k in ns_iters_list}
            results_bj = {}

            for t in range(trials):
                h = (np.random.randn(nr, nt) + 1j * np.random.randn(nr, nt)) / np.sqrt(2)

                # Newton-Schulz at varying iterations
                for k in ns_iters_list:
                    results_ns[k].append(ns_se(h, noise_var, k, ec(nr, nt)))

                # BJ baselines (only first SNR to find best)
                if t == 0 or snr_db >= 15:
                    for B in Bs:
                        if B not in results_bj:
                            results_bj[B] = {}
                        for L in [1, 2, 3, 4, 6, 8]:
                            if L in results_bj[B]:
                                continue
                            try:
                                results_bj[B][L] = bj_se_ref(h, noise_var, B, L, ec(nr, nt))
                            except Exception:
                                results_bj[B][L] = 0

            # Print NS results
            ns_means = {k: np.mean(v) for k, v in results_ns.items()}
            ns_str = "  ".join(f"K={k}:{ns_means[k]:.1f}" for k in ns_iters_list)
            print(f"  SNR={snr_db:2d}dB | NS: {ns_str}")

            # Find minimum K to reach cap-0.5
            for k in ns_iters_list:
                if ns_means[k] >= cap - 0.5:
                    print(f"    NS reaches cap at K={k} (SE={ns_means[k]:.1f})")
                    break

        # Print BJ baseline snapshot
        print(f"  BJ baselines (SNR=20dB, best per B):")
        for B in Bs:
            best_L = None
            best_se = 0
            for L, se in results_bj.get(B, {}).items():
                if se > best_se:
                    best_se = se
                    best_L = L
            if best_L:
                print(f"    B={B}: L={best_L} SE={best_se:.1f}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
