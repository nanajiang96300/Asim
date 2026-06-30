#!/usr/bin/env python3
"""End-to-end verification: Python Cholesky/LDL NoBlock vs numpy.linalg.inv."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from channel import RayleighChannel, CDLBChannel
from algo import cholesky_noblock_inverse, ldl_noblock_inverse


def _fp16_complex(x):
    """Quantize real and imag separately to FP16."""
    real = x.real.astype(np.float16).astype(np.float64)
    imag = x.imag.astype(np.float16).astype(np.float64)
    return real + 1j * imag


def gram_and_regularize(H, lam=0.1):
    """Compute A = H^H @ H + lambda * I (FP16 at each step)."""
    B, M, U = H.shape
    A = np.zeros((B, U, U), dtype=np.complex128)
    for b in range(B):
        G = H[b].conj().T @ H[b]
        G = _fp16_complex(G)
        for i in range(U):
            G[i, i] = G[i, i].real + lam
        A[b] = G
    return A


def relative_error(A_est, A_ref):
    """Mean relative Frobenius error across batches."""
    B = A_ref.shape[0]
    errs = []
    for b in range(B):
        num = np.linalg.norm(A_est[b] - A_ref[b], 'fro')
        den = np.linalg.norm(A_ref[b], 'fro')
        errs.append(num / max(den, 1e-15))
    return float(np.mean(errs))


def run_verification(batch_size=96, nr=64, nt=16, lam=0.1, noise_power=0.1):
    print("=" * 60)
    print("Asim NoBlock Baseline Verification")
    print(f"  Batch={batch_size}, nr={nr}, nt={nt}, lambda={lam}")
    print("=" * 60)

    results = {}
    for ch_name, ch_gen in [
        ("Rayleigh", RayleighChannel()),
        ("CDL-B", CDLBChannel()),
    ]:
        print(f"\n--- {ch_name} Channel ---")
        H = ch_gen.generate(batch_size, nr, nt, seed=42)
        A = gram_and_regularize(H, lam)
        A_ref = np.linalg.inv(A)

        # Test Cholesky NoBlock
        A_chol = np.zeros_like(A_ref)
        for b in range(batch_size):
            A_chol[b] = cholesky_noblock_inverse(A[b].copy())
        err_chol = relative_error(A_chol, A_ref)

        # Test LDL NoBlock
        A_ldl = np.zeros_like(A_ref)
        for b in range(batch_size):
            A_ldl[b] = ldl_noblock_inverse(A[b].copy())
        err_ldl = relative_error(A_ldl, A_ref)
        err_cross = relative_error(A_chol, A_ldl)

        # SE comparison (simplified: use sum-rate)
        se_chol = 0.0
        se_ref = 0.0
        for b in range(batch_size):
            W = A_chol[b] @ H[b].conj().T
            for i in range(nt):
                w = W[i]
                sig = np.abs(w @ H[b, :, i]) ** 2
                interf = sum(np.abs(w @ H[b, :, j]) ** 2 for j in range(nt) if j != i)
                n = noise_power * np.sum(np.abs(w) ** 2)
                se_chol += np.log2(1 + sig / max(interf + n, 1e-15))
        se_chol /= batch_size

        for b in range(batch_size):
            W_ref = A_ref[b] @ H[b].conj().T
            for i in range(nt):
                w = W_ref[i]
                sig = np.abs(w @ H[b, :, i]) ** 2
                interf = sum(np.abs(w @ H[b, :, j]) ** 2 for j in range(nt) if j != i)
                n = noise_power * np.sum(np.abs(w) ** 2)
                se_ref += np.log2(1 + sig / max(interf + n, 1e-15))
        se_ref /= batch_size

        print(f"  RelErr Chol vs Ref:  {err_chol:.6e}")
        print(f"  RelErr LDL vs Ref:   {err_ldl:.6e}")
        print(f"  RelErr Chol vs LDL:  {err_cross:.6e}")
        print(f"  SE Chol: {se_chol:.4f}  SE Ref: {se_ref:.4f}  Δ={se_chol-se_ref:+.4f}")

        passed = (err_chol < 0.01 and err_ldl < 0.01 and err_cross < 0.02)
        print(f"  Status: {'PASS' if passed else 'FAIL'}")
        results[ch_name] = {"passed": passed, "err_chol": err_chol, "err_ldl": err_ldl}

    print("\n" + "=" * 60)
    all_pass = all(r["passed"] for r in results.values())
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_verification())
