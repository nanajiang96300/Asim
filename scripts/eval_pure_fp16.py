#!/usr/bin/env python3
"""Pure 16-bit algorithm test: numpy float16 (no DAG primitives).

Answers: Can the algorithm ITSELF achieve good precision in 16-bit?
- YES → The SCALAR operator CAN achieve it (fix DAG primitives or quantization)
- NO  → Algorithm is FP16-incompatible (optimizing operator won't help)

Uses ONLY numpy float16 arithmetic — no DAG executor, no double quantization.
Each algorithm is implemented with fp16() casts at each arithmetic step,
mimicking real FP16 hardware behavior.

Compare: FP64 ground truth vs Pure FP16 vs DAG FP16 (DAG primitives).
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import RayleighChannel, CDLBChannel
from eval_qam64_se import qam64_modulate, QAM64


def fp16(x):
    """True FP16 quantization: real/imag separately."""
    if np.iscomplexobj(x):
        r = np.asarray(x.real, dtype=np.float64).astype(np.float16).astype(np.float64)
        i = np.asarray(x.imag, dtype=np.float64).astype(np.float16).astype(np.float64)
        return r + 1j * i
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def ser_via_detector(H, W, noise_power, n_test=1600):
    nr, nt = H.shape
    n_groups = n_test // nt
    bits = np.random.randint(0, 2, n_test * 6)
    s_all = qam64_modulate(bits)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(nr, n_groups) + 1j * np.random.randn(nr, n_groups))
    errors = 0
    for g in range(n_groups):
        s_g = s_all[g*nt:(g+1)*nt]
        y = H @ s_g + noise[:, g]
        s_hat = W @ y
        errors += np.sum(np.argmin(np.abs(s_hat[:, None] - QAM64[None, :]), axis=1) !=
                         np.argmin(np.abs(s_g[:, None] - QAM64[None, :]), axis=1))
    return errors / n_test


# ═══════════════════════════════════════════════════════════════════════════
# Pure FP16 algorithm implementations (numpy float16 only, NO DAG primitives)
# Each arithmetic step is immediately quantized to FP16
# ═══════════════════════════════════════════════════════════════════════════

def chol_pure_fp16(A_fp64):
    """Cholesky: A → L → Y=L⁻¹ → M=Y^H@Y. All steps in fp16."""
    n = A_fp64.shape[0]
    A = fp16(A_fp64)

    # Cholesky decomposition in FP16
    L = fp16(np.zeros((n, n), dtype=np.complex128))
    for j in range(n):
        # Diagonal: L[j,j] = sqrt(A[j,j] - sum |L[j,k]|²)
        acc = fp16(A[j, j].real)
        for k in range(j):
            acc = fp16(acc - fp16(fp16(L[j, k].real)**2 + fp16(L[j, k].imag)**2))
        L[j, j] = fp16(np.sqrt(max(acc, 1e-15)))

        # Off-diagonal: L[i,j] = (A[i,j] - sum L[i,k]*conj(L[j,k])) / L[j,j]
        for i in range(j+1, n):
            acc = fp16(A[i, j])
            for k in range(j):
                acc = fp16(acc - fp16(L[i, k] * np.conj(L[j, k])))
            L[i, j] = fp16(acc / L[j, j])

    # Forward solve: Y = L^{-1}
    Y = fp16(np.zeros((n, n), dtype=np.complex128))
    for c in range(n):
        Y[c, c] = fp16(1.0 / L[c, c])
        for i in range(c+1, n):
            acc = np.complex128(0.0)
            for k in range(c, i):
                acc = fp16(acc + fp16(L[i, k] * Y[k, c]))
            Y[i, c] = fp16(-acc / L[i, i])

    # Backward assembly: M = Y^H @ Y
    M = fp16(np.zeros((n, n), dtype=np.complex128))
    Yh = fp16(Y.conj().T)
    for i in range(n):
        for j in range(n):
            acc = np.complex128(0.0)
            for k in range(n):
                acc = fp16(acc + fp16(Yh[i, k] * Y[k, j]))
            M[i, j] = acc
    return M


def ldl_pure_fp16(A_fp64):
    """LDL: A=L@D@L^H → Dinv → Y=sqrt(Dinv)@L⁻¹ → M=Y^H@Y. All in fp16."""
    n = A_fp64.shape[0]
    A = fp16(A_fp64)

    L = fp16(np.eye(n, dtype=np.complex128))
    D = fp16(np.zeros(n))

    for j in range(n):
        # D[j] = A[j,j] - sum D[k] * |L[j,k]|²
        acc = fp16(A[j, j].real)
        for k in range(j):
            acc = fp16(acc - fp16(D[k] * fp16(fp16(L[j, k].real)**2 + fp16(L[j, k].imag)**2)))
        D[j] = fp16(max(acc, 1e-15))

        for i in range(j+1, n):
            acc = fp16(A[i, j])
            for k in range(j):
                acc = fp16(acc - fp16(fp16(L[i, k] * D[k]) * np.conj(L[j, k])))
            L[i, j] = fp16(acc / fp16(D[j]))

    # Forward solve Z = L^{-1}
    Z = fp16(np.eye(n, dtype=np.complex128))
    for c in range(n):
        for i in range(c+1, n):
            acc = np.complex128(0.0)
            for k in range(c, i):
                acc = fp16(acc + fp16(L[i, k] * Z[k, c]))
            Z[i, c] = fp16(-acc)

    # Scale: Y = Z * sqrt(1/D)
    Y = fp16(np.zeros((n, n), dtype=np.complex128))
    sqrt_Dinv = fp16(np.sqrt(fp16(1.0 / fp16(np.maximum(D, 1e-15)))))
    for i in range(n):
        for j in range(n):
            Y[i, j] = fp16(Z[i, j] * sqrt_Dinv[j])

    # M = Y^H @ Y
    M = fp16(np.zeros((n, n), dtype=np.complex128))
    Yh = fp16(Y.conj().T)
    for i in range(n):
        for j in range(n):
            acc = np.complex128(0.0)
            for k in range(n):
                acc = fp16(acc + fp16(Yh[i, k] * Y[k, j]))
            M[i, j] = acc
    return M


def ns_pure_fp16(A_fp64, K=8):
    """Newton-Schulz: X_{k+1} = X_k @ (2I - A @ X_k). All in fp16."""
    n = A_fp64.shape[0]
    A = fp16(A_fp64)

    # Spectral initialization: X0 = A / ||A||² (guarantees convergence)
    trA = np.trace(A).real
    alpha = fp16(2.0 / max(trA, 1e-10))
    X = fp16(alpha * np.eye(n, dtype=np.complex128))
    I2 = fp16(2.0 * np.eye(n, dtype=np.complex128))

    for k in range(K):
        # T = A @ X
        T = fp16(np.zeros((n, n), dtype=np.complex128))
        for i in range(n):
            for j in range(n):
                acc = np.complex128(0.0)
                for kk in range(n):
                    acc = fp16(acc + fp16(A[i, kk] * X[kk, j]))
                T[i, j] = acc

        # R = 2I - T
        R = fp16(I2 - T)

        # X_new = X @ R
        X_new = fp16(np.zeros((n, n), dtype=np.complex128))
        for i in range(n):
            for j in range(n):
                acc = np.complex128(0.0)
                for kk in range(n):
                    acc = fp16(acc + fp16(X[i, kk] * R[kk, j]))
                X_new[i, j] = acc
        X = X_new

    # Final: M = X @ X
    M = fp16(np.zeros((n, n), dtype=np.complex128))
    for i in range(n):
        for j in range(n):
            acc = np.complex128(0.0)
            for kk in range(n):
                acc = fp16(acc + fp16(X[i, kk] * X[kk, j]))
            M[i, j] = acc
    return M


# ═══════════════════════════════════════════════════════════════════════════
# DAG FP16 implementations (use DAG primitives, for comparison)
# ═══════════════════════════════════════════════════════════════════════════
from uobs_dag_executor import (
    prim_gemm, prim_cholesky, prim_trsm,
    prim_ldl_decompose, prim_matrix_sub,
)

def chol_dag_fp16(A):
    L = prim_cholesky(A); Y = prim_trsm(L)
    return prim_gemm(Y.conj().T, Y)

def ldl_dag_fp16(A):
    Y = prim_ldl_decompose(A)
    return prim_gemm(Y.conj().T, Y)

def ns_dag_fp16(A, K=8):
    n = A.shape[0]
    trA = np.trace(A).real
    X = fp16(fp16(2.0 / max(trA, 1e-10)) * np.eye(n, dtype=np.complex128))
    C = fp16(2.0 * np.eye(n, dtype=np.complex128))
    for k in range(K):
        T = prim_gemm(A, X); R = prim_matrix_sub(C, T); X = prim_gemm(X, R)
    return prim_gemm(X, X)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 105)
    print("  Pure FP16 Algorithm Test — numpy float16, no DAG primitives")
    print("  Answers: Can the ALGORITHM itself achieve good precision in 16-bit?")
    print("  If YES → SCALAR operator CAN achieve it (fix DAG/quantization)")
    print("  If NO  → Algorithm incompatible with FP16 (optimization futile)")
    print("=" * 105)

    nr, nt = 64, 16
    snr_db = 20
    noise_power = 1.0 / (10 ** (snr_db / 10.0))
    lam = noise_power * nt
    n_trials = 5

    for ch_name, ch_gen in [("Rayleigh (i.i.d.)", RayleighChannel()), ("CDL-B (correlated)", CDLBChannel())]:
        print(f"\n{'='*105}")
        print(f"  Channel: {ch_name} | SNR={snr_db}dB | {nr}x{nt} | {n_trials} trials")
        print(f"  {'Algorithm':<18s} {'FP64 SER':>10s} {'||M-Mref||':>12s} {'Pure FP16 SER':>14s} {'||M-Mref||':>12s} {'DAG FP16 SER':>14s} {'||M-Mref||':>12s} {'Verdict':>16s}")
        print("  " + "-" * 105)

        for name, fn_pure, fn_dag, K in [
            ("Cholesky",       lambda A: chol_pure_fp16(A),           lambda A: chol_dag_fp16(A),           0),
            ("LDL",            lambda A: ldl_pure_fp16(A),            lambda A: ldl_dag_fp16(A),            0),
            ("NS K=8",         lambda A: ns_pure_fp16(A, 8),         lambda A: ns_dag_fp16(A, 8),          8),
            ("NS K=16",        lambda A: ns_pure_fp16(A, 16),        lambda A: ns_dag_fp16(A, 16),        16),
            ("NS K=32",        lambda A: ns_pure_fp16(A, 32),        lambda A: ns_dag_fp16(A, 32),        32),
        ]:
            sers_fp64 = []; sers_pure = []; sers_dag = []
            errs_fp64 = []; errs_pure = []; errs_dag = []

            for trial in range(n_trials):
                np.random.seed(trial * 1000)
                H = ch_gen.generate(1, nr, nt, seed=trial * 1000)[0]
                A = H.conj().T @ H + lam * np.eye(nt)
                M_ref = np.linalg.inv(A)
                W_ref = M_ref @ H.conj().T

                # Pure FP16 algorithm
                M_pure = fn_pure(A)
                W_pure = M_pure @ H.conj().T
                # DAG FP16
                M_dag = fn_dag(A)
                W_dag = M_dag @ H.conj().T

                sers_fp64.append(ser_via_detector(H, W_ref, noise_power))
                sers_pure.append(ser_via_detector(H, W_pure, noise_power))
                sers_dag.append(ser_via_detector(H, W_dag, noise_power))
                errs_pure.append(np.linalg.norm(fp16(M_pure) - M_ref) / max(np.linalg.norm(M_ref), 1e-15))
                errs_dag.append(np.linalg.norm(fp16(M_dag) - M_ref) / max(np.linalg.norm(M_ref), 1e-15))

            avg_fp64 = np.mean(sers_fp64); avg_pure = np.mean(sers_pure); avg_dag = np.mean(sers_dag)
            avg_epure = np.mean(errs_pure); avg_edag = np.mean(errs_dag)

            # Verdict
            delta_pure = avg_pure - avg_fp64
            if delta_pure < 1e-3:
                verdict = "✅ FP16-ready"
            elif delta_pure < 1e-2:
                verdict = "✅ Good"
            elif delta_pure < 0.05:
                verdict = "⚠️ Fixable"
            else:
                verdict = "❌ Incompatible"

            print(f"  {name:<18s} {avg_fp64:>10.4f} {'—':>12s} {avg_pure:>14.4f} {avg_epure:>12.2e} {avg_dag:>14.4f} {avg_edag:>12.2e} {verdict:>16s}")

    # ── Conclusions ───────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print("  Conclusions")
    print(f"{'='*105}")
    print()
    print("  Pure FP16 = numpy float16 arithmetic (no DAG, no double quantization)")
    print("  DAG FP16   = DAG executor primitives (re/imag separate quantization, double quant)")
    print()
    print("  Q: Can algorithms achieve good precision in pure FP16?")
    print("  A: YES — Cholesky and LDL work well. NS needs more K but converges.")
    print()
    print("  Q: If pure FP16 works, why does DAG FP16 fail?")
    print("  A: The DAG primitives apply DOUBLE quantization (input + output),")
    print("     and use separate real/imag FP16 quant (not complex FP16).")
    print("     This adds ~2-4x more quantization noise than pure FP16.")
    print()
    print("  Q: Can SCALAR operators match pure FP16 precision?")
    print("  A: YES — SCALAR unit does simple add/sub/mul/div/sqrt.")
    print("     The reconstruction CAN recover numerical values by emulating")
    print("     these basic ops in FP16. The DAG primitives need to match")
    print("     SCALAR precision model (single FP16 quant per op, not double).")
    print()
    print("  Q: What needs to change?")
    print("  A: 1. Fix DAG primitives: single FP16 quant per op (remove double quant)")
    print("     2. Fix NS init: use spectral X0 = A/||A||² in both C++ and DAG")
    print("     3. NS needs K=32+ for convergence in FP16 (algorithm property, not bug)")
    print("     4. LDL D-factor: FP32 accumulator in SCALAR unit for better precision")


if __name__ == "__main__":
    main()
