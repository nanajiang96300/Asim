#!/usr/bin/env python3
"""BRI CDL-B improvement experiments: NS-init hybrid + other attempts.

Idea 1: NS+BRI Hybrid
  Run NS for K_NS iterations → Y0 ≈ A^{-1} @ D → BRI with warm start

Idea 2: Damping (cap Chebyshev omega)
  omega_k = min(1/t_k, omega_max) to prevent overshoot on ill-conditioned B

Idea 3: Regularized preconditioner
  D_inv += alpha*I to push eigenvalues away from 0

Idea 4: Over-relaxation smoothing
  Y_{k+1} = (1-gamma)*Y_k + gamma*(Y_k + omega_k*R)
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import RayleighChannel, CDLBChannel
from eval_qam64_se import qam64_modulate, QAM64


def fp16(x):
    if np.iscomplexobj(x):
        return (x.real.astype(np.float16).astype(np.float64) +
                1j * x.imag.astype(np.float16).astype(np.float64))
    return np.asarray(x, dtype=np.float16).astype(np.float64)

def ser_detect(H, W, noise_power, n_test=1600):
    nr, nt = H.shape; ng = n_test // nt
    bits = np.random.randint(0, 2, n_test * 6)
    s_all = qam64_modulate(bits)
    noise = np.sqrt(noise_power/2)*(np.random.randn(nr, ng)+1j*np.random.randn(nr, ng))
    e = 0
    for g in range(ng):
        y = H @ s_all[g*nt:(g+1)*nt] + noise[:, g]
        sh = W @ y
        e += np.sum(np.argmin(np.abs(sh[:,None]-QAM64[None,:]), axis=1) !=
                    np.argmin(np.abs(s_all[g*nt:(g+1)*nt,None]-QAM64[None,:]), axis=1))
    return e / n_test


def ns_warm_start(A, K_ns):
    """NS iteration in complex64, returns A^{-1} approximation."""
    n = A.shape[0]
    A_c64 = A.astype(np.complex64)
    ra = np.abs(A_c64)
    alpha = 1.0 / max(np.max(np.sum(ra, axis=0)) * np.max(np.sum(ra, axis=1)), 1e-12)
    X = (alpha * A_c64.conj().T).astype(np.complex64)
    I2 = (2.0 * np.eye(n)).astype(np.complex64)
    for _ in range(K_ns):
        X = X @ (I2 - A_c64 @ X)
    return X.astype(np.complex128)  # returns X ≈ A^{-1}


def bri_base(A, L, blk, Y0=None, omega_cap=None, damp_gamma=None, reg_alpha=0.0):
    """BRI with optional warm start Y0, omega capping, damping, regularization."""
    n = A.shape[0]
    D_inv = np.zeros_like(A, dtype=np.complex128)
    for b in range(0, n, blk):
        D_inv[b:b+blk, b:b+blk] = np.linalg.inv(A[b:b+blk, b:b+blk])

    if reg_alpha > 0:
        D_inv = D_inv + reg_alpha * np.eye(n)

    B = fp16(D_inv @ A)
    D_inv = fp16(D_inv)

    # Chebyshev weights
    ev = np.linalg.eigvals(B); ev = np.real(ev[ev > 1e-10])
    emin = max(float(np.min(ev)) if len(ev) > 0 else 0.1, 0.001)
    emax = max(float(np.max(ev)) if len(ev) > 0 else 2.0, emin * 2)

    Y = fp16(np.zeros_like(A)) if Y0 is None else fp16(Y0)
    I = np.eye(n, dtype=np.complex128)

    for k in range(L):
        omega = 1.0 / (0.5*(emax+emin) + 0.5*(emax-emin)*np.cos(np.pi*(2*k+1)/(2*L)))
        if omega_cap is not None:
            omega = min(omega, omega_cap)

        R = fp16(I - B @ Y)
        Y_new = Y + omega * R

        if damp_gamma is not None:
            Y_new = fp16((1-damp_gamma)*Y + damp_gamma*Y_new)

        Y = fp16(Y_new)

    return fp16(Y @ D_inv)


def main():
    print("=" * 110)
    print("  BRI CDL-B Improvement Experiments")
    print("=" * 110)

    nr, nt = 64, 16; snr = 20; npwr = 1.0/(10**(snr/10)); lam = npwr*nt; trials = 10

    configs = [
        # (label, L_bri, blk, K_ns, omega_cap, damp, reg)
        ("Baseline: B=2 L=8        ",  8, 2, 0, None, None, 0.0),
        ("Baseline: B=2 L=16       ", 16, 2, 0, None, None, 0.0),

        # NS-init hybrids
        ("NS(2)+BRI B=2 L=4        ",  4, 2, 2, None, None, 0.0),
        ("NS(2)+BRI B=2 L=8        ",  8, 2, 2, None, None, 0.0),
        ("NS(4)+BRI B=2 L=4        ",  4, 2, 4, None, None, 0.0),
        ("NS(4)+BRI B=2 L=8        ",  8, 2, 4, None, None, 0.0),
        ("NS(8)+BRI B=2 L=4        ",  4, 2, 8, None, None, 0.0),

        # Omega capping
        ("B=2 L=8  omega<0.5       ",  8, 2, 0, 0.5, None, 0.0),
        ("B=2 L=16 omega<0.5       ", 16, 2, 0, 0.5, None, 0.0),
        ("B=2 L=8  omega<1.0       ",  8, 2, 0, 1.0, None, 0.0),

        # Damping
        ("B=2 L=16 damp=0.5        ", 16, 2, 0, None, 0.5, 0.0),
        ("B=2 L=16 damp=0.8        ", 16, 2, 0, None, 0.8, 0.0),

        # Regularization
        ("B=2 L=8  reg=0.01         ",  8, 2, 0, None, None, 0.01),
        ("B=2 L=16 reg=0.01         ", 16, 2, 0, None, None, 0.01),

        # Combined: NS + damp + cap
        ("NS(4)+B=2 L=8 damp=0.5    ",  8, 2, 4, 0.5, 0.5, 0.0),
        ("NS(8)+B=2 L=4 damp=0.5    ",  4, 2, 8, 0.5, 0.5, 0.0),

        # Reference
        ("NS K=16 (ref)             ",  0, 0, 16, None, None, 0.0),
        ("Cholesky (ref)            ",  0, 0, 0, None, None, 0.0),
    ]

    for ch_name, ch_gen in [("Rayleigh", RayleighChannel()), ("CDL-B", CDLBChannel())]:
        print(f"\n  Channel: {ch_name} | SNR={snr}dB | {nr}x{nt} | {trials} trials")
        print(f"  {'Config':<30s} {'ΔSER':>10s} {'||ΔM||':>12s} {'Status':>10s}")
        print(f"  {'-'*64}")

        for label, L_bri, blk, K_ns, ocap, damp, reg in configs:
            if K_ns > 0 and L_bri == 0:  # NS-only ref
                pass  # handled below
            if K_ns == 0 and L_bri == 0 and "Cholesky" not in label and "NS K=" not in label:
                continue

            sers, errs = [], []
            for t in range(trials):
                np.random.seed(t*1000)
                H = ch_gen.generate(1, nr, nt, seed=t*1000)[0]
                A = H.conj().T@H + lam*np.eye(nt)
                M_ref = np.linalg.inv(A); W_ref = M_ref @ H.conj().T

                if "Cholesky" in label:
                    M = np.linalg.inv(A)
                elif K_ns > 0 and L_bri == 0:
                    # NS only
                    M = ns_warm_start(A, K_ns)
                    M = fp16(M @ M)  # X@X for NS output
                elif K_ns > 0:
                    # NS + BRI hybrid
                    X_ns = ns_warm_start(A, K_ns)
                    # Convert X ≈ A^{-1} to Y0 ≈ A^{-1} @ D
                    D = np.zeros_like(A, dtype=np.complex128)
                    for b in range(0, nt, blk):
                        D[b:b+blk, b:b+blk] = fp16(A[b:b+blk, b:b+blk])
                    Y0 = fp16(X_ns @ D)  # Y0 = A^{-1} @ D
                    M = bri_base(A, L_bri, blk, Y0=Y0, omega_cap=ocap, damp_gamma=damp, reg_alpha=reg)
                else:
                    M = bri_base(A, L_bri, blk, omega_cap=ocap, damp_gamma=damp, reg_alpha=reg)

                W = M @ H.conj().T
                sers.append(ser_detect(H, W, npwr) - ser_detect(H, W_ref, npwr))
                errs.append(np.linalg.norm(fp16(M)-M_ref)/max(np.linalg.norm(M_ref),1e-15))

            s, e = np.mean(sers), np.mean(errs)
            st = ("✅" if s < 1e-3 else ("✓" if s < 0.01 else ("△" if s < 0.05 else ("〜" if s < 0.5 else "❌"))))
            print(f"  {label:<30s} {s:>+10.4f} {e:>12.2e} {st:>10s}")


if __name__ == "__main__":
    main()
