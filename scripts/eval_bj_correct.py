#!/usr/bin/env python3
"""Block-Jacobi corrected test — matching the old bj_inverse.py implementation.

Key differences from my broken BRI test:
1. Y0 = ZEROS (not identity)
2. B = D^{-1} @ A  (preconditioned system, not just D^{-1})
3. Iteration: Y += ω*(I - B@Y)  on the full preconditioned system
4. Recovery: A^{-1} = Y @ D^{-1}  at the END
5. Chebyshev acceleration with adaptive bounds

My old BRI was wrong in ALL 5 of these aspects.
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


def bj_inverse(A, cfg):
    """Block-Jacobi: matches scripts/aaa/bj_inverse.py exactly."""
    n = A.shape[0]; blk = cfg.get("block_size", 2); L = cfg.get("num_layers", 4)
    use_fp16 = cfg.get("use_fp16_quant", True)
    adaptive = cfg.get("adaptive_omega", True)

    # ── Step 1: Build preconditioner D^{-1} (block-diagonal) ──
    D_inv = np.zeros_like(A, dtype=np.complex128)
    for b in range(0, n, blk):
        block = A[b:b+blk, b:b+blk]
        if blk == 2:
            a00, a01 = block[0,0], block[0,1]; a10, a11 = block[1,0], block[1,1]
            det = a00*a11 - a01*a10 + 1e-12
            D_inv[b,b] = a11/det; D_inv[b,b+1] = -a01/det
            D_inv[b+1,b] = -a10/det; D_inv[b+1,b+1] = a00/det
        else:
            D_inv[b:b+blk, b:b+blk] = np.linalg.inv(block)

    # ── Step 2: B = D^{-1} @ A  (preconditioned matrix, eigenvalues near 1) ──
    B = D_inv @ A
    if use_fp16:
        B = fp16(B); D_inv = fp16(D_inv)

    # ── Step 3: Chebyshev weights ──
    if adaptive:
        ev = np.linalg.eigvals(B); ev = np.real(ev[ev > 1e-8])
        if len(ev) == 0: ev = np.array([0.1, 1.0])
        emin, emax = float(np.min(ev)), float(np.max(ev))
    else:
        emin, emax = 0.1, 1.2
    omegas = [1.0 / (0.5*(emax+emin) + 0.5*(emax-emin)*np.cos(np.pi*(2*k+1)/(2*L)))
              for k in range(L)]

    # ── Step 4: Iterate Y_{k+1} = Y_k + ω_k * (I - B @ Y_k) ──
    Y = np.zeros_like(A, dtype=np.complex128)  # Y0 = 0 (NOT identity!)
    I = np.eye(n, dtype=np.complex128)
    for omega in omegas:
        R = I - B @ Y
        if use_fp16: R = fp16(R)
        Y = Y + omega * R
        if use_fp16: Y = fp16(Y)

    # ── Step 5: A^{-1} = Y @ D^{-1} ──
    A_inv = Y @ D_inv
    if use_fp16: A_inv = fp16(A_inv)
    return A_inv


def main():
    print("=" * 110)
    print("  Block-Jacobi (BJ) Corrected Test — matching scripts/aaa/bj_inverse.py")
    print("  5 fixes: Y0=0, B=D^{-1}@A, ω Chebyshev, Y@D^{-1} recovery, adaptive ω")
    print("=" * 110)

    nr, nt = 64, 16; snr = 20; npwr = 1.0/(10**(snr/10)); lam = npwr*nt; trials = 10

    configs = [
        ("BJ B=2 L=4  FP16+Cheb",  {"block_size":2, "num_layers":4, "use_fp16_quant":True, "adaptive_omega":True}),
        ("BJ B=2 L=8  FP16+Cheb",  {"block_size":2, "num_layers":8, "use_fp16_quant":True, "adaptive_omega":True}),
        ("BJ B=2 L=16 FP16+Cheb",  {"block_size":2, "num_layers":16, "use_fp16_quant":True, "adaptive_omega":True}),
        ("BJ B=2 L=8  FP64+Cheb",  {"block_size":2, "num_layers":8, "use_fp16_quant":False, "adaptive_omega":True}),
        ("BJ B=2 L=8  FP16 noCheb", {"block_size":2, "num_layers":8, "use_fp16_quant":True, "adaptive_omega":False}),
        ("Cholesky (ref)", None),
        ("NS K=16  (ref)", None),
    ]

    for ch_name, ch_gen in [("Rayleigh", RayleighChannel()), ("CDL-B", CDLBChannel())]:
        print(f"\n  Channel: {ch_name} | SNR={snr}dB | {nr}x{nt} | {trials} trials")
        print(f"  {'Algorithm':<28s} {'ΔSER':>10s} {'||A^{-1}-A^{-1}_ref||':>20s} {'Status':>10s}")
        print("  " + "-" * 72)

        for name, cfg in configs:
            sers, errs = [], []
            for t in range(trials):
                np.random.seed(t*1000)
                H = ch_gen.generate(1, nr, nt, seed=t*1000)[0]
                A = H.conj().T@H + lam*np.eye(nt)
                M_ref = np.linalg.inv(A)
                W_ref = M_ref @ H.conj().T

                if cfg is None:
                    # Baseline: numpy or NS
                    if "Cholesky" in name:
                        Lmat = np.linalg.cholesky(A)
                        Ymat = np.linalg.solve(Lmat, np.eye(nt))
                        M = Ymat.conj().T @ Ymat
                    else:
                        A_c64 = A.astype(np.complex64)
                        ra = np.abs(A_c64)
                        alpha = 1.0/max(np.max(np.sum(ra,axis=0))*np.max(np.sum(ra,axis=1)), 1e-12)
                        X = (alpha*A_c64.conj().T).astype(np.complex64)
                        I2 = (2.0*np.eye(nt)).astype(np.complex64)
                        for _ in range(16): X = X @ (I2 - A_c64 @ X)
                        M = X.astype(np.complex128)
                    W = M @ H.conj().T
                else:
                    M = bj_inverse(A, cfg)
                    W = M @ H.conj().T

                sers.append(ser_detect(H, W, npwr) - ser_detect(H, W_ref, npwr))
                errs.append(np.linalg.norm(fp16(M)-M_ref)/max(np.linalg.norm(M_ref),1e-15))

            avg_s = np.mean(sers); avg_e = np.mean(errs)
            if cfg is None: st = "— ref"
            elif avg_s < 1e-3: st = "✅ Perfect"
            elif avg_s < 0.01: st = "✅ Good"
            elif avg_s < 0.05: st = "⚠️ Marginal"
            else: st = "❌ Fail"
            print(f"  {name:<28s} {avg_s:>+10.4f} {avg_e:>20.2e} {st:>10s}")

    # ── What was wrong ────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  What was wrong with my old BRI test (5 bugs)")
    print(f"{'='*110}")
    print(f"  {'Aspect':<30s} {'My broken BRI':<35s} {'Correct BJ':<35s}")
    print(f"  {'-'*100}")
    for aspect, wrong, correct in [
        ("Initial Y",          "Y0 = I (identity)",              "Y0 = 0 (zeros)"),
        ("Preconditioner",     "B = blockdiag(A_ii^{-1})",       "B = D^{-1} @ A (full system)"),
        ("Iteration",          "Y = B@Y etc (wrong direction)",  "Y += ω*(I - B@Y)"),
        ("Acceleration",       "None (constant step)",           "Chebyshev via eigenvalues of B"),
        ("Recovery",           "Use Y directly (no recovery)",   "A^{-1} = Y @ D^{-1}"),
        ("FP16 strategy",      "Every op quantized",             "Optional, I/O only"),
    ]:
        print(f"  {aspect:<30s} {wrong:<35s} {correct:<35s}")


if __name__ == "__main__":
    main()
