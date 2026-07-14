#!/usr/bin/env python3
"""BRI Block-Tridiagonal Schur-Corrected Preconditioner.

Replaces blockdiag(A_ii^{-1}) with Schur-complement corrected blocks:
  D_0 = A_00
  For i>0: D_i = A_ii - A_{i,i-1} @ D_{i-1}^{-1} @ A_{i-1,i}

This captures nearest-neighbor coupling. The preconditioner becomes:
  M = blockdiag(D_0^{-1}, D_1^{-1}, ..., D_{nB-1}^{-1})
  B = M @ A  (should have eigenvalues much closer to 1 on CDL-B)

Then BRI iteration is unchanged: Y_{k+1}=Y_k+ω_k(I-B@Y_k), A^{-1}=Y@M.
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


def bri_tridiag_precond(A, blk=2, use_fp16=True):
    """Build block-tridiagonal Schur-corrected preconditioner M.

    Block-Thomas forward sweep:
      D_0 = A_00
      For i=1..nB-1: L_i = A_{i,i-1} @ D_{i-1}^{-1}
                     D_i = A_ii - L_i @ A_{i-1,i}

    Returns M = blockdiag(D_0^{-1}, ..., D_{nB-1}^{-1})
    """
    n = A.shape[0]; nB = n // blk
    M = np.zeros_like(A, dtype=np.complex128)

    # D_0 = A_00, M_00 = D_0^{-1}
    D_prev = fp16(A[0:blk, 0:blk]) if use_fp16 else A[0:blk, 0:blk].copy()
    M[0:blk, 0:blk] = np.linalg.inv(D_prev)

    for i in range(1, nB):
        ri = slice(i*blk, (i+1)*blk)
        rp = slice((i-1)*blk, i*blk)

        # A_{i,i-1} and A_{i-1,i}
        A_i_prev = A[ri, rp]
        A_prev_i = A[rp, ri]

        # D_{i-1}^{-1}
        inv_D_prev = M[rp, rp]

        # L_i = A_{i,i-1} @ D_{i-1}^{-1}
        L_i = A_i_prev @ inv_D_prev
        if use_fp16: L_i = fp16(L_i)

        # D_i = A_ii - L_i @ A_{i-1,i}
        D_i = A[ri, ri] - L_i @ A_prev_i
        if use_fp16: D_i = fp16(D_i)

        M[ri, ri] = np.linalg.inv(D_i)
        if use_fp16: M[ri, ri] = fp16(M[ri, ri])

    return M


def bri_iterate(A, M, L=4, use_fp16=True):
    """BRI iteration with given preconditioner M. Returns A^{-1}."""
    n = A.shape[0]
    B = M @ A
    if use_fp16: B = fp16(B); M = fp16(M)

    ev = np.linalg.eigvals(B); ev = np.real(ev[ev > 1e-10])
    emin = max(float(np.min(ev)) if len(ev) > 0 else 0.1, 0.001)
    emax = max(float(np.max(ev)) if len(ev) > 0 else 2.0, emin * 2)

    Y = fp16(np.zeros_like(A))
    I = np.eye(n, dtype=np.complex128)
    for k in range(L):
        omega = 1.0 / (0.5*(emax+emin) + 0.5*(emax-emin)*np.cos(np.pi*(2*k+1)/(2*L)))
        R = fp16(I - B @ Y)
        Y = fp16(Y + omega * R)
    return fp16(Y @ M)


def bri_standard(A, L=4, blk=2, use_fp16=True):
    """Standard BRI with blockdiag preconditioner."""
    n = A.shape[0]
    M = np.zeros_like(A, dtype=np.complex128)
    for b in range(0, n, blk):
        M[b:b+blk, b:b+blk] = np.linalg.inv(A[b:b+blk, b:b+blk])
    return bri_iterate(A, M, L, use_fp16)


def main():
    print("=" * 105)
    print("  BRI: Block-Tridiagonal Schur-Corrected Preconditioner")
    print("  vs Standard Block-Diagonal Preconditioner")
    print("=" * 105)

    nr, nt = 64, 16; snr = 20; npwr = 1.0/(10**(snr/10)); lam = npwr*nt; trials = 10
    blk = 2

    for ch_name, ch_gen in [("Rayleigh", RayleighChannel()), ("CDL-B", CDLBChannel())]:
        print(f"\n  Channel: {ch_name} | {nr}x{nt} | SNR={snr}dB | blk={blk}")
        print(f"  {'Method':<35s} {'L':>4s} {'min eig(B)':>12s} {'max eig(B)':>12s} {'ΔSER':>10s} {'Status':>8s}")
        print(f"  {'-'*85}")

        # First: show eigenvalue comparison
        for t in [0]:
            np.random.seed(t*1000)
            H = ch_gen.generate(1, nr, nt, seed=t*1000)[0]
            A = H.conj().T@H + lam*np.eye(nt)

            # Standard
            M_std = np.zeros_like(A, dtype=np.complex128)
            for b in range(0, nt, blk):
                M_std[b:b+blk, b:b+blk] = np.linalg.inv(A[b:b+blk, b:b+blk])
            B_std = M_std @ A
            ev_std = np.linalg.eigvals(B_std); ev_std = np.real(ev_std[ev_std > 1e-10])

            # Tridiag
            M_tri = bri_tridiag_precond(A, blk, use_fp16=False)
            B_tri = M_tri @ A
            ev_tri = np.linalg.eigvals(B_tri); ev_tri = np.real(ev_tri[ev_tri > 1e-10])

            emin_s = float(np.min(ev_std)); emax_s = float(np.max(ev_std))
            emin_t = float(np.min(ev_tri)); emax_t = float(np.max(ev_tri))

            print(f"  {'Standard (blockdiag)':<35s} {'—':>4s} {emin_s:>12.4f} {emax_s:>12.4f} {'—':>10s} {'—':>8s}")
            print(f"  {'Tridiag (Schur-corrected)':<35s} {'—':>4s} {emin_t:>12.4f} {emax_t:>12.4f} {'—':>10s} {'—':>8s}")

            ratio_s = emin_s/emax_s if emax_s > 0 else 0
            ratio_t = emin_t/emax_t if emax_t > 0 else 0
            print(f"  Ratio improvement: {ratio_s:.3f} → {ratio_t:.3f} ({(ratio_t/ratio_s-1)*100:.0f}% better)")

        # Now SER comparison
        for label, fn, L_list in [
            ("Standard B=2", lambda A: bri_standard(A, L=4, blk=2), [4, 8]),
            ("Standard B=2", lambda A: bri_standard(A, L=8, blk=2), [4, 8]),
            ("Tridiag B=2", lambda A: bri_iterate(A, bri_tridiag_precond(A, blk=2), L=4), [4, 8]),
            ("Tridiag B=2", lambda A: bri_iterate(A, bri_tridiag_precond(A, blk=2), L=8), [4, 8]),
            ("Tridiag B=2", lambda A: bri_iterate(A, bri_tridiag_precond(A, blk=2), L=16), [16]),
        ]:
            for L in L_list:
                sers = []
                for t in range(trials):
                    np.random.seed(t*1000)
                    H = ch_gen.generate(1, nr, nt, seed=t*1000)[0]
                    A = H.conj().T@H + lam*np.eye(nt)
                    M_ref = np.linalg.inv(A); W_ref = M_ref @ H.conj().T
                    M_algo = fn(A) if L in L_list else bri_iterate(A, bri_tridiag_precond(A, blk=2), L=L)
                    W = M_algo @ H.conj().T
                    sers.append(ser_detect(H, W, npwr) - ser_detect(H, W_ref, npwr))
                s = np.mean(sers)
                st = "✅" if s < 0.01 else ("△" if s < 0.05 else "❌")
                print(f"  {label+' L='+str(L):<35s} {L:>4d} {'—':>12s} {'—':>12s} {s:>+10.4f} {st:>8s}")


if __name__ == "__main__":
    main()
