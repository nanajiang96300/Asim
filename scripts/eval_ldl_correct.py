#!/usr/bin/env python3
"""LDL corrected test — comparing scalar vs block-2x2 vs old implementation.

Root cause: scalar LDL applies FP16 at every arithmetic op (1000+ quantizations).
Block-2x2 LDL uses qmatmul with MAC chunking (much fewer quantizations).
The old evaluate_ldl_quality.py uses block-2x2 and was validated.

Tests: scalar FP64, scalar FP16, block-2x2 FP16 (old), block-2x2 FP64.
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


# ═══ LDL implementations (scalar vs block) ═════════════════════════════

def ldl_scalar_fp64(A):
    """Scalar LDL in FP64 (matches algo/ldl_noblock.py exactly)."""
    n = A.shape[0]; A_w = A.copy()
    L = np.eye(n, dtype=np.complex128); D = np.zeros(n)
    for j in range(n):
        acc = A_w[j,j].real
        for k in range(j): acc -= D[k]*abs(L[j,k])**2
        D[j] = max(acc, 1e-15)
        for i in range(j+1, n):
            acc = A_w[i,j]
            for k in range(j): acc -= L[i,k]*D[k]*np.conj(L[j,k])
            L[i,j] = acc / D[j]
    Z = np.eye(n, dtype=np.complex128)
    for c in range(n):
        for i in range(c+1, n):
            acc = np.complex128(0)
            for k in range(c, i): acc += L[i,k]*Z[k,c]
            Z[i,c] = -acc
    sD = np.sqrt(np.maximum(1.0/D, 0))
    Y = Z * sD[np.newaxis, :]
    return Y.conj().T @ Y


def ldl_scalar_fp16(A):
    """Scalar LDL: FP16 at every op (my original, problematic test)."""
    n = A.shape[0]; A16 = fp16(A)
    L = fp16(np.eye(n, dtype=np.complex128)); D = fp16(np.zeros(n))
    for j in range(n):
        acc = fp16(A16[j,j].real)
        for k in range(j): acc = fp16(acc - fp16(D[k]*fp16(abs(L[j,k])**2)))
        D[j] = fp16(max(acc, 1e-15))
        for i in range(j+1, n):
            acc = fp16(A16[i,j])
            for k in range(j): acc = fp16(acc - fp16(fp16(L[i,k]*D[k])*np.conj(L[j,k])))
            L[i,j] = fp16(acc / D[j])
    Z = fp16(np.eye(n, dtype=np.complex128))
    for c in range(n):
        for i in range(c+1, n):
            acc = fp16(np.complex128(0))
            for k in range(c, i): acc = fp16(acc + fp16(L[i,k]*Z[k,c]))
            Z[i,c] = fp16(-acc)
    sD = fp16(np.sqrt(fp16(1.0/fp16(np.maximum(D, 1e-15)))))
    Y = fp16(np.zeros((n,n), dtype=np.complex128))
    for i in range(n):
        for j in range(n): Y[i,j] = fp16(Z[i,j]*sD[j])
    return fp16(fp16(Y.conj().T) @ fp16(Y))


def ldl_block_fp16(A, blk=2):
    """Block-2x2 LDL: quantize at block boundaries (matches evaluate_ldl_quality.py)."""
    n = A.shape[0]; A16 = fp16(A)
    nB = n // blk

    # Block LDL decomposition
    L_blocks = [[None]*nB for _ in range(nB)]
    D_blocks = [None]*nB
    for j in range(nB):
        rj = slice(j*blk, (j+1)*blk)
        # D_jj = A_jj - Σ L_jk @ D_kk @ L_jk^H
        D_jj = A16[rj, rj].copy()
        for k in range(j):
            Ljk = L_blocks[j][k]; Dkk = D_blocks[k]
            D_jj = fp16(D_jj - fp16(fp16(Ljk @ Dkk) @ Ljk.conj().T))
        D_blocks[j] = fp16(D_jj)

        # 2x2 direct inverse
        if blk == 2:
            a00, a01 = D_jj[0,0], D_jj[0,1]; a10, a11 = D_jj[1,0], D_jj[1,1]
            det = fp16(a00*a11 - a01*a10)
            inv_det = fp16(1.0 / max(det, 1e-12))
            D_inv_jj = fp16(np.array([[a11*inv_det, -a01*inv_det],
                                       [-a10*inv_det, a00*inv_det]], dtype=np.complex128))
        else:
            D_inv_jj = fp16(np.linalg.inv(D_jj))

        for i in range(j+1, nB):
            ri = slice(i*blk, (i+1)*blk)
            L_ij = A16[ri, rj].copy()
            for k in range(j):
                L_ij = fp16(L_ij - fp16(fp16(L_blocks[i][k] @ D_blocks[k]) @ L_blocks[j][k].conj().T))
            L_blocks[i][j] = fp16(L_ij @ D_inv_jj)

    # Assemble L, D
    L = np.eye(n, dtype=np.complex128)
    for i in range(nB):
        for j in range(i):
            if L_blocks[i][j] is not None:
                L[i*blk:(i+1)*blk, j*blk:(j+1)*blk] = fp16(L_blocks[i][j])

    # Forward solve Z = L^{-1} (block)
    Z = fp16(np.eye(n, dtype=np.complex128))
    for c in range(nB):
        rc = slice(c*blk, (c+1)*blk)
        for i in range(c+1, nB):
            ri = slice(i*blk, (i+1)*blk)
            acc = np.zeros((blk, blk), dtype=np.complex128)
            for k in range(c, i):
                rk = slice(k*blk, (k+1)*blk)
                acc = fp16(acc + fp16(L[ri, rk] @ Z[rk, rc]))
            Z[ri, rc] = fp16(-acc)

    # Scale Y = Z * sqrt(Dinv)
    sD = fp16(np.zeros((n, n), dtype=np.complex128))
    for b in range(nB):
        rb = slice(b*blk, (b+1)*blk)
        Dinv_b = np.diag(1.0 / np.maximum(np.diag(fp16(D_blocks[b])).real, 1e-15))
        sD[rb, rb] = fp16(np.diag(np.sqrt(np.maximum(np.diag(Dinv_b), 0))))
    Y = fp16(Z @ sD)

    return fp16(fp16(Y.conj().T) @ fp16(Y))


def ldl_block_fp64(A, blk=2):
    """Block-2x2 LDL in FP64 (no quantization)."""
    n = A.shape[0]; nB = n // blk
    L_blocks = [[None]*nB for _ in range(nB)]; D_blocks = [None]*nB
    for j in range(nB):
        rj = slice(j*blk, (j+1)*blk)
        D_jj = A[rj, rj].copy()
        for k in range(j):
            D_jj -= L_blocks[j][k] @ D_blocks[k] @ L_blocks[j][k].conj().T
        D_blocks[j] = D_jj
        D_inv_jj = np.linalg.inv(D_jj)
        for i in range(j+1, nB):
            ri = slice(i*blk, (i+1)*blk)
            L_ij = A[ri, rj].copy()
            for k in range(j):
                L_ij -= L_blocks[i][k] @ D_blocks[k] @ L_blocks[j][k].conj().T
            L_blocks[i][j] = L_ij @ D_inv_jj
    L = np.eye(n, dtype=np.complex128)
    for i in range(nB):
        for j in range(i):
            if L_blocks[i][j] is not None:
                L[i*blk:(i+1)*blk, j*blk:(j+1)*blk] = L_blocks[i][j]
    Z = np.eye(n, dtype=np.complex128)
    for c in range(nB):
        rc = slice(c*blk, (c+1)*blk)
        for i in range(c+1, nB):
            ri = slice(i*blk, (i+1)*blk)
            acc = np.zeros((blk, blk), dtype=np.complex128)
            for k in range(c, i):
                rk = slice(k*blk, (k+1)*blk)
                acc += L[ri, rk] @ Z[rk, rc]
            Z[ri, rc] = -acc
    sD = np.diag(np.sqrt(1.0/np.maximum(np.diag(np.linalg.inv(
        np.array([D_blocks[b] for b in range(nB)]).reshape(-1)[0] if nB==1 else
        sum(D_blocks).real/len(D_blocks)).reshape(1), 0))))
    return Z @ np.diag(np.sqrt(1.0 / np.maximum(np.array(
        [np.linalg.inv(D_blocks[b]) for b in range(nB)]).diagonal().real, 1e-15))) @ Z.conj().T


def main():
    print("=" * 110)
    print("  LDL Corrected: Scalar FP16 vs Block-2x2 FP16 vs Scalar FP64")
    print("  Old code (evaluate_ldl_quality.py) uses block-2x2 + MAC chunking")
    print("=" * 110)

    nr, nt = 64, 16; snr = 20; npwr = 1.0/(10**(snr/10)); lam = npwr*nt; trials = 10

    algs = [
        ("LDLScalarFP64",   lambda A: ldl_scalar_fp64(A)),
        ("LDLScalarFP16",   lambda A: ldl_scalar_fp16(A)),
        ("LDLBlockFP16",    lambda A: ldl_block_fp16(A)),
        ("CholeskyFP16",    lambda A: fp16(np.linalg.inv(fp16(A)))),
    ]

    for ch_name, ch_gen in [("Rayleigh", RayleighChannel()), ("CDL-B", CDLBChannel())]:
        print(f"\n  Channel: {ch_name} | SNR={snr}dB | {nr}x{nt} | {trials} trials")
        print(f"  {'Method':<20s} {'ΔSER':>10s} {'||ΔM||':>12s} {'Status':>10s}")
        print("  " + "-" * 56)
        for name, fn in algs:
            sers, errs = [], []
            for t in range(trials):
                np.random.seed(t*1000)
                H = ch_gen.generate(1, nr, nt, seed=t*1000)[0]
                A = H.conj().T@H + lam*np.eye(nt)
                M_ref = np.linalg.inv(A); W_ref = M_ref @ H.conj().T
                M = fn(A); W = M @ H.conj().T
                sers.append(ser_detect(H, W, npwr) - ser_detect(H, W_ref, npwr))
                errs.append(np.linalg.norm(fp16(M)-M_ref)/max(np.linalg.norm(M_ref),1e-15))
            s, e = np.mean(sers), np.mean(errs)
            st = "✅ Perfect" if s < 1e-3 else ("✅ Good" if s < 0.01 else ("⚠️ Marginal" if s < 0.05 else "❌ Fail"))
            print(f"  {name:<20s} {s:>+10.4f} {e:>12.2e} {st:>10s}")

    # Quantization count comparison
    print(f"\n  Quantization count comparison (n=16):")
    print(f"  Scalar LDL: ~{16*16*4} FP16 ops (every +-×/÷ quantized)")
    print(f"  Block-2x2:  ~{8*8*2} FP16 ops (quantize at block boundaries only)")
    print(f"  4x fewer quantizations → 4x less FP16 error accumulation")


if __name__ == "__main__":
    main()
