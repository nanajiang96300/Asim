"""LDL NoBlock — Python reference implementation (float64 precision).

A = L·D·L^H (L unit lower triangular, D real diagonal)
A^{-1} = L^{-H}·D^{-1}·L^{-1}
"""

import numpy as np


def ldl_noblock_inverse(A: np.ndarray) -> np.ndarray:
    """LDL NoBlock inverse.

    Column-by-column decomposition. No SQRT. Extra multiply for D-factor.

    Args:
        A: Hermitian, shape (U, U), complex128
    Returns:
        A_inv: shape (U, U), complex128
    """
    U = A.shape[0]
    A_work = A.copy()
    L = np.eye(U, dtype=np.complex128)
    D = np.zeros(U, dtype=np.float64)
    D_inv = np.zeros(U, dtype=np.float64)

    # Phase 1: LDL Decomposition
    for j in range(U):
        # D_UPDATE: D[j] = A[j,j] - sum_{k<j} D[k]·|L[j,k]|^2
        acc = 0.0
        for k in range(j):
            acc += D[k] * (L[j, k].real ** 2 + L[j, k].imag ** 2)
        d_jj = A_work[j, j].real - acc
        if d_jj <= 0:
            d_jj = 1e-15
        D[j] = d_jj
        D_inv[j] = 1.0 / d_jj

        # L_UPDATE: L[i,j] = (A[i,j] - sum)/D[j] for i>j
        for i in range(j + 1, U):
            dot = np.complex128(0.0)
            for k in range(j):
                dot += L[i, k] * D[k] * np.conj(L[j, k])
            L[i, j] = (A_work[i, j] - dot) * D_inv[j]

    # Phase 2: Forward Solve — Z = L^{-1} (unit triangular)
    Z = np.zeros((U, U), dtype=np.complex128)
    for c in range(U):
        Z[c, c] = 1.0
        for i in range(c + 1, U):
            acc = np.complex128(0.0)
            for k in range(c, i):
                acc += L[i, k] * Z[k, c]
            Z[i, c] = -acc

    # Phase 3: Weight by D^{-1/2} for symmetric assembly
    Y = Z * np.sqrt(np.maximum(D_inv, 0))[:, None]
    return Y.conj().T @ Y


def ldl_noblock_inverse_batched(A_batch: np.ndarray) -> np.ndarray:
    """Batched version."""
    B = A_batch.shape[0]
    result = np.zeros_like(A_batch)
    for b in range(B):
        result[b] = ldl_noblock_inverse(A_batch[b])
    return result
