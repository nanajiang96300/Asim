"""Cholesky NoBlock — Python reference implementation (float64 precision).

This is the mathematical ground truth. Every step matches one C++ instruction.
The C++ output should match this within FP16 tolerance (~1e-2 rel error).
"""

import numpy as np


def cholesky_noblock_inverse(A: np.ndarray) -> np.ndarray:
    """Cholesky NoBlock: A = L·L^H → A^{-1} = L^{-H}·L^{-1}.

    Column-by-column (B=1) decomposition.
    Each step maps to SCALAR instructions in C++ operator.

    Args:
        A: Hermitian positive definite, shape (U, U), complex128
    Returns:
        A_inv: shape (U, U), complex128
    """
    U = A.shape[0]
    A_work = A.copy()
    L = np.zeros((U, U), dtype=np.complex128)

    # Phase 1: Cholesky Decomposition
    for j in range(U):
        # C1: POTRF diag update — A[j,j] -= sum_{k<j} |L[j,k]|^2
        acc = np.complex128(0.0)
        for k in range(j):
            acc += L[j, k] * np.conj(L[j, k])
        a_jj = (A_work[j, j] - acc).real
        if a_jj <= 0:
            a_jj = 1e-15
        L[j, j] = np.sqrt(a_jj)

        # C2: TRSM — L[i,j] = (A[i,j] - sum)/L[j,j] for i>j
        for i in range(j + 1, U):
            dot = np.complex128(0.0)
            for k in range(j):
                dot += L[i, k] * np.conj(L[j, k])
            L[i, j] = (A_work[i, j] - dot) / L[j, j]

    # Phase 2: Forward Solve — Y = L^{-1}
    Y = np.zeros((U, U), dtype=np.complex128)
    for c in range(U):
        Y[c, c] = 1.0 / L[c, c]
        for i in range(c + 1, U):
            acc = np.complex128(0.0)
            for k in range(c, i):
                acc += L[i, k] * Y[k, c]
            Y[i, c] = -acc / L[i, i]

    # Phase 3: Backward Assembly — Ainv = Y @ Y^H
    return Y.conj().T @ Y


def cholesky_noblock_inverse_batched(A_batch: np.ndarray) -> np.ndarray:
    """Batched version."""
    B = A_batch.shape[0]
    result = np.zeros_like(A_batch)
    for b in range(B):
        result[b] = cholesky_noblock_inverse(A_batch[b])
    return result
