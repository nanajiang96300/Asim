"""
Stable comparison of 4 matrix inversion algorithms for n=16.

Algorithms:
  1. Direct Cholesky  — scalar decomposition + 1 matmul
  2. Direct LDL^T     — scalar decomposition + diagonal scaling + 1 matmul
  3. Block Cholesky   — recursive block (b=4), traces 13 matmul calls
  4. Block LDL^T      — recursive block (b=4), traces 20 matmul calls

Note: LDL algorithms treat D_inv as dense in matmul (L^T @ D_inv @ L).
This overcounts FLOPs vs an optimized diagonal-scaling implementation,
but correctly models hardware that folds scaling into GEMM.
"""

import sys
sys.path.insert(0, '/home/gu/workspace/AlgorithmInfo')

from algolib import (
    kernel, SymMatrix, analyze, compare,
    _cost_cholesky_decomp, _cost_ldl_decomp,
    _cost_tril_inv, _cost_diag_inv, _cost_matmul, _cost_newton_init,
    FlopBreakdown, OpCounts, ParallelismInfo, KernelResult,
    get_kernel_library, _trace,
)


# ============================================================
# Kernel definitions (with return_shape for multi-output)
# ============================================================

@kernel(name="cholesky_decomp", cost_fn=_cost_cholesky_decomp)
def cholesky_decomp(A, n):
    """Scalar Cholesky decomposition A = L @ L^T."""
    import numpy as np
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j: L[i, i] = np.sqrt(A[i, i] - s)
            else:      L[i, j] = (A[i, j] - s) / L[j, j]
    return L


@kernel(name="ldl_decomp", cost_fn=_cost_ldl_decomp,
        return_shape=lambda n, **kw: [(n, n), (n, n)])
def ldl_decomp(A, n):
    """Scalar LDL^T decomposition A = L @ D @ L^T."""
    import numpy as np
    L = np.eye(n); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            s = sum(L[i, k] * L[j, k] * D[k, k] for k in range(j))
            L[i, j] = (A[i, j] - s) / D[j, j]
        s = sum((L[i, k] ** 2) * D[k, k] for k in range(i))
        D[i, i] = A[i, i] - s
    return L, D


@kernel(name="tril_inv", cost_fn=_cost_tril_inv)
def tril_inv(L, n, unit_diag=False):
    """Scalar lower-triangular inversion."""
    import numpy as np
    X = np.zeros((n, n))
    for i in range(n):
        X[i, i] = 1.0 if unit_diag else 1.0 / L[i, i]
        for j in range(i):
            s = sum(L[i, k] * X[k, j] for k in range(j, i))
            X[i, j] = -s if unit_diag else -s / L[i, i]
    return X


@kernel(name="diag_inv", cost_fn=_cost_diag_inv)
def diag_inv(D, n):
    """Diagonal reciprocal D^{-1}."""
    import numpy as np
    D_inv = np.zeros((n, n))
    for i in range(n):
        D_inv[i, i] = 1.0 / D[i, i]
    return D_inv


# ============================================================
# Algorithm 1: Direct Cholesky inversion
# ============================================================

def cholesky_inversion(A, n):
    """A^{-1} = L^{-T} @ L^{-1}"""
    L = cholesky_decomp(A, n)
    L_inv = tril_inv(L, n, unit_diag=False)
    return L_inv.T @ L_inv


# ============================================================
# Algorithm 2: Direct LDL^T inversion
# ============================================================

def ldl_inversion(A, n):
    """A^{-1} = L^{-T} @ D^{-1} @ L^{-1}"""
    L, D = ldl_decomp(A, n)
    L_inv = tril_inv(L, n, unit_diag=True)
    D_inv = diag_inv(D, n)
    return L_inv.T @ D_inv @ L_inv


# ============================================================
# Algorithm 3: Block Cholesky inversion (mirrors inv.py)
# ============================================================

def block_cholesky_core(A, n, block_size):
    """Recursive block Cholesky: returns (L, L_inv)."""
    if n <= block_size:
        L = cholesky_decomp(A, n)
        L_inv = tril_inv(L, n, unit_diag=False)
        return L, L_inv

    mid = n // 2
    A11, A21, A22 = A[:mid, :mid], A[mid:, :mid], A[mid:, mid:]

    L11, L11_inv = block_cholesky_core(A11, mid, block_size)
    L21 = A21 @ L11_inv.T
    S = A22 - L21 @ L21.T
    L22, L22_inv = block_cholesky_core(S, mid, block_size)
    _ = L22_inv @ L21 @ L11_inv     # L_inv bottom-left block assembly

    return SymMatrix((n, n)), SymMatrix((n, n))


def block_cholesky_inversion(A, n, block_size):
    """Top-level: A^{-1} = L^{-T} @ L^{-1}"""
    _, L_inv = block_cholesky_core(A, n, block_size)
    return L_inv.T @ L_inv


# ============================================================
# Algorithm 4: Block LDL^T inversion (mirrors inv.py)
# ============================================================

def block_ldl_core(A, n, block_size):
    """Recursive block LDL^T: returns (L, D, L_inv)."""
    if n <= block_size:
        L, D = ldl_decomp(A, n)
        L_inv = tril_inv(L, n, unit_diag=True)
        return L, D, L_inv

    mid = n // 2
    A11, A21, A22 = A[:mid, :mid], A[mid:, :mid], A[mid:, mid:]

    L11, D11, L11_inv = block_ldl_core(A11, mid, block_size)
    D11_inv = diag_inv(D11, n=mid)
    L21 = A21 @ L11_inv.T @ D11_inv
    S = A22 - L21 @ D11 @ L21.T
    L22, D22, L22_inv = block_ldl_core(S, mid, block_size)
    _ = L22_inv @ L21 @ L11_inv     # L_inv bottom-left block assembly

    return SymMatrix((n, n)), SymMatrix((n, n)), SymMatrix((n, n))


def block_ldl_inversion(A, n, block_size):
    """Top-level: A^{-1} = L^{-T} @ D^{-1} @ L^{-1}"""
    _, D, L_inv = block_ldl_core(A, n, block_size)
    D_inv = diag_inv(D, n=n)
    return L_inv.T @ D_inv @ L_inv


# ============================================================
# Algorithm 5: Newton-Schulz iteration
# ============================================================

# Newton initial guess kernel (traceable)
@kernel(name="newton_init", cost_fn=_cost_newton_init,
        return_shape=lambda n, **kw: (n, n))
def newton_init(A, n):
    """X_0 = A^T / ||A||^2_F.  Cost: ~3n^2 vector FLOPs + 1 scalar div."""
    import numpy as np
    alpha = 1.0 / np.sum(A**2)
    return alpha * A.T


# ============================================================
# Algorithm 5: Block-Jacobi iterative inversion
# ============================================================

def bj_inversion(A, n, block_size=2, num_layers=4):
    """
    Block-Jacobi with Chebyshev acceleration.

    Step 1: D_inv = blkdiag(A_ii^{-1})  (independent block inversions)
            B = D_inv @ A               (1 matmul)
    Step 2: Y_0 = 0
            For k=0..L-1:
              residual = I - B @ Y_k    (1 matmul + 1 matrix_add)
              Y_{k+1} = Y_k + ω_k · residual  (1 vector_scale + 1 matrix_add)
    Step 3: A^{-1} = Y_L @ D_inv       (1 matmul)
    """
    import numpy as np

    # Step 1: preconditioner
    _trace("bj_block_inv", n=n, block_size=block_size)
    D_inv = SymMatrix((n, n))
    B = D_inv @ A               # matmul: D_inv @ A

    # Step 2: Chebyshev iteration (Chebyshev weight computation is O(L) scalar, ignored)
    Y = SymMatrix((n, n))       # Y_0 = 0
    I_mat = SymMatrix((n, n))   # identity (shape placeholder)

    for _ in range(num_layers):
        BY = B @ Y              # matmul
        residual = I_mat - BY   # matrix_add (negate BY, +1 on diag → O(n²))
        Y = Y + 0.8 * residual  # vector_scale (ω·res) + matrix_add (Y + ...)
        # ω ≈ 0.8–1.2 typical Chebyshev weight; cost independent of actual value

    # Step 3: recover inverse
    return Y @ D_inv            # matmul: A^{-1} = Y @ D_inv


# ============================================================
# Algorithm 6: Newton-Schulz iteration
# ============================================================

def newton_inversion(A, n, iterations=10):
    """
    Newton-Schulz: X_0 = A^T/||A||^2_F, then X_{k+1} = 2X_k - X_k @ A @ X_k.

    Each iteration: 2 matmuls.
    The 2X_k - ... subtraction is element-wise (vector unit).
    """
    X = newton_init(A, n)   # X_0 = A^T / ||A||^2_F  (vector ops)

    for _ in range(iterations):
        AX = A @ X           # matmul #1: A @ X_k
        X = X @ AX           # matmul #2: X_k @ A @ X_k
        # X_{k+1} = 2X_k - X_k @ A @ X_k  → O(n^2) sub ignored (not traced)

    return X


# ============================================================
# Main: compare all 5 algorithms for n=16
# ============================================================

if __name__ == "__main__":
    N = 16
    B = 2
    ITER = 10

    lib = get_kernel_library()
    print(f"Registered kernels: {lib.list_kernels()}")
    print(f"\n  (Note: LDL path treats D_inv as dense in matmul — ")
    print(f"   overcounts FLOPs vs opt. diagonal scaling, but models")
    print(f"   AI-accelerator GEMM-fused execution correctly.)\n")

    # Run all five and compare
    compare(
        ("Direct Cholesky",  cholesky_inversion,       {"n": N}),
        ("Direct LDL",       ldl_inversion,            {"n": N}),
        ("Block Cholesky",   block_cholesky_inversion, {"n": N, "block_size": B}),
        ("Block LDL",        block_ldl_inversion,      {"n": N, "block_size": B}),
        ("Newton(10 iter)",  newton_inversion,         {"n": N, "iterations": ITER}),
        ("BJ(blk=2, L=4)",  bj_inversion,             {"n": N, "block_size": B, "num_layers": 4}),
    )
