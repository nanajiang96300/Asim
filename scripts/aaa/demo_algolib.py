"""
Demo: using algolib.py to analyze matrix inversion algorithms.

Shows how to:
  1. Decorate kernels with @kernel(name, cost_fn)
  2. Write algorithms using kernels + SymMatrix operations
  3. Use analyze() for one-shot reporting
  4. Use compare() for head-to-head comparison

Includes both direct (scalar) and block-recursive versions.
"""

import sys
sys.path.insert(0, '/home/gu/workspace/AlgorithmInfo')

from algolib import (
    kernel, SymMatrix, analyze, compare,
    _cost_cholesky_decomp, _cost_ldl_decomp,
    _cost_tril_inv, _cost_diag_inv, _cost_matmul, _cost_matmul_chain,
    get_kernel_library,
)


# ============================================================
# Register kernels (they auto-register via decorator)
# In practice you'd put these in a shared module
# ============================================================

@kernel(name="cholesky_decomp", cost_fn=_cost_cholesky_decomp)
def cholesky_decomp(A, n):
    """Scalar Cholesky decomposition. Body not executed in trace mode."""
    import numpy as np
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, i] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


@kernel(name="ldl_decomp", cost_fn=_cost_ldl_decomp,
        return_shape=lambda n, **kw: [(n, n), (n, n)])
def ldl_decomp(A, n):
    import numpy as np
    L = np.eye(n)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            s = sum(L[i, k] * L[j, k] * D[k, k] for k in range(j))
            L[i, j] = (A[i, j] - s) / D[j, j]
        s = sum((L[i, k] ** 2) * D[k, k] for k in range(i))
        D[i, i] = A[i, i] - s
    return L, D


@kernel(name="tril_inv", cost_fn=_cost_tril_inv)
def tril_inv(L, n, unit_diag=False):
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
# Algorithm 3: Block Cholesky inversion
#   (mirrors _block_cholesky_core in inv.py)
# ============================================================

def block_cholesky_core(A, n, block_size):
    """Recursive block Cholesky: returns (L, L_inv)."""
    if n <= block_size:
        L = cholesky_decomp(A, n)
        L_inv = tril_inv(L, n, unit_diag=False)
        return L, L_inv

    mid = n // 2
    A11 = A[:mid, :mid]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    L11, L11_inv = block_cholesky_core(A11, mid, block_size)

    L21 = A21 @ L11_inv.T           # matmul (mid × mid)
    S = A22 - L21 @ L21.T           # matmul (mid × mid)
    L22, L22_inv = block_cholesky_core(S, mid, block_size)

    # L_inv bottom-left block:
    _ = L22_inv @ L21 @ L11_inv     # 2 matmuls (mid × mid)

    return SymMatrix((n, n)), SymMatrix((n, n))   # (L, L_inv)


def block_cholesky_inversion(A, n, block_size):
    """Top-level: A^{-1} = L^{-T} @ L^{-1}"""
    _, L_inv = block_cholesky_core(A, n, block_size)
    return L_inv.T @ L_inv           # matmul (n × n)


# ============================================================
# Algorithm 4: Block LDL^T inversion
#   (mirrors _block_ldl_core in inv.py)
# ============================================================

def block_ldl_core(A, n, block_size):
    """Recursive block LDL^T: returns (L, D, L_inv)."""
    if n <= block_size:
        L, D = ldl_decomp(A, n)
        L_inv = tril_inv(L, n, unit_diag=True)
        return L, D, L_inv

    mid = n // 2
    A11 = A[:mid, :mid]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    L11, D11, L11_inv = block_ldl_core(A11, mid, block_size)
    D11_inv = diag_inv(D11, n=mid)

    L21 = A21 @ L11_inv.T @ D11_inv  # 2 matmuls
    S = A22 - L21 @ D11 @ L21.T      # 2 matmuls
    L22, D22, L22_inv = block_ldl_core(S, mid, block_size)

    _ = L22_inv @ L21 @ L11_inv      # 2 matmuls

    return SymMatrix((n, n)), SymMatrix((n, n)), SymMatrix((n, n))


def block_ldl_inversion(A, n, block_size):
    """Top-level: A^{-1} = L^{-T} @ D^{-1} @ L^{-1}"""
    _, D, L_inv = block_ldl_core(A, n, block_size)
    D_inv = diag_inv(D, n=n)
    return L_inv.T @ D_inv @ L_inv    # 2 matmuls


# ============================================================
# Run analysis
# ============================================================

if __name__ == "__main__":
    lib = get_kernel_library()
    print(f"Registered kernels: {lib.list_kernels()}\n")

    # ---- Single analysis ----
    print("="*60)
    print("  Direct Cholesky (n=16)")
    print("="*60)
    analyze(cholesky_inversion, n=16)

    # ---- Multi-algorithm comparison ----
    compare(
        (cholesky_inversion, {"n": 16}),
        (ldl_inversion, {"n": 16}),
        (block_cholesky_inversion, {"n": 16, "block_size": 4}),
        (block_ldl_inversion, {"n": 16, "block_size": 4}),
    )

    # ---- Different scales ----
    compare(
        (block_cholesky_inversion, {"n": 16, "block_size": 4}),
        (block_cholesky_inversion, {"n": 32, "block_size": 4}),
        (block_cholesky_inversion, {"n": 64, "block_size": 4}),
    )
