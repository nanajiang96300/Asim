"""
Trace-based static analysis for matrix algorithms.

Usage pattern:
  1. Decorate kernel functions with @kernel(name, cost_fn)
  2. Write your algorithm using these kernels + SymMatrix operations
  3. Call analyze(your_algo, n=16) for automatic metric reporting

The algorithm code executes in "trace mode" — SymMatrix objects carry
shape but no values, and each kernel call is recorded for analysis.
This means recursive/block algorithms are traced naturally without
any manual recipe construction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Any, Optional, Union
import inspect
import math


# ============================================================
# Data types (shared conceptual model with analyze.py)
# ============================================================

@dataclass
class FlopBreakdown:
    """FLOPs categorized by operation level. matrix + vector + scalar = total."""
    matrix: int = 0   # BLAS3
    vector: int = 0   # BLAS1
    scalar: int = 0

    @property
    def total(self) -> int:
        return self.matrix + self.vector + self.scalar

    def __add__(self, o: 'FlopBreakdown') -> 'FlopBreakdown':
        return FlopBreakdown(self.matrix+o.matrix, self.vector+o.vector, self.scalar+o.scalar)

    def __iadd__(self, o: 'FlopBreakdown') -> 'FlopBreakdown':
        self.matrix += o.matrix; self.vector += o.vector; self.scalar += o.scalar
        return self


@dataclass
class OpCounts:
    """Number of operations by level."""
    matrix: int = 0
    vector: int = 0
    scalar: int = 0

    @property
    def total(self) -> int:
        return self.matrix + self.vector + self.scalar

    def __add__(self, o: 'OpCounts') -> 'OpCounts':
        return OpCounts(self.matrix+o.matrix, self.vector+o.vector, self.scalar+o.scalar)

    def __iadd__(self, o: 'OpCounts') -> 'OpCounts':
        self.matrix += o.matrix; self.vector += o.vector; self.scalar += o.scalar
        return self


@dataclass
class ParallelismInfo:
    sequential_depth: int
    reduction_log2: int
    parallel_width: int
    description: str = ""

    @property
    def critical_path(self) -> int:
        return self.sequential_depth * (1 + self.reduction_log2)


@dataclass
class KernelResult:
    flops: FlopBreakdown = field(default_factory=FlopBreakdown)
    ops: OpCounts = field(default_factory=OpCounts)
    parallelism: ParallelismInfo = field(default_factory=lambda: ParallelismInfo(1,0,1))
    detail_mul: int = 0
    detail_add: int = 0
    detail_div: int = 0
    detail_sqrt: int = 0


# ============================================================
# SymMatrix — symbolic matrix (shape only, no values)
# ============================================================

class SymMatrix:
    """Symbolic matrix for trace mode. Carries shape; operations trigger traces."""

    def __init__(self, shape: Tuple[int, int]):
        self.shape = tuple(shape)

    @property
    def T(self) -> 'SymMatrix':
        return SymMatrix((self.shape[1], self.shape[0]))

    def __matmul__(self, other: 'SymMatrix') -> 'SymMatrix':
        m, k = self.shape
        k2, n = other.shape
        if k != k2:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        _trace("matmul", m=m, n=n, k=k)
        return SymMatrix((m, n))

    def __add__(self, other: 'SymMatrix') -> 'SymMatrix':
        result = _broadcast_shape(self.shape, other.shape)
        _trace("matrix_add", m=result[0], n=result[1])
        return SymMatrix(result)

    def __sub__(self, other: 'SymMatrix') -> 'SymMatrix':
        result = _broadcast_shape(self.shape, other.shape)
        _trace("matrix_add", m=result[0], n=result[1])
        return SymMatrix(result)

    def __neg__(self) -> 'SymMatrix':
        return SymMatrix(self.shape)

    def __mul__(self, scalar) -> 'SymMatrix':
        """Scalar × matrix: vector scaling op."""
        m, n = self.shape
        _trace("vector_scale", n=max(m, n))
        return SymMatrix(self.shape)

    def __rmul__(self, scalar) -> 'SymMatrix':
        return self.__mul__(scalar)

    def __getitem__(self, idx) -> 'SymMatrix':
        """Support slicing like A[:mid, :mid] or A[mid:, :mid]."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) != 2:
            raise IndexError("SymMatrix requires 2D indexing")

        def _slice_dim(size: int, slc) -> int:
            if isinstance(slc, slice):
                start = slc.start or 0
                stop = slc.stop or size
                step = slc.step or 1
                return len(range(start, stop, step))
            elif isinstance(slc, int):
                return 1
            else:
                raise IndexError(f"Unsupported index: {slc}")

        new_rows = _slice_dim(self.shape[0], idx[0])
        new_cols = _slice_dim(self.shape[1], idx[1])
        return SymMatrix((new_rows, new_cols))

    def __repr__(self):
        return f"SymMatrix({self.shape[0]}×{self.shape[1]})"


def _broadcast_shape(s1, s2):
    """Element-wise addition/subtraction shape."""
    if s1 == s2:
        return s1
    # Allow broadcasting for scalar-like ops
    r = max(s1[0], s2[0]) if s1[0] == 1 or s2[0] == 1 else s1[0]
    c = max(s1[1], s2[1]) if s1[1] == 1 or s2[1] == 1 else s1[1]
    return (r, c)


# ============================================================
# Tracer — global trace context
# ============================================================

_tracer: Optional['Tracer'] = None


class Tracer:
    """Context manager that collects kernel invocations during trace execution."""

    def __init__(self):
        self.trace: List[Tuple[str, dict]] = []
        self.enabled = False

    def record(self, name: str, **params):
        if self.enabled:
            self.trace.append((name, params))

    def __enter__(self) -> 'Tracer':
        global _tracer
        self.enabled = True
        _tracer = self
        return self

    def __exit__(self, *args):
        global _tracer
        self.enabled = False
        _tracer = None

    def print_trace(self):
        print(f"\n  Trace ({len(self.trace)} entries):")
        for i, (name, params) in enumerate(self.trace):
            p = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    [{i}] {name}({p})")


def _trace(name: str, **params):
    """Record a kernel invocation if tracing is active."""
    if _tracer is not None and _tracer.enabled:
        _tracer.record(name, **params)


# ============================================================
# Kernel library
# ============================================================

class KernelDef:
    """A registered kernel with its cost model."""

    def __init__(self, name: str, cost_fn: Callable[..., KernelResult],
                 description: str = ""):
        self.name = name
        self.cost_fn = cost_fn
        self.description = description

    def analyze(self, **params) -> KernelResult:
        return self.cost_fn(**params)


class KernelLibrary:
    def __init__(self):
        self._kernels: Dict[str, KernelDef] = {}

    def register(self, kernel: KernelDef):
        self._kernels[kernel.name] = kernel

    def get(self, name: str) -> KernelDef:
        if name not in self._kernels:
            raise KeyError(f"Unknown kernel: '{name}'. Registered: {list(self._kernels.keys())}")
        return self._kernels[name]

    def list_kernels(self) -> List[str]:
        return list(self._kernels.keys())


# Global kernel library (populated by @kernel decorator and builtins)
_kernel_lib = KernelLibrary()


def get_kernel_library() -> KernelLibrary:
    return _kernel_lib


# ============================================================
# @kernel decorator
# ============================================================

def kernel(name: str, cost_fn: Callable[..., KernelResult],
           description: str = "",
           return_shape: Union[Callable, Tuple, List, None] = None):
    """
    Decorator that registers a function as a traceable kernel.

    In trace mode: function body is NOT executed. Instead, the call is
    recorded with its parameters, and a SymMatrix (or tuple thereof) is
    returned based on return_shape.

    In normal mode: the original function executes normally.

    Args:
        name: kernel name (used in trace records and library lookup)
        cost_fn: function(**params) -> KernelResult (cost model)
        description: human-readable description
        return_shape: function(**params) -> shape, or a constant shape,
                      or a callable returning a shape, or list thereof.
                      If None, shape is inferred from cost_fn params
                      (looks for 'n' or 'm'/'n' keys).
    """
    def decorator(func):
        # Register cost model
        _kernel_lib.register(KernelDef(name, cost_fn, description))

        def wrapper(*args, **kwargs):
            if _tracer is not None and _tracer.enabled:
                # Trace mode: extract params, record, return symbolic shape
                params = _bind_params(func, args, kwargs)
                _trace(name, **params)
                return _make_return(return_shape, params, name)
            else:
                # Normal mode: execute
                return func(*args, **kwargs)

        # Preserve signature for introspection
        wrapper.__name__ = func.__name__
        wrapper.__signature__ = inspect.signature(func)
        wrapper.__wrapped__ = func
        return wrapper

    return decorator


def _bind_params(func, args, kwargs) -> dict:
    """Bind positional and keyword args to parameter names."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        # Remove 'self' if present (method call)
        params.pop('self', None)
        # Remove SymMatrix args (A, L, etc.) — keep only scalar params
        return {k: v for k, v in params.items() if not isinstance(v, SymMatrix)}
    except Exception:
        # Fallback: just use kwargs
        return {k: v for k, v in kwargs.items() if not isinstance(v, SymMatrix)}


def _make_return(return_shape, params: dict, name: str):
    """Build the return value (SymMatrix or tuple) for trace mode."""
    if return_shape is None:
        # Infer: most kernels return an n×n matrix
        n = params.get('n')
        if n is not None:
            return SymMatrix((n, n))
        m, n = params.get('m'), params.get('n')
        if m is not None and n is not None:
            return SymMatrix((m, n))
        # Default: return None (void kernel)
        return None

    if callable(return_shape):
        shape = return_shape(**params)
    else:
        shape = return_shape

    if isinstance(shape, (list, tuple)) and len(shape) > 0:
        if isinstance(shape[0], (list, tuple)):
            # Multiple shapes → return tuple of SymMatrices
            return tuple(SymMatrix(tuple(s)) for s in shape)
        elif isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[0], int):
            # Single shape (m,n)
            return SymMatrix(tuple(shape))
        else:
            return SymMatrix(tuple(shape))
    elif isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[0], int):
        return SymMatrix(shape)

    return None


# ============================================================
# Built-in cost functions
# ============================================================

def _ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return (x - 1).bit_length()


def _cost_cholesky_decomp(**params) -> KernelResult:
    """
    Left-looking scalar Cholesky decomposition.

    for i in 0..n-1:                          ← sequential (row i needs all prior rows)
      for j in 0..i:                          ← SEQUENTIAL (L[i,j] needs L[i,0..j-1])
        dot(L[i,:j], L[j,:j])                 ← j muls + j-1 adds  (BLAS1 dot product)
        A[i,j]-s ; if i==j: sqrt else: /L[j,j]  ← scalar ops

    Only 1 dot product active at a time (j-loop serial).
    Within each dot, j multiplications reducible in log2(j) steps.
    """
    n = params["n"]
    flops = FlopBreakdown()
    ops = OpCounts()
    dmul, dadd, ddiv, dsqrt = 0, 0, 0, 0
    longest_dot = 0

    for i in range(n):
        for j in range(i + 1):
            if j > 0:
                dmul += j; dadd += j - 1
                flops.vector += j + (j - 1); ops.vector += 1
                longest_dot = max(longest_dot, j)
            dadd += 1; flops.scalar += 1; ops.scalar += 1     # subtraction
            if i == j:
                dsqrt += 1; flops.scalar += 1; ops.scalar += 1  # sqrt
            else:
                ddiv += 1; flops.scalar += 1; ops.scalar += 1  # /L[j,j]

    para = ParallelismInfo(
        sequential_depth=n, reduction_log2=_ceil_log2(longest_dot),
        parallel_width=1,
        description=(
            f"Left-looking: i-loop ({n} steps) and inner j-loop are both sequential "
            f"(L[i,j] depends on L[i,0..j-1]). 1 dot product at a time."
        ),
    )
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=dmul, detail_add=dadd, detail_div=ddiv, detail_sqrt=dsqrt)


def _cost_ldl_decomp(**params) -> KernelResult:
    """
    Left-looking scalar LDL^T decomposition. Same sequential structure as Cholesky.
    Each dot product element costs 2 muls (L[i,k]*L[j,k]*D[k,k]), no sqrt.
    """
    n = params["n"]
    flops = FlopBreakdown()
    ops = OpCounts()
    dmul, dadd, ddiv, dsqrt = 0, 0, 0, 0
    longest_dot = 0

    for i in range(n):
        for j in range(i):
            if j > 0:
                dmul += 2*j; dadd += j-1
                flops.vector += 2*j + (j-1); ops.vector += 1
                longest_dot = max(longest_dot, j)
            dadd += 1; ddiv += 1; flops.scalar += 2; ops.scalar += 2  # sub + div
        if i > 0:
            dmul += 2*i; dadd += i-1
            flops.vector += 2*i + (i-1); ops.vector += 1
            longest_dot = max(longest_dot, i)
        dadd += 1; flops.scalar += 1; ops.scalar += 1                # diagonal sub

    para = ParallelismInfo(
        sequential_depth=n, reduction_log2=_ceil_log2(longest_dot),
        parallel_width=1,
        description=(
            f"j-loop within each row is sequential (same as Cholesky). "
            f"1 active dot product at a time; longest has {longest_dot} elements, "
            f"each costing 2 muls (with D scaling)."
        ),
    )
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=dmul, detail_add=dadd, detail_div=ddiv, detail_sqrt=dsqrt)


def _cost_tril_inv(**params) -> KernelResult:
    n = params["n"]
    unit_diag = params.get("unit_diag", False)
    flops = FlopBreakdown()
    ops = OpCounts()
    dmul, dadd, ddiv, dsqrt = 0, 0, 0, 0
    longest_dot = 0

    for i in range(n):
        if not unit_diag:
            ddiv += 1; flops.scalar += 1; ops.scalar += 1
        for j in range(i):
            k = i - j
            dmul += k; dadd += k - 1
            flops.vector += k + (k - 1); ops.vector += 1
            longest_dot = max(longest_dot, k)
        if i > 0:
            dadd += i; flops.scalar += i; ops.scalar += i
            if not unit_diag:
                ddiv += i; flops.scalar += i; ops.scalar += i

    para = ParallelismInfo(
        sequential_depth=n, reduction_log2=_ceil_log2(longest_dot),
        parallel_width=1,
        description=(
            f"j-loop within each row is sequential (X[i,j] depends on X[k,j] for k<j). "
            f"1 active dot product at a time; longest has {longest_dot} elements."
        ),
    )
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=dmul, detail_add=dadd, detail_div=ddiv, detail_sqrt=dsqrt)


def _cost_diag_inv(**params) -> KernelResult:
    n = params["n"]
    para = ParallelismInfo(1, 0, 1, f"1 kernel call ({n} independent divisions inside)")
    return KernelResult(flops=FlopBreakdown(scalar=n), ops=OpCounts(scalar=n),
                        parallelism=para, detail_div=n)


def _cost_matmul(**params) -> KernelResult:
    m, n_, k = params["m"], params["n"], params["k"]
    mul = m * n_ * k
    add = m * n_ * (k - 1)
    para = ParallelismInfo(
        1, _ceil_log2(k), 1,
        f"1 BLAS3 call. All {m*n_} outputs independent — parallelism inside the hardware kernel.",
    )
    return KernelResult(flops=FlopBreakdown(matrix=mul+add), ops=OpCounts(matrix=1),
                        parallelism=para, detail_mul=mul, detail_add=add)


def _cost_matrix_add(**params) -> KernelResult:
    """Element-wise matrix ± : modeled as min(m,n) vector ops, m*n vector FLOPs."""
    m, n_ = params["m"], params["n"]
    vec_ops = min(m, n_)
    flops = FlopBreakdown(vector=m * n_)
    ops = OpCounts(vector=vec_ops)
    para = ParallelismInfo(1, 0, 1,
        f"{m}×{n_} element-wise ±: {m*n_} FLOPs, modeled as {vec_ops} vector ops")
    return KernelResult(flops=flops, ops=ops, parallelism=para, detail_add=m * n_)


def _cost_newton_init(**params) -> KernelResult:
    """
    X_0 = A^T / ||A||^2_F.
    - n² squares (mul) + n²-1 adds = sum reduction
    - 1 div for reciprocal
    - n² muls to scale A^T by alpha
    Total: ~3n² vector FLOPs + 1 scalar div.
    """
    n = params["n"]
    flops = FlopBreakdown(vector=3 * n * n - 1, scalar=1)
    ops = OpCounts(vector=2 * n, scalar=1)
    para = ParallelismInfo(2, _ceil_log2(n * n), 1,
        f"X_0 = A^T/||A||²_F: {n} vector reductions (norm) + {n} vector scalings (transpose scale)")
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=2 * n * n, detail_add=n * n - 1, detail_div=1)


def _cost_bj_block_inv(**params) -> KernelResult:
    """
    Invert n/blk independent diagonal blocks of A.
    For 2×2 blocks ("direct2x2"):
      det = a00*a11 - a01*a10,  inv = [[a11,-a01],[-a10,a00]]/det
      Per block: 6 muls + 1 add + 1 div ≈ 8 FLOPs.
    """
    n = params["n"]
    blk = params.get("block_size", 2)
    n_blk = n // blk
    per_blk_mul, per_blk_add, per_blk_div = 6, 1, 1
    flops = FlopBreakdown(scalar=n_blk * (per_blk_mul + per_blk_add + per_blk_div))
    ops = OpCounts(scalar=n_blk)
    para = ParallelismInfo(1, 0, n_blk,
        f"{n_blk} independent {blk}×{blk} block inversions (fully parallel)")
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=n_blk*per_blk_mul, detail_add=n_blk*per_blk_add,
                        detail_div=n_blk*per_blk_div)


def _cost_vector_scale(**params) -> KernelResult:
    """Scalar × n×n matrix: n² muls, modeled as n vector scalings."""
    n = params["n"]
    flops = FlopBreakdown(vector=n * n)
    ops = OpCounts(vector=n)
    para = ParallelismInfo(1, 0, 1, f"Scalar × {n}×{n} matrix: {n} independent vector scalings")
    return KernelResult(flops=flops, ops=ops, parallelism=para, detail_mul=n * n)


def _cost_matmul_chain(**params) -> KernelResult:
    n = params["n"]
    has_d = params.get("has_d", False)
    if has_d:
        mm = _cost_matmul(m=n, n=n, k=n)
        return KernelResult(
            flops=FlopBreakdown(matrix=mm.flops.matrix, vector=n*n, scalar=0),
            ops=OpCounts(matrix=1, vector=n, scalar=0),
            parallelism=mm.parallelism,
            detail_mul=mm.detail_mul + n*n, detail_add=mm.detail_add,
        )
    return _cost_matmul(m=n, n=n, k=n)


# ============================================================
# Register built-in kernels
# ============================================================

def _register_builtins():
    lib = _kernel_lib
    # Only register if not already present (allow user overrides)
    if "cholesky_decomp" not in lib._kernels:
        lib.register(KernelDef("cholesky_decomp", _cost_cholesky_decomp,
                               "Cholesky decomposition A=L@L^T"))
    if "ldl_decomp" not in lib._kernels:
        lib.register(KernelDef("ldl_decomp", _cost_ldl_decomp,
                               "LDL^T decomposition A=L@D@L^T"))
    if "tril_inv" not in lib._kernels:
        lib.register(KernelDef("tril_inv", _cost_tril_inv,
                               "Lower-triangular inverse (forward substitution)"))
    if "diag_inv" not in lib._kernels:
        lib.register(KernelDef("diag_inv", _cost_diag_inv,
                               "Diagonal reciprocal D^{-1}"))
    if "matmul" not in lib._kernels:
        lib.register(KernelDef("matmul", _cost_matmul,
                               "Dense matrix multiply (BLAS3)"))
    if "matmul_chain" not in lib._kernels:
        lib.register(KernelDef("matmul_chain", _cost_matmul_chain,
                               "Final assembly L^{-T}@[D^{-1}]@L^{-1}"))
    if "matrix_add" not in lib._kernels:
        lib.register(KernelDef("matrix_add", _cost_matrix_add,
                               "Element-wise matrix addition/subtraction (vector unit)"))
    if "newton_init" not in lib._kernels:
        lib.register(KernelDef("newton_init", _cost_newton_init,
                               "Newton initial guess X_0 = A^T/||A||^2_F"))
    if "bj_block_inv" not in lib._kernels:
        lib.register(KernelDef("bj_block_inv", _cost_bj_block_inv,
                               "Block-Jacobi: invert diagonal blocks (2×2 direct formula)"))
    if "vector_scale" not in lib._kernels:
        lib.register(KernelDef("vector_scale", _cost_vector_scale,
                               "Scalar × matrix: vector scaling (Vector unit)"))


_register_builtins()


# ============================================================
# Analyzer (reads trace, computes metrics, prints report)
# ============================================================

@dataclass
class AlgorithmAnalysis:
    name: str
    trace: List[Tuple[str, dict]]
    flops: FlopBreakdown = field(default_factory=FlopBreakdown)
    ops: OpCounts = field(default_factory=OpCounts)
    critical_path: int = 0
    max_parallelism: int = 0
    min_parallelism: int = 2**31  # bottleneck step (lower = worse)
    detail_mul: int = 0
    detail_add: int = 0
    detail_div: int = 0
    detail_sqrt: int = 0


class Analyzer:
    def __init__(self, library: KernelLibrary = None):
        self.library = library or _kernel_lib

    def analyze(self, name: str, trace: List[Tuple[str, dict]]) -> AlgorithmAnalysis:
        a = AlgorithmAnalysis(name=name, trace=trace)
        for kernel_name, params in trace:
            kern = self.library.get(kernel_name)
            r = kern.analyze(**params)
            a.flops += r.flops
            a.ops += r.ops
            a.critical_path += r.parallelism.critical_path
            pw = r.parallelism.parallel_width
            a.max_parallelism = max(a.max_parallelism, pw)
            a.min_parallelism = min(a.min_parallelism, pw)
            a.detail_mul += r.detail_mul
            a.detail_add += r.detail_add
            a.detail_div += r.detail_div
            a.detail_sqrt += r.detail_sqrt
        return a

    def print_report(self, a: AlgorithmAnalysis):
        f = a.flops
        o = a.ops
        t = f.total
        ot = o.total

        print(f"\n{'='*60}")
        print(f"  {a.name}")
        print(f"{'='*60}")

        # Section 1: FLOPs
        print(f"\n  1. Total FLOPs{' ' * 33} {t:>10,}")
        print(f"     ├─ Matrix FLOPs (BLAS3):  {f.matrix:>10,}  ({f.matrix/t*100:5.1f}%)")
        print(f"     ├─ Vector FLOPs (BLAS1):  {f.vector:>10,}  ({f.vector/t*100:5.1f}%)")
        print(f"     └─ Scalar FLOPs:          {f.scalar:>10,}  ({f.scalar/t*100:5.1f}%)")
        print(f"     (detail: mul={a.detail_mul:,} add={a.detail_add:,} "
              f"div={a.detail_div:,} sqrt={a.detail_sqrt:,})")

        # Section 2: Operation count
        print(f"\n  2. Operation Count{' ' * 26} {ot:>10}")
        print(f"     ├─ Matrix ops:  {o.matrix:>10}")
        print(f"     ├─ Vector ops:  {o.vector:>10}")
        print(f"     └─ Scalar ops:  {o.scalar:>10}")

        # Section 3: Parallelism
        # ---- Section 3: Parallelism ----
        print(f"\n  3. Parallelism (ops-level: each kernel call = 1 operation)")
        print(f"     ├─ Critical path:       {a.critical_path:>8}  steps (serial depth)")
        print(f"     ├─ Total kernel calls:  {len(a.trace):>8}")
        print(f"     └─ Ops per crit-path:   {len(a.trace)/a.critical_path:>8.2f}  (avg concurrency)")

        # Trace details
        print(f"\n  Kernel call trace ({len(a.trace)} calls):")
        from collections import Counter
        # Convert params dict → hashable tuple of sorted items
        hashable_trace = [(name, tuple(sorted(p.items()))) for name, p in a.trace]
        call_counts = Counter(hashable_trace)
        for (name, params_tuple), count in call_counts.most_common():
            params_str = ", ".join(f"{k}={v}" for k, v in params_tuple)
            suffix = f"  ×{count}" if count > 1 else ""
            print(f"    {name}({params_str}){suffix}")

    def print_comparison(self, analyses: List[AlgorithmAnalysis]):
        if len(analyses) < 2:
            return
        print(f"\n{'='*70}")
        print(f"  Comparison")
        print(f"{'='*70}")
        header = f"  {'Metric':<36}"
        for a in analyses:
            header += f"  {a.name:>16}"
        print(header)
        print(f"  {'─'*68}")

        def _show(label, *vals):
            line = f"  {label:<36}"
            for v in vals:
                line += f"  {v:>16}"
            print(line)

        def _row(label, getter, fmt="{:>,}"):
            vals = [fmt.format(getter(a)) for a in analyses]
            _show(label, *vals)

        def _pct(label, getter):
            vals = []
            for a in analyses:
                v = getter(a)
                den = a.flops.total
                vals.append(f"{v/den*100:.1f}%" if den else "-")
            _show(label, *vals)

        def _sep(label):
            _show(label, *([""] * len(analyses)))

        _sep("── 1. Total FLOPs ──")
        _row("  Total FLOPs",      lambda a: a.flops.total)
        _row("  Matrix FLOPs",     lambda a: a.flops.matrix)
        _row("  Vector FLOPs",     lambda a: a.flops.vector)
        _row("  Scalar FLOPs",     lambda a: a.flops.scalar)
        _pct("  Matrix FLOPs %",   lambda a: a.flops.matrix)
        _pct("  Vector FLOPs %",   lambda a: a.flops.vector)
        _pct("  Scalar FLOPs %",   lambda a: a.flops.scalar)
        _show("")
        _sep("── 2. Operation Count ──")
        _row("  Total ops",        lambda a: a.ops.total)
        _row("  Matrix ops",       lambda a: a.ops.matrix)
        _row("  Vector ops",       lambda a: a.ops.vector)
        _row("  Scalar ops",       lambda a: a.ops.scalar)
        _show("")
        _sep("── 3. Parallelism ──")
        _row("  Critical path",       lambda a: a.critical_path)
        _row("  Total kernel calls",  lambda a: len(a.trace))
        _row("  Ops / crit-path",     lambda a: f"{len(a.trace)/a.critical_path:.2f}" if a.critical_path else "-", fmt="{:>}")


# ============================================================
# Top-level analyze() function
# ============================================================

def analyze(fn: Callable, n: int = None, block_size: int = None,
            **extra_params) -> AlgorithmAnalysis:
    """
    Run a matrix algorithm in trace mode and produce a static analysis report.

    Args:
        fn: the algorithm function. Should accept `A` (SymMatrix) as first arg.
        n: matrix size. If given, creates A = SymMatrix((n,n)).
        block_size: optionally passed to fn for block algorithms.
        **extra_params: additional keyword arguments passed to fn.

    Returns:
        AlgorithmAnalysis with full metrics.

    Example:
        analyze(cholesky_inversion, n=16)
        analyze(block_cholesky_inversion, n=16, block_size=4)
    """
    # Build parameters
    params = dict(extra_params)
    if n is not None:
        params.setdefault("A", SymMatrix((n, n)))
        params.setdefault("n", n)
    if block_size is not None:
        params["block_size"] = block_size

    # Run in trace mode
    with Tracer() as tracer:
        fn(**params)

    # Analyze trace
    lib = _kernel_lib
    analyzer = Analyzer(lib)
    name = getattr(fn, '__name__', 'unknown')
    if n:
        name = f"{name} (n={n}" + (f", block={block_size})" if block_size else ")")

    result = analyzer.analyze(name, tracer.trace)
    analyzer.print_report(result)
    return result


def compare(*algos_and_params) -> List[AlgorithmAnalysis]:
    """
    Run multiple algorithms in trace mode and print a comparison.

    Each argument can be:
      (fn, params)           — name inferred from fn.__name__
      ("Name", fn, params)   — explicit display name

    Example:
        compare(
            (cholesky_inversion, {"n": 16}),
            ("Block LDL", block_ldl_inversion, {"n": 16, "block_size": 4}),
        )
    """
    lib = _kernel_lib
    analyzer = Analyzer(lib)
    analyses = []

    for entry in algos_and_params:
        if len(entry) == 3:
            display_name, fn, params = entry
        else:
            fn, params = entry
            display_name = None
        n = params.get("n")
        params = dict(params)
        if "A" not in params and n:
            params["A"] = SymMatrix((n, n))
        if "n" not in params and n:
            params["n"] = n

        with Tracer() as tracer:
            fn(**params)

        name = display_name or getattr(fn, '__name__', 'unknown')
        if n:
            bs = params.get("block_size")
            name = f"{name} (n={n}" + (f", b={bs})" if bs else ")")

        a = analyzer.analyze(name, tracer.trace)
        analyses.append(a)
        analyzer.print_report(a)

    analyzer.print_comparison(analyses)
    return analyses
