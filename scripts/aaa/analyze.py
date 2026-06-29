"""
Static analysis framework for matrix algorithms.

Architecture:
  KernelDef       — primitive operation with built-in cost model
  KernelLibrary   — registry of all known kernels
  AlgorithmRecipe — named sequence of (kernel_name, params) steps
  Analyzer        — takes recipes, aggregates metrics, prints reports

Output (3 sections, each sums to its own total):
  1. Total FLOPs    = matrix FLOPs + vector FLOPs + scalar FLOPs
  2. Operation count = matrix ops  + vector ops  + scalar ops
  3. Parallelism: critical path & peak parallelism
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Any
import math


# ============================================================
# Data structures
# ============================================================

@dataclass
class FlopBreakdown:
    """FLOPs categorized by operation level.  matrix + vector + scalar = total."""
    matrix: int = 0   # BLAS3: matmul FLOPs
    vector: int = 0   # BLAS1: dot-product FLOPs
    scalar: int = 0   # individual element ops (div, sqrt, single ±)

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
    """Number of operations (instructions) by level.  matrix + vector + scalar = total."""
    matrix: int = 0   # BLAS3 kernel calls
    vector: int = 0   # BLAS1 kernel calls (dot, axpy, scal)
    scalar: int = 0   # individual element ops

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
    """Parallelism model for a kernel.

    sequential_depth — must-be-sequential layers (e.g. i-loop iterations)
    reduction_log2  — log2 depth of reduction within each layer
    parallel_width  — max independent operations that can run concurrently
    """
    sequential_depth: int
    reduction_log2: int
    parallel_width: int
    description: str = ""

    @property
    def critical_path(self) -> int:
        return self.sequential_depth * (1 + self.reduction_log2)


@dataclass
class KernelResult:
    """What a kernel invocation contributes to the analysis."""
    flops: FlopBreakdown       # FLOPs by BLAS level (sums to total)
    ops: OpCounts              # operation count by level
    parallelism: ParallelismInfo
    # optional: detailed mul/add/div/sqrt counts
    detail_mul:  int = 0
    detail_add:  int = 0
    detail_div:  int = 0
    detail_sqrt: int = 0

    @property
    def detail_total(self) -> int:
        return self.detail_mul + self.detail_add + self.detail_div + self.detail_sqrt


# ============================================================
# Kernel definition & library
# ============================================================

CostFn = Callable[..., KernelResult]


class KernelDef:
    def __init__(self, name: str, description: str, cost_fn: CostFn):
        self.name = name
        self.description = description
        self.cost_fn = cost_fn

    def analyze(self, **params) -> KernelResult:
        return self.cost_fn(**params)


class KernelLibrary:
    def __init__(self):
        self._kernels: Dict[str, KernelDef] = {}

    def register(self, kernel: KernelDef):
        self._kernels[kernel.name] = kernel

    def get(self, name: str) -> KernelDef:
        if name not in self._kernels:
            raise KeyError(f"Unknown kernel: {name}. Available: {list(self._kernels.keys())}")
        return self._kernels[name]

    def list_kernels(self) -> List[str]:
        return list(self._kernels.keys())


# ============================================================
# Helper
# ============================================================

def _ceil_log2(x: int) -> int:
    """ceil(log2(x)), or 0 when no reduction is needed (x <= 1)."""
    if x <= 1:
        return 0
    return (x - 1).bit_length()


# ============================================================
# Cost functions — each returns a KernelResult
# ============================================================

def _cost_cholesky_decomp(**params) -> KernelResult:
    """
    Scalar Cholesky  A = L @ L^T.

    for i in 0..n-1:
      for j in 0..i:
        dot(L[i,:j], L[j,:j])       ← vector (BLAS1): j muls + (j-1) adds
        A[i,j] - s                  ← scalar subtraction
        if i==j: sqrt  else: /L[j,j] ← scalar
    """
    n = params["n"]
    flops = FlopBreakdown()
    ops = OpCounts()
    dmul, dadd, ddiv, dsqrt = 0, 0, 0, 0
    longest_dot = 0

    for i in range(n):
        for j in range(i + 1):
            if j > 0:
                dmul += j
                dadd += j - 1
                flops.vector += j + (j - 1)
                ops.vector += 1
                longest_dot = max(longest_dot, j)
            dadd += 1; flops.scalar += 1         # A[i,j] - s
            if i == j:
                dsqrt += 1; flops.scalar += 1
            else:
                ddiv += 1; flops.scalar += 1

    para = ParallelismInfo(
        sequential_depth=n,
        reduction_log2=_ceil_log2(longest_dot),
        parallel_width=n * n // 2,
        description=(
            f"i-loop: sequential ({n} iterations). "
            f"Within each i: up to n parallel j-iterations, "
            f"each a dot product reducible in ceil(log2({longest_dot}))={_ceil_log2(longest_dot)} steps."
        ),
    )
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=dmul, detail_add=dadd, detail_div=ddiv, detail_sqrt=dsqrt)


def _cost_ldl_decomp(**params) -> KernelResult:
    """
    Scalar LDL^T  A = L @ D @ L^T.

    Same structure as Cholesky but dot products include D-scaling
    (2 muls per element: L[i,k]*L[j,k] and *D[k,k]), and no sqrt.
    """
    n = params["n"]
    flops = FlopBreakdown()
    ops = OpCounts()
    dmul, dadd, ddiv, dsqrt = 0, 0, 0, 0
    longest_dot = 0

    for i in range(n):
        # off-diagonal
        for j in range(i):
            if j > 0:
                dmul += 2 * j
                dadd += j - 1
                flops.vector += 2 * j + (j - 1)
                ops.vector += 1
                longest_dot = max(longest_dot, j)
            dadd += 1; ddiv += 1               # (A[i,j]-s) / D[j,j]
            flops.scalar += 2

        # diagonal
        if i > 0:
            dmul += 2 * i
            dadd += i - 1
            flops.vector += 2 * i + (i - 1)
            ops.vector += 1
            longest_dot = max(longest_dot, i)
        dadd += 1                               # A[i,i] - s
        flops.scalar += 1

    para = ParallelismInfo(
        sequential_depth=n,
        reduction_log2=_ceil_log2(longest_dot),
        parallel_width=n * n // 2,
        description=(
            f"i-loop: sequential ({n} iterations). "
            f"Same parallelism as Cholesky; each dot element costs 2 muls (with D scaling)."
        ),
    )
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=dmul, detail_add=dadd, detail_div=ddiv, detail_sqrt=dsqrt)


def _cost_tril_inv(**params) -> KernelResult:
    """
    Scalar lower-triangular inverse (forward substitution).

    for i in 0..n-1:
      X[i,i] = 1  or  1/L[i,i]                 ← scalar (div for non-unit)
      for j in 0..i-1:
        dot(L[i,j:i], X[j:i,j])                ← vector (BLAS1)
        X[i,j] = -s  or  -s / L[i,i]           ← scalar (neg [+ div])
    """
    n = params["n"]
    unit_diag = params.get("unit_diag", False)
    flops = FlopBreakdown()
    ops = OpCounts()
    dmul, dadd, ddiv, dsqrt = 0, 0, 0, 0
    longest_dot = 0

    for i in range(n):
        if not unit_diag:
            ddiv += 1; flops.scalar += 1; ops.scalar += 1   # X[i,i] = 1/L[i,i]

        for j in range(i):
            k = i - j
            dmul += k; dadd += k - 1
            flops.vector += k + (k - 1)
            ops.vector += 1
            longest_dot = max(longest_dot, k)

        if i > 0:
            dadd += i; flops.scalar += i; ops.scalar += i    # negations
            if not unit_diag:
                ddiv += i; flops.scalar += i; ops.scalar += i

    para = ParallelismInfo(
        sequential_depth=n,
        reduction_log2=_ceil_log2(longest_dot),
        parallel_width=n * n // 2,
        description=(
            f"i-loop: sequential ({n} iterations). "
            f"Within each i, up to i parallel j-iterations, "
            f"each a dot product of length up to {longest_dot}."
        ),
    )
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=dmul, detail_add=dadd, detail_div=ddiv, detail_sqrt=dsqrt)


def _cost_diag_inv(**params) -> KernelResult:
    """Element-wise reciprocal of diagonal matrix: D^{-1}[i,i] = 1/D[i,i]."""
    n = params["n"]
    para = ParallelismInfo(
        sequential_depth=1, reduction_log2=0, parallel_width=n,
        description=f"All {n} diagonal elements independent — fully parallel, 1 step.",
    )
    return KernelResult(
        flops=FlopBreakdown(scalar=n),
        ops=OpCounts(scalar=n),
        parallelism=para,
        detail_div=n,
    )


def _cost_matmul(**params) -> KernelResult:
    """
    Dense matrix multiply  C = A @ B   (A: m×k, B: k×n).

    Each C[i,j] = dot(row_i(A), col_j(B)) — length-k dot.
    Total: m·n·k muls + m·n·(k-1) adds.
    """
    m, n_, k = params["m"], params["n"], params["k"]
    mul = m * n_ * k
    add = m * n_ * (k - 1)
    para = ParallelismInfo(
        sequential_depth=1,
        reduction_log2=_ceil_log2(k),
        parallel_width=m * n_,
        description=(
            f"All {m*n_} outputs independent. "
            f"Each is a dot product of length {k}, "
            f"reducible in ceil(log2({k}))={_ceil_log2(k)} steps."
        ),
    )
    return KernelResult(
        flops=FlopBreakdown(matrix=mul + add),
        ops=OpCounts(matrix=1),
        parallelism=para,
        detail_mul=mul, detail_add=add,
    )


def _cost_matmul_chain(**params) -> KernelResult:
    """
    Final assembly:  L^{-T} @ [D^{-1}] @ L^{-1}.

    Without D: just matmul.
    With D:    first scale L^{-1} rows by D^{-1} (vector), then matmul.
    """
    n = params["n"]
    has_D = params.get("has_D", False)

    flops = FlopBreakdown()
    ops = OpCounts()
    dmul, dadd = 0, 0

    if has_D:
        dmul += n * n
        flops.vector += n * n
        ops.vector += n                     # n independent vector scalings

    mm = _cost_matmul(m=n, n=n, k=n)
    flops += mm.flops
    ops += mm.ops
    dmul += mm.detail_mul
    dadd += mm.detail_add

    para = ParallelismInfo(
        sequential_depth=1 + (1 if has_D else 0),
        reduction_log2=_ceil_log2(n) + (1 if has_D else 0),
        parallel_width=n * n,
        description=mm.parallelism.description,
    )
    return KernelResult(flops=flops, ops=ops, parallelism=para,
                        detail_mul=dmul, detail_add=dadd)


# ============================================================
# Kernel library
# ============================================================

def build_default_library() -> KernelLibrary:
    lib = KernelLibrary()
    lib.register(KernelDef("cholesky_decomp",
        "Cholesky decomposition A = L@L^T", _cost_cholesky_decomp))
    lib.register(KernelDef("ldl_decomp",
        "LDL^T decomposition A = L@D@L^T", _cost_ldl_decomp))
    lib.register(KernelDef("tril_inv",
        "Lower-triangular inverse (forward substitution)", _cost_tril_inv))
    lib.register(KernelDef("diag_inv",
        "Diagonal reciprocal D^{-1}", _cost_diag_inv))
    lib.register(KernelDef("matmul",
        "Dense matrix multiply (BLAS3)", _cost_matmul))
    lib.register(KernelDef("matmul_chain",
        "Final assembly L^{-T} @ [D^{-1}] @ L^{-1}", _cost_matmul_chain))
    return lib


# ============================================================
# Algorithm recipe
# ============================================================

@dataclass
class AlgorithmRecipe:
    name: str
    steps: List[Tuple[str, dict]]   # [(kernel_name, params), ...]


# ============================================================
# Analyzer
# ============================================================

@dataclass
class StepAnalysis:
    kernel_name: str
    params: dict
    result: KernelResult


@dataclass
class AlgorithmAnalysis:
    recipe: AlgorithmRecipe
    steps: List[StepAnalysis]

    # aggregated across all steps
    flops: FlopBreakdown = field(default_factory=FlopBreakdown)
    ops: OpCounts = field(default_factory=OpCounts)
    critical_path: int = 0
    max_parallelism: int = 0
    detail_mul: int = 0
    detail_add: int = 0
    detail_div: int = 0
    detail_sqrt: int = 0

    def __post_init__(self):
        for s in self.steps:
            r = s.result
            self.flops += r.flops
            self.ops += r.ops
            self.critical_path += r.parallelism.critical_path
            self.max_parallelism = max(self.max_parallelism, r.parallelism.parallel_width)
            self.detail_mul += r.detail_mul
            self.detail_add += r.detail_add
            self.detail_div += r.detail_div
            self.detail_sqrt += r.detail_sqrt


class Analyzer:
    def __init__(self, library: KernelLibrary = None):
        self.library = library or build_default_library()
        self.results: List[AlgorithmAnalysis] = []

    def analyze(self, recipe: AlgorithmRecipe) -> AlgorithmAnalysis:
        steps = []
        for kernel_name, params in recipe.steps:
            kernel = self.library.get(kernel_name)
            result = kernel.analyze(**params)
            steps.append(StepAnalysis(kernel_name, params, result))
        analysis = AlgorithmAnalysis(recipe=recipe, steps=steps)
        self.results.append(analysis)
        return analysis

    def analyze_all(self, recipes: List[AlgorithmRecipe]) -> List[AlgorithmAnalysis]:
        return [self.analyze(r) for r in recipes]

    # ================================================================
    # Display
    # ================================================================

    def print_algorithm(self, a: AlgorithmAnalysis, n: int = None):
        """Print a single algorithm's analysis in 3 clear sections."""
        f = a.flops
        o = a.ops
        total = f.total
        n = n or a.steps[0].params.get("n", "?")

        print(f"\n{'─'*60}")
        print(f"  {a.recipe.name} (n={n})")
        print(f"{'─'*60}")

        # ---- Section 1: Total FLOPs ----
        print(f"\n  1. Total FLOPs{' ' * 33} {total:>10,}")
        print(f"     ├─ Matrix FLOPs (BLAS3):  {f.matrix:>10,}  ({f.matrix/total*100:5.1f}%)")
        print(f"     ├─ Vector FLOPs (BLAS1):  {f.vector:>10,}  ({f.vector/total*100:5.1f}%)")
        print(f"     └─ Scalar FLOPs:          {f.scalar:>10,}  ({f.scalar/total*100:5.1f}%)")
        print(f"     (detail: mul={a.detail_mul:,}  add={a.detail_add:,}  "
              f"div={a.detail_div:,}  sqrt={a.detail_sqrt:,})")

        # ---- Section 2: Operation Count ----
        print(f"\n  2. Operation Count{' ' * 26} {o.total:>10}")
        print(f"     ├─ Matrix ops:  {o.matrix:>10}")
        print(f"     ├─ Vector ops:  {o.vector:>10}")
        print(f"     └─ Scalar ops:  {o.scalar:>10}")

        # ---- Section 3: Parallelism ----
        print(f"\n  3. Parallelism")
        print(f"     ├─ Critical path:     {a.critical_path:>8}  steps")
        print(f"     └─ Peak parallelism:  {a.max_parallelism:>8}  concurrent ops")

        # ---- Step details (collapsed) ----
        print(f"\n  Step breakdown:")
        for s in a.steps:
            r = s.result
            params_str = ", ".join(f"{k}={v}" for k, v in s.params.items())
            print(f"    {s.kernel_name}({params_str})")
            print(f"      → FLOPs {r.flops.total:,}  "
                  f"(M:{r.flops.matrix:,} V:{r.flops.vector:,} S:{r.flops.scalar:,})  "
                  f"crit_path={r.parallelism.critical_path}")

    def print_comparison(self):
        """Head-to-head comparison table."""
        if len(self.results) < 2:
            return

        # Determine column widths
        name_width = max(len(a.recipe.name) for a in self.results) + 2

        print(f"\n{'='*70}")
        print(f"  Comparison")
        print(f"{'='*70}")

        # Collect rows as (label, values_list, format)
        rows = []

        for a in self.results:
            f = a.flops; o = a.ops; t = f.total
            if t == 0: continue
            rows.append(("Total FLOPs", [
                f"{f.total:>12,}", f"{f.total:>12,}"]))

        # Build a simple side-by-side table
        # Headers
        header = f"  {'Metric':<36}"
        for a in self.results:
            header += f"  {a.recipe.name:>16}"
        print(header)
        print(f"  {'─'*68}")

        def show(label, *vals):
            line = f"  {label:<36}"
            for v in vals:
                line += f"  {v:>16}"
            print(line)

        def s(fmt, *vals):
            """All values use same format."""
            return [fmt.format(v) for v in vals]

        # Each row: label, then one value per algorithm
        def sep(label):
            vals = [""] * len(self.results)
            show(label, *vals)

        def row(label, getter, fmt_str="{:>,}"):
            vals = [fmt_str.format(getter(a)) for a in self.results]
            show(label, *vals)

        def pct_row(label, num_getter):
            """num_getter(a)->int, den_getter(a)->int → percentage."""
            vals = []
            for a in self.results:
                num = num_getter(a)
                den = a.flops.total
                vals.append(f"{num/den*100:.1f}%" if den else "-")
            show(label, *vals)

        sep("── 1. Total FLOPs ──")
        row("  Total FLOPs",      lambda a: a.flops.total)
        row("  Matrix FLOPs",     lambda a: a.flops.matrix)
        row("  Vector FLOPs",     lambda a: a.flops.vector)
        row("  Scalar FLOPs",     lambda a: a.flops.scalar)
        pct_row("  Matrix FLOPs %", lambda a: a.flops.matrix)
        pct_row("  Vector FLOPs %", lambda a: a.flops.vector)
        pct_row("  Scalar FLOPs %", lambda a: a.flops.scalar)
        show("")
        sep("── 2. Operation Count ──")
        row("  Total ops",        lambda a: a.ops.total)
        row("  Matrix ops",       lambda a: a.ops.matrix)
        row("  Vector ops",       lambda a: a.ops.vector)
        row("  Scalar ops",       lambda a: a.ops.scalar)
        show("")
        sep("── 3. Parallelism ──")
        row("  Critical path",    lambda a: a.critical_path)
        row("  Peak parallelism", lambda a: a.max_parallelism)


# ============================================================
# Built-in algorithm recipes
# ============================================================

def build_recipes(n: int) -> List[AlgorithmRecipe]:
    return [
        AlgorithmRecipe("Cholesky Inversion", [
            ("cholesky_decomp", {"n": n}),
            ("tril_inv",        {"n": n, "unit_diag": False}),
            ("matmul_chain",    {"n": n, "has_D": False}),
        ]),
        AlgorithmRecipe("LDL^T Inversion", [
            ("ldl_decomp",      {"n": n}),
            ("tril_inv",        {"n": n, "unit_diag": True}),
            ("diag_inv",        {"n": n}),
            ("matmul_chain",    {"n": n, "has_D": True}),
        ]),
    ]


# ============================================================
# Numerical verification
# ============================================================

def verify_algorithms(n: int = 16):
    import numpy as np

    H = np.random.randn(n, n)
    A = H.T @ H + 1e-5 * np.eye(n)
    inv_ref = np.linalg.inv(A)

    L = _scalar_cholesky(A)
    L_inv = _scalar_tril_inv(L)
    inv_chol = L_inv.T @ L_inv

    L2, D = _scalar_ldl(A)
    L2_inv = _scalar_tril_inv(L2, is_unit_diag=True)
    D_inv = np.diag(1.0 / np.diag(D))
    inv_ldl = L2_inv.T @ D_inv @ L2_inv

    print(f"\n  Verification vs numpy.linalg.inv:")
    print(f"    Cholesky error: {np.linalg.norm(inv_chol - inv_ref):.4e}")
    print(f"    LDL^T   error:  {np.linalg.norm(inv_ldl  - inv_ref):.4e}")


def _scalar_cholesky(A):
    import numpy as np
    n = A.shape[0]; L = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i,k]*L[j,k] for k in range(j))
            if i == j: L[i,i] = np.sqrt(A[i,i]-s)
            else:      L[i,j] = (A[i,j]-s) / L[j,j]
    return L

def _scalar_tril_inv(L, is_unit_diag=False):
    import numpy as np
    n = L.shape[0]; X = np.zeros((n, n))
    for i in range(n):
        X[i,i] = 1.0 if is_unit_diag else 1.0/L[i,i]
        for j in range(i):
            s = sum(L[i,k]*X[k,j] for k in range(j,i))
            X[i,j] = -s if is_unit_diag else -s/L[i,i]
    return X

def _scalar_ldl(A):
    import numpy as np
    n = A.shape[0]; L = np.eye(n); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            s = sum(L[i,k]*L[j,k]*D[k,k] for k in range(j))
            L[i,j] = (A[i,j]-s) / D[j,j]
        s = sum(L[i,k]**2 * D[k,k] for k in range(i))
        D[i,i] = A[i,i] - s
    return L, D


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N = 16

    library = build_default_library()
    print(f"Kernel library: {library.list_kernels()}\n")

    analyzer = Analyzer(library)

    # Single-size detailed analysis
    recipes = build_recipes(N)
    analyzer.analyze_all(recipes)
    for a in analyzer.results:
        analyzer.print_algorithm(a, n=N)

    # Multi-size comparison
    analyzer2 = Analyzer(library)
    for n in [16, 32, 64]:
        analyzer2.analyze_all(build_recipes(n))
    analyzer2.print_comparison()

    verify_algorithms(N)
