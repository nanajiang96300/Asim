# DAG 原语规范 v1.0

> 约束新旧算子，DAG executor 只增不减。新算子按需组合原语。

## 核心原语（6个，覆盖所有算子的通用操作）

| 原语 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `GEMM` | A, B | C = A@B | 矩阵乘法 |
| `DIAG_ADD` | A | A+λI | 对角正则化 |
| `MATRIX_SUB` | A, B | A-B | 逐元素减法 |
| `MATRIX_ADD` | A, B | A+B | 逐元素加法 |
| `SCALE` | A, α | α·A | 标量缩放 |
| `TRSM` | L | Y=L^{-1} | 三角求解（1输入=前向求解，2输入=透传） |

## 算法原语（2个，复用）

| 原语 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `CHOLESKY` | A | L=chol(A) | Cholesky 分解 |
| `LDL_FACTOR` | A | Y | LDL分解+前向求解+sqrt(Dinv)加权 |

## 算子专用原语（1个）

| 原语 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `BRI_PRECOND` | A | B=blockdiag(inv(A_ii)) | BRI 块对角预条件器 |

## 新增算子规则

1. **首选核心原语组合**：如果新算法的数学步骤能用核心原语表达，只修改 C++ 的 `emit_step()`，DAG executor 零改动。
2. **次选添加算法原语**：如果需要特殊分解（如 QR、LU），在 `uobs_dag_executor.py` 中添加一个 `prim_*` 函数并注册到 `PRIMITIVES` 字典。
3. **末选专用原语**：只用于无法用核心+算法原语表达的步骤。

## 各算子使用的原语

| 算子 | 核心原语 | 算法原语 | 专用原语 |
|------|---------|---------|---------|
| Cholesky NoBlock | GEMM, DIAG_ADD, TRSM, GEMM | CHOLESKY | — |
| LDL NoBlock | GEMM, DIAG_ADD | LDL_FACTOR | — |
| Cholesky Block | GEMM, DIAG_ADD, TRSM, GEMM | CHOLESKY | — |
| LDL Block | GEMM, DIAG_ADD | LDL_FACTOR | — |
| Newton-Schulz | GEMM×K, MATRIX_SUB×K | — | — |
| BRI | GEMM, DIAG_ADD, MATRIX_SUB×L, MATRIX_ADD×L | — | BRI_PRECOND |

## 反耦合证明

- NS 只用了核心原语 → 新增 NS 算子时 DAG executor **零改动**
- Cholesky/LDL 各需要一个算法原语 → 仅新增 1 个函数 + 1 行注册
- BRI 需要一个专用原语 → 仅新增 1 个函数 + 1 行注册
- 未来新算子：大概率只用核心原语组合
