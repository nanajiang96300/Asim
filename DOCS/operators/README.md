# Asim 求逆算子文档索引

每个算子文档包含：数学公式推导 → Python 参考 → C++ 指令序列 → SPAD 布局 → FormulaLogger 覆盖表 → 验证数据。

## NoBlock 算子（逐元素分解）

| # | 算子 | 文档 | Python | C++ | 周期 (U=16) |
|---|------|------|--------|-----|------------|
| 1 | Cholesky NoBlock v2 | [01_cholesky_noblock_v2.md](01_cholesky_noblock_v2.md) | `scripts/algo/cholesky_noblock.py` | `CholeskyNoBlockBaselineOp` | 23,439 |
| 2 | LDL NoBlock v2 | [02_ldl_noblock_v2.md](02_ldl_noblock_v2.md) | `scripts/algo/ldl_noblock.py` | `LDLNoBlockBaselineOp` | 25,628 |

## Block 算子（块分解）

| # | 算子 | 文档 | C++ | 周期 (U=16, B=2) |
|---|------|------|-----|------------------|
| 3 | Cholesky Block v3 | [03_cholesky_block_v3.md](03_cholesky_block_v3.md) | `CholeskyBlockBaselineOp` | 6,440 |
| 4 | LDL Block v3 | [04_ldl_block_v3.md](04_ldl_block_v3.md) | `LDLBlockBaselineOp` | 4,959 |

## 迭代算子

| # | 算子 | 文档 | C++ | 周期 |
|---|------|------|-----|------|
| 5 | Newton-Schulz v3 | [05_newton_schulz_v3.md](05_newton_schulz_v3.md) | `NewtonSchulzBaselineOp` | 2,591 (N=32, K=8) |
| 6 | Block-Richardson v3 | [06_block_richardson_v3.md](06_block_richardson_v3.md) | `BlockRichardsonBaselineOp` | 1,993 (U=16, B=2, L=8) |

## 开发标准

- [算子开发标准 v3.0](../OPERATOR_DEVELOPMENT_STANDARD_V3.md) — 指令集、SPAD 管理、SCALAR 模式、反模式、审查清单
