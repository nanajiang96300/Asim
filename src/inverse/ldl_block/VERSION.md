# LDL-Block (ldl_block)

## 算法概述
块 LDL 分解求逆，是 Cholesky 的重要变体。核心优势：**避免 SQRT 开方运算**，将 Cholesky 中计算密集的 SCALAR_SQRT 替换为 MUL (Vector Pipeline)，延迟从 scalar_sqrt_latency 降为 mul_latency=1。

## 文件
| 文件 | 角色 |
|------|------|
| `LDLDecompOp.h/.cc` | 算子：block LDL 分解 + $2\times2 \rightarrow 16\times16$ 拼接 |
| `LDLModel.h/.cc` | 模型 |

## 核心特征
- **分块参数**: `block_size` (默认 B=2), `pack_blocks` (拼接因子, 默认 2)
- **$2\times2 \rightarrow 16\times16$ 拼接**: 将 8 组 $2\times2$ 子块拼成 $16\times16$ GEMM，Cube 利用率从 0.2% 提升至 ~33%
- **依赖深度**: $O(n_b^2)$ vs Cholesky 的 $O(n_b^3)$ — D/L 分离使非对角块可并行
- **Cycle**: 比 Cholesky-Block 减少 42%（U=16 时: 2869 vs 4978）

## 配置参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 2 | 对角块尺寸 |
| `bwd_steps` | 3 | 回代步数 |
| `pack_blocks` | 2 | 拼接因子 (拼接成 16×16 的组数) |
| `batch_size` | 96 | 批次数量 |
| `matrix_m` | 64 | 接收天线数 |
| `matrix_k` | 16 | 发射天线数 |

## 版本历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06 | 从 operations/ 迁移到 inverse/ldl_block/ |
