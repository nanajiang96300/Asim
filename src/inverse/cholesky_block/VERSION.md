# Cholesky-Block (cholesky_block)

## 算法概述
Hermitian 正定矩阵的块 Cholesky 分解求逆。将矩阵划分为 B×B 子块，按 POTRF/TRSM/RK_UPDATE 三步执行。

## 文件
| 文件 | 角色 |
|------|------|
| `CholeskyInvOp.h/.cc` | 算子：Block Cholesky 分解 + 三角回代 |
| `CholeskyModel.h/.cc` | 模型：创建 Tensor 图并实例化算子 |
| `CholeskyInvChainOp.h/.cc` | 变体：Cholesky Chain 求解模式（不同回代路径） |
| `CholeskyChainModel.h/.cc` | Chain 变体的 Model |

## 核心特征
- **分块参数**: `block_size` (默认 B=2)
- **Cube 利用率**: ~1.8% (小维度) — 大量微小 GEMM 碎片化
- **瓶颈**: SCALAR_SQRT/DIV 串行依赖 + RK Update 的微小 GEMM
- **适用场景**: 中小维度（U≤32），Block 模式优于 NoBlock

## 配置参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 2 | 对角块尺寸 |
| `batch_size` | 96 | 批次数量 |
| `matrix_m` | 64 | 接收天线数 |
| `matrix_k` | 16 | 发射天线数 |

## 版本历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06 | 从 operations/ 迁移到 inverse/cholesky_block/，基线版本 |
