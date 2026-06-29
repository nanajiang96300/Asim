# Cholesky-NoBlock (cholesky_noblock)

## 算法概述
逐列 Cholesky 分解求逆，B=1 退化形式。每列使用 SCALAR_SQRT 开方 + SCALAR_DIV 归一化，无 Cube GEMM。

## 文件
| 文件 | 角色 |
|------|------|
| `CholeskyInvNoBlockOp.h/.cc` | 算子：逐列 Cholesky 分解 |
| `CholeskyNoBlockModel.h/.cc` | 模型 |

## 核心特征
- **分块参数**: B=1（无分块）
- **Cube 利用率**: ~1.3% (极低)
- **瓶颈**: 每列 1 次 SCALAR_SQRT + 多次 SCALAR_DIV，全部串行化在 Scalar Pipeline 上
- **适用场景**: 大维度（U=64）时优于 Block（避免了分块 barrier 开销）

## 配置参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 96 | 批次数量 |
| `matrix_m` | 64 | 接收天线数 |
| `matrix_k` | 16 | 发射天线数 |
| `strict_iso_lowering` | 1 | 严格各向同性 lowering |

## 版本历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06 | 从 operations/ 迁移到 inverse/cholesky_noblock/ |
