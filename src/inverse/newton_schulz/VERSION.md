# Newton-Schulz (newton_schulz)

## 算法概述
Newton-Schulz 迭代求逆，具有二次收敛性。迭代公式：$X_{k+1} = X_k(2I - AX_k)$。
初始猜测 $X_0 = I/\|A\|_2$ 保证任意正定矩阵收敛。

## 版本

### v1.0 Baseline (NewtonSchulzOp)
- **文件**: `NewtonSchulzOp.h/.cc` + `NewtonSchulzModel.h/.cc`
- **特点**: 原始实现，基础指令调度
- **Mode**: `newton_schulz_test`
- **注册**: 在 OperationFactory 中注册（支持 ONNX 路径）

### v2.0 Optimized (NewtonSchulzOptOp)
- **文件**: `NewtonSchulzOptOp.h/.cc` + `NewtonSchulzOptModel.h/.cc`
- **特点**: 当前"生产"版本。接入 FormulaLogger 用于 UOBS 黑盒打分；采用每 Batch 独立 Tile 的 Round-Robin 多核分发；基于 Cube 阵列 $16\times16\times16$ GEMM 分块；Scalar 单元参与的流水线屏障
- **Mode**: `newton_schulz_opt_test`
- **迭代数**: K=8 (i.i.d. Rayleigh 信道)

## 区别
| 特性 | Baseline | Optimized |
|------|----------|-----------|
| 代码量 | 299 行 | 486 行 |
| FormulaLogger | 未接入 | 已接入 |
| 指令调度 | 基础 | 优化 barrier + GEMM 分解 |
| ONNX 支持 | 是 | 否（仅 C++ Model） |
| UOBS 评估 | 不支持 | 支持 |

## 配置参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iterations` | 10 | NS 迭代次数 |
| `batch_size` | 96 | 批次数量 |
| `matrix_m` | 64 | 接收天线数 |
| `matrix_k` | 16 | 发射天线数 |

## 版本历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06 | 从 operations/ 迁移到 inverse/newton_schulz/ |
