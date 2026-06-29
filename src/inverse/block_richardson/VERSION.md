# Block-Richardson (block_richardson / BRI)

## 算法概述
Block-Richardson 预条件迭代求逆，是**周期最优方案**。通过块对角预条件器 $B = \text{blockdiag}(A_{11}^{-1}, ..., A_{bb}^{-1})$ 将大部分计算转化为 GEMM，利用 Chebyshev 多项式加速收敛。单次迭代仅需 $O(B \cdot N^2)$ GEMM，远低于 NS 的 $O(N^3)$。

## 文件
| 文件 | 角色 |
|------|------|
| `BlockRichardsonOp.h/.cc` | 算子：最大的算子文件（871行），包含 4 种预条件器路径 |
| `BlockRichardsonModel.h/.cc` | 模型 |

## 预条件器变体（运行时 attribute 控制）
| 变体 | `block_size` | `precond_solver` | 实现路径 |
|------|-------------|-------------------|----------|
| 2x2 Direct | B=2 | (自动) | 8 条 SCALAR 指令直接计算 $2\times2$ 矩阵逆 |
| Direct (Gauss-Jordan) | B≥4 | `"direct"` | GEMM 化的 Gauss-Jordan 消元 |
| Cholesky Block | B≥4 | `"cholesky"` | 块 Cholesky: POTRF + TRSM + RK + 前向/后向回代 |
| Simple Fallback | 任意 | (默认) | Vector ADD 拷贝（无实际求逆） |

## 调度参数（运行时 attribute 控制）
| 参数 | 默认值 | 作用 |
|------|--------|------|
| `layers` | 16 | Richardson 迭代层数 L |
| `block_size` | 2 | 预条件器对角块大小 B |
| `group_sync` | 2 | 每 N 层插入一次 PIPE_BARRIER |
| `fused_by_gemm` | false | 主循环用 GEMM 还是 GEMM_PRELOAD |
| `by_preload_period` | 4 | GEMM_PRELOAD 复用间隔 |
| `by_kernel_fuse_factor` | 1 | 核融合强度 |
| `fuse_residual_update` | false | 残差计算融合到 BY 缓冲区 |
| `iter_weight` | true | Chebyshev 自适应权重 |
| `omega_relaxed` | true | 松弛 omega 参数 |
| `adaptive_bounds` | false | 自适应谱界估计 |
| `precond_solver` | `"cholesky"` | 预条件器求解器选择 |

## 编译时选项
| Flag | 默认 | 作用 |
|------|------|------|
| `BRI_ENABLE_PRECOND_ELEM_ADDR` | OFF | 元素地址访问（实验性，有死锁风险） |
| `BRI_ENABLE_WEIGHTED_UPDATE` | OFF | 加权残差更新（实验性） |

## 已知问题
- **SCALAR 死锁**: 当使用 `BRI_ENABLE_PRECOND_ELEM_ADDR` 时，SCALAR 指令的逐元素地址访问可能导致 SPAD 依赖永久不满足
- **Block 交叉点**: B=4 处 Direct 与 Cholesky 周期接近，需要根据维度选择

## 版本历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06 | 从 operations/ 迁移到 inverse/block_richardson/ |
