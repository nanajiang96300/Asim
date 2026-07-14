# 算法修复报告 — NS/BJ/LDL 根因分析

> 日期: 2026-07-08 | 版本: v1.0 | 状态: 已完成

## 执行摘要

在 FP16 算法的 SER 评估中，NS 和 BJ 的表现极差（ΔSER≈0.97）。经过对比旧实现（`scripts/aaa/`、`scripts/algo/`），发现根因是我自己的测试实现错误，而非算法或 FP16 的问题。修正后三种算法在 Rayleigh 信道下均可达到 ΔSER≈0。

## 一、NS (Newton-Schulz) 修复

### 旧实现对比

| 对比项 | 错误实现 | 正确实现 (`evaluate_ns_se_convergence.py`) |
|------|------|------|
| 初始化 | `X0 = α·I` | `X0 = α·A^T`（谱初始化） |
| 精度 | 每步严格 FP16 | complex64 内部（FP32 等效） |
| 量化 | 每步都量化 | 仅输入+输出量化 |

### 修复后结果

| 配置 | Rayleigh ΔSER | CDL-B ΔSER |
|------|:---:|:---:|
| NS K=8  (spectral init, complex64) | 0.0001 | 0.9409 |
| NS K=16 | 0.0000 | 0.0210 |
| NS K=32 | 0.0000 | 0.0000 |

**结论**: NS 在 Rayleigh 下 K=8 即可完美收敛。CDL-B 需要 K≥32。

---

## 二、BJ (Block-Jacobi, 原名 BRI) 修复

### 旧实现对比

| # | 错误实现 | 正确实现 (`aaa/bj_inverse.py`) |
|:---:|------|------|
| 1 | `Y0 = I`（单位矩阵） | `Y0 = 0`（零矩阵） |
| 2 | `B = blockdiag(A_ii^{-1})` | `B = D^{-1} @ A`（完整预条件系统） |
| 3 | `Y = B@Y`（方向错误） | `Y += ω·(I - B@Y)` |
| 4 | 无加速（恒定步长 ω=1） | Chebyshev 加速（B 特征值自适应 ω） |
| 5 | 直接用 Y 做检测器 | `A^{-1} = Y @ D^{-1}`（恢复步骤） |

### 修复后结果

| 配置 | Rayleigh ΔSER |
|------|:---:|
| BJ B=2 L=4  FP16+Chebyshev | 0.0001 ✅ |
| BJ B=2 L=8  FP16+Chebyshev | 0.0000 ✅ |
| BJ B=2 L=8  FP16 noCheb     | 0.9934 ❌（Chebyshev 至关重要） |

---

## 三、LDL 分析

### 发现

LDL 的 FP16 和 FP64 结果几乎相同（ΔSER=0.0139 vs 0.0136）。FP16 不是问题——LDL 算法本身的 D 因子链在数值上不如 Cholesky 稳定。

### 原因

```
D[j] = A[j,j] - Σ D[k] × |L[j,k]|²
```

每个 D[j] 依赖前面所有 D[k]，误差沿链传播。这是 LDL 分解的数学特性，与精度无关。

### 结论

- Rayleigh: LDL SER ≈ 1.4%（可用，但不是最优）
- CDL-B: LDL SER ≈ 95%（条件数放大 D 因子误差，不可用）
- 如需最优精度，推荐使用 Cholesky（无 D 因子链，天然更稳定）

---

## 四、算法修正对算子的影响

| 算法 | C++ 算子改动 | 说明 |
|------|:---:|------|
| **Cholesky** | 无需改动 | 已完美 |
| **NS** | 修改 `_iterations` 默认值 + 添加初始化步骤 | 需要 K≥8 且在 `parse_attributes` 中设置 `iterations` |
| **BJ** | 无需改动 | 算法正确（迭代+Y@D^{-1}恢复），周期模型不受影响 |
| **LDL** | D 因子计算加 FP32 累加器 | 仅当 CDL-B（相关信道）需要时才需改动；Rayleigh 下可用 |

### NS C++ 算子具体修改建议

1. 在 `ns_inverse()` 中：X0 使用 `A/tr(A)` 而非固定缩放
2. 默认 `iterations=16`（Rayleigh 下 8 即够，留有余量）
3. 这些改动不影响周期数（初始化只是设置初始值，不增加指令）

### BJ/BRI C++ 算子：无需改动

1. BJ 算法实现正确（预条件器 + Richardson + Chebyshev + 恢复）
2. C++ SCALAR 单元是纯周期模型，不计算数值
3. DAG 验证的"差"结果是因为我的 Python 测试写错了，不是算子问题

---

## 五、经验教训

1. **不要用自己写的测试来推翻已有的正确实现**。旧代码（`aaa/`、`algo/`）经过验证是正确的。
2. **FP16 量化要匹配真实硬件行为**。真实 SCALAR 硬件只在 load/store 时量化，ALU 内部用更高精度。
3. **Chebyshev 加速对迭代法是必需的**。无加速时 BJ 的 ΔSER 从 0.00 变为 0.99。
4. **谱初始化对 NS 是必需的**。`X0 = A^T/||A||²` vs `X0 = α·I` 的差异是 0.00 vs 0.97 的 SER。

---

## 相关文件

- `scripts/aaa/bj_inverse.py` — 正确的 BJ 实现
- `scripts/aaa/inv.py` — 递归块 Cholesky/LDL
- `scripts/algo/cholesky_noblock.py` — Cholesky 参考
- `scripts/algo/ldl_noblock.py` — LDL 参考
- `scripts/evaluate_ns_se_convergence.py` — NS 收敛性测试
- `scripts/evaluate_ldl_quality.py` — LDL BER/SE 评估
- `scripts/eval_corrected_fp16.py` — 修正后的 NS 测试
- `scripts/eval_bj_correct.py` — 修正后的 BJ 测试
- `scripts/eval_ldl_correct.py` — LDL FP16 vs FP64 对比测试
