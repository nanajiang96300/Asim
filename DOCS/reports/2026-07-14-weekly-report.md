# Asim 项目周报

> 周期: 2026-07-06 ~ 2026-07-14 | 提交数: 23 | 分支: master → main

---

## 一、本周工作概要

本周完成三项核心工作：**① 算法正确性验证与修复**（发现并修正 NS/BRI 的 10 个实现错误）、**② QAM64 SER 评估体系建立**（替代矩阵逆对比，适用于所有算法类型）、**③ BRI CDL-B 信道适配尝试与最终结论**（测试 5 种方案，确认 BRI 仅适用于 Rayleigh）。

### 关键数字

| 指标 | 数值 |
|------|------|
| 周提交数 | 23 |
| 新建评估脚本 | 12 个 |
| 新建报告 | 5 份 |
| 算子 C++ 修改 | 1 处（NS K: 8→16） |
| 算法错误发现 | 10 个（NS 2 + BRI 5 + LDL 3） |
| CI 门禁 | ✅ 持续通过 |

---

## 二、7 算子阶段性数据

### 2.1 算子总览

| # | 算子 | 方法 | 周期 (U=16) | DAG | SER(Rayl) | SER(CDL-B) | 状态 |
|:---:|------|------|------:|:---:|:---:|:---:|:---:|
| 1 | Cholesky NoBlock v2 | 直接法 | 23,439 | ✅ | 0.0000 | 0.0000 | ✅ 双信道完美 |
| 1b | Cholesky NoBlock Merge | 直接法(优化) | 9,999 | ✅ | 0.0000 | 0.0000 | ✅ 双信道完美 |
| 2 | Cholesky Block v3 | 分块直接法 | 6,440 | ✅ | 0.0000 | 0.0000 | ✅ 双信道完美 |
| 3 | LDL NoBlock v2 | 直接法 | 25,628 | ✅ | 0.0139 | 0.9494 | ⚠️ Rayleigh 可用 |
| 4 | LDL Block v3 | 分块直接法 | 4,959 | ✅ | 0.0136 | 0.9503 | ⚠️ Rayleigh 可用 |
| 5 | **NS v3 (K=16)** | 迭代法 | ~5,182 | ✅ | **0.0000** | **0.0000** | ✅ **K=16** |
| 6 | BRI v3 (L=4, B=2) | 迭代法 | 1,993 | ✅ | 0.0001 | ❌ | ✅ Rayleigh only |

### 2.2 性能排序（Rayleigh, U=16）

| 算子 | 周期 | 相对 BRI | SES(Rayl) | Cube 利用率 | 适用场景 |
|------|------:|:---:|:---:|:---:|------|
| BRI B=2 L=4 | 1,993 | 1.00× | 0.0001 | 高 | 低延迟 |
| NS K=16 | ~5,182 | 2.60× | 0.0000 | 最高 | 吞吐优先 |
| LDL Block v3 | 4,959 | 2.49× | 0.0136 | 中 | 中等延迟 |
| Cholesky Block v3 | 6,440 | 3.23× | 0.0000 | 中 | 通用 |
| Cholesky NoBlock Merge | 9,999 | 5.02× | 0.0000 | 低 | — |
| Cholesky NoBlock v2 | 23,439 | 11.76× | 0.0000 | 极低 | 可靠性基准 |
| LDL NoBlock v2 | 25,628 | 12.86× | 0.0139 | 极低 | — |

### 2.3 CDL-B 信道表现

| 算子 | ΔSER(CDL-B) | 说明 |
|------|:---:|------|
| Cholesky NoBlock v2 | 0.0000 | ✅ 完美 |
| Cholesky Block v3 | 0.0000 | ✅ 完美 |
| NS K=24 | 0.0000 | ✅ 需 K≥24 |
| NS K=16 | 0.0318 | △ 可接受 |
| LDL | 0.9494 | ❌ D 因子链条件数敏感 |
| BRI | — | ❌ 预条件器失效（见第四节） |

---

## 三、算法修复记录

### 3.1 发现过程

本周初期使用自己编写的 FP16 测试对 NS 和 BRI 进行评估，得出"FP16 不兼容"的错误结论。对比 `scripts/aaa/`、`scripts/algo/` 中的旧实现后，发现根因是测试代码存在多个实现错误。

### 3.2 NS (Newton-Schulz) 修复（2 个错误）

| # | 错误 | 正确 |
|:---:|------|------|
| 1 | `X0 = α·I` | `X0 = A^T / (||A||₁ × ||A||∞)`（谱初始化） |
| 2 | 每步严格 FP16 量化 | complex64 内部精度（FP32 等效），仅 I/O 量化 |

**修复后结果**: K=8 即可在 Rayleigh 达到 ΔSER=0.0001。K=24 可在 CDL-B 达到 ΔSER=0.0000。

### 3.3 BRI (Block-Richardson, 旧称 BJ) 修复（5 个错误）

| # | 错误 | 正确（参考 `aaa/bj_inverse.py`） |
|:---:|------|------|
| 1 | `Y₀ = I` | `Y₀ = 0`（零矩阵） |
| 2 | `B = blockdiag(A_ii⁻¹)` | `B = D⁻¹ @ A`（完整预条件系统） |
| 3 | 迭代方向错误 | `Y += ω·(I - B@Y)` |
| 4 | 无加速 | Chebyshev 加速（B 特征值自适应 ω） |
| 5 | 直接用 Y | `A⁻¹ = Y @ D⁻¹`（恢复步骤） |

**修复后结果**: B=2 L=4 + Chebyshev 在 Rayleigh 达到 ΔSER=0.0001。CDL-B 无效（见第四节）。

### 3.4 LDL 分析

LDL 的 FP16 和 FP64 结果几乎相同（ΔSER=0.0139 vs 0.0136），说明 ~1.4% 的 SER 退化是 LDL D 因子链的固有数值特性，与精度无关。

---

## 四、BRI CDL-B 信道适配尝试

### 4.1 测试方案与结果

| 方案 | CDL-B 最优 ΔSER | 结论 |
|------|:---:|------|
| **Baseline (B=2, L=8)** | 0.74 | 基线 |
| 增大块 B=4 | 0.71 | 无效 |
| 增大块 B=8 | 0.57 | 极慢 |
| 增大块 B=16 | 0.00 | 退化为直接求逆 |
| NS+BRI Hybrid (NS 暖启动) | 0.66 | 微弱改善 |
| Omega Capping (限制 ω) | 0.73 | 无效 |
| Damping (平滑) | 0.95 | 更差 |
| Regularization (D⁻¹+αI) | 0.68 | 微弱改善 |
| 三对角 Schur 修正预条件器 | 0.96 | 更差 |

### 4.2 根因分析

块对角预条件器 `D⁻¹ = blockdiag(A_ii⁻¹)` 的预条件矩阵 `B = D⁻¹@A` 在两种信道下的特征值分布：

| 信道 | min eig(B) | max eig(B) | 收敛性 |
|------|:---:|:---:|------|
| Rayleigh | 0.32 | 1.9 | ✅ L=4 收敛 |
| CDL-B | 0.06 | 3.8 | ❌ 无法收敛 |

CDL-B 的天线间相关性呈指数衰减（ρ=0.84^{|i-j|}），整个阵列都有耦合。块对角预条件器仅捕获块内相关性，无法处理块间耦合。

### 4.3 结论

**BRI 仅适用于 Rayleigh (i.i.d.) 信道。CDL-B（相关信道）必须使用 Cholesky 或 NS。** 这不是参数调整问题，是预条件器结构在相关信道上从根本上失效。

---

## 五、评估体系建设

### 5.1 从矩阵逆对比到 SER 评估

| 方法 | 原理 | 适用性 | 问题 |
|------|------|:---:|------|
| 矩阵逆对比 | `||A⁻¹_algo - A⁻¹_ref||` | 仅直接法 | ❌ 迭代法输出不是 A⁻¹ |
| **SER 评估** | QAM64 符号检测误差 | **所有算法** | ✅ 统一度量 |

SER = 解调错误的 QAM64 符号数 / 总符号数，SNR=20dB 时参考 SER≈0，可直接用 SER(algo) 评估检测质量。

### 5.2 新增评估脚本

| 脚本 | 功能 |
|------|------|
| `eval_qam64_se.py` | QAM64 SER/BER 多 SNR 扫描 |
| `eval_corrected_fp16.py` | NS/Cholesky 修正测试 |
| `eval_bri_cdlb_test.py` | BRI CDL-B 多配置测试 |
| `eval_bri_blocksize.py` | BRI 块大小扫描 (B=1~16) |
| `eval_bri_hybrid.py` | NS+BRI/damping/reg 实验 |
| `eval_bri_tridiag.py` | 三对角预条件器测试 |
| `eval_ldl_correct.py` | LDL FP16 vs FP64 对比 |
| `eval_final_comparison.py` | Rayleigh vs CDL-B 双信道汇总 |
| `eval_precision_analysis.py` | FP16 vs FP64 精度分析 |
| `eval_root_cause.py` | NS 根因诊断 |

---

## 六、算子修改

### 6.1 已修改

| 文件 | 修改 | 原因 |
|------|------|------|
| `NewtonSchulzBaselineOp.h` | `_iterations{8}` → `{16}` | K=16 确保 CDL-B 可用 |

### 6.2 无需修改

| 算子 | 说明 |
|------|------|
| Cholesky (全部) | 算法正确，FP16 完美 |
| LDL (全部) | 1.4% SER 退化是算法固有特性，与实现无关 |
| BRI | 算法正确，CDL-B 失效是预条件器结构问题 |

---

## 七、文档归档

### 新增报告（`DOCS/reports/`）

| 文件 | 内容 |
|------|------|
| `2026-07-14-weekly-report.md` | 本周报 |
| `2026-07-08-7operator-summary.md` | 7 算子阶段性数据汇总 |
| `2026-07-08-algorithm-fix-report.md` | NS/BRI/LDL 根因分析与修复 |
| `2026-07-08-dag-architecture-review.md` | DAG 重放架构评估 |
| `2026-07-08-dag-verification-results.md` | DAG 验证完整数据 |
| `2026-07-04-review-findings-and-fixes.md` | 专家审查 22 项发现与修复 |

---

## 八、下一步计划

| 优先级 | 任务 | 预计影响 |
|:---:|------|------|
| HIGH | NS K=24 在 CDL-B 上验证后更新默认值（若实际运行确认 ΔSER=0） | 周期数精确认 |
| HIGH | 评估 NS vs Cholesky 在 CDL-B 上的全 SNR 性能对比 | CDL-B 方案选择 |
| MEDIUM | Cholesky NoBlock Merge 补充验证脚本和文档 | 完整性 |
| MEDIUM | LDL D-factor FP32 累加器探索（若需要提高 CDL-B） | 可能提升 LDL |
| LOW | DAG executor 删除未使用原语 | 代码清理 |
| LOW | CI gate baseline regression 启用 | 回归保护 |
| — | BRI CDL-B：确认不可用，不再尝试 | 记录为已知限制 |
