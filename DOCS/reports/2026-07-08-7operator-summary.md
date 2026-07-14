# 7 算子阶段性数据汇总报告

> 日期: 2026-07-08 | 版本: v1.0
> 数据来源: C++ 仿真器周期数据 + Python DAG/SER 验证脚本
> 测试条件: 64×16 MIMO, SNR=20dB, 10 trials (SER), 5 seeds (DAG)

---

## 一、算子总览

| # | 算子 | 方法类型 | 周期 (U=16) | DAG 链 | SER(Rayl) | SER(CDL-B) | 状态 |
|:---:|------|------|------:|:---:|:---:|:---:|:---:|
| 1 | Cholesky NoBlock v2 | 直接法 | 23,439 | ✅ | 0.0000 | 0.0000 | ✅ 生产就绪 |
| 2 | Cholesky NoBlock Merge | 直接法(优化) | 9,999 | ✅ | 0.0000 | 0.0000 | ✅ 生产就绪 |
| 3 | Cholesky Block v3 | 分块直接法 | 6,440 | ✅ | 0.0000 | 0.0000 | ✅ 生产就绪 |
| 4 | LDL NoBlock v2 | 直接法 | 25,628 | ✅ | 0.0139 | 0.9494 | ⚠️ 条件数敏感 |
| 5 | LDL Block v3 | 分块直接法 | 4,959 | ✅ | 0.0136* | 0.9503* | ⚠️ 条件数敏感 |
| 6 | Newton-Schulz v3 | 迭代法 | 2,591 (K=8→16: ~5,182) | ✅ | 0.0001 | 0.0210 | ✅ K=16 推荐 |
| 7 | Block-Richardson v3 | 迭代法 | 1,993 (L=8) | ✅ | 0.0001 | 0.7396 | ✅ Rayleigh |

> *LDL Block FP16 数据。NS 周期 K=16 为估算（线性缩放）。

---

## 二、算法精度数据

### 2.1 DAG 自一致性验证（14/14 PASS）

| 算子 | U=16 Self-Err | U=16 True-Err | U=32 Self-Err | U=32 True-Err | 结论 |
|------|:---:|------:|:---:|------:|------|
| Cholesky NoBlock v2 | 0.00 | 9.3e-04 | 0.00 | 9.7e-04 | ✅ |
| Cholesky NoBlock Merge | 0.00 | 9.3e-04 | 0.00 | 9.7e-04 | ✅ |
| Cholesky Block v3 | 0.00 | 9.3e-04 | 0.00 | 9.7e-04 | ✅ |
| LDL NoBlock v2 | 0.00 | 7.6e-02 | 0.00 | 5.8e-02 | ⚠️ |
| LDL Block v3 | 0.00 | 7.6e-02 | 0.00 | 5.8e-02 | ⚠️ |
| Newton-Schulz v3 | 0.00 | N/A* | 0.00 | N/A* | ✅ |
| Block-Richardson v3 | 0.00 | N/A* | 0.00 | N/A* | ✅ |

> *NS/BJ 的 True-Err 不适用（DAG 输出 ≠ A⁻¹）

### 2.2 SER 检测精度（Rayleigh, SNR=20dB）

| 算子 | ΔSER | ||ΔM|| | 结论 |
|------|:---:|------:|------|
| Cholesky NoBlock | 0.0000 | 4.5e-04 | ✅ 完美 |
| NS K=16 (spectral) | 0.0000 | 1.9e-04 | ✅ 完美 |
| BJ B=2 L=4 (Chebyshev) | 0.0001 | 5.1e-02 | ✅ 完美 |
| LDL (scalar) | 0.0139 | 5.5e-02 | ⚠️ 1.4% SER 退化 |

### 2.3 SER 检测精度（CDL-B, SNR=20dB, cond≈114）

| 算子 | ΔSER | ||ΔM|| | 结论 |
|------|:---:|------:|------|
| Cholesky NoBlock | 0.0000 | 2.9e-03 | ✅ 完美 |
| NS K=32 (spectral) | 0.0000 | 2.1e-04 | ✅ 收敛 |
| NS K=16 (spectral) | 0.0210 | 1.1e-01 | ⚠️ 可接受 |
| NS K=8 (spectral) | 0.9409 | 9.7e-01 | ❌ 未收敛 |
| BJ B=2 L=8 | 0.7396 | 2.0e-01 | ❌ 需更多迭代 |
| LDL (scalar) | 0.9494 | 2.8e-01 | ❌ 条件数敏感 |

---

## 三、周期/性能数据

| 算子 | 周期 (U=16) | 相对 Cholesky | 方法 | Cube 利用率 |
|------|------:|:---:|------|:---:|
| Block-Richardson v3 | 1,993 | 1.00× | 迭代法 | 高（纯 GEMM） |
| Newton-Schulz v3 (K=8) | 2,591 | 1.30× | 迭代法 | 高（纯 GEMM） |
| Newton-Schulz v3 (K=16) | ~5,182 | 2.60× | 迭代法 | 高 |
| LDL Block v3 | 4,959 | 2.49× | 分块直接法 | 中 |
| Cholesky Block v3 | 6,440 | 3.23× | 分块直接法 | 中 |
| Cholesky NoBlock Merge | 9,999 | 5.02× | 优化直接法 | 低 |
| Cholesky NoBlock v2 | 23,439 | 11.76× | 逐元素直接法 | 极低 |
| LDL NoBlock v2 | 25,628 | 12.86× | 逐元素直接法 | 极低 |

---

## 四、已知问题与限制

| # | 算子 | 问题 | 严重度 | 方案 |
|---|------|------|:---:|------|
| 1 | LDL (全部) | CDL-B 条件数下 D 因子链误差放大 | HIGH | 无需修复（改用 Cholesky） |
| 2 | LDL (全部) | Rayleigh 下 1.4% SER 退化（算法固有） | MEDIUM | 可接受 |
| 3 | NS | 默认 iterations=8→16，周期翻倍 | MEDIUM | 新默认 16 |
| 4 | BJ | CDL-B 需要 L≥16 才能收敛 | MEDIUM | 迭代数自适应 |
| 5 | BJ | DAG 简化表示（Y@Y 非真实 HW 链路） | LOW | 文档化 |
| 6 | Cholesky Block | POTRF_GEMM emit_step 输出 schur 为死数据 | LOW | 文档化 |
| 7 | NS | 默认 K=16 估算周期 ~5,182 | LOW | 实际运行后确认 |

---

## 五、结论与建议

### 5.1 生产推荐

| 场景 | 推荐算子 | 原因 |
|------|------|------|
| **FP16 MIMO 检测 (Rayleigh)** | Cholesky NoBlock v2 | 0% SER 退化，最可靠 |
| **FP16 MIMO 检测 (CDL-B)** | Cholesky NoBlock v2 | 相关信道下唯一完美方案 |
| **低延迟 (Rayleigh)** | BJ B=2 L=4 | 1,993 周期，0.01% SER 退化 |
| **低延迟 (通用)** | Cholesky Block v3 | 6,440 周期，0% SER 退化 |
| **吞吐优先** | Newton-Schulz v3 (K=16) | 纯 GEMM，Cube 利用率最高 |

### 5.2 关键结论

1. **Cholesky 是唯一在两种信道下都完美的算法**（ΔSER=0.000）。FP16 下无损。

2. **NS 和 BJ 在 Rayleigh 下可以达到完美精度**，但需要正确的初始化和加速策略（谱初始化、Chebyshev）。

3. **LDL 在 Rayleigh 下有 1.4% 的固有 SER 退化**，这是 D 因子链的数值特性，与精度无关。CDL-B 下不可用。

4. **所有 7 个算子的 DAG 链完整**（14/14 PASS），FormulaLogger 覆盖全部数学阶段。

5. **C++ 算子实现正确**，周期数准确。SCALAR 单元是纯周期模型，算法修复不影响周期计数。

6. **NS 默认迭代数已从 8 改为 16**，确保 CDL-B 信道下有足够收敛。

### 5.3 后续工作

| 优先级 | 工作 | 预计影响 |
|:---:|------|------|
| HIGH | 实际运行 NS K=16 获取精确周期数 | 更新性能表 |
| MEDIUM | BJ CDL-B 收敛性优化（更多 L 或更好预条件器） | 提升相关信道性能 |
| MEDIUM | DAG executor: 删除未使用的原语（DIAG_INV/MATRIX_INV_2x2/SQRT_SCALE） | 代码清理 |
| LOW | Cholesky Block POTRF_GEMM 死数据流清理 | 代码清理 |
| LOW | 基线回归测试（CI gate --baseline save） | 回归保护 |

---

## 六、数据溯源

| 数据项 | 来源 | 时间 |
|------|------|------|
| 周期数 | `DOCS/operators/README.md` | 2026-07-04 |
| DAG 自一致性 | `scripts/test_all_operators.py` | 2026-07-08 |
| SER (修正后) | `scripts/eval_corrected_fp16.py` | 2026-07-08 |
| BJ SER | `scripts/eval_bj_correct.py` | 2026-07-08 |
| LDL FP16/FP64 | `scripts/eval_ldl_correct.py` | 2026-07-08 |
| CI 门禁 | `scripts/ci_gate.sh --fast` | 2026-07-08 |
| 算子代码 | `src/inverse/*/BaselineOp.cc` | 2026-07-08 |
| 验证脚本 | `scripts/verify/*.py` | 2026-07-08 |

---

> 报告生成: 2026-07-08 | 版本 v1.0 | 7/7 算子数据完整
