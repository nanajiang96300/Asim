# Newton-Schulz v3 — 公式·算法·指令 全对应文档

> 版本: v3.0 | 日期: 2026-06-30

## 1. 文件清单

| 角色 | 路径 |
|------|------|
| C++ 算子 | `src/inverse/newton_schulz/NewtonSchulzBaselineOp.{h,cc}` |
| C++ 模型 | `src/inverse/newton_schulz/NewtonSchulzBaselineModel.{h,cc}` |
| 测试配置 | `example/newton_schulz_v3_test.json` |
| 运行 | `--mode newton_schulz_v3_test` |

## 2. 数学公式

迭代求逆，二次收敛。$X_0 = I/\|A\|_2$。

对 $k = 0,\ldots,K-1$：
$$T_k = A \cdot X_k$$
$$R_k = 2I - T_k$$
$$X_{k+1} = X_k \cdot R_k$$

最终：$A^{-1} \approx X_K$

**特点：纯 GEMM + Vector ADD，无 SCALAR 操作。**

## 3. SPAD 布局

| 区域 | 变量 | 用途 |
|------|------|------|
| aA | A | 输入矩阵 (N×N) |
| aX | X_k | 迭代变量 (N×N) |
| aC | 2I | 常数矩阵 (N×N) |
| aT | T_k | A·X_k |
| aR | R_k | 2I - T_k |
| aAinv (ACCUM) | — | 输出 |

## 4. 指令映射（K=8）

### Phase 1: MOVIN

| 指令 | src | dest |
|------|-----|------|
| MOVIN | dram_A+bOff | aA |
| MOVIN | dram_X0+bOff | aX |
| MOVIN | dram_C | aC |
| PIPE_BARRIER type=1 | — | — |

### Phase 2: 迭代（每轮 k）

| 公式 | 指令 | ID | src | dest | FormulaLogger |
|------|------|-----|-----|------|-------------|
| $T_k = A X_k$ | `GEMM_PRELOAD` | NS_T_k | {aA, aX} | aT | `emit_step("NS_GEMM_T_k", "GEMM", …)` |
| 同步 | `PIPE_BARRIER` type=2 | NS_T2R_k | — | — | |
| $R_k = 2I - T_k$ | `ADD` | NS_R_k | {aC, aT} | aR | `emit_step("NS_RESIDUAL_k", "MATRIX_SUB", …)` |
| 同步 | `PIPE_BARRIER` type=3 | NS_R2X_k | — | — | |
| $X_{k+1} = X_k R_k$ | `GEMM` | NS_X_k | {aX, aR} | aX | `emit_step("NS_UPDATE_k", "GEMM", …)` |
| 同步 (非最后一轮) | `PIPE_BARRIER` type=4 | NS_ITER_k | — | — | |

**总指令数/轮: 2 GEMM + 1 ADD + 2~3 BARRIER**

### Phase 3: 输出

| 指令 | src | dest |
|------|-----|------|
| GEMM_PRELOAD | {aX, aX} | aAinv |
| PIPE_BARRIER type=6 | — | — |
| MOVOUT | aAinv | dram_Out |

## 5. FormulaLogger 覆盖表

| step_id | op_type | 输入 | 输出 |
|---------|---------|------|------|
| NS_GEMM_T_k (×K) | GEMM | A, X | T |
| NS_RESIDUAL_k (×K) | MATRIX_SUB | 2I, T | R |
| NS_UPDATE_k (×K) | GEMM | X, R | X_new |

总计: 3K 个 FormulaLogger 步骤。

## 6. 验证数据

| 维度 | K | 周期 |
|------|---|------|
| N=32 | 8 | 2,591 |
