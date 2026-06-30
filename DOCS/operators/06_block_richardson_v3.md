# Block-Richardson (BRI) v3 — 公式·算法·指令 全对应文档

> 版本: v3.0 | 日期: 2026-06-30

## 1. 文件清单

| 角色 | 路径 |
|------|------|
| C++ 算子 | `src/inverse/block_richardson/BlockRichardsonBaselineOp.{h,cc}` |
| C++ 模型 | `src/inverse/block_richardson/BlockRichardsonBaselineModel.{h,cc}` |
| 测试配置 | `example/bri_v3_test.json` |
| 运行 | `--mode bri_v3_test` |

## 2. 数学公式

### Phase 2: Gram + 正则化
$$A = H^H H + \lambda I$$

### Phase 3: 块对角预条件器（B=2 直接求逆）

将 $A$ 划分为 $n_B = U/B$ 个 $B \times B$ 对角块。

对每个 $B \times B$ 对角块 $A_{bb} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$：
$$B_{bb} = A_{bb}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

预条件器：$B = \text{blockdiag}(B_{00}, B_{11}, \ldots, B_{n_B-1,n_B-1})$

### Phase 5: Richardson 迭代（$L$ 层）

初始化 $Y_0 = I$。

对 $l = 0,\ldots,L-1$：
$$BY_l = B \cdot Y_l$$
$$R_l = I - BY_l$$
$$Y_{l+1} = Y_l + \omega \cdot R_l$$

其中 $\omega = 1$（基线 Chebyshev）。

### Phase 6: 滤波输出

$$W = A^{-1} \cdot H^H$$
$$\hat{X} = W \cdot Y_{in}$$

## 3. SPAD 布局

| 区域 | 变量 | 用途 |
|------|------|------|
| aH | H | 输入信道 (M×U) |
| aReg | λI | 正则化 (U×U) |
| aYin | Y | 接收信号 (M×U) |
| aA | A | Gram+reg (U×U) |
| aB | B | 预条件器 (U×U) |
| aYk | Y_l | 迭代变量 (U×U) |
| aBY | B·Y_l | 中间结果 (U×U) |
| aR | R_l | 残差 (U×U) |
| aYnext | Y_{l+1} | 更新 (U×U) |
| aW | W | 滤波中间 (U×M) |
| aTmp | — | SCALAR 临时 |
| aXhat (ACCUM) | — | 输出 $\hat{X}$ |

## 4. 指令映射

### 预条件器构造（B=2，每个对角块 b）

| 公式 | 指令 | src | dest |
|------|------|-----|------|
| $p_1 = a \cdot d$ | `SCALAR_MUL` | {aA, aA} | aTmp |
| $p_2 = b \cdot c$ | `SCALAR_MUL` | {aA, aA} | aBY |
| $\det = p_1 - p_2$ | `SCALAR_SUB` | {aTmp, aBY} | aTmp |
| $1 = λ/λ$ | `SCALAR_DIV` | {aReg, aReg} | aBY |
| $1/\det$ | `SCALAR_DIV` | {aBY, aTmp} | aBY |
| $B_{00} = d/\det$ | `SCALAR_MUL` | {aA, aBY} | aB |
| $B_{01} = -b/\det$ | `SCALAR_MUL` | {aA, aBY} | aB |
| $B_{10} = -c/\det$ | `SCALAR_MUL` | {aA, aBY} | aB |
| $B_{11} = a/\det$ | `SCALAR_MUL` | {aA, aBY} | aB |

→ `FormulaLogger::emit_step("BRI_PRECOND", "MATRIX_INV_2x2", …)`

### 迭代（每层 l）

| 公式 | 指令 | ID | src | dest | FormulaLogger |
|------|------|-----|-----|------|-------------|
| $B Y_l$ | `GEMM_PRELOAD` | BRI_BY_l | {aB, aYk} | aBY | `emit_step("BRI_BY_l", "GEMM", …)` |
| $R_l = I - BY_l$ | `ADD` | BRI_RESIDUAL_l | {aReg, aBY} | aR | `emit_step("BRI_RESIDUAL_l", "MATRIX_SUB", …)` |
| $Y_{l+1} = Y_l + R_l$ | `ADD` | BRI_Y_UPDATE_l | {aYk, aR} | aYnext | `emit_step("BRI_Y_UPDATE_l", "MATRIX_ADD", …)` |
| 同步 (每 4 层) | `PIPE_BARRIER` type=5 | BRI_SYNC_l | — | — | |

### 输出

| 公式 | 指令 |
|------|------|
| $W = A^{-1} H^H$ | `GEMM_PRELOAD` BRI_W {aYk, aH} → aW |
| $\hat{X} = W Y$ | `GEMM_PRELOAD` BRI_XHAT {aW, aYin} → aXhat |
| 写回 | `MOVOUT` BRI_STORE aXhat → dram_Out |

## 5. FormulaLogger 覆盖表

| step_id | op_type | 输入 | 输出 |
|---------|---------|------|------|
| BRI_GRAM | GEMM | H, H^H | A |
| BRI_REG | DIAG_ADD | A, λI | A_reg |
| BRI_PRECOND | MATRIX_INV_2x2 | A | B |
| BRI_BY_l (×L) | GEMM | B, Y | BY |
| BRI_RESIDUAL_l (×L) | MATRIX_SUB | I, BY | R |
| BRI_Y_UPDATE_l (×L) | MATRIX_ADD | Y, ωR | Y_new |

总计: 2 + 3L 个 FormulaLogger 步骤。

## 6. 验证数据

| 维度 | B | L | 周期 |
|------|---|---|------|
| U=16 | 2 | 8 | 1,993 |
