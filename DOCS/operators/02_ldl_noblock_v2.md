# LDL NoBlock v2 — 公式·算法·指令 全对应文档

> 版本: v2.0 | 日期: 2026-06-30

## 1. 文件清单

| 角色 | 路径 |
|------|------|
| Python 参考 | `scripts/algo/ldl_noblock.py` |
| C++ 算子 | `src/inverse/ldl_noblock/LDLNoBlockBaselineOp.{h,cc}` |
| C++ 模型 | `src/inverse/ldl_noblock/LDLNoBlockBaselineModel.{h,cc}` |
| 测试配置 | `example/ldl_noblock_v2_test.json` |
| 运行命令 | `--mode ldl_noblock_v2_test` |

## 2. 数学公式

$A = L \cdot D \cdot L^H$，$L$ 为单位下三角（对角全 1），$D$ 为实对角矩阵。

### Phase 3: LDL 分解（逐列）

对 $j = 0,\ldots,U-1$：

**D_UPDATE:**
$$D_j = A_{jj} - \sum_{k=0}^{j-1} D_k \cdot |L_{jk}|^2$$
$$D_j^{inv} = 1 / D_j$$

**L_UPDATE:** 对每个 $i > j$：
$$L_{ij} = \left(A_{ij} - \sum_{k=0}^{j-1} L_{ik} \cdot D_k \cdot L_{jk}^*\right) \cdot D_j^{inv}$$

### Phase 4: 前向求解 $Z = L^{-1}$

$L$ 单位下三角，故 $Z_{cc} = 1$。对 $i > c$：
$$Z_{ic} = -\sum_{k=c}^{i-1} L_{ik} \cdot Z_{kc}$$

### Phase 5: $\sqrt{D^{inv}}$ 加权

$$Y_{ic} = Z_{ic} \cdot \sqrt{D_c^{inv}}$$

### Phase 6: 后向装配
$$A^{-1} = Y^H \cdot Y$$

## 3. SPAD 内存布局

与 Cholesky NoBlock 相比，多了 2 个区域：

| 额外区域 | 变量 | 用途 |
|---------|------|------|
| `aL + size_uu` | `aD` | 对角因子 D（实对角）|
| `+ size_uu` | `aDinv` | D 的倒数 |

**初始化**: 所有新区域 (aD, aDinv, aTmp, aL, aY) 在 Phase 2 后通过 `ADD dest=X, src={aReg, aReg}` 初始化。

## 4. 核心指令映射（D_UPDATE / L_UPDATE）

### D_UPDATE（j=1, k=0 示例）

| 公式 | 指令 | src | dest |
|------|------|-----|------|
| $|L_{1,0}|^2$ | `SCALAR_MUL` | {aL, aL} | aTmp |
| $D_0 \cdot |L_{1,0}|^2$ | `SCALAR_MUL` | {aTmp, aD} | aTmp |
| $A_{1,1} \mathrel{-}= D_0|L_{1,0}|^2$ | `SCALAR_SUB` | {aA, aTmp} | aA |
| $1 = λ/λ$ | `SCALAR_DIV` | {aReg, aReg} | aTmp |
| $D_1^{inv} = 1/A_{1,1}$ | `SCALAR_DIV` | {aTmp, aA} | aDinv |
| $D_1 = A_{1,1} \cdot 1$ | `SCALAR_MUL` | {aA, aTmp} | aD |

→ `FormulaLogger::emit_step("LDL_NB_DUPDATE_1", "DIAG_INV", …)`

### L_UPDATE（i=2, j=1, k=0 示例）

| 公式 | 指令 | src | dest |
|------|------|-----|------|
| $L_{2,0} \cdot D_0$ | `SCALAR_MUL` | {aL, aD} | aTmp |
| $(L_{2,0}D_0) \cdot L_{1,0}^*$ | `SCALAR_MUL` | {aTmp, aL} | aTmp |
| $A_{2,1} \mathrel{-}= …$ | `SCALAR_SUB` | {aA, aTmp} | aA |
| $L_{2,1} = A_{2,1} \cdot D_1^{inv}$ | `SCALAR_MUL` | {aA, aDinv} | aL |

→ `FormulaLogger::emit_step("LDL_NB_LUPDATE_2_1", "TRSM", …)`

### $\sqrt{D^{inv}}$ 加权（每列 c）

| 公式 | 指令 | src | dest |
|------|------|-----|------|
| $\sqrt{D_c^{inv}}$ | `SCALAR_SQRT` | {aDinv} | aTmp |
| $Y_{:,c} \mathrel{*}= \sqrt{}$ | `SCALAR_MUL` | {aY, aTmp} | aY (tile_m=U) |

## 5. FormulaLogger 覆盖表

| step_id | op_type | 输入 | 输出 |
|---------|---------|------|------|
| LDL_NB_GRAM | GEMM | H, H^H | G |
| LDL_NB_REG | DIAG_ADD | G, λI | A |
| LDL_NB_DUPDATE_j | DIAG_INV | A | D_inv |
| LDL_NB_LUPDATE_i_j | TRSM | A, D_inv | L_ij |
| LDL_NB_BWD_ASSEMBLE | GEMM | Y^H, Y | Ainv |

## 6. 与 Cholesky NoBlock 的关键差异

| 特性 | Cholesky | LDL |
|------|----------|-----|
| SQRT | 每列 1 次 | 0（仅在 $\sqrt{D^{inv}}$ 处用） |
| 内层 MUL/列 | j 次 | 3j 次（D 因子 ×2 + conj ×1） |
| SPAD 区域 | 8 | 10（多 aD, aDinv） |
| $A^{-1}$ 装配 | $Y^H Y$ | $Y^H Y$（Y 已含 $\sqrt{D^{inv}}$ 加权） |

## 7. 验证数据

| 维度 | 周期 | LDL/Cholesky 比 | Python vs numpy |
|------|------|----------------|----------------|
| U=16 | 25,628 | 1.09× | 1e-15 ✅ |
