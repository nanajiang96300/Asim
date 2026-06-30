# LDL Block v3 — 公式·算法·指令 全对应文档

> 版本: v3.0 | 日期: 2026-06-30

## 1. 文件清单

| 角色 | 路径 |
|------|------|
| C++ 算子 | `src/inverse/ldl_block/LDLBlockBaselineOp.{h,cc}` |
| C++ 模型 | `src/inverse/ldl_block/LDLBlockBaselineModel.{h,cc}` |
| 测试配置 | `example/ldl_block_v3_test.json` |
| 运行 | `--mode ldl_block_v3_test` |

## 2. 数学公式

Block LDL: $A = L \cdot D \cdot L^H$，$L$ 块单位下三角，$D$ 块对角。

### Phase 3: 块 LDL 分解

对 $j = 0,\ldots,n_B-1$：

**D_UPDATE (对角块):**
$$D_j = \text{ldl}\left(A_{jj} - \sum_{k=0}^{j-1} L_{jk} \cdot D_k \cdot L_{jk}^H\right)$$

块内逐元素 LDL（$B \times B$）：
$$d_{pp} = a_{pp} - \sum_{q=0}^{p-1} d_q \cdot |l_{pq}|^2$$
$$d_p^{inv} = 1/d_p$$
$$l_{rp} = \left(a_{rp} - \sum_{q=0}^{p-1} l_{rq} \cdot d_q \cdot l_{pq}^*\right) \cdot d_p^{inv}$$

**L_UPDATE (非对角块):** 对 $i > j$：
$$L_{ij} = \left(A_{ij} - \sum_{k=0}^{j-1} L_{ik} \cdot D_k \cdot L_{jk}^H\right) \cdot D_j^{inv}$$

### Phase 4: 前向求解 $Z = L^{-1}$

$L$ 块单位下三角，$Z_{cc} = I$。对 $i > c$：
$$Z_{ic} = -\sum_{k=c}^{i-1} L_{ik} \cdot Z_{kc}$$

### Phase 5: $\sqrt{D^{inv}}$ 加权 + 装配

$$Y_{ic} = Z_{ic} \cdot \sqrt{D_c^{inv}}$$
$$A^{-1} = Y^H \cdot Y$$

## 3. 指令映射（B=2 示例）

### D_UPDATE 对角块

| 层次 | 公式 | 指令 |
|------|------|------|
| 块间 | $\sum L_{jk} D_k L_{jk}^H$ | `GEMM` (B×B×B) per k |
| 块内 | $|l_{pq}|^2$ | `SCALAR_MUL` |
| 块内 | $d_q \cdot |l_{pq}|^2$ | `SCALAR_MUL` |
| 块内 | $a_{pp} \mathrel{-}= …$ | `SCALAR_SUB` |
| 块内 | $1 = λ/λ$ | `SCALAR_DIV` (Reg,Reg) |
| 块内 | $d_p^{inv} = 1/a_{pp}$ | `SCALAR_DIV` |
| 块内 | $d_p = a_{pp} \cdot 1$ | `SCALAR_MUL` (存入 aD) |

### L_UPDATE 非对角块

| 层次 | 公式 | 指令 |
|------|------|------|
| 块间 | $\sum L_{ik} D_k L_{jk}^H$ | `GEMM` per k |
| 块内 | $l_{rp} = (a_{rp} - …) \cdot d_p^{inv}$ | `SCALAR_MUL` (乘以 Dinv) |

### $\sqrt{D^{inv}}$ 加权

| 公式 | 指令 |
|------|------|
| $\sqrt{D_c^{inv}}$ | `SCALAR_SQRT` |
| $Y_{:,c} \mathrel{*}= \sqrt{}$ | `SCALAR_MUL` (tile_m=B) |

## 4. FormulaLogger 覆盖表

| step_id | op_type | 粒度 |
|---------|---------|------|
| LDL_BLK_GRAM | GEMM | U×U |
| LDL_BLK_REG | DIAG_ADD | U×U |
| LDL_BLK_DUPDATE_j | DIAG_INV | B×B 对角块 |
| LDL_BLK_LUPDATE_i_j | TRSM | B×B 非对角块 |
| LDL_BLK_BWD_ASSEMBLE | GEMM | U×U |

## 5. SPAD 布局（vs Cholesky Block 额外区域）

| 额外区域 | 用途 |
|---------|------|
| aD | D 对角因子 |
| aDinv | D 倒数 |

## 6. 验证数据

| 维度 | B | 周期 |
|------|---|------|
| U=16 | 2 | 4,959 |
