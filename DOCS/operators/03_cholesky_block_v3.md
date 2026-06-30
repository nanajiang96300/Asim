# Cholesky Block v3 — 公式·算法·指令 全对应文档

> 版本: v3.0 | 日期: 2026-06-30

## 1. 文件清单

| 角色 | 路径 |
|------|------|
| C++ 算子 | `src/inverse/cholesky_block/CholeskyBlockBaselineOp.{h,cc}` |
| C++ 模型 | `src/inverse/cholesky_block/CholeskyBlockBaselineModel.{h,cc}` |
| 测试配置 | `example/cholesky_block_v3_test.json` |
| 运行 | `--mode cholesky_block_v3_test` |

## 2. 数学公式

将 $U \times U$ 矩阵划分为 $n_B = U/B$ 个 $B \times B$ 子块。Block Cholesky: $A = L \cdot L^H$。

### Phase 3: 块分解

对每个对角块 $j = 0,\ldots,n_B-1$：

**POTRF (对角块 Cholesky):**
$$L_{jj} = \text{chol}\left(A_{jj} - \sum_{k=0}^{j-1} L_{jk} \cdot L_{jk}^H\right)$$

其中 $L_{jj} \in \mathbb{C}^{B \times B}$ 通过对角块的逐元素 Cholesky 分解获得：
$$l_{pp} = \sqrt{a_{pp} - \sum_{q=0}^{p-1} l_{pq} \cdot l_{pq}^*}, \quad p = 0,\ldots,B-1$$
$$l_{rp} = \frac{a_{rp} - \sum_{q=0}^{p-1} l_{rq} \cdot l_{pq}^*}{l_{pp}}, \quad r > p$$

**TRSM (非对角块求解):** 对每个 $i > j$：
$$L_{ij} = \left(A_{ij} - \sum_{k=0}^{j-1} L_{ik} \cdot L_{jk}^H\right) \cdot L_{jj}^{-H}$$

**RK_UPDATE:** 对 $i,k > j$：
$$A_{ik} \leftarrow A_{ik} - L_{ij} \cdot L_{kj}^H$$

### Phase 4: 前向求解（块三角）

对每个块列 $c = 0,\ldots,n_B-1$：
$$Y_{cc} = L_{cc}^{-1}$$
对 $i > c$：
$$Y_{ic} = -L_{ii}^{-1} \cdot \sum_{k=c}^{i-1} L_{ik} \cdot Y_{kc}$$

### Phase 5: 后向装配
$$A^{-1} = Y^H \cdot Y$$

## 3. 指令映射（B=2 示例）

### POTRF 对角块

| 层次 | 公式 | 指令 |
|------|------|------|
| 块间 | $\sum_{k<j} L_{jk} L_{jk}^H$ | `GEMM` (B×B×B) per k |
| 块内 | $l_{pp} = \sqrt{…}$ | `SCALAR_MUL` × p + `SCALAR_SUB` × p + `SCALAR_SQRT` |
| 块内 | $l_{rp} = (…)/l_{pp}$ | `SCALAR_MUL` × p + `SCALAR_SUB` × p + `SCALAR_DIV` |

### TRSM 非对角块

| 层次 | 公式 | 指令 |
|------|------|------|
| 块间 | $\sum_{k<j} L_{ik} L_{jk}^H$ | `GEMM` per k |
| 块内 | $L_{ij} \cdot L_{jj}^{-H}$ | `SCALAR_DIV` × B (逐行除以对角元) |

### RK_UPDATE

| 层次 | 公式 | 指令 |
|------|------|------|
| 块间 | $A_{ik} \mathrel{-}= L_{ij} L_{kj}^H$ | `GEMM` (B×B×B) per (i,k) |

## 4. SPAD 布局

| 区域 | 用途 |
|------|------|
| aH, aReg, aG, aA | 同 NoBlock |
| aL | L 因子（全 U×U） |
| aTmp | 临时 B×B 块 |
| aY | 前向求解结果 |
| aAinv (ACCUM) | 输出 |

## 5. FormulaLogger 覆盖表

| step_id | op_type | 粒度 |
|---------|---------|------|
| CHOL_BLK_GRAM | GEMM | U×U |
| CHOL_BLK_REG | DIAG_ADD | U×U |
| CHOL_BLK_POTRF_GEMM_j_k | GEMM | B×B 块间 Schur |
| CHOL_BLK_POTRF_j | CHOLESKY | B×B 对角块 |
| CHOL_BLK_TRSM_i_j | TRSM | B×B 非对角块 |
| CHOL_BLK_BWD_ASSEMBLE | GEMM | U×U |

## 6. 验证数据

| 维度 | B | 周期 |
|------|---|------|
| U=16 | 2 | 6,440 |
