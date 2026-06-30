# Cholesky NoBlock v2 — 公式·算法·指令 全对应文档

> 版本: v2.0 | 日期: 2026-06-30 | 对应分支: `fix/scalar-unit-and-noblock-operators`

## 1. 文件清单

| 角色 | 路径 |
|------|------|
| Python 参考 | `scripts/algo/cholesky_noblock.py` |
| C++ 算子头 | `src/inverse/cholesky_noblock/CholeskyNoBlockBaselineOp.h` |
| C++ 算子实现 | `src/inverse/cholesky_noblock/CholeskyNoBlockBaselineOp.cc` |
| C++ 模型 | `src/inverse/cholesky_noblock/CholeskyNoBlockBaselineModel.{h,cc}` |
| 测试配置 | `example/cholesky_noblock_v2_test.json` |
| 运行命令 | `./build/bin/Simulator --config configs/ascend_910b_quiet.json --models_list example/cholesky_noblock_v2_test.json --mode cholesky_noblock_v2_test` |

## 2. 数学公式

输入 $H \in \mathbb{C}^{M \times U}$，$\lambda \in \mathbb{R}^+$。

### Phase 2: Gram + 正则化
$$G = H^H \cdot H \in \mathbb{C}^{U \times U}$$
$$A = G + \lambda I$$

### Phase 3: Cholesky 分解（逐列）
对 $j = 0,\ldots,U-1$：

**C1 (POTRF 对角):**
$$L_{jj} = \sqrt{A_{jj} - \sum_{k=0}^{j-1} L_{jk} \cdot L_{jk}^*}$$

**C2 (TRSM 非对角):** 对每个 $i > j$：
$$L_{ij} = \frac{A_{ij} - \sum_{k=0}^{j-1} L_{ik} \cdot L_{jk}^*}{L_{jj}}$$

### Phase 4: 前向求解 $Y = L^{-1}$
对 $c = 0,\ldots,U-1$：
$$Y_{cc} = \frac{1}{L_{cc}}$$
对 $i > c$：
$$Y_{ic} = -\frac{1}{L_{ii}} \sum_{k=c}^{i-1} L_{ik} \cdot Y_{kc}$$

### Phase 5: 后向装配
$$A^{-1} = Y^H \cdot Y$$

## 3. SPAD 内存布局

| 地址 | 变量 | 大小 | 用途 |
|------|------|------|------|
| `SPAD_BASE` | `aH` | M×U×2 | 输入 H |
| `aH + size_mu` | `aReg` | U×U×2 | 正则化 λI |
| `+ size_uu` | `aG` | U×U×2 | Gram = H^H·H |
| `+ size_uu` | `aA` | U×U×2 | A = G + λI |
| `+ size_uu` | `aL` | U×U×2 | L 因子 |
| `+ size_uu` | `aInv` | U×U×2 | 临时/逆元 |
| `+ size_uu` | `aY` | U×U×2 | Y = L^{-1} |
| `+ size_uu` | `aTmp` | U×U×2 | 临时空间 |
| `ACCUM_SPAD_BASE` | `aAinv` | U×U×2 | 输出 A^{-1} |

## 4. 指令映射表（U=16 示例，j=1, k=0）

### Phase 1: 数据搬运

| 公式 | 指令 | ID | src | dest | compute_size |
|------|------|-----|-----|------|-------------|
| Load H | `MOVIN` | — | dram_H | aH | M×U |
| Load Reg | `MOVIN` | — | dram_Reg | aReg | U×U |
| 同步 | `PIPE_BARRIER` | CHOL_NB_LOAD | — | — | type=1 |

### Phase 2: Gram + 正则化

| 公式 | 指令 | ID | src | dest | FormulaLogger |
|------|------|-----|-----|------|-------------|
| $G = H^H H$ | `GEMM_PRELOAD` | CHOL_NB_GRAM | {aH, aH} | aG | `emit_step("CHOL_NB_GRAM", "GEMM", {"H","H^H"}, "G", …)` |
| $A = G + λI$ | `ADD` | CHOL_NB_REG | {aG, aReg} | aA | `emit_step("CHOL_NB_REG", "DIAG_ADD", …)` |
| 同步 | `PIPE_BARRIER` | CHOL_NB_REG2DECOMP | — | — | type=3 |

### Phase 3: 分解（j=1, k=0 示例）

| 公式 | 指令 | ID | src | dest |
|------|------|-----|-----|------|
| $L_{1,0} \cdot L_{1,0}^*$ | `SCALAR_MUL` | CHOL_NB_POTRF_SQ_1_0 | {aL, aL} | aTmp |
| $A_{1,1} \mathrel{-}= |L_{1,0}|^2$ | `SCALAR_SUB` | CHOL_NB_POTRF_SUB_1_0 | {aA, aTmp} | aA |
| $L_{1,1} = \sqrt{A_{1,1}}$ | `SCALAR_SQRT` | CHOL_NB_POTRF_SQRT_1 | {aA} | aL |

→ `FormulaLogger::emit_step("CHOL_NB_POTRF_1", "CHOLESKY", …)`

| 公式 | 指令 | ID | src | dest |
|------|------|-----|-----|------|
| $L_{2,0} \cdot L_{1,0}^*$ | `SCALAR_MUL` | CHOL_NB_TRSM_MUL_2_1_0 | {aL, aL} | aTmp |
| $A_{2,1} \mathrel{-}= …$ | `SCALAR_SUB` | CHOL_NB_TRSM_SUB_2_1_0 | {aA, aTmp} | aA |
| $L_{2,1} = A_{2,1} / L_{1,1}$ | `SCALAR_DIV` | CHOL_NB_TRSM_DIV_2_1 | {aA, aL} | aL |

→ `FormulaLogger::emit_step("CHOL_NB_TRSM_2_1", "TRSM", …)`

每列结束 `PIPE_BARRIER type=4`

### Phase 4: 前向求解（c=0, i=1 示例）

| 公式 | 指令 | ID | src | dest |
|------|------|-----|-----|------|
| $1 = λ/λ$ | `SCALAR_DIV` | CHOL_NB_FWD_DIAG_0 | {aReg, aReg} | aInv |
| $Y_{0,0} = 1/L_{0,0}$ | `SCALAR_DIV` | CHOL_NB_FWD_DIAG2_0 | {aInv, aL} | aY |
| $L_{1,0} \cdot Y_{0,0}$ | `SCALAR_MUL` | CHOL_NB_FWD_MUL_1_0_0 | {aL, aY} | aTmp |
| $-sum$ | `SCALAR_SUB` | CHOL_NB_FWD_NEG_1_0 | {aReg, aTmp} | aTmp |
| $-sum / L_{1,1}$ | `SCALAR_DIV` | CHOL_NB_FWD_DIV_1_0 | {aTmp, aL} | aY |

每列结束 `PIPE_BARRIER type=5`

### Phase 5: 后向装配

| 公式 | 指令 | ID | src | dest | FormulaLogger |
|------|------|-----|-----|------|-------------|
| $A^{-1} = Y^H Y$ | `GEMM_PRELOAD` | CHOL_NB_BWD_GEMM | {aY, aY} | aAinv | `emit_step("CHOL_NB_BWD_ASSEMBLE", "GEMM", …)` |
| 同步 | `PIPE_BARRIER` | CHOL_NB_PRE_MOVOUT | — | — | type=6 |

### Phase 6: 写回

| 公式 | 指令 | ID | dest | 备注 |
|------|------|-----|------|------|
| Store $A^{-1}$ | `MOVOUT` | CHOL_NB_STORE | aAinv | `src_from_accum=true, last_inst=true` |

## 5. FormulaLogger 覆盖表

| step_id | op_type | 输入 | 输出 | 关联指令 |
|---------|---------|------|------|---------|
| CHOL_NB_GRAM | GEMM | H, H^H | G | CHOL_NB_GRAM |
| CHOL_NB_REG | DIAG_ADD | G, λI | A | CHOL_NB_REG |
| CHOL_NB_POTRF_0 … U-1 | CHOLESKY | A | L_jj | CHOL_NB_POTRF_SQRT_j |
| CHOL_NB_TRSM_i_j | TRSM | A, L | L_ij | CHOL_NB_TRSM_DIV_i_j |
| CHOL_NB_BWD_ASSEMBLE | GEMM | Y^H, Y | Ainv | CHOL_NB_BWD_GEMM |

## 6. 验证数据

| 维度 | 周期 | Python vs numpy | C++ 状态 |
|------|------|----------------|----------|
| U=4 (M=8) | 320 | 1e-15 | ✅ |
| U=16 (M=64) | 23,439 | 1e-15 (Rayleigh+CDL-B) | ✅ |

运行: `bash scripts/run_benchmark_suite.sh cholesky_noblock_v2 cholesky_noblock_v2_test configs/ascend_910b_quiet.json example/cholesky_noblock_v2_test.json`
