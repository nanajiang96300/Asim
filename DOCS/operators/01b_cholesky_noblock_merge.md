# Cholesky NoBlock — SCALAR 合并优化 (Opt1)

> 基线: `CholeskyNoBlockBaselineOp` (23,439 cyc) | 优化: `CholeskyNoBlockMergeOp` (9,999 cyc) | 加速比: **2.34×**

## 1. 优化动机

SCALAR Pipeline 是单发射通道，指令数直接决定周期。基线的内层循环将 j 次点积拆为 j 条 `SCALAR_MUL compute_size=1`，全部串行化在 Scalar Pipeline 上。合并为 1 条 `compute_size=j` 后，利用 SCALAR 的向量化处理能力并行完成。

## 2. 公式推导

### 2.1 数学等价性

优化基于**累加操作的结合律**和**点积的向量化重写**。

POTRF 的 Schur complement 公式为：
$$A_{jj} \leftarrow A_{jj} - \sum_{k=0}^{j-1} |L_{jk}|^2$$

令 $\mathbf{v} = L_{j,0:j} \in \mathbb{C}^j$（第 j 行的前 j 个元素），则：
$$\sum_{k=0}^{j-1} |L_{jk}|^2 = \sum_{k=0}^{j-1} L_{jk} \cdot L_{jk}^* = \mathbf{v} \cdot \mathbf{v}^H = \|\mathbf{v}\|_2^2$$

**基线**: 逐个减去 $|L_{jk}|^2$（j 步）  
**合并**: 一次性计算 $\|\mathbf{v}\|_2^2$ 向量点积并减去结果

由于减法满足结合律：$A - t_0 - t_1 - \dots - t_{j-1} = A - (t_0 + t_1 + \dots + t_{j-1}) = A - \sum t_k$，两者数学等价。

同理，TRSM：
$$A_{ij} \leftarrow A_{ij} - \sum_{k=0}^{j-1} L_{ik} \cdot L_{jk}^* = A_{ij} - \mathbf{L}_{i,0:j} \cdot \mathbf{L}_{j,0:j}^H$$

FWD：
$$Y_{ic} = -\frac{1}{L_{ii}} \sum_{k=c}^{i-1} L_{ik} \cdot Y_{kc} = -\frac{1}{L_{ii}} \cdot \mathbf{L}_{i,c:i} \cdot \mathbf{Y}_{c:i,c}$$

**约束条件**: SCALAR_SUB 在 `compute_size=j` 时对一个 j 维向量执行逐元素减法。源地址指向的 SPAD 区域必须连续存储 j 个元素。这在 Asim 基地址模型中成立（SPAD 区域整体分配）。

### 2.2 指令流对比

```
SCALAR_MUL  id=POTRF_SQ_3_0  compute_size=1    // |L[3,0]|^2
SCALAR_SUB  id=POTRF_SUB_3_0  compute_size=1    // A -= term0
SCALAR_MUL  id=POTRF_SQ_3_1  compute_size=1    // |L[3,1]|^2
SCALAR_SUB  id=POTRF_SUB_3_1  compute_size=1    // A -= term1
SCALAR_MUL  id=POTRF_SQ_3_2  compute_size=1    // |L[3,2]|^2
SCALAR_SUB  id=POTRF_SUB_3_2  compute_size=1    // A -= term2
```
共 2j = 6 条指令，6 cycles（Scalar Pipeline 串行）

### 优化后

```
SCALAR_MUL  id=POTRF_SQ_3  compute_size=3  tile_m=1 tile_k=3  // |L[3,:]|^2 全向量
SCALAR_SUB  id=POTRF_SUB_3  compute_size=3  tile_m=1 tile_k=3  // A -= vec
```
共 2 条指令，约 2 × ceil(j/128) ≈ 2 cycles（j=3 时），实际接近 2 cycles

**理论加速**: 从 O(j) 降到 O(1)，对大 U 加速更显著。

### TRSM 合并（同样模式）

**基线**: 每个 (i,j) 对内 j 条 MUL+SUB → 2j 条指令
**优化**: 每个 (i,j) 对内 1 条 MUL(j)+1 条 SUB(j) → 2 条指令

### FWD 合并

**基线**: 每个 (i,c) 对内 (i-c) 条 MUL → (i-c) 条指令
**优化**: 每个 (i,c) 对内 1 条 MUL(len) → 1 条指令

## 3. 指令数对比 (U=16)

| 阶段 | 基线指令数 | 合并后指令数 | 减少 |
|------|-----------|-------------|------|
| POTRF (全列) | Σ(2j) ≈ U² ≈ 256 | Σ(2×1) ≈ 2U ≈ 32 | 87.5% |
| TRSM (全列) | Σ(2j×(U-j-1)) ≈ U³/3 ≈ 1365 | Σ(2×(U-j-1)) ≈ U² ≈ 256 | 81% |
| FWD (全列) | Σ((i-c)×(U-c-1)) ≈ U³/6 ≈ 682 | Σ(1×(U-c-1)) ≈ U²/2 ≈ 128 | 81% |

## 4. 适用条件

- ✅ NoBlock (B=1): j 越大加速越显著
- ✅ Block (B≥2): 块内 SCALAR 同样可合并
- ❌ 向量化 penalty: compute_size 很大且超过 SPAD 单次处理能力时退化

## 5. 实测数据

| 配置 | 基线 | Merge | 加速比 |
|------|------|-------|--------|
| U=16, M=64, bs=96 | 23,439 | 9,999 | 2.34× |

运行: `--mode cholesky_noblock_merge_test`
