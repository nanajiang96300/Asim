# Cholesky/LDL NoBlock 优化计划

> 日期: 2026-06-30 | 基准: v2 Baseline

## 1. 旧算子中发掘的优化技术

### 1.1 SCALAR 指令合并（`_strict_iso_lowering`）

**来源**: `CholeskyInvNoBlockOp.cc` (旧)

**原理**: SCALAR Pipeline 是单发射通道，指令数直接决定延迟。在 POTRF/TRSM 的内层循环中，j 条 `SCALAR_MUL` 各自 `compute_size=1`，串行执行耗时 j × 1 = j cycles。如果合并为 1 条 `SCALAR_MUL compute_size=j`，Vector Unit 的并行宽度（2048-bit = 128 FP16/拍）可以在 ≈ j/128 拍内完成，远快于 j 拍。

**公式变换**:
```
旧:  for k=0..j-1: SCALAR_MUL(compute_size=1)  // j 条指令，j cycles
新:  SCALAR_MUL(compute_size=j, tile_m=j)        // 1 条指令，vec_op_iter(j) ≈ ceil(j/128) cycles
```

**适用位置**（Cholesky NoBlock）:
- POTRF 对角更新: j 条 `SCALAR_MUL(aL[j,k] × aL[j,k]*)` → 1 条
- TRSM 点积: j 条 `SCALAR_MUL(L[i,k] × L[j,k]*)` → 1 条
- FWD 点积: (i-c) 条 `SCALAR_MUL(L[i,k] × Y[k,c])` → 1 条

**适用位置**（LDL NoBlock）:
- D_UPDATE: j 条 `SCALAR_MUL(|L|²)` + j 条 `SCALAR_MUL(*D)` → 2 条（各自合并）
- L_UPDATE: j 条 `SCALAR_MUL(L*D)` + j 条 `SCALAR_MUL(*conj(L))` → 2 条

**⚠️ 约束**: SCALAR_SUB 不能合并（每步减法影响后续累加的正确语义），但可以用连续的 MUL+SUB 模式来表达"先全部乘完，再一次性减"。

### 1.2 GEMM_PRELOAD 复用

**原理**: `GEMM_PRELOAD` 每次发射前有 preload 开销（`core_height + core_height - 1` 周期），而 `GEMM`（不带 preload）没有。对于连续使用相同权重矩阵的 GEMM 操作，第二次起用 `GEMM` 代替 `GEMM_PRELOAD`。

**适用位置**:
- Newton-Schulz 的迭代循环: 第 1 次 `GEMM_PRELOAD(A@X)`，第 2 次起可尝试复用
- BRI 的 `B @ Y_l` 循环: 预条件器 B 不变，可以复用

### 1.3 Barrier 密度调优

**原理**: PIPE_BARRIER 等待前序所有流水线排空。过密增加等待开销，过疏导致依赖错误。

**优化方向**:
- NoBlock 算子当前每列 1 个 barrier → 可尝试每 2 列 1 个 barrier（Schur complement 需要前 1 列结果，但不需要前 2 列）
- 前向求解的 barrier 可以放宽：Y[c,c] 就绪后，所有 i>c 的 Y[i,c] 可以并行处理

## 2. 优化方案设计

### Opt1: SCALAR 合并优化 (Cholesky + LDL NoBlock)

**新文件**: `CholeskyNoBlockMergeOp.{h,cc}`, `LDLNoBlockMergeOp.{h,cc}`

**改动**: 从 Baseline 复制，将内层 MUL 循环替换为单条合并指令。

**公式推导**（以 Cholesky POTRF 为例）:

基线:
```
for k=0..j-1:
    SCALAR_MUL(id=POTRF_SQ_j_k, compute_size=1, src={aL,aL}, dest=aTmp)
    SCALAR_SUB(id=POTRF_SUB_j_k, compute_size=1, src={aA,aTmp}, dest=aA)
```

合并后:
```
SCALAR_MUL(id=POTRF_SQ_j, compute_size=j, src={aL,aL}, dest=aTmp)  // j elements at once
SCALAR_SUB(id=POTRF_SUB_j, compute_size=j, src={aA,aTmp}, dest=aA)  // subtract all
```

**预期效果**: 
- 指令数减少: 每列从 2j+2 条降至 4 条（Cholesky POTRF），总体 ~50% 减少
- 周期减少: ~30-50%（Scalar pipeline 瓶颈大幅缓解）

### Opt2: Barrier 密度优化 (Cholesky + LDL NoBlock)

**新文件**: `CholeskyNoBlockBarrierOptOp.{h,cc}`

**改动**: 从 Opt1 复制，减少 barrier 频率。

**Barrier 分析**:
- BARRIER type=4 (每列分解后): 每 2 列保留 1 个即可（第 j+1 列只依赖第 j 列，不依赖第 j-1 列）
- BARRIER type=5 (每列 FWD 后): 可以全部移除（FWD 的解向量元素间无依赖，只是列间有依赖）

**预期效果**: 周期减少 5-10%

### Opt3: 全优化合并 (Cholesky + LDL NoBlock)

**新文件**: `CholeskyNoBlockOptOp.{h,cc}`, `LDLNoBlockOptOp.{h,cc}`

**改动**: Opt1 + Opt2 的组合，加上 GEMM_PRELOAD 复用。

## 3. 文件规划

```
src/inverse/cholesky_noblock/
├── CholeskyNoBlockBaselineOp.{h,cc}       # 基线（已完成）
├── CholeskyNoBlockMergeOp.{h,cc}          # Opt1: SCALAR 合并
├── CholeskyNoBlockBarrierOptOp.{h,cc}     # Opt2: Barrier 优化
└── CholeskyNoBlockOptOp.{h,cc}            # Opt3: 全优化

src/inverse/ldl_noblock/
├── LDLNoBlockBaselineOp.{h,cc}            # 基线（已完成）
├── LDLNoBlockMergeOp.{h,cc}               # Opt1: SCALAR 合并
├── LDLNoBlockBarrierOptOp.{h,cc}          # Opt2: Barrier 优化
└── LDLNoBlockOptOp.{h,cc}                 # Opt3: 全优化

# Model 复用: 所有优化共用 BaselineModel，仅 Op 不同
```

## 4. 实施步骤

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 从 Baseline 复制 → CholeskyNoBlockMergeOp | Merge 版本 |
| 2 | 编写 Opt1 文档: 公式推导 + 指令映射对比 | DOCS/operators/01b_cholesky_noblock_merge.md |
| 3 | 编译运行，对比周期 | 数据 |
| 4 | 从 Merge 复制 → CholeskyNoBlockOptOp | 全优化版本 |
| 5 | 编写 Opt3 文档 | DOCS/operators/01c_cholesky_noblock_opt.md |
| 6 | 对 LDL 重复步骤 1-5 | 4 个 LDL 优化文件 + 文档 |
| 7 | 运行全量 benchmark | results/ 数据 |

## 5. 文档标准（每份优化文档）

1. **优化动机**: 为什么做这个优化
2. **公式推导**: 从基线公式出发，推导优化后的等价形式
3. **指令流对比**: 基线 vs 优化后的指令序列（表格）
4. **理论加速比**: 基于指令数和延迟模型的计算
5. **实测数据**: finish cycle, 加速比
6. **适用条件**: 什么情况下优化有效，什么情况下退化
