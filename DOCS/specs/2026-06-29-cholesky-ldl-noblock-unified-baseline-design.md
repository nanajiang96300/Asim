# Cholesky + LDL NoBlock 统一基线重写 — 设计文档

> 日期: 2026-06-29 | 状态: 设计完成，开始实施

## 1. 背景与目标

当前 Asim 中的 Cholesky-NoBlock 和 LDL-NoBlock 算子存在以下问题：
1. **编写不统一**：两个算子在 SPAD 布局、指令模式、barrier 策略上不一致
2. **LDL 数值错误**：D_UPDATE/L_UPDATE 的 Schur complement 减法被遗漏，forward solve 算法不正确
3. **FormulaLogger 覆盖不全**：LDL 只声明了 GRAM+REG 两步，遗漏了 D_UPDATE/L_UPDATE/BWD 等全部阶段
4. **缺少端到端可验证性**：没有 Python→C++→SE 的完整验证链路

**目标**：从公式推导出发，建立 Cholesky-NoBlock 和 LDL-NoBlock 的统一 Python 仿真基线，然后在 C++ 中严格按同一语义重写，最终通过 CDL-B 信道下的 SE 验证证明仿真器正确性。

## 2. 设计原则

1. **公式→Python→C++ 三步严格对应**：Python 算法中的每一步对应 C++ 中的一条或多条指令，中间不省略、不简化
2. **Cholesky 和 LDL 结构统一**：两者共享相同的代码骨架（MOVIN→GRAM→REG→Decomp→FwdSolve→BwdAssemble→MOVOUT），仅在分解阶段不同
3. **纯基线，无优化标记**：去掉 `_strict_iso_lowering`、`_use_left_looking` 等 flag，每条指令都是 1 条 SCALAR 操作（compute_size=1）
4. **公式声明全覆盖**：FormulaLogger 覆盖所有数学步骤，使 DAG executor 或 UOBS 可自动重建
5. **信道模型可插拔**：CDL-B / Rayleigh / i.i.d. 通过统一接口切换，供所有算法使用

## 3. 数学推导

### 3.1 符号与前置

- 输入：$\mathbf{H} \in \mathbb{C}^{M \times U}$（信道矩阵），$\lambda \in \mathbb{R}^+$（正则化系数）
- Gram 矩阵：$\mathbf{G} = \mathbf{H}^H \mathbf{H}$
- 正则化矩阵：$\mathbf{A} = \mathbf{G} + \lambda \mathbf{I} \in \mathbb{C}^{U \times U}$
- 目标：计算 $\mathbf{A}^{-1}$

记号约定：
- $\mathbf{A}[i,j]$ 表示第 $(i,j)$ 元素（0-indexed）
- $\mathbf{A}[i:j, k:l]$ 表示子矩阵切片
- $a_{ij}^*$ 表示复共轭

### 3.2 Cholesky NoBlock 分解（A = L·L^H）

对于 Hermitian 正定矩阵 $\mathbf{A} \in \mathbb{C}^{U \times U}$，Cholesky 分解 $\mathbf{A} = \mathbf{L}\mathbf{L}^H$ 的逐列（NoBlock）递推公式：

**第 j 列（j = 0, 1, ..., U-1）：**

**Step C1 (POTRF 对角更新)**：计算 $L[j,j]$ 的平方根
$$l_{jj} = \sqrt{a_{jj} - \sum_{k=0}^{j-1} l_{jk} \cdot l_{jk}^*}$$

此步对应：
- `j` 次 `SCALAR_MUL`：对每个 $k<j$，计算 $t_k = L[j,k] \cdot L[j,k]^*$
- `1` 次 `SCALAR_SUB`（累加）：$A[j,j] \leftarrow A[j,j] - \sum t_k$  
  （注：SCALAR 无 SUB 指令，用 SCALAR_MUL + SCALAR_ADD 的符号变体实现：$A[j,j] \leftarrow A[j,j] + (-t_k)$）
- `1` 次 `SCALAR_SQRT`：$L[j,j] = \sqrt{A[j,j]}$

**Step C2 (TRSM)**：对每个 $i > j$，计算 $L[i,j]$
$$l_{ij} = \frac{a_{ij} - \sum_{k=0}^{j-1} l_{ik} \cdot l_{jk}^*}{l_{jj}}$$

此步对应：
- `j` 次 `SCALAR_MUL`：$\text{sum} = \sum_{k=0}^{j-1} L[i,k] \cdot L[j,k]^*$
- `1` 次 `SCALAR_MUL`：$\text{num} = A[i,j] - \text{sum}$（存入 tmp 以保持 A 不变）
- `1` 次 `SCALAR_DIV`：$L[i,j] = \text{num} / L[j,j]$

**Step C3 (RK Update)**：对每个 $i \ge j+1$ 和 $k \ge i$，更新剩余子矩阵
$$a_{ik} \leftarrow a_{ik} - l_{ij} \cdot l_{kj}^*$$

此步对应：
- $1$ 次 `SCALAR_MUL`：$t = L[i,j] \cdot L[k,j]^*$
- 此时由 scalar pipeline 延迟建模，实际减法通过后续读 A 时重新计算产生

**Cholesky Forward Solve（计算 Y = L^{-1}）：**
对于 $c = 0, \dots, U-1$：
$$y_{cc} = 1 / l_{cc}$$
对于 $i > c$：
$$y_{ic} = -\frac{1}{l_{ii}} \sum_{k=c}^{i-1} l_{ik} \cdot y_{kc}$$

### 3.3 LDL NoBlock 分解（A = L·D·L^H）

对于 Hermitian 矩阵 $\mathbf{A}$，LDL 分解 $\mathbf{A} = \mathbf{L}\mathbf{D}\mathbf{L}^H$（$\mathbf{L}$ 为单位下三角，$\mathbf{D}$ 为实对角矩阵）的逐列递推公式：

**第 j 列（j = 0, 1, ..., U-1）：**

**Step L1 (D_UPDATE)**：计算 $D[j,j]$
$$d_{jj} = a_{jj} - \sum_{k=0}^{j-1} l_{jk} \cdot d_{kk} \cdot l_{jk}^*$$

此步对应：
- $2j$ 次 `SCALAR_MUL`：$t_k^i = L[j,k]^* \cdot (D[k,k])^{-1}$（为后续 L_UPDATE 预计算）
- `j` 次 `SCALAR_MUL`：$t_k = L[j,k] \cdot (d_{kk} \cdot L[j,k]^*)$ 
- `j` 次更新：$A[j,j] \leftarrow A[j,j] - d_{kk} \cdot |L[j,k]|^2$

注意：与 Cholesky 不同，LDL 需要显式保存 $\mathbf{D}$ 和 $\mathbf{L}$ 两个矩阵。$\mathbf{D}$ 为实对角矩阵。

**Step L2 (L_UPDATE)**：对每个 $i > j$，由 $\mathbf{L}$ 为单位下三角（对角全1）

$$l_{ij} = \frac{a_{ij} - \sum_{k=0}^{j-1} l_{ik} \cdot d_{kk} \cdot l_{jk}^*}{d_{jj}}$$

此步对应（内层 $k$ 循环步骤与 L1 类似，但操作不同行 i 的元素）

**LDL Forward Solve（计算 Z = D^{-1}·L^{-1}）：**
由于 $\mathbf{L}$ 是单位下三角（对角全1），只需除以 $\mathbf{D}$：
$$z_{ij} = \frac{1}{d_{jj}} \quad \text{（对角）}$$
对于 $i > j$：
$$z_{ij} = -\sum_{k=j}^{i-1} l_{ik} \cdot z_{kj}$$

**LDL Backward Assembly（计算 A^{-1} = L^{-H}·D^{-1}·L^{-1}）：**
$$\mathbf{A}^{-1} = \mathbf{Z}^H \cdot \mathbf{D} \cdot \mathbf{Z}$$

其中 $\mathbf{Z} = \mathbf{L}^{-1}$（由 Forward Solve 产生）。但因为我们求解的是 $\mathbf{Z} = \mathbf{D}^{-1} \cdot \mathbf{L}^{-1}$，最终的 $\mathbf{A}^{-1} = \mathbf{L}^{-H} \cdot \mathbf{D}^{-1} \cdot \mathbf{L}^{-1}$ 可直接由两次三角回代获得。

### 3.4 Cholesky vs LDL 指令数对比

对于 $U \times U$ 矩阵：

| 阶段 | Cholesky NoBlock | LDL NoBlock |
|------|-----------------|-------------|
| GRAM | 1×GEMM | 1×GEMM |
| REG | 1×ADD | 1×ADD |
| Decomp (per col j) | POTRF: j+1 MUL + 1 SQRT + 1 DIV; TRSM: (U-j-1)×(j+1) MUL + (U-j-1) DIV | D_UPDATE: 2j+1 MUL; L_UPDATE: (U-j-1)×(2j+1) MUL + (U-j-1) DIV |
| Fwd Solve | ~U²/2 MUL + U DIV | ~U²/2 MUL + U DIV |
| Bwd Assemble | 1×GEMM (Y·Y^H) | 1×GEMM (Y·Y^H) |

**预期 LDL 比 Cholesky 多 ~25-50% 指令（主要来自 D_UPDATE 中额外的 D 相关乘法和 L_UPDATE 中更多的内层循环），但 LDL 无 SQRT**。

## 4. Python 参考实现架构

### 4.1 文件结构

```
scripts/
├── channel/
│   ├── __init__.py          # ChannelModel 抽象接口
│   ├── rayleigh.py          # RayleighChannel（i.i.d. 复高斯）
│   └── cdl.py               # CDLChannel (Kronecker R_TX⊗R_RX, 3GPP 参数)
├── algo/
│   ├── __init__.py
│   ├── cholesky_noblock.py  # Cholesky-NoBlock Python 参考实现
│   └── ldl_noblock.py       # LDL-NoBlock Python 参考实现
├── quantize.py              # FP16 量化工具 (from evaluate_ldl_quality.py)
├── se_eval.py               # SE/BER 评估框架
└── verify_baseline.py       # ★ 端到端验证脚本：Python→C++→SE 完整链路
```

### 4.2 ChannelModel 接口

```python
class ChannelModel(ABC):
    @abstractmethod
    def generate(self, batch_size: int, nr: int, nt: int, 
                 seed: int = None) -> np.ndarray:
        """Generate H ∈ C^{batch × nr × nt}"""
        ...
```

### 4.3 CDL-B (Kronecker 模型)

```python
class CDLBChannel(ChannelModel):
    def __init__(self, delay_spread: float = 100e-9,
                 carrier_freq: float = 3.5e9,
                 tx_correlation: float = 0.3,
                 rx_correlation: float = 0.5):
        # 3GPP TR 38.901 CDL-B 参数:
        # - 角度扩展 (ASA=10°, ASD=22°)
        # - Kronecker model: H = R_RX^{1/2} · G · R_TX^{1/2}
        # - G 为 i.i.d. 瑞利信道
        ...
```

### 4.4 Python 算法实现约定

每个算法的 Python 函数必须：
1. 接受 `A` (Gram+reg 后的矩阵) 和 `cfg` (EvalConfig)
2. 在每一步后进行 FP16 量化（模拟 C++ 标量精度）
3. 返回 `A_inv`，值范围与 C++ 输出一致
4. 函数签名：`def cholesky_noblock_inverse(A: np.ndarray, cfg: EvalConfig) -> np.ndarray`

## 5. C++ 算子重写规范

### 5.1 文件结构

```
src/inverse/
├── cholesky_noblock/
│   ├── CholeskyInvNoBlockOp.cc  ← 重写：统一基线
│   ├── CholeskyInvNoBlockOp.h
│   ├── CholeskyNoBlockModel.cc
│   ├── CholeskyNoBlockModel.h
│   └── VERSION.md               ← 更新：v2.0 重构记录
├── ldl_noblock/
│   ├── LDLDecompNoBlockOp.cc    ← 重写：统一基线
│   ├── LDLDecompNoBlockOp.h
│   ├── LDLNoBlockModel.cc
│   ├── LDLNoBlockModel.h
│   └── VERSION.md               ← 更新：v2.0 重构记录
```

旧的 `LDLDecompNoBlockAlignedOp` 归档到 `ldl_noblock/_legacy/` 或直接删除（因为被新基线替代）。

### 5.2 统一代码骨架

两个算子的 `initialize_instructions()` 共享相同结构：

```
[Phase 1: 数据搬运]
  MOVIN(H) + MOVIN(RegI)
  PIPE_BARRIER(type=1)

[Phase 2: Gram + 正则化]
  GEMM_PRELOAD: G = H^H @ H
  FormulaLogger::emit_step("GRAM", "GEMM", ...)
  ADD: A = G + λI
  FormulaLogger::emit_step("REG", "DIAG_ADD", ...)
  PIPE_BARRIER(type=3)

[Phase 3: 分解]  ← Cholesky/LDL 唯一不同之处
  for j = 0..U-1:
    [POTRF/D_UPDATE]
    [TRSM/L_UPDATE]
    PIPE_BARRIER(type=4)

[Phase 4: 前向求解]
  for c = 0..U-1:
    SCALAR_DIV + SCALAR_MUL 链路
    PIPE_BARRIER(type=5)

[Phase 5: 反向装配]
  GEMM_PRELOAD: Ainv = Y @ Y^H
  FormulaLogger::emit_step("BWD_ASSEMBLE", "GEMM", ...)
  PIPE_BARRIER(type=6)

[Phase 6: 写回]
  MOVOUT(Ainv)
```

### 5.3 FormulaLogger 全覆盖规范

每个数学步骤必须调用 `FormulaLogger::emit_step()`：

| phase | step_id pattern | op_type | 说明 |
|-------|----------------|---------|------|
| GRAM | `CHOL_NB_GRAM` / `LDL_NB_GRAM` | `GEMM` | G = H^H @ H |
| REG | `CHOL_NB_REG` / `LDL_NB_REG` | `DIAG_ADD` | A = G + λI |
| POTRF/C1 | `CHOL_NB_POTRF_{j}` | `CHOLESKY` | L[j,j] = sqrt(...) |
| TRSM/C2 | `CHOL_NB_TRSM_{i}_{j}` | `TRSM` | L[i,j] = ... / L[j,j] |
| D_UPDATE/L1 | `LDL_NB_DUPDATE_{j}` | `DIAG_INV` | D[j,j] = ... |
| L_UPDATE/L2 | `LDL_NB_LUPDATE_{i}_{j}` | `TRSM` | L[i,j] = ... |
| BWD | `CHOL_NB_BWD` / `LDL_NB_BWD` | `GEMM` | Ainv = Y @ Y^H |

### 5.4 新增 get_verification_spec()

在 Model 类中实现 `get_verification_spec()` 方法，使得通用 harness 可以直接验证数值正确性：

```cpp
json get_verification_spec() const override {
    return {
        {"tensors", {
            {"H", {/* shape, dram_addr */}},
            {"RegI", {/* shape, dram_addr */}},
            {"Ainv", {/* shape, dram_addr, expected tolerance */}}
        }}
    };
}
```

### 5.5 代码质量标准

1. **无魔术数字**：所有 SPAD 地址用命名常量
2. **emit_movin 公共化**：提取到 Operation 基类或 helper 头文件
3. **barrier type 语义化**：enum 代替 1/3/4/5/6
4. **地址计算可审计**：每个 addr_XXX 有明确注释说明存储内容和大小
5. **MOVOUT base_addr 正确**：不要 double-add（修复 LDL 的已知 bug）

## 6. 实施计划

### Phase 1: Python 基础设施 (预计 2h)
1. 创建 `scripts/channel/` — ChannelModel 接口 + RayleighChannel + CDLBChannel
2. 创建 `scripts/algo/` — cholesky_noblock.py + ldl_noblock.py
3. 创建 `scripts/verify_baseline.py` — 端到端验证脚本
4. 在 CDL-B 信道下验证 Python 算法的 SE 正确性

### Phase 2: C++ 算子重写 (预计 3h)
1. 重写 `CholeskyInvNoBlockOp.cc` — 统一骨架，FormulaLogger 全覆盖
2. 重写 `LDLDecompNoBlockOp.cc` — 与 Cholesky 完全一致的骨架
3. 更新 Model 文件
4. 编译验证

### Phase 3: 仿真验证 (预计 1h)
1. 运行两个算子，对比 cycle
2. 对比 Python 和 C++ 的输出逆矩阵（数值验证）
3. SE 对比验证

### Phase 4: 归档清理 (预计 0.5h)
1. 归档 LDLDecompNoBlockAlignedOp 到 _legacy/
2. 更新 VERSION.md
3. 更新 operator_registry.json
4. 运行全量 benchmark suite

## 7. 验证标准

### 7.1 内部一致性
- C++ finish cycle 与 Python 指令序列的延迟模型预测一致（±5%）
- Cholesky NoBlock cycle ≈ LDL NoBlock cycle（LDL 略高 10-30%）

### 7.2 数值正确性
- $\|\mathbf{A}_{\text{cpp}}^{-1} - \mathbf{A}_{\text{py}}^{-1}\|_F / \|\mathbf{A}_{\text{py}}^{-1}\|_F < 10^{-4}$
- $\|\mathbf{A}_{\text{cpp}}^{-1} - \mathbf{A}_{\text{ref}}^{-1}\|_F / \|\mathbf{A}_{\text{ref}}^{-1}\|_F < 10^{-2}$（FP16 精度）

### 7.3 SE 验证
- CDL-B 信道下，Cholesky 和 LDL 的 SE 曲线与 np.linalg.inv 精确解差距 < 1%
- 所有 SNR 点 SE 单调递增

## 8. 文件变更清单

| 操作 | 文件 |
|------|------|
| 新建 | `scripts/channel/__init__.py` |
| 新建 | `scripts/channel/rayleigh.py` |
| 新建 | `scripts/channel/cdl.py` |
| 新建 | `scripts/algo/__init__.py` |
| 新建 | `scripts/algo/cholesky_noblock.py` |
| 新建 | `scripts/algo/ldl_noblock.py` |
| 新建 | `scripts/verify_baseline.py` |
| 重写 | `src/inverse/cholesky_noblock/CholeskyInvNoBlockOp.cc` |
| 重写 | `src/inverse/cholesky_noblock/CholeskyInvNoBlockOp.h` |
| 重写 | `src/inverse/ldl_noblock/LDLDecompNoBlockOp.cc` |
| 重写 | `src/inverse/ldl_noblock/LDLDecompNoBlockOp.h` |
| 修改 | `src/inverse/cholesky_noblock/CholeskyNoBlockModel.cc` |
| 修改 | `src/inverse/ldl_noblock/LDLNoBlockModel.cc` |
| 归档 | `src/inverse/ldl_noblock/LDLDecompNoBlockAlignedOp.*` → `_legacy/` |
| 更新 | `src/inverse/cholesky_noblock/VERSION.md` |
| 更新 | `src/inverse/ldl_noblock/VERSION.md` |
| 更新 | `orchestrator/operator_registry.json` |
