# Asim 算子开发标准 v3.0

> 版本: 3.0 | 日期: 2026-06-30
> 适用: 所有新算子及旧算子重写
> 参考实现: `CholeskyNoBlockBaselineOp`, `LDLNoBlockBaselineOp`

## 1. 指令集总览

### 1.1 三类流水线

| 流水线 | 指令 | 延迟模型 | 用途 |
|--------|------|---------|------|
| **Cube** | `GEMM_PRELOAD`, `GEMM` | `1+(M+N-2)+max(Bm*Bn*Bk,1)` @16³ | 矩阵乘法 |
| **Vector** | `ADD`, `MUL`, `MAC`, `DIV`, `SQRT`, `EXP`, `GELU`, `ADDTREE`, `COMP`, `PIPE_BARRIER` | `vec_op_iter × latency` | 向量/SIMD 操作、同步 |
| **Scalar** | `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`, `SCALAR_SQRT` | 固定延迟 (1-4 cycles) | 逐元素标量操作 |

### 1.2 SCALAR 指令详解

| 指令 | 延迟 | 语义 | 源操作数 | 目标 |
|------|------|------|---------|------|
| `SCALAR_ADD` | `scalar_add_latency` (1) | `dst = src1 + src2` | 2 | 1 |
| `SCALAR_SUB` | `scalar_add_latency` (1) | `dst = src1 - src2` | 2 | 1 |
| `SCALAR_MUL` | `scalar_mul_latency` (1) | `dst = src1 × src2` | 2 | 1 |
| `SCALAR_DIV` | `div_latency` (4) | `dst = src1 / src2` | 2 | 1 |
| `SCALAR_SQRT` | `scalar_sqrt_latency` (4) | `dst = √src1` | 1 | 1 |

**关键约束：**
- Scalar Pipeline **单发射**：同一时刻只能执行 1 条 SCALAR 指令
- 所有 SCALAR 指令必须 `compute_size = 1`
- SCALAR 使用 SPAD **基地址**（不是元素地址）
- 地址仅用于 SPAD 分配/命中检查，不参与实际数值路由
- SCALAR 是**纯周期模型**：公式语义通过 `FormulaLogger` 记录，数值正确性通过 Python 参考验证

### 1.3 Vector 数据搬运指令

| 指令 | 队列 | 粒度 |
|------|------|------|
| `MOVIN` | LD (Load) | 64B 分包，经 ICNT→DRAM |
| `MOVOUT` | ST (Store) | 64B 分包，经 ICNT→DRAM |
| `PIPE_BARRIER` | EX (Vector) | 1 cycle，等待前序所有流水线排空 |

## 2. SPAD 地址管理规范

### 2.1 地址常量

```cpp
#define SPAD_BASE        0x10000000  // 通用 SPAD
#define ACCUM_SPAD_BASE  0x20000000  // 累加器 SPAD (用于 GEMM 输出)
```

### 2.2 区域分配

每个 SPAD 区域按**元素大小 × 矩阵维度**连续分配：

```cpp
const uint32_t P = _config.precision;  // 2 (FP16)
const addr_type size_mu = M * U * P;   // [M × U] 矩阵
const addr_type size_uu = U * U * P;   // [U × U] 矩阵

const addr_type aH   = SPAD_BASE;          // H [M×U]
const addr_type aReg = aH   + size_mu;     // λI [U×U]
const addr_type aG   = aReg + size_uu;     // Gram [U×U]
const addr_type aA   = aG   + size_uu;     // A = G + λI [U×U]
const addr_type aL   = aA   + size_uu;     // L 因子 [U×U]
// ... 继续按需分配
const addr_type aAinv = ACCUM_SPAD_BASE;   // 输出逆矩阵
```

### 2.3 区域初始化 ⚠️ 关键

**任何 SCALAR 指令读写的新 SPAD 区域，必须在首次使用前初始化**，否则 SPAD `check_hit` 失败导致死锁。

初始化方法：
```cpp
// 用 Vector ADD 初始化整个区域（从已加载的 aReg 复制）
tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
    .opcode = Opcode::ADD,
    .id = "INIT_MYREGION",
    .dest_addr = aMyRegion,           // 新区域
    .compute_size = U * U,             // 元素数
    .src_addrs = {aReg, aReg},         // 从已分配区域复制
    .tile_m = U, .tile_k = U, .tile_n = U,
    .my_tile = tile}));
```

### 2.4 MOVIN base_addr ⚠️ 关键

**`base_addr` 必须设为 0**，因为 `src_addrs` 中已经包含了完整的 `dram_base + offset`：

```cpp
// ✅ 正确
auto movin = [&](addr_type dram, addr_type spad, ...) {
    addrs.insert(dram + make_address({r, c}, shape));  // 完整地址
    ...
    .operand_id = op_id,
    .base_addr = 0,   // ← 必须为 0，避免 double-add
};
```

**禁止**将 `dram_base` 同时放入 `src_addrs` 和 `base_addr`。

### 2.5 MOVOUT 规范

```cpp
std::set<addr_type> outs;
for (uint32_t r = 0; r < U; ++r)
    for (uint32_t c = 0; c < U; c += elems_per_access)
        outs.insert(dram_out + make_address({r, min(c, U-1)}, shape_uu));

tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
    .opcode = Opcode::MOVOUT,
    .dest_addr = aOutput,              // ACCUM SPAD 源
    .size = static_cast<uint32_t>(outs.size()),
    .src_addrs = vector<addr_type>(outs.begin(), outs.end()),
    .operand_id = _OUTPUT_OPERAND,
    .src_from_accum = true,            // 从 ACCUM 读取
    .last_inst = true,                 // 标记为最后一条指令
    .tile_m = U, .tile_k = U, .tile_n = U,
    .my_tile = tile}));
```

## 3. SCALAR 单元使用模式

### 3.1 恒等元合成

当公式需要常数 `1.0`（如 `1/L[c,c]`、`1/D[j]`），**禁止直接用 `addr_Reg`**（其对角元是 λ 不是 1）。必须合成：

```cpp
// Step 1: unity = Reg/Reg = 1
SCALAR_DIV  dest=aTmp,  src={aReg, aReg}

// Step 2: result = 1 / operand
SCALAR_DIV  dest=aResult, src={aTmp, aOperand}
```

### 3.2 Schur Complement 累减

公式 `A -= Σ term_k` 拆为：每条 term 一个 `MUL`（算 term）+ 一个 `SUB`（累减）：

```cpp
for (k = 0; k < j; ++k) {
    SCALAR_MUL  dest=aTmp,  src={...}     // term = L[j,k] * conj(L[j,k])
    SCALAR_SUB  dest=aA,    src={aA, aTmp} // A[j,j] -= term
}
```

### 3.3 取负

公式 `y = -sum` 用 `SCALAR_SUB` 从恒等元减去：

```cpp
SCALAR_SUB  dest=aResult,  src={aUnity, aSum}  // result = 0 - sum = -sum
```

其中 `aUnity` 是 3.1 合成的恒等元。

### 3.4 列缩放

公式 `Y[:,c] *= sqrt(Dinv[c])`：

```cpp
SCALAR_SQRT  dest=aTmp,  src={aDinv}           // sqrt_val = sqrt(Dinv[c])
SCALAR_MUL   dest=aY,    src={aY, aTmp},        // Y[:,c] *= sqrt_val
             .tile_m = U, .tile_k = 1, .tile_n = 1   // 整列
```

## 4. FormulaLogger 集成规范

### 4.1 必须声明算法身份

```cpp
FormulaLogger::instance().set_algorithm(
    "algorithm_name",   // 与 operator_registry.json 一致
    block_size,         // 分块大小（无分块填 1）
    layers,             // 迭代层数（直接法填 0）
    U);                 // 矩阵维度
```

### 4.2 必须覆盖所有数学阶段

| 阶段 | emit_step 要求 |
|------|---------------|
| Gram | `("GRAM", "GEMM", {"H","H^H"}, "G", ...)` |
| 正则化 | `("REG", "DIAG_ADD", {"G","lambda*I"}, "A", ...)` |
| 分解（每列） | `("POTRF_j"\|"D_UPDATE_j", "CHOLESKY"\|"DIAG_INV", ...)` |
| 三角求解（每步） | `("TRSM_i_j"\|"L_UPDATE_i_j", "TRSM", ...)` |
| 后向装配 | `("BWD_ASSEMBLE", "GEMM", {"Y^H","Y"}, "Ainv", ...)` |

```cpp
FormulaLogger::instance().emit_step(
    "STEP_ID",           // 全局唯一标识
    "OP_TYPE",           // UOBS 原语: GEMM/DIAG_ADD/CHOLESKY/TRSM/DIAG_INV
    {"input1","input2"}, // 输入张量名
    "output",            // 输出张量名
    {{M,U},{U,M}},       // 输入形状列表
    {U,U},               // 输出形状
    tile->batch,         // batch 索引
    "RELATION_ID");      // 关联指令 ID 前缀
```

### 4.3 UOBS 原语全集

| op_type | 语义 | 输入数 |
|---------|------|--------|
| `GEMM` | C = A @ B | 2 |
| `DIAG_ADD` | A += λI | 1+λ |
| `CHOLESKY` | L = chol(A) | 1 |
| `TRSM` | X = L⁻¹B | 2 |
| `DIAG_INV` | D_inv = 1/D | 1 |

## 5. 代码结构模板

### 5.1 文件命名

新算子文件命名：`<Algorithm><Variant>BaselineOp.{h,cc}` + `<Algorithm><Variant>BaselineModel.{h,cc}`

与旧算子（不带 `Baseline` 后缀）分开存储在同一目录下。

### 5.2 initialize_instructions() 结构

```cpp
void MyOp::initialize_instructions(Tile* tile, Mapping) {
    // 1. 维度常量
    const uint32_t M = ..., U = ...;
    
    // 2. SPAD 地址布局（逐区域声明 + 注释）
    const addr_type aH   = SPAD_BASE;
    const addr_type aReg = aH   + size_mu;  // [U×U] 正则化矩阵
    // ...
    
    // 3. Helper lambdas（movin, barrier）
    auto movin = [&](...) { ... };
    auto barrier = [&](const string& id, uint32_t type) { ... };
    
    // 4. Algorithm metadata
    FormulaLogger::instance().set_algorithm(...);
    
    // 5. Phase 1: 数据搬运（MOVIN × N + BARRIER + 区域初始化）
    // 6. Phase 2: Gram + 正则化（GEMM_PRELOAD + ADD + FormulaLogger × 2 + BARRIER）
    // 7. Phase 3: 分解主循环（SCALAR ops + FormulaLogger per col + BARRIER per col）
    // 8. Phase 4: 前向求解
    // 9. Phase 5: 后向装配（GEMM_PRELOAD + FormulaLogger + BARRIER）
    // 10. Phase 6: 写回（MOVOUT）
}
```

### 5.3 必须的 barrier 位置

| Barrier | Type | 位置 |
|---------|------|------|
| LOAD | 1 | MOVIN 完成后 |
| REG | 3 | Gram+正则化完成后 |
| COL_j | 4 | 每列分解完成后 |
| FWD_c | 5 | 每列前向求解完成后 |
| PRE_MOVOUT | 6 | GEMM 写回前 |

## 6. 常见错误与反模式 ⚠️

### 6.1 禁止直接使用 addr_Reg 作为数值 1

```cpp
// ❌ 错误：addr_Reg 对角元 = λ，不是 1
SCALAR_DIV dest=aY, src={aReg, aL}  // 计算 λ/L 不是 1/L

// ✅ 正确：先合成为 1
SCALAR_DIV dest=aUnity, src={aReg, aReg}  // unity = 1
SCALAR_DIV dest=aY,     src={aUnity, aL}   // 1/L
```

### 6.2 禁止 MOVIN double-add base

```cpp
// ❌ 错误：src_addrs 已包含 dram + offset，再加 base_addr 导致 double-add
.base_addr = dram_base

// ✅ 正确
.base_addr = 0
```

### 6.3 禁止未初始化的 SPAD 区域被 SCALAR 读取

```cpp
// ❌ 错误：aD, aDinv 未初始化就作为 src
SCALAR_MUL src={aL, aD}  // aD 从未被写过 → SPAD check_hit 失败 → 死锁

// ✅ 正确：初始化所有新区域
ADD dest=aD, src={aReg, aReg}, compute_size=U  // 先分配
```

### 6.4 禁止 SCALAR_MUL 用于除法

```cpp
// ❌ 错误：公式是除但用了乘
SCALAR_MUL dest=aY, src={aSum, aL}  // 乘法代替除法

// ✅ 正确
SCALAR_DIV dest=aY, src={aSum, aL}  // aSum / aL
```

### 6.5 禁止遗漏取负

```cpp
// ❌ 错误：公式 y = -sum 缺少取负
SCALAR_DIV dest=aY, src={aSum, aL}  // +sum/L

// ✅ 正确
SCALAR_SUB dest=aNeg, src={aUnity, aSum}  // -sum
SCALAR_DIV dest=aY,   src={aNeg, aL}      // -sum/L
```

### 6.6 禁止 FormulaLogger 覆盖不全

```cpp
// ❌ 错误：只记录了 GRAM 和 REG，遗漏了分解和求解步骤
// ✅ 正确：每个数学阶段都调用 emit_step
```

## 7. 审查检查清单

写完算子后，必须通过以下检查：

- [ ] 所有公式步骤有对应 opcode（MUL/DIV/SUB/SQRT/ADD）
- [ ] 恒等元通过 DIV(Reg/Reg) 合成
- [ ] Schur complement 累减用 MUL+SUB 对
- [ ] 取负用 SCALAR_SUB
- [ ] 所有 SPAD 区域在首次读取前已初始化
- [ ] MOVIN base_addr = 0
- [ ] MOVOUT src_from_accum = true, last_inst = true
- [ ] FormulaLogger 覆盖全部数学阶段
- [ ] 每列 / 每阶段有 PIPE_BARRIER
- [ ] 可运行 `/audit-operator` 自动审查通过
