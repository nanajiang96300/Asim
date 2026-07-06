# Asim 项目周报

> 周期: 2026-07-01 ~ 2026-07-06 | 分支: master → main
> 远程: https://github.com/nanajiang96300/Asim.git

---

## 一、总体概况

本周完成从"混乱的多版本算子"到"标准化、可验证、有门禁"的架构转型。核心成果：

| 维度 | 变更前 | 变更后 |
|------|------|------|
| 算子组织 | 散落在 src/operations/，新旧混存 | 统一在 src/inverse/，按算子分目录 |
| 验证方式 | 无自动化验证 | 6 个算子专用验证脚本 + DAG 重放 |
| 开发流程 | 无门禁 | 3 层 CI 门禁 + 9 阶段 /op-flow 流水线 |
| 文档 | 分散三处 | 统一清单 + v3 标准 + 原语规范 |
| 数值正确性 | 无保证 | 双路径 DAG 数值验证（FP16 量化） |

---

## 二、架构调整

### 2.1 整体架构图

```mermaid
graph TB
    subgraph "用户层 (User Layer)"
        A["/op-flow 流水线<br/>新建/优化算子"]
        B["/audit-operator<br/>公式-代码审查"]
        C["/verify-operator<br/>数值验证"]
        D["/eval-patch<br/>黑盒评估"]
        E["/opt-round<br/>AI 驱动优化"]
    end

    subgraph "CI 门禁 (CI Gate)"
        F["scripts/ci_gate.sh<br/>构建 → 测试 → 验证"]
    end

    subgraph "仿真层 (Simulator Layer)"
        G["C++ Simulator<br/>cycle-level NPU model"]
        H["FormulaLogger<br/>数学语义嵌入"]
        I["SCALAR Pipeline<br/>单发射, 固定延迟"]
        J["Cube Pipeline<br/>GEMM_PRELOAD/GEMM"]
        K["Vector Pipeline<br/>ADD/MUL/DIV/BARRIER"]
    end

    subgraph "验证层 (Verification Layer)"
        L["DAG Executor<br/>formula_steps.json 重放"]
        M["Per-Op Verify Scripts<br/>6 个算子双路径对比"]
        N["Trace Audit<br/>GEMM 覆盖率检查"]
    end

    subgraph "数据层 (Data Layer)"
        O[("算子文档<br/>DOCS/operators/")]
        P[("formula_steps.json<br/>DAG 步骤记录")]
        Q[("trace.csv<br/>指令级轨迹")]
        R[("基线快照<br/>results/baselines/")]
    end

    A --> G
    B --> G
    C --> H
    C --> L
    D --> G
    E --> D
    F --> G
    F --> L
    F --> N
    G --> H
    H --> P
    G --> Q
    L --> M
    M --> O
    P --> L
    N --> M
```

### 2.2 算子矩阵

6 个 Baseline 算子 + 1 个优化变体，统一在 `src/inverse/` 下，旧版移入 `legacy_operators/`：

| 算子 | 算法名 | 周期数 (U=16) | emit_step | 指令数 | 方法类型 | 方阵求逆方法 |
|------|------|------|:---:|:---:|------|------|
| Cholesky NoBlock v2 | `cholesky_noblock_v2` | 23,439 | 5 | 20 | 直接法基线 | A=LL^H → Y=L⁻¹ → A⁻¹=Y^HY |
| Cholesky NoBlock Merge | `cholesky_noblock_merge` | 9,999 | 5 | 20 | SCALAR 合并优化 | 同上（合并 per-column SCALAR） |
| LDL NoBlock v2 | `ldl_noblock_v2` | 25,628 | 5 | 26 | 直接法基线 | A=LDL^H → Y=√D⁻¹L⁻¹ → A⁻¹=Y^HY |
| Cholesky Block v3 | `cholesky_block_v3` | 6,440 (B=2) | 7 | 19 | 分块直接法 | Block Cholesky + Block TRSM |
| LDL Block v3 | `ldl_block_v3` | 4,959 (B=2) | 5 | 25 | 分块直接法 | Block LDL + Block Forward Solve |
| Newton-Schulz v3 | `newton_schulz_v3` | 2,591 (N=32,K=8) | 4 | 7 | 迭代法 | X_{k+1}=X_k(2I-AX_k) |
| Block-Richardson v3 | `block_richardson_v3` | 1,993 (B=2,L=8) | 7 | 19 | 迭代预处理 | Y_{l+1}=Y_l(2I-BY_l) |

```
src/inverse/
├── cholesky_noblock/        # Cholesky 无分块基线 (v2)
│   ├── CholeskyNoBlockBaselineOp.{h,cc}
│   ├── CholeskyNoBlockMergeOp.{h,cc}     # SCALAR merge 优化
│   └── CholeskyNoBlockBaselineModel.{h,cc}
├── ldl_noblock/             # LDL 无分块基线 (v2)
│   ├── LDLNoBlockBaselineOp.{h,cc}
│   └── LDLNoBlockBaselineModel.{h,cc}
├── cholesky_block/          # Cholesky 分块 (v3)
├── ldl_block/               # LDL 分块 (v3)
├── newton_schulz/           # Newton-Schulz 迭代 (v3)
├── block_richardson/        # Block-Richardson 迭代 (v3)
```

### 2.3 算子版本演进策略

| 版本 | 标记 | 含义 | 适用场景 |
|------|------|------|------|
| v1 (已废弃) | `*Op` (src/operations/) | 旧版实现，公式-代码不一致 | 仅作考古参考 |
| v2 | `*BaselineOp`（无分块）| 纯净基线：逐元素 SCALAR，完整 FormulaLogger 覆盖 | 正确性基准、回归测试 |
| v2 | `*MergeOp`（无分块）| 优化变体：合并 SCALAR 操作减少指令数 | 性能优化探索 |
| v3 | `*BaselineOp`（分块/迭代）| 分块或迭代实现，`_optype` 区分 | 性能优化 + 大规模矩阵 |

### 2.4 关键架构决策

1. **DAG 原语三层分级**：Core（6 个）→ Algorithm（3 个）→ Operator-specific（3 个），严格防耦合。新增算子优先用 Core 原语组合
2. **Per-Operator 验证**替代通用验证：每个算子独立验证脚本，自由组合原语，独立设定阈值
3. **FormulaLogger DAG 链规范**：emit_step 输出名 ≡ 下游输入名，形成 H → ... → Ainv 完整链路
4. **分支开发 + 审查 + 门禁合入**：feature branch → build → /audit-operator → /verify-operator → CI gate → merge
5. **SCALAR 单元抽象为周期模型**：基地址模型，不追踪逐元素数值；公式语义通过 FormulaLogger 记录，数值正确性通过 Python 参考验证

---

## 三、仿真链路问题修补

### 3.1 已修复的关键缺陷

| # | 算子 | 问题 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | SCALAR 单元 | 缺失 SCALAR_SUB 指令 | 硬件模型不完整 | 新增 SCALAR_SUB opcode |
| 2 | Cholesky Block | DAG 链断裂（Y 未产出） | 缺少 FWD_SOLVE emit_step | 添加 TRSM L→Y 步骤 |
| 3 | LDL Block | DAG 链破坏（逐块覆盖） | 逐块 DUPDATE/LUPDATE 覆盖 Y | 替换为单次 LDL_DECOMPOSE |
| 4 | Newton-Schulz | DAG 链不完整（无 Ainv） | 缺少 BWD_ASSEMBLE + 2I 未注册 | 添加最终 GEMM + 2I 特判 |
| 5 | BRI | 硬编码 Y_7（仅 L=8 有效） | 代码直接写死 | 改为动态 Y_{L-1} |
| 6 | Cholesky Block | TRSM 引用未注册的 L_j | 输出名不一致 | 统一为 L 输出 |
| 7 | Multi-seed | 5 个验证脚本 lambda bug | s vs seed 变量名错误 | 统一修复为 seed |
| 8 | GTest 构建 | CMake 4.x + GCC 新版本兼容性 | GTest 1.8.1 硬编码 -Werror | PATCH_COMMAND 修复 |
| 9 | CI 门禁 | 6 个算子 mode 名错误 | _baseline vs _v2_test 不匹配 | 统一为 main.cc 中实际 mode |

### 3.2 SPAD 死锁修复

LDL 算子新增 SPAD 区域（aD, aDinv, aTmp）未初始化，SCALAR 指令读取时 check_hit 失败导致死锁。修复：使用 `ADD(dest=X, src={aReg, aReg})` 在首次使用前初始化所有新区域。

### 3.3 GRAM 输入顺序修正

所有算子 GRAM emit_step 输入从 `{"H", "H^H"}` 改为 `{"H^H", "H"}`，确保 DAG executor 中 `prim_gemm(H^H, H)` 输出正确维度 (U×M) @ (M×U) → (U×U)。

---

## 四、DAG 原理与实现

### 4.1 FormulaLogger ↔ DAG Executor 全链路

```mermaid
graph TB
    subgraph "C++ 仿真器 (initialize_instructions)"
        A["set_algorithm(name, B, L, U)"]
        B["emit_step(GRAM, GEMM, {H^H,H}, G)"]
        C["emit_step(REG, DIAG_ADD, {G,lambda*I}, A)"]
        D["emit_step(POTRF, CHOLESKY, {A}, L)"]
        E["emit_step(FWD, TRSM, {L}, Y)"]
        F["emit_step(BWD, GEMM, {Y^H,Y}, Ainv)"]
    end

    subgraph "formula_steps.json"
        G["[{step_id: GRAM, op_type: GEMM, inputs: [H^H,H], output: G},<br/>{step_id: REG, op_type: DIAG_ADD, inputs: [G,lambda*I], output: A},<br/>{step_id: POTRF, op_type: CHOLESKY, inputs: [A], output: L},<br/>{step_id: FWD, op_type: TRSM, inputs: [L], output: Y},<br/>{step_id: BWD, op_type: GEMM, inputs: [Y^H,Y], output: Ainv}]"]
    end

    subgraph "Python DAG Executor"
        H["FormulaDAG(steps)<br/>构建 DAG 节点图"]
        I["dag.execute({H: matrix}, {lambda: 0.1})"]
        J["拓扑执行每个节点"]
        K["输入解析链:<br/>batch → batch-0 → aux<br/>→ I/2I/lambda*I<br/>→ H^H/Y^H 共轭转置"]
        L["查找 PRIMITIVES[node.op_type]<br/>调用原语函数"]
        M["registry[(batch, name)] = FP16(result)"]
        N["返回 {Ainv: np.ndarray}"]
    end

    A --> B --> C --> D --> E --> F
    F --> G
    G --> H --> I --> J
    J --> K --> L --> M --> N
```

### 4.2 双路径验证原理

```
Path A (DAG)                          Path B (Reference)
─────────────────                     ─────────────────
formula_steps.json                    Python primitives
      │                                      │
      ▼                                      ▼
FormulaDAG.execute(H)           prim_cholesky(prim_diag_add(
      │                         prim_gemm(H^H, H), 0.1))
      ▼                                      │
   Ainv_dag                                 ▼
      │                                 Ainv_ref
      │                                      │
      └────────── error = ||A_dag - A_ref|| ─┘
                          ────────────────
                             ||A_ref||

   PASS if error < THRESHOLD
```

### 4.3 输入名特殊解析链

DAG Executor 按以下优先级解析输入名：

1. `registry[(batch, name)]` — 精确 batch 匹配
2. `registry[(0, name)]` — 回退到 batch 0
3. `aux_params[name]` — 辅助参数（如 lambda）
4. `"I"` → `np.eye(N)` — 单位矩阵
5. `"2I"` → `2.0 * np.eye(N)` — 两倍单位矩阵
6. `"lambda*I"` → `lambda * np.eye(N)` — 正则化矩阵
7. `"H^H"` → 从 registry 查找 H，返回 `H.conj().T`
8. `"Y^H"` → 从 registry 查找 Y，返回 `Y.conj().T`

---

## 五、DAG 原语表

### 5.1 原语三级体系

```
Core Primitives (6)
├── GEMM          C = A @ B
├── DIAG_ADD      A += λI
├── TRSM          Y = L⁻¹ (1-input) / pass-through (2-input)
├── MATRIX_SUB    C = A - B
├── MATRIX_ADD    C = A + B
└── SCALE         A ← α·A

Algorithm Primitives (3)
├── CHOLESKY      L = chol(A)
├── LDL_DECOMPOSE Full LDL: L·D·L^H + forward solve + sqrt(Dinv)
└── DIAG_INV      D⁻¹ = 1/D

Operator-Specific Primitives (3)
├── BRI_PRECOND   B = blockdiag(A_ii⁻¹)
├── MATRIX_INV_2x2  Direct 2×2 inversion
└── SQRT_SCALE    Y *= sqrt(Dinv)
```

### 5.2 各算子 DAG 链

| 算子类型 | DAG 链 | 原语 |
|---------|------|------|
| Cholesky (NoBlock/Block) | `GRAM(GEMM) → REG(DIAG_ADD) → POTRF(CHOLESKY) → FWD_SOLVE(TRSM) → BWD_ASSEMBLE(GEMM)` | GEMM, DIAG_ADD, CHOLESKY, TRSM |
| LDL (NoBlock/Block) | `GRAM(GEMM) → REG(DIAG_ADD) → LDL_DECOMPOSE → BWD_ASSEMBLE(GEMM)` | GEMM, DIAG_ADD, LDL_DECOMPOSE |
| Newton-Schulz (K iter) | `K×(GEMM → MATRIX_SUB → GEMM) + BWD_ASSEMBLE(GEMM)` | GEMM, MATRIX_SUB |
| Block-Richardson (L iter) | `GRAM(GEMM) → REG(DIAG_ADD) → BRI_PRECOND → L×(GEMM → MATRIX_SUB → MATRIX_ADD)` | GEMM, DIAG_ADD, BRI_PRECOND, MATRIX_SUB, MATRIX_ADD |

---

## 六、验证体系

### 6.1 三层验证架构

```
Layer 1: Code Audit          Layer 2: Numerical          Layer 3: Integration
─────────────────          ─────────────────          ──────────────────
/audit-operator <name>     /verify-operator <name>    scripts/ci_gate.sh
│                          │                          │
├─ opcode 正确性            ├─ formula_steps.json       ├─ Build + Test
├─ 操作数正确性             ├─ FormulaDAG 执行          ├─ DAG 自测
├─ 循环结构                 ├─ Python Reference         ├─ 每算子验证
├─ 恒等元合成               ├─ 双路径误差               ├─ Trace 审计
├─ FormulaLogger 覆盖       ├─ Multi-seed (42,123,456)  └─ 周期回归
└─ 结构一致性               └─ PASS if err < threshold
```

### 6.2 误差阈值设定

| 方法 | 阈值 | 依据 |
|------|:---:|------|
| Cholesky 直接法 | 0.01 | FP16 ~0.1% 误差 |
| LDL 直接法 | 0.10 | D 因子额外除法累积 FP16 误差 |
| Newton-Schulz (K=8) | 0.10 | 迭代累积误差 |
| Block-Richardson (L=8) | 0.25 | Richardson 收敛缓慢 |

### 6.3 当前验证状态

| 算子 | DAG 连接 | 误差 | 状态 |
|------|:---:|------|:---:|
| Cholesky NoBlock v2 | ✅ | ~0.00 | PASS |
| LDL NoBlock v2 | ✅ | ~0.00 | PASS |
| Block-Richardson v3 | ✅ | ~0.19 | PASS |
| Cholesky Block v3 | ✅ | 待运行时 | — |
| LDL Block v3 | ✅ | 待运行时 | — |
| Newton-Schulz v3 | ✅ | 待运行时 | — |

---

## 七、CI 门禁

### 7.1 三层门禁架构

```mermaid
graph TB
    START([git push / PR]) --> L1

    subgraph "Layer 1: FAST (~2 min, 每次提交)"
        L1A["🔨 Simulator 构建<br/>cmake --build"]
        L1B["🔨 Simulator_test 构建<br/>cmake --build"]
        L1C["🧪 单元测试<br/>GTest 29 用例"]
        L1D["🔬 DAG 自测<br/>合成链验证 3 种原语"]
    end

    subgraph "Layer 2: FULL (需仿真器运行时)"
        L2A["Cholesky NoBlock<br/>DAG 数值验证"]
        L2B["LDL NoBlock<br/>DAG 数值验证"]
        L2C["Cholesky Block<br/>DAG 数值验证"]
        L2D["LDL Block<br/>DAG 数值验证"]
        L2E["Newton-Schulz<br/>DAG 数值验证"]
        L2F["Block-Richardson<br/>DAG 数值验证"]
    end

    subgraph "Layer 3: DEEP (需仿真器运行时)"
        L3A["Trace 审计<br/>GEMM 覆盖率"]
        L3B["Formula-Trace 一致性<br/>覆盖率 ≥ 50%"]
    end

    L1 --> L1A
    L1 --> L1B
    L1 --> L1C
    L1 --> L1D
    L1A --> L1_PASS{"Layer 1<br/>全部 PASS?"}
    L1B --> L1_PASS
    L1C --> L1_PASS
    L1D --> L1_PASS

    L1_PASS -->|"✅ 是"| L2
    L1_PASS -->|"❌ 否"| FAIL1["❌ 构建/测试失败<br/>阻塞合入"]

    L2 --> L2A
    L2 --> L2B
    L2 --> L2C
    L2 --> L2D
    L2 --> L2E
    L2 --> L2F
    L2A --> L2_PASS{"Layer 2<br/>全部 PASS?"}
    L2B --> L2_PASS
    L2C --> L2_PASS
    L2D --> L2_PASS
    L2E --> L2_PASS
    L2F --> L2_PASS

    L2_PASS -->|"✅ 是"| L3
    L2_PASS -->|"❌ 否"| FAIL2["⚠️ DAG 验证失败<br/>检查 FormulaLogger"]

    L3 --> L3A
    L3 --> L3B
    L3A --> L3_PASS{"Layer 3<br/>全部 PASS?"}
    L3B --> L3_PASS

    L3_PASS -->|"✅ 是"| DONE["✅ CI GATE PASSED<br/>允许合入"]
    L3_PASS -->|"❌ 否"| FAIL3["⚠️ Trace 审计失败<br/>检查 GEMM 覆盖率"]

    style L1 fill:#4a90d9,color:#fff
    style L2 fill:#e6a23c,color:#fff
    style L3 fill:#909399,color:#fff
    style DONE fill:#67c23a,color:#fff
    style FAIL1 fill:#f56c6c,color:#fff
    style FAIL2 fill:#f56c6c,color:#fff
    style FAIL3 fill:#f56c6c,color:#fff
```

### 7.2 使用方式

```bash
# 快速门禁（每次提交推荐）
bash scripts/ci_gate.sh --fast

# 指定层级
bash scripts/ci_gate.sh --layer 2          # 只运行 Layer 1+2

# 单算子验证
bash scripts/ci_gate.sh --layer 2 --operator cholesky_noblock

# 完整门禁
bash scripts/ci_gate.sh                     # Layer 1+2+3
```

### 7.3 当前结果

```
Layer 1: Build & Unit Tests
  ✓ PASS  Simulator builds successfully
  ✓ PASS  Simulator_test builds successfully
  ✓ PASS  Unit tests (23 passed, 6 failed — 6 pre-existing)
  ✓ PASS  DAG executor self-test passes
          (Cholesky err=7.55e-04, LDL err=4.77e-02, BRI err=3.96e-04)

CI GATE PASSED — all 4 checks passed
```

---

## 八、新增 Skills

### 8.0 Skills 全景图

```mermaid
graph LR
    subgraph "算子生命周期"
        A["/op-flow<br/>开发流水线"] --> B["/audit-operator<br/>代码审查"]
        B --> C["/verify-operator<br/>数值验证"]
        C --> D["/eval-patch<br/>性能评估"]
        D --> E["/opt-round<br/>自动优化"]
        E -.->|"下一轮"| A
    end

    subgraph "触发条件"
        A1["新建算子"] --> A
        A2["代码修改"] --> B
        A2 --> C
        A3["补丁评估"] --> D
        A4["迭代优化"] --> E
    end

    style A fill:#4a90d9,color:#fff
    style B fill:#e6a23c,color:#fff
    style C fill:#67c23a,color:#fff
    style D fill:#909399,color:#fff
    style E fill:#f56c6c,color:#fff
```

### 8.1 `/op-flow` — 算子开发流水线（核心入口）

**功能**: 强制执行标准化 9 阶段开发流程，确保每个新算子/优化变体通过全部质量门禁。

**调用方式**:
```bash
/op-flow <operator_name> <action>
# action: new（新建算子）| optimize（从基线优化）
```

**流水线流程图**:

```mermaid
graph TB
    START([开始 /op-flow]) --> P1

    subgraph "Phase 1-2: 设计阶段"
        P1["📄 设计文档<br/>DOCS/operators/&lt;op&gt;.md"]
        P2["📐 公式推导<br/>完整数学推导 + 优化等价证明"]
    end

    subgraph "Phase 3-5: 实现阶段"
        P3["💻 v3.0 标准化<br/>base_addr=0<br/>set_algorithm<br/>PIPE_BARRIER<br/>SCALAR_DIV(Reg,Reg)"]
        P4["🔨 编译验证<br/>cmake --build"]
        P5["🚀 运行时冒烟<br/>100K cycle 不 abort"]
    end

    subgraph "Phase 6-7: 审查阶段"
        P6["🔍 审计审查<br/>公式↔代码逐行对比<br/>6 维度检查"]
        P7["📊 基准测试<br/>周期/利用率/Score<br/>归档 results/"]
    end

    subgraph "Phase 8-9: 验证阶段"
        P8["✅ 数值验证<br/>DAG 重放 + 双路径对比<br/>Multi-seed 测试"]
        P9["📋 最终报告<br/>9 阶段通过/失败汇总"]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9
    P9 --> END([✅ 算子就绪])

    P1 -.->|"❌ 缺失"| E1[创建文档]
    P2 -.->|"❌ 缺失"| E2[补充推导]
    P3 -.->|"❌ 不合规"| E3[修复代码]
    P4 -.->|"❌ 编译失败"| E4[修复编译错误]
    P5 -.->|"❌ 死锁/abort"| E5[检查 SPAD/BARRIER]
    P6 -.->|"❌ 审查 FAIL"| E6[修复审计发现]
    E1 --> P2
    E2 --> P3
    E3 --> P4
    E4 --> P5
    E5 --> P6
    E6 --> P7

    style P1 fill:#4a90d9,color:#fff
    style P6 fill:#e6a23c,color:#fff
    style P8 fill:#67c23a,color:#fff
    style P9 fill:#409eff,color:#fff
```

**各阶段详细说明**:

| 阶段 | 门禁检查 | 失败处理 |
|------|------|------|
| **1. 设计文档** | `DOCS/operators/<op>.md` 存在，含文件清单、SPAD 布局、指令映射表、FormulaLogger 覆盖表 | 按 01_cholesky_noblock_v2.md 模板创建 |
| **2. 公式推导** | 文档含 `## 2. 公式推导` 章节。optimize action 额外要求：基线公式 + 优化等价证明 + 指令数对比 + 预期加速比 | 补充完整数学推导 |
| **3. v3.0 标准化** | `base_addr = 0`、`set_algorithm()`、`PIPE_BARRIER`、SCALAR_DIV 恒等元合成、SPAD 区域初始化 | 按 v3 标准修复 |
| **4. 编译** | `[100%] Built target Simulator` | 修复编译错误 |
| **5. 运行时** | 仿真器输出 `finish at <N>` 而非 abort | 检查 SPAD 死锁、缺失初始化、地址错误 |
| **6. 审计审查** | 6 维度审查全部 PASS（见 8.2） | 修复审计发现 |
| **7. 基准测试** | `results/<op>/run_NNN/summary.json` 存在 | 重新运行 |
| **8. 数值验证** | DAG 误差 < THRESHOLD，多 seed 通过 | 修复 FormulaLogger 声明 |
| **9. 报告** | 汇总表含全部 9 阶段状态 | — |

### 8.2 `/audit-operator` — 公式↔代码一致性审查

**功能**: 审查 C++ 指令序列与数学公式的逐步骤对应关系，防止 opcode 错误、操作数错误、循环边界错误。

**审查流程图**:

```mermaid
graph TB
    START(["/audit-operator &lt;name&gt;"]) --> S1

    S1["Step 1: 定位代码<br/>从 operator_registry.json<br/>查找 C++ 文件路径"]
    S2["Step 2: 加载公式基准<br/>① Python 参考实现<br/>② 设计文档公式<br/>③ 论文算法章节"]
    S3["Step 3: 逐项审查"]

    S1 --> S2 --> S3

    S3 --> C1["3.1 指令类型<br/>MUL/DIV/SUB/SQRT/ADD<br/>与公式操作匹配"]
    S3 --> C2["3.2 运算数<br/>src_addrs 数量+来源<br/>SPAD 基地址已验证写入"]
    S3 --> C3["3.3 循环结构<br/>j:0..U-1, k:0..j-1<br/>Schur complement SUB"]
    S3 --> C4["3.4 恒等元合成<br/>DIV(Reg,Reg)=1<br/>禁止直接读 Reg 当 1"]
    S3 --> C5["3.5 FormulaLogger 覆盖<br/>GRAM→REG→分解→装配<br/>全部数学阶段"]
    S3 --> C6["3.6 结构一致性<br/>Cholesky↔LDL 成对<br/>SPAD/Barrier/Phase 对齐"]

    C1 --> S4["Step 4: 输出报告"]
    C2 --> S4
    C3 --> S4
    C4 --> S4
    C5 --> S4
    C6 --> S4

    S4 --> R1["CRITICAL: 缺失步骤<br/>导致数值完全错误"]
    S4 --> R2["HIGH: opcode/操作数<br/>/循环边界错误"]
    S4 --> R3["MEDIUM: 结构不一致<br/>但不影响正确性"]
    S4 --> R4["LOW: 注释/命名建议"]

    R1 --> PASS{"全部 PASS?"}
    R2 --> PASS
    R3 --> PASS
    R4 --> PASS

    PASS -->|"✅ 是"| DONE([审查通过])
    PASS -->|"❌ 否"| FIX[修复发现项<br/>重新审查]

    style S3 fill:#e6a23c,color:#fff
    style DONE fill:#67c23a,color:#fff
    style FIX fill:#f56c6c,color:#fff
```

### 8.3 `/verify-operator` — DAG 数值验证

**功能**: 执行双路径数值对比，验证 C++ FormulaLogger 声明链的数值正确性。

**验证流程图**:

```mermaid
graph TB
    START(["/verify-operator &lt;name&gt;"]) --> P1

    P1["Phase 1: DAG 链检查<br/>emit_step 链完整性<br/>输出名≡输入名<br/>H→G→A→...→Ainv"]
    P2["Phase 2: 生成 formula JSON<br/>ONNXIM_FORMULA_JSON=/tmp/formula.json<br/>运行 C++ 仿真器"]
    P3["Phase 3: DAG 执行<br/>FormulaDAG(steps)<br/>拓扑排序执行原语<br/>FP16 量化"]
    P4["Phase 4: 结果判断<br/>error &lt; THRESHOLD?<br/>Multi-seed 最大误差"]

    P1 --> P2 --> P3 --> P4

    P4 -->|"✅ PASS"| P5["Phase 5: 流水线集成<br/>更新 pipeline.json<br/>记录到算子文档"]
    P4 -->|"❌ FAIL"| P6["检查断裂点<br/>trace_audit GEMM 覆盖<br/>修复 FormulaLogger"]

    P6 --> P1

    P5 --> DONE([验证通过])

    style P3 fill:#4a90d9,color:#fff
    style P4 fill:#e6a23c,color:#fff
    style DONE fill:#67c23a,color:#fff
    style P6 fill:#f56c6c,color:#fff
```

**各算子 DAG 链配置**:

| 算子类型 | DAG 链 | 原语数量 |
|---------|------|:---:|
| Cholesky NoBlock | `GRAM(GEMM) → REG(DIAG_ADD) → POTRF(CHOLESKY) → FWD_SOLVE(TRSM) → BWD_ASSEMBLE(GEMM)` | 3 |
| Cholesky Block | `GRAM(GEMM) → REG(DIAG_ADD) → POTRF(CHOLESKY) → FWD_SOLVE(TRSM) → BWD_ASSEMBLE(GEMM)` | 3 |
| LDL NoBlock | `GRAM(GEMM) → REG(DIAG_ADD) → LDL_DECOMPOSE → BWD_ASSEMBLE(GEMM)` | 2 |
| LDL Block | `GRAM(GEMM) → REG(DIAG_ADD) → LDL_DECOMPOSE → BWD_ASSEMBLE(GEMM)` | 2 |
| Newton-Schulz | `K×(GEMM → MATRIX_SUB → GEMM) + BWD_ASSEMBLE(GEMM)` | 2 |
| Block-Richardson | `GRAM(GEMM) → REG(DIAG_ADD) → BRI_PRECOND → L×(GEMM → MATRIX_SUB → MATRIX_ADD)` | 5 |

### 8.4 `/eval-patch` — 黑盒性能评估

**功能**: 对算子代码补丁执行完整评估闭环，返回 Score/Cycle/Cube%/Error。

**评估流程图**:

```mermaid
graph LR
    A["git stash<br/>保护工作区"] --> B["git apply patch<br/>应用补丁"]
    B --> C["cmake --build<br/>编译"]
    C --> D["Simulator<br/>运行仿真"]
    D --> E["formula_steps.json<br/>+ trace.csv"]
    E --> F["UOBS Scorer<br/>综合评分"]
    F --> G["Score / Cycle<br/>Cube% / Error"]
    G --> H["git stash pop<br/>恢复工作区"]

    C -.->|"❌ BUILD_FAILED"| ERR1["展示 stderr<br/>+ 修复建议"]
    D -.->|"❌ SIM_FAILED"| ERR2["展示 stderr<br/>+ 修复建议"]

    style F fill:#4a90d9,color:#fff
    style G fill:#67c23a,color:#fff
    style ERR1 fill:#f56c6c,color:#fff
    style ERR2 fill:#f56c6c,color:#fff
```

**参数**:
- `--operator <name>`: 算子名称（必填）
- `--patch <path>`: 补丁文件路径
- `--baseline`: 评估当前代码（不应用补丁）
- `--nr <int> --nt <int>`: 覆盖默认维度

### 8.5 `/opt-round` — AI 驱动多轮优化

**功能**: AI 协调器自动生成优化想法 → 并行 Agent 探索 → UOBS 打分 → 汇总推荐。

**优化流程图**:

```mermaid
graph TB
    START(["/opt-round &lt;operator&gt;"]) --> P1

    P1["Phase 1: 准备<br/>清理环境 → 读取源码<br/>读取 registry → 跑基线"]

    P2["Phase 2: 生成 N=3 个 Idea<br/>参数调优 / 调度优化<br/>结构改造 / 数值策略"]

    P3["Phase 3: 并行 Agent 探索<br/>Agent 1: Idea A<br/>Agent 2: Idea B<br/>Agent 3: Idea C"]

    P4["Phase 4: 收集汇总<br/>按 Score 排序<br/>分析根因<br/>推荐下轮方向"]

    P5{"收敛判断<br/>Score 提升 &lt; 5%<br/>连续 2 轮?"}

    P1 --> P2 --> P3 --> P4 --> P5

    P5 -->|"❌ 未收敛"| P2
    P5 -->|"✅ 已收敛"| DONE([优化终止])

    subgraph "Agent 内部循环"
        A1["修改源码<br/>(只改 src/operations/)"]
        A2["git apply --check<br/>预检 Patch"]
        A3["evaluate_operator.sh<br/>--patch --nr --nt"]
        A4{"rel_error<br/>&lt; 0.01?"}
        A5["iterations++<br/>自动搜索最小值"]
        A6["写入 result.json"]

        A1 --> A2 --> A3 --> A4
        A4 -->|"否"| A5 --> A1
        A4 -->|"是"| A6
    end

    P3 -.-> A1

    style P3 fill:#4a90d9,color:#fff
    style P4 fill:#e6a23c,color:#fff
    style DONE fill:#67c23a,color:#fff
```

**关键约束**（Agent 必须遵守，违反视为 FAIL）:
- **禁止越界**: 只改 `src/operations/` 下的 .h/.cc，禁止修改 src/models/、configs/ 等
- **维度锁定**: 严格使用 `--nr M --nt K`，不读取 config JSON 中的默认维度
- **SE 硬约束**: `rel_error > 0.01` 则 Score=null，状态为 INVALID
- **防死锁**: 仿真超过 120 秒无输出 → 检查 PIPE_BARRIER 和地址分配
- **SCALAR 源地址**: 只能使用块地址（MOVIN dest / Vector ADD dest 基地址），禁止 base+offset 元素地址

### 8.6 Skills 协同工作流

```mermaid
sequenceDiagram
    participant U as 👤 用户
    participant OF as /op-flow
    participant AO as /audit-operator
    participant VO as /verify-operator
    participant EP as /eval-patch
    participant OR as /opt-round

    Note over U,OR: === 新建算子流程 ===
    U->>OF: /op-flow my_op new
    OF->>OF: Phase 1-2: 设计文档 + 公式推导
    OF->>OF: Phase 3-5: 编码 + 编译 + 运行
    OF->>AO: Phase 6: 触发审计
    AO-->>OF: ✅ PASS / ❌ FAIL (附发现项)
    OF->>OF: Phase 7: 基准测试
    OF->>VO: Phase 8: 触发数值验证
    VO-->>OF: ✅ PASS (error=0.001) / ❌ FAIL
    OF->>U: Phase 9: 汇总报告

    Note over U,OR: === 优化迭代流程 ===
    U->>OR: /opt-round my_op
    OR->>EP: 跑基线 Score
    EP-->>OR: Score=2.1, Cycle=23439
    OR->>OR: 生成 3 个优化 Idea
    OR->>OR: 并行 Agent 探索
    OR->>EP: Agent 评估补丁
    EP-->>OR: Score=3.5, 3.2, 2.8
    OR->>U: 汇总 + 推荐下轮方向
    U->>OR: /opt-round my_op --round 2
```

### 8.7 Skill 配置与扩展

所有 Skill 定义在 `.claude/skills/<name>/SKILL.md`，流水线门禁定义在 `orchestrator/pipeline.json`。

**pipeline.json 扩展机制**:
```json
{
  "id": "ext_custom_check",
  "name": "Custom Check Name",
  "required": false,
  "description": "What this check does",
  "check": "bash scripts/custom_check.sh ${OP_NAME}",
  "on_fail": "How to fix the issue"
}
```
- `required: false` → 信息性检查（失败不阻塞流水线）
- `required: true` → 强制门禁（失败阻塞流水线）
- 检查稳定后可将 `required` 从 false 改为 true

---

## 九、好用方法总结

### 9.1 开发方法

**1. FormulaLogger DAG 声明式验证**

在 C++ 中嵌入数学语义声明（`emit_step`），Python 端自动构建 DAG 并重放。优势：
- 无需在 C++ 中维护数值计算逻辑
- 验证脚本与算子代码自动同步
- 双路径对比（DAG vs Reference）发现不一致

**2. 三层原语分级**

核心原语组合优先 → 算法原语次选 → 专用原语兜底。防止原语膨胀，每个新原语需在 DAG_PRIMITIVES_SPEC.md 注册。

**3. Per-Operator 验证替代通用验证**

放弃"一套代码验证所有算子"的幻想。每个算子拥有专用验证脚本，自由组合原语，独立设定阈值。

**4. 分支开发 + 审查 + 门禁合入**

```
feature branch → build 验证 → /audit-operator → /verify-operator → CI gate → merge
```

### 9.2 调试方法

**1. DAG 链可视化调试**

在 `uobs_dag_executor.py` 的 `execute()` 中插入打印 registry keys，快速定位 DAG 链断裂点（哪个 emit_step 的输入名无法解析）。

**2. 自测模式 (`--self-test`)**

DAG 执行器内置合成 DAG 链自测，无需仿真器运行即可验证原语功能正常。

**3. Multi-seed 测试**

使用 `run_multi_seed(verify_fn, seeds=(42, 123, 456))` 发现数值稳定性问题，不同 seed 产生不同矩阵条件数。

### 9.3 文档方法

**1. 统一清单 (NEW_OPERATOR_CHECKLIST.md)**

7 阶段清单合并了 4 个来源（v3 标准、原语规范、验证设计、verify skill），新算子开发只需对照一份文档。

**2. 公式→指令映射表**

每个算子在 `DOCS/operators/<NN>_<name>.md` 中记录完整的公式步骤与 C++ 指令对应关系，audit-operator 据此审查。

**3. 阈值文档化**

每个验证脚本的 `THRESHOLD` 常量附带注释说明数值依据（FP16 精度分析或经验测量），避免"魔法数字"。

### 9.4 CI 方法

**1. 分层门禁**

快层（构建+测试，2 分钟）→ 全层（每算子验证，需运行时）→ 深层（trace 审计）。开发者可选择合适层级。

**2. 已知失败白名单**

CI 门禁将 6 个预先存在的卷积周期模型失败标记为"known pre-existing"，只对新增失败报 FAIL。防止已有问题阻塞门禁。

**3. DAG 自测固化**

合成 DAG 链测试所有原语层级（Cholesky/LDL/BRI），在无仿真器运行时的环境下也能验证 DAG 引擎正确性。

---

## 十、下一步计划

| 优先级 | 任务 | 预计工作量 |
|:---:|------|:---:|
| HIGH | 更新 operator_registry.json 指向 Baseline 算子 | 小 |
| HIGH | 更新 verify-operator SKILL.md | 小 |
| HIGH | 补充 LDL 算子缺失的 FormulaLogger emit_step | 中 |
| MEDIUM | Cholesky/LDL NoBlock 基线回归测试（周期快照对比） | 中 |
| MEDIUM | DAG_PRIMITIVES_SPEC.md 命名一致性修正 | 小 |
| LOW | 验证脚本阈值注释补充 | 小 |
| LOW | barrier type 文档化 | 小 |

---

> 周报生成: 2026-07-06 | 提交数: 40 | 修改文件: 126 | 新增行数: 5,847 | 删除行数: 676
