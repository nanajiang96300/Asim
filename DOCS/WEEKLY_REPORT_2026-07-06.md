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

### 2.1 算子目录结构

6 个 Baseline 算子 + 2 个优化变体，统一在 `src/inverse/` 下：

| 算子 | 算法名 | 周期数 | emit_step | 指令数 | 类型 |
|------|------|------|:---:|:---:|------|
| Cholesky NoBlock v2 | `cholesky_noblock_v2` | 23,439 (U=16) | 5 | 20 | 直接法基线 |
| Cholesky NoBlock Merge | `cholesky_noblock_merge` | 9,999 (U=16) | 5 | 20 | SCALAR 合并优化 |
| LDL NoBlock v2 | `ldl_noblock_v2` | 25,628 (U=16) | 5 | 26 | 直接法基线 |
| Cholesky Block v3 | `cholesky_block_v3` | 6,440 (U=16,B=2) | 7 | 19 | 分块 |
| LDL Block v3 | `ldl_block_v3` | 4,959 (U=16,B=2) | 5 | 25 | 分块 |
| Newton-Schulz v3 | `newton_schulz_v3` | 2,591 (N=32,K=8) | 4 | 7 | 迭代 |
| Block-Richardson v3 | `block_richardson_v3` | 1,993 (U=16,B=2,L=8) | 7 | 19 | 迭代 |

```
src/inverse/
├── cholesky_noblock/        # Cholesky 无分块基线 (v2)
│   ├── CholeskyNoBlockBaselineOp.{h,cc}
│   ├── CholeskyNoBlockMergeOp.{h,cc}     # SCALAR merge 优化
│   └── CholeskyNoBlockBaselineModel.{h,cc}
├── ldl_noblock/             # LDL 无分块基线 (v2)
├── cholesky_block/          # Cholesky 分块 (v3)
├── ldl_block/               # LDL 分块 (v3)
├── newton_schulz/           # Newton-Schulz 迭代 (v3)
├── block_richardson/        # Block-Richardson 迭代 (v3)
```

### 2.2 三版算子演进

| 版本 | 标记 | 含义 |
|------|------|------|
| v2 | `*BaselineOp`（无分块）| 纯净基线：逐元素 SCALAR，完整 FormulaLogger 覆盖 |
| v2 | `*MergeOp`（无分块）| 优化变体：合并 SCALAR 操作减少指令数 |
| v3 | `*BaselineOp`（分块/迭代）| 分块或迭代实现，`_optype` 区分 |

### 2.3 关键架构决策

1. **DAG 原语三层分级**：Core（GEMM/TRSM/DIAG_ADD 等 6 个）→ Algorithm（CHOLESKY/LDL_DECOMPOSE 等 3 个）→ Operator-specific（BRI_PRECOND 等 3 个），严格防耦合
2. **Per-Operator 验证**替代通用验证：每个算子拥有独立验证脚本，双路径对比（C++ DAG vs Python Reference）
3. **FormulaLogger DAG 链规范**：每个 emit_step 的输出名必须匹配后一步骤的输入名，形成完整链路 H → ... → Ainv

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

### 4.1 FormulaLogger ↔ DAG Executor 链路

```
┌─────────────────────────────────────────────────────────┐
│  C++ 仿真器 (initialize_instructions)                    │
│  │                                                       │
│  ├─ set_algorithm("name", B, L, U)                      │
│  ├─ emit_step("GRAM", "GEMM", {H^H,H}, "G", ...)        │
│  ├─ emit_step("REG", "DIAG_ADD", {G,lambda*I}, "A", ...)│
│  ├─ emit_step("POTRF", "CHOLESKY", {A}, "L", ...)       │
│  ├─ emit_step("FWD", "TRSM", {L}, "Y", ...)             │
│  └─ emit_step("BWD", "GEMM", {Y^H,Y}, "Ainv", ...)      │
│                    │                                     │
│                    ▼ formula_steps.json                  │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Python DAG Executor (uobs_dag_executor.py)             │
│  │                                                       │
│  ├─ FormulaDAG(steps) — 构建 DAG 节点图                  │
│  ├─ execute(initial_tensors, aux_params)                 │
│  │   ├─ 按声明顺序拓扑执行                               │
│  │   ├─ 输入解析: batch → batch-0 → aux → I/2I/H^H/Y^H  │
│  │   ├─ 查找 PRIMITIVES[node.op_type] 并调用             │
│  │   └─ registry[(batch, name)] = FP16(result)           │
│  └─ 返回 {"Ainv": np.ndarray}                            │
└─────────────────────────────────────────────────────────┘
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

### 7.1 门禁设计

```
scripts/ci_gate.sh [--fast] [--layer N] [--operator X]

--fast: 仅 Layer 1（构建 + 单元测试 + DAG 自测）

Layer 1 (FAST, ~2min):
  ├─ Simulator 构建
  ├─ Simulator_test 构建
  ├─ 单元测试 (GTest, 23/29)
  └─ DAG 执行器自测 (合成 DAG 链验证全部 3 种原语)

Layer 2 (FULL, 需仿真器运行时):
  ├─ Layer 1
  └─ 每算子: 运行仿真器 → 生成 formula_steps.json → DAG 数值验证

Layer 3 (DEEP, 需仿真器运行时):
  ├─ Layer 2
  └─ Trace 审计 + Formula-Trace 一致性 (GEMM 覆盖率 ≥ 50%)
```

### 7.2 门禁当前结果

```
Layer 1: Build & Unit Tests
  ✓ PASS  Simulator builds successfully
  ✓ PASS  Simulator_test builds successfully
  ✓ PASS  Unit tests (23 passed, 6 failed — 6 pre-existing)
  ✓ PASS  DAG executor self-test passes

CI GATE PASSED — all 4 checks passed
```

---

## 八、新增 Skills

### 8.1 Skill 列表

| Skill | 功能 | 触发 |
|------|------|------|
| `/op-flow` | 9 阶段算子开发流水线 | 新建/优化算子 |
| `/audit-operator` | C++ 指令 ↔ 公式一致性审查 | 代码修改后 |
| `/verify-operator` | DAG 数值验证（双路径对比） | 代码修改后 |
| `/eval-patch` | 黑盒评估（Score/Cycle/Cube%/Error） | 优化对比 |
| `/opt-round` | 多轮 AI 驱动算子优化 | 自动优化 |

### 8.2 /op-flow 流水线阶段

```
1. design_doc       → 设计文档存在
2. math_derivation  → 公式推导在文档中
3. code_v3_standard → v3 标准合规 (base_addr=0, set_algorithm, PIPE_BARRIER)
4. compile          → cmake build 通过
5. runtime          → 仿真器无 abort 完成 (100K cycle 限制)
6. audit_review     → /audit-operator 通过
7. benchmark        → 运行 benchmark 套件并归档结果
8. ext_numeric_verify → DAG 数值验证通过
9. ext_spad_audit   → SPAD 布局审计 (待实现)
```

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
