# Asim: Multi-Core NPU Cycle-Level Simulator

Asim 是一个多核 NPU 周期级仿真器（forked from ONNXim），显式建模 Core（Cube/Vector/Scalar/MTE 流水线）、片上存储（SRAM/SPAD/ACCUM）、NoC 互连（Simple/Booksim2）、以及片外 DRAM（Simple/Ramulator1/Ramulator2 HBM2），输出周期计数、利用率指标和指令级 Trace。

**核心应用场景**：在类昇腾 910B 脉动阵列架构上，利用 LLM 驱动的自动化探索，进行 MIMO 通信系统中矩阵求逆算子的指令级微架构优化与数值正确性验证。

---

## 快速开始

```bash
# 环境准备
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3
source .venv/bin/activate
pip install cmake conan==1.66.0 numpy scipy matplotlib pandas

# 编译
mkdir -p build && cd build
CMAKE_POLICY_VERSION_MINIMUM=3.5 conan install .. --build=missing
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake .. -DCMAKE_BUILD_TYPE=Release
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build . --target Simulator -j$(nproc)

# 运行
export ONNXIM_FORMULA_JSON=/tmp/formula.json
./build/bin/Simulator --config configs/ascend_910b_quiet.json \
  --models_list example/cholesky_noblock_v2_test.json \
  --mode cholesky_noblock_v2_test --log_level info

# 测试
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build . --target Simulator_test -j$(nproc)
./build/bin/Simulator_test

# CI 门禁
bash scripts/ci_gate.sh --fast
```

---

## 仿真器架构

### 三层执行模型

```
Model (张量图)  →  Operation (Tile 生成)  →  Instruction 序列
       ↓
Scheduler (Tile 分发到多核)
       ↓
Core::cycle()  →  LD / EX / ST 指令队列
       ↓
SystolicWS::cycle()  →  Cube Pipeline  +  Vector Pipeline  +  Scalar Pipeline
       ↓
TraceLogger  →  trace.csv          FormulaLogger  →  formula_steps.json
```

### 三类流水线

| 流水线 | 指令 | 延迟模型 | 用途 |
|--------|------|---------|------|
| **Cube** | `GEMM_PRELOAD`, `GEMM` | `1+(M+N-2)+max(Bm×Bn×Bk,1)` @ 16³ 基本块 | 矩阵乘法 |
| **Vector** | `ADD`, `MUL`, `MAC`, `DIV`, `SQRT`, `EXP`, `GELU`, `ADDTREE`, `COMP`, `PIPE_BARRIER` | `vec_op_iter × latency` | 向量/SIMD、同步屏障 |
| **Scalar** | `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`, `SCALAR_SQRT` | 固定延迟 (1-4 cycles) | 逐元素标量操作 |

**Scalar 流水线关键约束**：单发射，所有指令 `compute_size = 1`，使用 SPAD 基地址（非元素地址）。Scalar 是纯周期模型——公式语义通过 `FormulaLogger` 记录，数值正确性通过 Python 参考验证。

### SPAD 地址管理

- `SPAD_BASE = 0x10000000`（通用 SPAD），`ACCUM_SPAD_BASE = 0x20000000`（累加器 SPAD，用于 GEMM 输出）
- 区域按元素大小 × 矩阵维度连续分配，首次使用前必须通过 `ADD(dest, {aReg, aReg})` 初始化
- MOVIN `base_addr = 0`（避免 double-add），MOVOUT `src_from_accum = true, last_inst = true`

### 硬件参数（类昇腾 910B）

| 参数 | 值 |
|------|-----|
| 核心数 | 24 |
| Cube 基本块 | 16×16×16 |
| Vector 宽度 | 2048-bit (128 FP16/拍) |
| Scalar 流水线 | 单发射 |
| SPAD/ACCUM | 各 256KB/核 |
| DRAM | Ramulator2 HBM2, 32 通道 |
| 精度 | FP16 |

---

## 矩阵求逆算子

7 个算子（6 个 Baseline + 1 个优化变体），统一在 `src/inverse/` 下，旧版已移入 `legacy_operators/`：

| 算子 | 算法名 | 周期 (U=16) | 方法类型 | 求逆方法 |
|------|------|------|------|------|
| Cholesky NoBlock v2 | `cholesky_noblock_v2` | 23,439 | 直接法基线 | A=LL^H → Y=L⁻¹ → A⁻¹=Y^HY |
| Cholesky NoBlock Merge | `cholesky_noblock_merge` | 9,999 | SCALAR 合并优化 | 同上（合并 per-column SCALAR） |
| LDL NoBlock v2 | `ldl_noblock_v2` | 25,628 | 直接法基线 | A=LDL^H → Y=√D⁻¹L⁻¹ → A⁻¹=Y^HY |
| Cholesky Block v3 | `cholesky_block_v3` | 6,440 (B=2) | 分块直接法 | Block Cholesky + Block TRSM |
| LDL Block v3 | `ldl_block_v3` | 4,959 (B=2) | 分块直接法 | Block LDL + Block Forward Solve |
| Newton-Schulz v3 | `newton_schulz_v3` | 2,591 (N=32,K=8) | 迭代法 | X_{k+1}=X_k(2I-AX_k) |
| Block-Richardson v3 | `block_richardson_v3` | 1,993 (B=2,L=8) | 迭代预处理 | Y_{l+1}=Y_l(2I-BY_l) |

**版本策略**：v1（已废弃，`src/operations/`）→ v2（无分块基线，逐元素 SCALAR）→ v3（分块/迭代优化）。`*BaselineOp` 为纯净基线，`*MergeOp` 为 SCALAR 合并优化变体。

---

## DAG 数值验证体系

### 原理

C++ 端通过 `FormulaLogger::emit_step()` 在 `initialize_instructions()` 中声明每个数学步骤（操作类型、输入名、输出名），输出 `formula_steps.json`。Python 端 `FormulaDAG` 读取 JSON，按声明顺序拓扑执行，每个步骤调用对应的原语函数（FP16 量化），与独立 Python 参考实现做双路径误差对比。

```
Path A (DAG from C++)              Path B (Python Reference)
─────────────────────              ────────────────────────
formula_steps.json                 Python primitives
       │                                  │
FormulaDAG.execute(H)              prim_cholesky(prim_diag_add(
       │                            prim_gemm(H^H, H), 0.1))
   Ainv_dag                               │
       │                              Ainv_ref
       └──── error = ||A_dag - A_ref|| / ||A_ref|| ────┘
                  PASS if error < THRESHOLD
```

### DAG 原语（12 个，三级体系）

| 层级 | 原语 | 数量 |
|------|------|:---:|
| **Core** | GEMM, DIAG_ADD, TRSM, MATRIX_SUB, MATRIX_ADD, SCALE | 6 |
| **Algorithm** | CHOLESKY, LDL_DECOMPOSE, DIAG_INV | 3 |
| **Operator-Specific** | BRI_PRECOND, MATRIX_INV_2x2, SQRT_SCALE | 3 |

新增算子优先用 Core 原语组合，次选 Algorithm 原语，末选 Operator-Specific 原语。

### 验证状态

| 算子 | DAG 连接 | 误差 | 状态 |
|------|:---:|------|:---:|
| Cholesky NoBlock v2 | ✅ | ~0.00 | PASS |
| LDL NoBlock v2 | ✅ | ~0.00 | PASS |
| Block-Richardson v3 | ✅ | ~0.19 | PASS |
| Cholesky Block v3 | ✅ | 待运行时验证 | — |
| LDL Block v3 | ✅ | 待运行时验证 | — |
| Newton-Schulz v3 | ✅ | 待运行时验证 | — |

---

## 项目结构

```
Asim/
├── src/
│   ├── main.cc                    # 入口：mode 分发 + Simulator 启动
│   ├── Simulator.{h,cc}           # 全局事件循环（Core/ICNT/DRAM 三域推进）
│   ├── Core.{h,cc}                # 核心基类：指令队列 + SPAD/ACCUM 状态
│   ├── SystolicWS.{h,cc}          # 昇腾核实现：Cube/Vector/Scalar 周期模型
│   ├── Dram.{h,cc}                # DRAM 接口（Simple / Ramulator2）
│   ├── Interconnect.{h,cc}        # NoC 模型（SimpleInterconnect / BookSim2）
│   ├── Common.{h,cc}              # 核心类型：Opcode 枚举、Instruction 结构体
│   ├── FormulaLogger.{h,cc}       # 数学语义嵌入式声明 → formula_steps.json
│   ├── TraceLogger.{h,cc}         # 指令级 Trace CSV 输出
│   ├── SimulationConfig.h         # 硬件参数配置
│   ├── inverse/                   # ★ 6 种矩阵求逆算法（按算法分目录）
│   │   ├── cholesky_noblock/      # Cholesky 无分块基线 + Merge 优化
│   │   ├── cholesky_block/        # Cholesky 分块基线
│   │   ├── ldl_noblock/           # LDL 无分块基线
│   │   ├── ldl_block/             # LDL 分块基线
│   │   ├── newton_schulz/         # Newton-Schulz 迭代
│   │   └── block_richardson/      # Block-Richardson 迭代预处理
│   ├── operations/                # 其他算子（ONNX NN + 非求逆自定义算子）
│   ├── models/                    # 模型定义
│   ├── allocator/                 # SPAD 地址分配器
│   ├── scheduler/                 # Tile 调度器
│   └── helper/                    # 命令行解析等工具
├── configs/                       # JSON 硬件配置
│   ├── ascend_910b.json           # 类昇腾 910B（24 核, Cube 16³, HBM2）
│   └── ascend_910b_quiet.json     # 同上，关闭周期性日志
├── example/                       # JSON 模型配置（每个算子对应测试文件）
├── results/                       # Benchmark 结果（标准化目录结构）
│   └── <algorithm>/run_NNN/       # 每次运行独立目录
├── scripts/
│   ├── ci_gate.sh                 # ★ CI 门禁（3 层自动验证）
│   ├── uobs_dag_executor.py       # ★ DAG 执行器（formula JSON → 原语重放）
│   ├── uobs_scorer.py             # UOBS 综合评分
│   ├── trace_audit.py             # Trace GEMM 覆盖率审计
│   └── verify/                    # ★ 每算子专用验证脚本（6 个）
├── DOCS/
│   ├── OPERATOR_DEVELOPMENT_STANDARD_V3.md  # v3 编码标准（含验证要求）
│   ├── NEW_OPERATOR_CHECKLIST.md            # 统一算子开发清单（7 阶段）
│   ├── DAG_PRIMITIVES_SPEC.md               # DAG 原语规范（三级体系）
│   ├── ASIM_VERIFICATION_REPORT.md          # 验证报告
│   ├── operators/                           # 每算子公式文档（6 个）
│   └── specs/                               # 方案设计文档
├── orchestrator/
│   ├── pipeline.json              # /op-flow 流水线门禁定义（9 阶段）
│   └── operator_registry.json     # 算子注册表
├── .claude/skills/                # Claude Code 技能（5 个）
│   ├── op-flow/                   # 算子开发流水线
│   ├── audit-operator/            # 公式↔代码一致性审查
│   ├── verify-operator/           # DAG 数值验证
│   ├── eval-patch/                # 黑盒性能评估
│   └── opt-round/                 # AI 驱动多轮优化
├── tests/                         # GTest 单元测试
├── legacy_operators/              # 旧版算子（已废弃，仅作参考）
└── extern/                        # 外部依赖 (booksim/protobuf/onnx/ramulator2)
```

---

## 开发流程

### /op-flow 流水线（9 阶段）

```
设计文档 → 公式推导 → v3.0 编码 → 编译 → 冒烟测试 → 审计审查 → 基准测试 → 数值验证 → 报告
```

| 阶段 | 门禁 | 说明 |
|------|------|------|
| 1. 设计文档 | `DOCS/operators/<op>.md` 存在 | 文件清单、SPAD 布局、指令映射、FormulaLogger 覆盖 |
| 2. 公式推导 | 完整数学推导章节 | optimize 额外要求：基线公式 + 等价证明 + 加速比分析 |
| 3. v3.0 编码 | `base_addr=0`, `set_algorithm()`, `PIPE_BARRIER` | SCALAR_DIV 恒等元合成、SPAD 区域初始化 |
| 4. 编译 | `[100%] Built target Simulator` | cmake --build 零错误零警告 |
| 5. 冒烟测试 | `finish at <N>`（非 abort/死锁） | 100K cycle 限制 |
| 6. 审计审查 | 6 维度全部 PASS | 指令类型/运算数/循环/恒等元/FormulaLogger/结构一致性 |
| 7. 基准测试 | `results/<op>/summary.json` 归档 | 周期/利用率/Score |
| 8. 数值验证 | DAG 误差 < THRESHOLD, multi-seed | 3 seeds (42/123/456)，取最大误差 |
| 9. 报告 | 汇总表 | 9 阶段通过/失败状态 |

### CI 门禁

```bash
bash scripts/ci_gate.sh --fast           # Layer 1: 构建+测试 (~2 min)
bash scripts/ci_gate.sh --layer 2        # Layer 2: + 每算子 DAG 验证
bash scripts/ci_gate.sh --operator X     # 单算子模式
```

| 层级 | 内容 | 状态 |
|:---:|------|:---:|
| **Layer 1** | Simulator 构建 + Test 构建 + 单元测试 (23/29) + DAG 自测 | ✅ 4/4 PASS |
| **Layer 2** | Layer 1 + 每算子 DAG 数值验证 | 需仿真器运行时 |
| **Layer 3** | Layer 2 + Trace 审计 + Formula-Trace 一致性 | 需仿真器运行时 |

### Skills

| Skill | 功能 |
|------|------|
| `/op-flow <name> new\|optimize` | 9 阶段算子开发流水线（自动调用 audit/verify） |
| `/audit-operator <name>` | C++ 指令 ↔ 公式一致性审查（6 维度，CRITICAL→LOW 分级） |
| `/verify-operator <name>` | DAG 数值验证（双路径对比 + Multi-seed） |
| `/eval-patch --operator <name> --patch <path>` | 黑盒评估（Score/Cycle/Cube%/Error） |
| `/opt-round <name>` | AI 驱动多轮优化（生成 Idea → 并行 Agent → 打分汇总） |

---

## 仿真链路关键修复 (2026-07)

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | SCALAR_SUB 缺失 | 硬件模型不完整 | 新增 opcode |
| 2 | Cholesky Block DAG 链断裂 | 缺失 FWD_SOLVE emit_step | 添加 TRSM L→Y |
| 3 | LDL Block DAG 链破坏 | 逐块 DUPDATE/LUPDATE 覆盖 Y | 替换为单次 LDL_DECOMPOSE |
| 4 | Newton-Schulz DAG 不完整 | 缺失 BWD_ASSEMBLE + 2I 未注册 | 添加 GEMM + 2I 特判 |
| 5 | BRI 硬编码 Y_7 | 仅 L=8 有效 | 改为动态 Y_{L-1} |
| 6 | Cholesky Block TRSM 引用断裂 | 输出名 L_j 未注册 | 统一 L 输出 |
| 7 | Multi-seed lambda bug | s vs seed 变量名错误 | 全部修复为 seed |
| 8 | GTest CMake 兼容性 | GTest 1.8.1 硬编码 -Werror | PATCH_COMMAND 修复 |
| 9 | CI 门禁 mode 名错误 | _baseline vs _v2_test 不匹配 | 统一 main.cc 实际 mode |

---

## 代码规范

- C++20 (`CMAKE_CXX_STANDARD 20`)，`_GLIBCXX_USE_CXX11_ABI=0`（legacy ABI）
- ASan 在 Debug 构建中启用 (`-fsanitize=address`)
- 算子属性通过 `std::map<std::string, std::string>` 传递，在 `parse_attributes()` 中解析
- Tiles 使用 `std::unique_ptr`，容器为 `std::deque<std::unique_ptr<Tile>>`
- `spdlog` 日志（trace/debug/info），`nlohmann/json` JSON 处理，`robin_hood` 哈希表
- **必须**调用 `FormulaLogger::set_algorithm()` + `FormulaLogger::emit_step()` 覆盖全部数学阶段
- **必须**通过 `/audit-operator` 审查和 `/verify-operator` 数值验证后方可合入

---

## 本周统计

| 指标 | 数值 |
|------|------|
| 提交数 | 40 |
| 修改文件 | 126（52 新增 + 57 修改 + 8 删除 + 92 重命名） |
| 新增代码 | 5,847 行 |
| 新增文档 | 9 个 (DOCS/) |
| 新增脚本 | 9 个 (scripts/) |
| 新增 Skills | 2 个 (op-flow, verify-operator) |

---

## License

Asim is forked from ONNXim. See original project for license details.
