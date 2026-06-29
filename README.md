# Asim: Multi-Core NPU Cycle-Level Simulator

Asim 是一个多核 NPU 周期级仿真器（forked from ONNXim），显式建模 Core（Cube/Vector/Scalar/MTE 流水线）、SRAM/SPAD/ACCUM 片上存储、NoC 片上互连（Simple/Booksim2）、以及 DRAM 片外存储（Simple/Ramulator1/Ramulator2 HBM2），输出周期计数、利用率指标和指令级 Trace。

本项目的核心应用场景是：**在类昇腾 910B 的脉动阵列架构上，利用 LLM 驱动的大规模自动探索，对 MIMO 通信系统中的矩阵求逆算子进行指令级微架构优化。**

## 快速开始

### 环境准备

```bash
# Python 虚拟环境
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3
source .venv/bin/activate
pip install cmake conan==1.66.0 numpy scipy matplotlib pandas

# C++ 依赖（Conan）
mkdir -p build && cd build
CMAKE_POLICY_VERSION_MINIMUM=3.5 conan install .. --build=missing

# 编译
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake .. -DCMAKE_BUILD_TYPE=Release
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build . --target Simulator -j$(nproc)
```

### 运行单个算子

```bash
# 设置环境变量
export ONNXIM_TRACE_CSV=results/<algo>/run_001/trace.csv
export ONNXIM_FORMULA_JSON=results/<algo>/run_001/formula_steps.json
export ONNXIM_MAX_CORE_CYCLES=200000

# 运行（以 Cholesky-NoBlock 为例）
./build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/cholesky_noblock_test.json \
  --mode cholesky_noblock_test \
  --log_level info
```

### 跑 Benchmark Suite

```bash
bash scripts/run_benchmark_suite.sh <algorithm> <mode> <config_json> [models_list_json]
```

## 项目结构

```
Asim/
├── src/
│   ├── main.cc                  # 入口：mode 分发 + Simulator 启动
│   ├── Simulator.{h,cc}         # 全局事件循环（Core/ICNT/DRAM 三域推进）
│   ├── Core.{h,cc}              # 核心基类：指令队列 + SPAD/ACCUM 状态
│   ├── SystolicWS.{h,cc}        # 昇腾核实现：Cube/Vector/Scalar 周期模型
│   ├── Dram.{h,cc}              # DRAM 接口（Simple / Ramulator2）
│   ├── Interconnect.{h,cc}      # NoC 模型（SimpleInterconnect / Booksim2）
│   ├── Common.{h,cc}            # 核心类型：Opcode 枚举、Instruction 结构体
│   ├── FormulaLogger.{h,cc}     # 数学语义嵌入式声明 → UOBS 自动评分
│   ├── TraceLogger.{h,cc}       # 指令级 Trace CSV 输出
│   ├── SimulationConfig.h       # 硬件参数配置
│   ├── inverse/                 # ★ 6 种矩阵求逆算法（按算法归档）
│   │   ├── cholesky_block/      # Cholesky-Block (基线)
│   │   ├── cholesky_noblock/    # Cholesky-NoBlock
│   │   ├── ldl_block/           # LDL-Block (无 SQRT)
│   │   ├── ldl_noblock/         # LDL-NoBlock (basic + aligned 两个变体)
│   │   ├── newton_schulz/       # Newton-Schulz (baseline + optimized 两个版本)
│   │   └── block_richardson/    # Block-Richardson (BRI, 周期最优)
│   ├── operations/              # 其他算子（ONNX NN 算子 + 非求逆自定义算子）
│   ├── models/                  # 其他模型
│   ├── allocator/               # SPAD 地址分配器
│   ├── scheduler/               # Tile 调度器
│   └── helper/                  # 命令行解析等工具
├── configs/                     # JSON 硬件配置
│   ├── ascend_910b.json         # 类昇腾 910B（24 核, Cube 16³, HBM2）
│   └── ascend_910b_quiet.json   # 同上，关闭周期性日志（基准测试用）
├── example/                     # JSON 模型配置（每个算子的矩阵维度+参数）
├── results/                     # ★ Benchmark 结果（标准化目录结构）
│   └── <algorithm>/run_NNN/     # 每次运行独立目录，含 timestamp/version/trace
├── scripts/                     # 评测、可视化、benchmark 脚本
├── orchestrator/                # LLM 优化框架注册表
├── .claude/skills/              # Claude Code 技能 (eval-patch/op-dev/opt-round)
├── extern/                      # 外部依赖 (booksim/protobuf/onnx/ramulator2)
├── tests/                       # GTest 单元测试
└── DOCS/                        # 论文源文件 (Ascend.tex) + 图片
```

## 矩阵求逆算法一览

| 算法 | 目录 | 核心特征 | Cycle (U=16) |
|------|------|----------|-------------|
| Cholesky-Block | `inverse/cholesky_block/` | POTRF/TRSM/RK, SQRT 瓶颈, B=2 | 3,326 |
| Cholesky-NoBlock | `inverse/cholesky_noblock/` | 逐列分解, B=1, 无 Cube | 6,142 |
| LDL-Block | `inverse/ldl_block/` | 无 SQRT + $2\times2\rightarrow16\times16$ 拼接 | 1,682 |
| LDL-NoBlock | `inverse/ldl_noblock/` | 逐列 LDL (basic+aligned 双变体) | 3,673 |
| Newton-Schulz | `inverse/newton_schulz/` | 二次收敛迭代 (baseline+opt 双版本) | 38,806 |
| Block-Richardson | `inverse/block_richardson/` | 块对角预条件 + Chebyshev 加速 (周期最优) | 1,979 |

## 三层执行模型

```
Model (张量图) → Operation (Tile 生成) → Instruction 序列
    ↓
Scheduler (Tile 分发到 24 核) 
    ↓
Core::cycle() → LD/EX/ST 指令队列
    ↓
SystolicWS::cycle() → Cube / Vector / Scalar 流水线 + ICNT + DRAM
    ↓
TraceLogger → trace.csv  +  FormulaLogger → formula_steps.json
```

## 硬件参数（类昇腾 910B）

| 参数 | 值 |
|------|-----|
| 核心数 | 24 |
| Cube 基本块 | 16×16×16 |
| Vector 宽度 | 2048-bit (128 FP16/拍) |
| Scalar 流水线 | 单发射 |
| SPAD/ACCUM | 各 256KB/核 |
| DRAM | Ramulator2 HBM2, 32 通道 |
| 精度 | FP16 |

## 代码规范

- C++20 (`CMAKE_CXX_STANDARD 20`)
- `_GLIBCXX_USE_CXX11_ABI=0` (legacy ABI)
- ASan 在 Debug 构建中启用
- 算子属性通过 `std::map<std::string, std::string>` 传递，在 `parse_attributes()` 中解析
- `spdlog` 用于日志，级别: trace/debug/info
- 必须调用 `FormulaLogger::set_algorithm()` + `FormulaLogger::emit_step()` 接入 UOBS 评分

## License

Asim is forked from ONNXim. See original project for license details.
