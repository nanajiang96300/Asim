# Asim 仿真器架构 - Design Spec

> Human-readable design narrative. Machine-readable execution contract: `spec_lock.md`.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | Asim 仿真器架构与流水线 |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 7 |
| **Design Style** | blueprint |
| **Target Audience** | 技术团队 / 新成员 — 需要理解仿真器架构的工程师 |
| **Use Case** | 技术分享 / 内部培训 / 架构讲解 |
| **Delivery Purpose** | `presentation` — 投影演示，一页一个核心概念 |
| **Content Strategy** | 平衡 — 基于源码事实，重新组织为清晰的叙事结构 |
| **Created Date** | 2026-07-08 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 60px, top/bottom 50px |
| **Content Area** | 1160×620 |

---

## III. Visual Theme

### Theme Style

- **Mode**: instructional — 概念分解，逐步展开
- **Visual style**: blueprint — 工程图纸风格，深色背景 + 蓝图线框
- **Theme**: Dark theme
- **Tone**: 技术、精确、工程感、专业

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#0D1B2A` | 蓝图深色背景 |
| **Secondary bg** | `#1B2838` | 卡片/面板背景 |
| **Primary** | `#2196F3` | 主线框、标题强调、关键路径 |
| **Accent** | `#00BCD4` | 数据高亮、关键标注 |
| **Secondary accent** | `#4DD0E1` | 次要强调 |
| **Body text** | `#ECEFF1` | 主文本 |
| **Secondary text** | `#90A4AE` | 辅助说明、标注 |
| **Tertiary text** | `#607D8B` | 网格线、页码 |
| **Border/divider** | `#263238` | 分割线、面板边框 |
| **Success** | `#4CAF50` | 正向指标 |
| **Warning** | `#FF9800` | 注意标记 |

---

## IV. Typography System

### Font Plan

**Typography direction**: 技术风格 —— 清晰的 CJK 无衬线 + 等宽代码字体

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei", "PingFang SC"` | `Arial` | `sans-serif` |
| **Body** | `"Microsoft YaHei", "PingFang SC"` | `Arial` | `sans-serif` |
| **Code** | — | `Consolas, "Courier New"` | `monospace` |

**Per-role font stacks**:
- Title: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Body: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline (unitless px)**: Body = 32 (presentation)

| Role | Size (px) | Weight |
| ---- | --------- | ------ |
| Cover title | 72 | Bold |
| Page title | 48 | Bold |
| Subtitle | 36 | SemiBold |
| **Body** | **32** | Regular |
| Annotation | 22 | Regular |
| Footnote | 18 | Regular |
| Code | 24 | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: Title at top-left, 48px, with thin accent underline
- **Content area**: Center-aligned, flexible layout per rhythm tag
- **Footer area**: Page number bottom-right, 18px; grid-line decorations

### Spacing

| Element | Value |
| ------- | ----- |
| Safe margin | 60px |
| Content block gap | 32px |
| Icon-text gap | 12px |

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `tabler-outline` (线性图标，与 blueprint 风格匹配)
- **Stroke width**: 2px

### Recommended Icon List

| Purpose | Icon Path |
| ------- | --------- |
| 架构/系统 | `tabler-outline/building-arch` |
| CPU/Core | `tabler-outline/cpu` |
| 内存/存储 | `tabler-outline/database` |
| 网络/互联 | `tabler-outline/network` |
| 流程/流水线 | `tabler-outline/arrows-sort` |
| 层/分层 | `tabler-outline/stack-2` |
| 循环 | `tabler-outline/repeat` |
| 速度/性能 | `tabler-outline/bolt` |
| 矩阵/GEMM | `tabler-outline/grid-dots` |
| 向量 | `tabler-outline/vector` |
| 标量 | `tabler-outline/circle` |

---

## VII. Visualization Reference List

No chart templates needed — all diagrams are custom SVG architecture drawings.

---

## VIII. Image Resource List

No external images — all visuals are native SVG line drawings.

---

## IX. Content Outline

### Part 1: 仿真器架构

#### P01 - 封面

- **Cover impact**: 核心隐喻——"芯片蓝图"：一张抽象的 NPU 架构蓝图，用几何线条勾勒出 Core/NoC/DRAM 的连接关系，标题浮于其上。深色蓝图上发光的蓝色线条传达"技术深度"和"工程精确感"。
- **Layout**: 全幅蓝图背景 + 浮动标题，右下角渐变网格线
- **Title**: Asim：多核 NPU 周期级仿真器
- **Subtitle**: 架构、流水线与基础算子
- **Info**: 2026-07

#### P02 - 项目概览

- **Layout**: 左侧标题 + 右侧三列卡片（定位、场景、指标）
- **Title**: 项目概览
- **Core message**: Asim 是一个显式建模 Core/NoC/DRAM 的多核 NPU 周期级仿真器，服务于 MIMO 矩阵求逆算子优化。
- **Content**:
  - 定位：多核 NPU 周期级仿真器（forked from ONNXim）
  - 硬件建模：24 核 + 3 流水线（Cube/Vector/Scalar）+ NoC + DRAM
  - 核心场景：MIMO 通信系统矩阵求逆算子的指令级微架构优化
  - 输出：周期计数、利用率指标、指令级 Trace、数学语义验证

#### P03 - 整体架构

- **Layout**: 中心架构图（7 大组件关系）+ 四周标注说明，线框连接
- **Title**: 仿真器整体架构
- **Core message**: Simulator 作为中心调度者，统一管理 Core/DRAM/Interconnect/Scheduler/Model/FormulaLogger 六大子系统。
- **Content**:
  - Simulator：全局事件循环，三域时间推进
  - Core[0..23]：SystolicWS 核心，Cube/Vector/Scalar 三流水线
  - DRAM：HBM2 32 通道，Ramulator2 周期模型
  - Interconnect：NoC（Simple/Booksim2）
  - Scheduler：Tile 分发到多核
  - Model → Operation → Tile → Instruction
  - FormulaLogger：数学语义记录 → formula_steps.json

#### P04 - 三层执行模型

- **Layout**: 自上而下的三层流程图，每层展示关键类和转换关系
- **Title**: 三层执行模型
- **Core message**: Model → Operation → Tile → Instruction 是仿真器最核心的设计范式，将张量计算图逐步分解为原子执行单元。
- **Content**:
  - Layer 1 — Model：张量计算图，创建 Operation 节点，连接 Tensor 边
  - Layer 2 — Operation：算子实现，initialize_tiles() 生成 Tile，initialize_instructions() 生成指令序列
  - Layer 3 — Instruction：原子执行单元，携带 opcode、地址、维度信息

#### P05 - 仿真主循环

- **Layout**: 左侧阶段式流程图 + 右侧代码块注解
- **Title**: 仿真主循环：三域时间推进
- **Core message**: set_cycle_mask() 比较 Core/DRAM/ICNT 三域时间戳，最小值对应域推进，实现异构频率的精确交错仿真。
- **Content**:
  - while(running): 全局循环
  - set_cycle_mask(): 选择本拍推进的域（Core/DRAM/ICNT）
  - Core 域：handle_model → issue tile → core->cycle()
  - DRAM 域：dram->cycle()
  - ICNT 域：Core↔ICNT↔DRAM 请求/响应路由
  - 仿真结束 → 统计输出 + Trace CSV + Formula JSON

#### P06 - Core 微架构

- **Layout**: 左右分栏：左侧指令队列流程 + 右侧三流水线特性卡片
- **Title**: Core 微架构：三条并行流水线
- **Core message**: 每个 Core 内部通过 LD/EX/ST 指令队列将 Tile 指令分发到 Cube（脉动阵列）、Vector（SIMD）、Scalar（逐元素）三条物理流水线。
- **Content**:
  - 指令队列：LD Queue(MOVIN) → EX Queue(计算) → ST Queue(MOVOUT)
  - Cube Pipeline：GEMM/GEMM_PRELOAD，脉动阵列 16×16×16 基本块，多发射
  - Vector Pipeline：ADD/MUL/MAC/DIV/SQRT/EXP/...，单发射 SIMD
  - Scalar Pipeline：SCALAR_ADD/SUB/MUL/DIV/SQRT，严格单发射
  - 双缓冲 SPAD：SPAD_BASE(0x10000000) / ACCUM_SPAD_BASE(0x20000000)

#### P07 - 基础算子对比

- **Layout**: 全宽表格，三列对比 GEMM / Vector / Scalar 算子
- **Title**: 基础算子：GEMM · Vector · Scalar
- **Core message**: 仿真器提供三类基础计算算子，分别映射到 Cube/Vector/Scalar 三条物理流水线，覆盖矩阵乘法、向量 SIMD 和逐元素标量操作。
- **Content**: 对比表格：

| 特性 | GEMM (Cube) | Vector | Scalar |
|------|------------|--------|--------|
| 指令 | GEMM_PRELOAD, GEMM | ADD, MUL, MAC, DIV, SQRT, EXP, GELU, ADDTREE, COMP, PIPE_BARRIER | SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_SQRT |
| 硬件单元 | 脉动阵列 (Systolic Array) | Vector SIMD (2048-bit) | Scalar ALU |
| 并行度 | 多发射（深度=core_height） | 单发射（前一条未完成则阻塞） | 严格单发射（pipeline 只能 1 条） |
| 延迟模型 | 1+(M+N-2)+max(⌈M/16⌉⌈N/16⌉⌈K/16⌉,1) | vec_op_iter × latency | 固定 1-4 cycles |
| compute_size | tile_m×tile_k×tile_n | U×U 或向量宽度 | 1 |
| 典型用途 | 矩阵乘法、Gram 矩阵 | 逐元素运算、激活函数 | 对角线元素操作 |

---

## X. Speaker Notes Requirements

One speaker note file per page, saved to `notes/`:
- **Filename**: match SVG name (e.g., `01_封面.svg` → `notes/01_封面.md`)
- **Content**: 2-4 句自然口语，承载该页核心信息 + 页面过渡

---

## XI. Technical Constraints Reminder

### SVG Generation Must Follow:

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>` (`<foreignObject>` FORBIDDEN)
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` FORBIDDEN
5. FORBIDDEN: `mask`, `<style>`, `class`, `foreignObject`
6. FORBIDDEN: `textPath`, `animate*`, `script`
7. Raw Unicode for special chars; XML reserved chars escaped
8. `clipPath` allowed only on `<image>` elements
9. `<g opacity="...">` FORBIDDEN
10. Inline styles only; no external CSS
