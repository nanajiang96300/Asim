# 01_封面

欢迎来到 Asim 仿真器架构分享。Asim 是一个多核 NPU 周期级仿真器，今天我们来看它的架构设计、仿真流程和基础计算算子。

---

# 02_项目概览

Asim 是从 ONNXim 分支出来的多核 NPU 周期级仿真器。它显式建模了 24 核 SystolicWS 核心、三种流水线、两种 NoC 模型和三种 DRAM 模型，底层使用 Ramulator2 做 HBM2 的 32 通道周期精确仿真。它的核心应用场景是 MIMO 通信系统中的矩阵求逆算子优化，输出包括周期计数、利用率和指令级 Trace。

---

# 03_整体架构

仿真器的整体架构以 Simulator 为中心调度者，统一管理六大子系统。Core 数组是计算核心，每个 Core 内部有三条流水线。DRAM 通过 Ramulator2 建模 HBM2，Interconnect 负责 Core 和 DRAM 之间的数据路由。Scheduler 负责将 Tile 分发到多核，底层的 Model 到 Operation 到 Tile 到 Instruction 的三层模型是整个仿真器的核心设计范式。FormulaLogger 和 TraceLogger 分别记录数学语义和指令 Trace。

---

# 04_三层执行模型

三层执行模型是理解整个仿真器的关键。最上层 Model 描述张量计算图，创建 Operation 节点并通过 Tensor 边连接。中间层 Operation 是算子实现，它的 initialize_tiles 方法按 batch 拆分生成 Tile，每个 Tile 再调用 initialize_instructions 生成指令序列。最底层 Instruction 是原子执行单元，携带 opcode、地址、维度和计算量信息。从 MOVIN 搬入数据，经过 GEMM 和向量标量计算，到 MOVOUT 搬出结果，一整条指令链就这样形成了。

---

# 05_仿真主循环

仿真主循环的核心机制是三域时间推进。Core、DRAM 和 ICNT 各自有不同的时钟频率，set_cycle_mask 比较三域的时间戳，选择时间最小的域推进。Core 域负责模型调度、Tile 分发和核心计算，DRAM 域推进 Ramulator2 的周期模型，ICNT 域处理 Core 和 DRAM 之间的请求响应路由。三域频率不同自然形成交错执行，仿真结束后输出利用率统计、Trace CSV 和 Formula JSON。

---

# 06_Core微架构

每个 Core 内部通过 LD、EX、ST 三条指令队列将 Tile 指令分发到物理流水线。EX 队列根据 opcode 类型将指令路由到 Cube、Vector 或 Scalar 流水线。Cube 流水线是脉动阵列，支持多发射，以 16 乘 16 乘 16 为基本块做矩阵乘法。Vector 流水线是 2048 位的 SIMD 单元，单发射处理逐元素运算。Scalar 流水线是严格的单发射 ALU，compute_size 恒为 1，用于对角线元素操作。SPAD 和 ACCUM 各 256KB，通过双缓冲机制交替使用。

---

# 07_基础算子对比

最后来看三类基础算子的对比。GEMM 系列映射到 Cube 脉动阵列，支持多发射，延迟由分块数和流水线填充排空决定，compute_size 是三维矩阵乘积累。Vector 系列包括十种运算和同步屏障，映射到 SIMD 单元，单发射，延迟由向量迭代次数乘延迟常数。Scalar 系列只有五种逐元素运算，严格单发射，compute_size 恒为 1，固定 1 到 4 周期延迟。数据搬运统一遵循 DRAM 到 SPAD 到流水线到 SPAD 再回 DRAM 的路径。
