# 关注点一：Python算法如何经skill变成Asim算子

**这是一套约束Agent编写和验收代码的流程，不是Python源码自动转NPU机器码的编译器。** skill是可读的工作规范；Agent需要理解算法、选择原语和布局、写C++ Model/Operation，再用确定性工具验收。没有Agent账号也能手工按同样步骤实施；包内不含账号、密钥或模型权重。

## 输入与产物

输入：Python算法原型、数学公式、输入输出契约、允许的精度/尺寸/误差和预算。输出：设计文档、C++指令生成器、typed views、配置/注册项、独立参考、冻结版本及验证证据。不能只得到一份FormulaLogger声明就说转换成功。

以NS为例，任务先定义H/Y/λ→xhat；A=HᴴH+λI，B=HᴴY，X0=I/trace(R(A))，X(k+1)=Xk(2I−AXk)，最后xhat=XK B。矩阵求逆XK与均衡xhat是不同输出；不能误输出XK²，也不能把外部预计算A或B的时间省掉。

1. **Python可行性**：明确初始化、K、预条件、缩放和输出恢复。FP16模型逐原语写回，不以NumPy FP64默认中间值冒充FP16。参数优化先校准选K/冻结，再新留出；不能边看留出边改K。
2. **读取skill并写设计**：读取`.claude/skills/op-flow/SKILL.md`、`op-dev`、`audit-operator`、AGENTS及V3标准。设计先于代码，列出公式、张量/分块、SPAD/ACCUM区域、依赖和计时范围。Codex中实际读文件执行步骤，不声称调用不存在的slash命令。
3. **降低到已有原语**：矩阵乘→GEMM_PRELOAD；逐元素→Vector；真正单元素→SCALAR。Vector没有独立SUB时，使用MUL(-1)+ADD并单独写回/计费。CONST、COPY、归约、预条件和初始化也必须有真实操作，不能免费填数。
4. **生成实际Instruction和语义一起写**：参考`src/inverse/ns_corrected/NSCorrectedModel.cc`与`src/semantic/SourceViews.h`。在真实Instruction构造点派生opcode、物理src/dst、m/n/k、compute_size以及view(row/col/shape/transpose/conjugate/broadcast)。所有切片要具体，不用“按算法名推测”代替。
5. **布局与同步**：每次读前有LOAD或写；完整区域先分配再局部写；所有batch的DMA地址/stride匹配。复制、取负、状态更新和输出恢复不能省计时；同步必须实际发出。独立worktree/build/log避免互相覆盖。
6. **独立参考与注册**：`orchestrator/codex_operators.json`绑定mode、参数函数和独立reference。参考不得调用实际replay引擎生成期望值；replay核心也不得按“NS/LDL/BRI”名字改成标准求解器。
7. **构建/独立审计/冻结**：构建真实Simulator；另一审查者对公式、代码、layout、精度和依赖复核；固定源码、binary、库、配置、输入、模型/评估器版本。
8. **运行门禁**：Asim→事件与成本/性能预筛→实际指令重建→独立同算法→FP64任务→适用通信验收。失败/未知语义不升级，保留原件。性能不竞争可在数值前拒绝；初始基线创建不宣称性能优化。

## 六类源码在哪里

| 类别 | C++生成器 | 独立Python参考 |
|---|---|---|
| Cholesky NoBlock | src/inverse/cholesky_explicit/ | verification/reference/cholesky.py |
| LDL NoBlock | src/inverse/ldl_explicit/ | verification/reference/ldl.py |
| Cholesky Block | src/inverse/structured_solve/ | verification/reference/structured.py |
| LDL Block | src/inverse/structured_solve/ | verification/reference/structured.py |
| Newton–Schulz | src/inverse/ns_corrected/ | verification/reference/ns.py |
| Block-Richardson | src/inverse/bri_explicit/ | verification/reference/bri.py |

Block实现是实块4；上述BRI为实块2、ω=.5、x0=0的stationary版本，并不自动代表其他复块/Chebyshev方案。其他历史入口和改进BRI源码保留为研究实现，不能从“有代码”推断同样通过全部信道。

## 推荐给Agent的任务文本

> 先读取AGENTS、docs/PYTHON_TO_OPERATOR.md、docs/RECONSTRUCTION.md和三个skill。将给定Python算法冻结为H/Y/λ→xhat契约，明确FP16写回、初值/K/预条件。设计先于C++，用实际Instruction与typed views生成，未知原语报告UNSUPPORTED。保持评估器/独立参考/门限与候选修改范围分离，独立构建并先Asim后数值。保存基线与每轮候选commit/参数/周期/精度/失败，不覆盖历史，不自动推送。

Agent可提出候选，但不能自行降低质量门限、改输入或给自己打分。输入输出范围、迭代参数、计时与精度由任务契约约束；指令数不必相同，但有效工作和成本必须有真实证据。
