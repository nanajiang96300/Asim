# NS 数值影子执行器：首个可实施改造包

日期：2026-10-28（Asia/Hong_Kong）；源码基点：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。

状态：**可执行设计，尚未实现**。本轮只增加审计和设计文档；下述接口、目录、schema、命令均为后续实施约定，除非在任务表中明确标记，否则不能当作已有工具。上位契约、门禁与评分原则见 [VERIFICATION_DESIGN.md](../VERIFICATION_DESIGN.md)，问题依据见 [审计报告](VERIFICATION_AUTOMATION_AUDIT.md)。

2026-10-29优先级调整：负责人要求先完善通用框架，当前不修改具体NS/其他算子。共同builder、解释器与门禁先按 [FRAMEWORK_PLAN](../FRAMEWORK_PLAN.md) 在合成程序上验收；本文S0–S6保留为框架完成后的真实NS接入/回归任务，复用已完成基础设施，避免重复实现。算法接入与修复不混入框架首批补丁。

## 1. 首个交付边界

先迁移 `NewtonSchulzBaselineOp` 的小规模静态求逆内核，再扩展复数 MMSE 和异步事件检查。保留 C++ 算子及 Asim 周期模型；不建设完整 IR 编译器，不从周期 trace 或 FormulaLogger 猜数值。

首个交付必须同时做到：

1. 从实际构建出的 Instruction 对象导出全量数值操作，不另外填写一条可脱节的数学步骤列表。
2. 当前入口的输入绑定/shape/输出问题被明确拒绝，不能通过解释器自动修正。
3. 一份语义完整的最小 NS 程序得到逐轮 T/R/X 和最终 STORE，并与独立算法及高精度参考比较。
4. 修改实际操作对象后，受支持的变异保持 FormulaLogger 不变也被捕获。

初版顺序功能通过仅可记为 `STRUCTURE_VERIFIED`；硬件精度、异步调度和通信质量未完成时整体 `score=null`，不建立 verified 冠军。

## 2. 当前必须保留的失败基线

| 问题 | 源码证据 | 必须出现的诊断 |
|---|---|---|
| Model 连接 H/RegI/X_init/C，Op 当 A/X0/C 消费前三个 | `src/inverse/newton_schulz/NewtonSchulzBaselineModel.cc:17`、`NewtonSchulzBaselineOp.cc:41` | INPUT_BINDING_MISMATCH；batch/shape 越界另报 BOUNDS_ERROR |
| Op 按 M×M，Model 输出为 batch×U×U | `NewtonSchulzBaselineOp.cc:34` | SHAPE_MISMATCH，禁止只选 M=U 掩盖绑定错误 |
| 注释 C−T，但实际 opcode 是 ADD | `NewtonSchulzBaselineOp.cc:100` | 旧未迁移语义为 UNSUPPORTED；显式解释为 ADD 时与正确 NS 不一致 |
| X0、2I 只有 tensor 名称，没有数值初始化 | `src/Tensor.cc:21`、BaselineOp `:76` | INPUT_MISSING/UNINITIALIZED_READ；不得补 I/10、随机矩阵或2I |
| 默认 K=16，Model 未转发 iterations | BaselineOp `.h:20`、BaselineModel `.cc:23` | 记录实际 K，不从 config 宣称值替代 |
| 最终真实生成 X@X | BaselineOp `:127` | 保留执行，报告 OUTPUT_CONTRACT_MISMATCH/NUMERIC_FAILED |

另一个 ONNX `NewtonSchulzOp.cc:217` 存在跨指令覆盖旧 X 的疑点，应单独登记 program_id；不把它与 baseline、legacy optimized 混成一个实现。原有算法文件本轮不修复；后续修复以单独候选 commit、独立工作区进行，保留失败程序供回归。

## 3. 输入输出与三条数值路径

### 3.1 结构试点契约

`ns_inverse_structure_v1`：初始支持实数行主序方阵、batch=1、固定 N/K，显式输入 `A、X0、C`，并校验 C=2I。输出目标是 X_K（逆矩阵近似）；程序实际 STORE 的内容另行保存。参数 X0/K 不由解释器重新选择。输入、输出 shape、别名许可、有效区和精度 profile 固定并生成摘要。

先用 N=2/4 和可精确表示的已知解，随后加入 N=16、非对角 SPD、受控条件数、batch=2、不同矩阵输入以及尾块。不能用全单位阵作为唯一测试，因它可能掩盖平方输出和初值错误。

例如固定 A=diag(2,4)、X0=diag(1/2,1/4)、C=2I，正确 NS 每轮保持 X0；若最终做 X@X 则输出 diag(1/4,1/16)。主 Agent 已运行这个独立数学见证（[diagnostics.json](../../../experiments/audit/2026-10-28/diagnostics.json)），平方输出未归一化 Frobenius 残差约0.901388；它仍不是 C++ 实际指令重放，后续必须用真实操作对象覆盖此断言。

### 3.2 MMSE 任务参考始终独立

任务契约为 A=HᴴH+λI、b=Hᴴy、x_ref=solve(A,b)。固定 H/y/λ 后，由可信数据准备生成 A/b，并保存全部输入及摘要。初期只计给定 A/X0/C 的求逆内核周期，标注 `inverse_kernel_given_inputs`；HᴴH、Hᴴy、初值/谱估计和 Xb 的时间没有测量时，不得称为均衡全流程周期。

三路径分别实现、分别输出：

| 路径 | 输入 | 执行与比较 |
|---|---|---|
| P1 实际程序 | 同一实际操作对象导出的 program + 固定内存初值 | 逐条执行真实 LOAD/GEMM/ADD/SUB/STORE，得到 X_stored 及阶段快照 |
| P2 同算法 | A、X0、C、实际 K、同精度约定 | 独立 NS 循环 T=A@X、R=C−T、X=X@R；不调用P1 dispatcher，不按算法名更换初值/轮数 |
| P3 任务 oracle | 原始固定 H、y、λ | complex128/高精度 solve(A,b)；求逆结构检查另用 solve(A,I)，不调用NS |

P1/P2 不符是实现或降级语义问题；两者相符但与 P3 差距过大是收敛、初值、有限精度或任务适用域问题。保存相对输出误差、AX−I 和归一化后向残差。通信质量基于同符号/噪声的成对 SER/BER；不把逆误差均值直接称为通信验收。

复数布局、共轭和实数指令展开未完整定义时返回 UNSUPPORTED。不能让 NumPy complex128 自动完成当前 C++ 并未表达的共轭运算。实数结构试点不代表复数 MMSE 已通过。

## 4. 同一操作对象与最小 schema

拟在 `src/Common.h` 的实际 Instruction 附加不可变数值语义，新增受信 builder。builder 接收完整 typed spec，一次构造操作身份、读写 view、数值与计时字段；Asim 消费计时投影，导出器序列化该对象本身。迁移范围内禁止绕过 builder 直接 push 未标记 Instruction。旧对象没有完整语义即返回 UNSUPPORTED。

数学 SUB 可以由受信后端映射到 ADD 的周期延迟，但对象中必须保留 `numeric_opcode=SUB`；测试验证 SUB/ADD 数值不同且计时映射符合配置。不能在 Python 中通过 `NS_R_` 名称或 FormulaLogger 决定加减。FormulaLogger 后续由同对象派生，迁移期间只作诊断。

| 字段组 | 必需字段/规则 |
|---|---|
| 程序身份 | schema_version、program_id、source/config/contract/input hashes、numeric_profile、完整序号与终止计数 |
| 操作身份 | 全局唯一 op_id、tile_id、batch、logical_order、numeric_opcode、timing_opcode、iteration/phase（仅诊断） |
| 内存 | space=DRAM/SPAD/ACCUM、tensor_id、region_id、core_id、buffer_id、byte_offset、allocated_bytes、有效区域 |
| View | shape、stride_bytes、element_dtype、complex_layout、transpose、conjugate、mask；logical元素与DMA request分开 |
| 算术 | 有序src/dst views、立即数、alpha/beta或overwrite/accumulate、zero_init、别名许可、常量来源 |
| 精度 | input/multiply/accumulate/output/storage dtype、rounding、overflow/underflow、复数展开/归约顺序；未知项不得默认 |
| 同步 | 依赖op_id、barrier作用域/单元、后续event实例绑定 |

最小结构示意（不是现有 JSON 文件）：

```json
{
  "op_id": "tile0.op7",
  "numeric_opcode": "SUB",
  "timing_opcode": "ADD",
  "src": ["view:C", "view:T"],
  "dst": "view:R",
  "write_mode": "overwrite",
  "profile": "structure_f64",
  "depends_on": ["tile0.op5"]
}
```

引用的 view 必须在同一 program 的完整定义表中存在；示例不是可直接执行的完整记录。schema 禁止未知字段冒充缺失的必需语义。canonical JSON 排序后计算 SHA-256；每个 backend 的运行清单引用同一个 program hash。非有限数不以 JSON NaN 输出，错误值另记状态。

## 5. 最小执行器

新增拟议模块 `verification/shadow/{schema,memory,execute,compare}.py`，只实现首批实际 NS 需要的原语：MOVIN、MOVOUT、GEMM_PRELOAD/GEMM、ADD/SUB、PIPE_BARRIER。GEMM_PRELOAD 的预加载影响计时；其数值覆盖/累加规则必须显式定义，不由名称猜测。需要 COPY/FILL 时显式新增及测试，不用“其他指令”隐含代替。

内存按真实空间、core、buffer 和字节范围寻址。维护初始化掩码、dtype/view 校验和写入版本；局部写入只更新有效区域。DMA 对齐会触及 padding：需声明合法 padding、逻辑 offset 和 mask，不让一个 request 隐式初始化整张矩阵。

单指令别名可采用“读取完整输入快照后写回”的功能语义；跨指令覆盖按实际顺序生效。因此原地 X=X@R 可在明确定义下执行，而先覆盖 X 再读 X 的程序不会被临时 ndarray 偷偷修正。此规则只证明逻辑功能，物理异步读写仍归 G4。

`structure_f64` 用 FP64 计算确认数据流；地址与尺寸仍遵循程序声明的存储布局。该模式明确放松硬件舍入，仅输出结构诊断。随后新增如 `f16_mul_f32_acc_f16_store_v1` 的明确 profile：逐项描述乘法舍入、累加顺序、转换、饱和/溢出/下溢，独立校准。禁止仅在整次 matmul 前后 cast 就把 profile 称为硬件精度。

执行器结果至少包含：status、failed_gate、first_bad_op、reason、coverage supported/executed/total、uninitialized/unsupported列表、T/R/X/STORE快照引用、各误差指标、program/input/profile hash。功能失败、未实现和样本不足分别报告，不能统一成布尔 pass。

## 6. 失败状态与负例

| 状态 | 触发示例 |
|---|---|
| INVALID | 缺输入/输出、shape或摘要不符、未初始化读、越界、重复ID/缺序号、非有限输入输出 |
| UNSUPPORTED | 无SUB/元素view/复数布局/精度语义；未支持opcode；需要的profile未实现 |
| NUMERIC_FAILED | 运算序列与同算法或任务阈值不符，含当前真实末尾平方 |
| HAZARD_FAILED | G4发现RAW/WAR/WAW、错误barrier或过早覆盖 |
| INCONCLUSIVE | 通信样本不足或置信界不能确认；不评分 |
| INFRA_ERROR/TIMEOUT | 工具/构建/制品读写错误、资源预算超时；与数学失败分开 |

每个负例修改真实 builder 返回的操作对象，并让两个后端的程序摘要一起变化；不只改 P2 算法脚本。FormulaLogger 保持不变验证其不是最终真值。

| 变异 | 最早应捕获的检查 |
|---|---|
| 删除LOAD/初始化 | G0/G1 未初始化读 |
| 删除GEMM/一轮迭代 | G0序列完整性、P1/P2阶段或最终差异；选尚未收敛样本 |
| SUB→ADD | G1原语与P1/P2残差阶段 |
| 改X0或C内容 | 输入hash/契约检查；若是合法新算法变体须创建新contract/algorithm_id |
| 输出X→X@X或保留当前平方 | 求逆/任务oracle失败，已知对角例不能被自一致绕过 |
| 错buffer/错误覆盖/错误batch | 初始化、范围或阶段值失败；batch输入必须不同 |
| 缺/坏program、trace、输入或输出 | G0失败，不随机补值/回退算法/跳过 |
| 漏共轭、尾块越界 | 相应复数/尾块支持后启用，未支持前不得标为通过 |
| 删除barrier/提前异步覆盖 | G4事件检查；顺序执行器不能假称捕获所有调度问题 |

不要求每个变异在每个输入上都失败：例如精确逆初值会掩盖漏迭代。测试清单须明确每种变异使用的辨别样本、预期失败门禁及覆盖范围。

## 7. 独立调度与计时

G4 另行扩展 Core/SystolicWS/MTE 事件记录：每条实际指令的op_id、空间/core/buffer、读写区间、发射、读完成、写可见和完成事件。DMA fragments 必须绑定父op及fragment范围；现有Load/Store通用名称与GEMM重复ID不足以唯一关联。

检查读前写、未完成读期间覆盖、多个写冲突、barrier范围、buffer重用与越界；跨核心/批次内存身份不可仅用地址整数区分。只按 start_cycle 排序或按预定拓扑序执行不能替代它。周期trace缺失/损坏时 G4 INVALID。

计时分别记录 setup、iterate、restore/apply、搬运和总周期；循环内部或外部预处理不免费。独立测量范围分别建结果组，跨信道复用明确 R 并报告 T_setup+R·T_apply。模型预测周期与实机测量分开。

## 8. 文件改造范围与依赖顺序

以下为后续任务，不是本轮已修改文件。基础设施与算法修复分开提交，评估器不在候选可写范围内。

| 包 | 拟改造范围 | 依赖/退出条件 |
|---|---|---|
| S0 契约与固定样本 | `verification/contracts/`、可信fixtures与schema | 冻结小内核、MMSE oracle、布局/精度及已知解；缺项明确unknown |
| S1 builder/导出 | `src/Common.h`、新增受信builder/serializer、NS实际构造入口 | 同一对象驱动两backend；完整性/范围负例拒绝；旧入口明确失败 |
| S2 顺序功能 | `verification/shadow/`、独立P2/P3 | 最小原语与已知解；所有操作执行/支持覆盖100%；当前平方和输入错位可定位 |
| S3 迁移/修复候选 | 独立worktree内的NS Model/Op/配置 | 先读开发标准并实际完成op-flow对应步骤；冻结新commit，不覆盖失败基线 |
| S4 精度/复数 | profile、量化/复数原语与独立校准fixture | 乘累存/共轭/布局明确；对应负例；硬件适用性证据另记 |
| S5 异步调度 | `src/Core.cc`、`SystolicWS.cc`、事件schema/检查器 | 实际事件与program一一绑定；删barrier/提前覆盖负例被拒绝 |
| S6 有监督runner | 新隔离runner、结果schema、评分器v2 | 固定评估版本、串行build/sim、干净重跑、归档hash；全部必需门禁前score=null |

有人监督的首个开发批次聚焦 S0–S2。S3 的算法修复需要独立设计/候选，不与基础设施补丁混合；完整自动搜索及跨天控制器不在前置依赖中。

## 9. 后续可执行命令约定与预算

下列命令只有实施相应模块后才可运行；本轮没有这些工具。接口参数和退出码需随实现一起冻结：

```bash
python3 -m verification.shadow.execute --program "$RUN_DIR/program.json" --inputs "$RUN_DIR/inputs.npz" --contract "$CONTRACT" --profile structure_f64 --output "$RUN_DIR/shadow.json"
python3 -m verification.shadow.compare --run "$RUN_DIR" --contract "$CONTRACT" --output "$RUN_DIR/comparison.json"
```

建议退出码：0=请求范围全部通过；2=INVALID；3=UNSUPPORTED；4=NUMERIC_FAILED；5=HAZARD_FAILED；6=INCONCLUSIVE；7=INFRA_ERROR；124=TIMEOUT。JSON状态与退出码一致，日志文本不决定通过。

2 vCPU环境起点：只读审计/生成最多2个并发、编译和仿真各1个且避免同时重负载；构建 `-j1`。先估算磁盘与内存，结构fixture每项上限10秒、首批最多32项；首次构建最多10分钟、单小规模sim最多120秒。超预算记录 TIMEOUT，不自动增加预算或启动搜索。基础设施暂态故障最多重试1次；相同确定性失败不重试；任何源修复产生新attempt与commit。

本包完成的证据：机器可校验契约、实际导出program与摘要、固定输入、阶段快照、独立P2/P3误差、逐个负例结果、构建与小规模运行记录。G4/G5/G6尚未完成时仍明确列出，不用“整体PASS”掩盖未实现能力。
