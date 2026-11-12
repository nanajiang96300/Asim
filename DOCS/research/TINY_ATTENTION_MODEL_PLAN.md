# 小型通用Attention模型：源码审计与实施计划

2026-11-12，状态DESIGN_ONLY。本轮只读审计和制定计划，不修改算子、Asim或验证器，不宣称模型已运行。专家目标是证明小型多算子模型的真实周期仿真及输出正确性；不以CSI恢复、训练、语言生成或通信质量为前置。

## 结论与依据

架构可复用，尚不能直接签发已有Attention/语言模型入口的端到端数值通过。现有仿射任务已证明可信BACKEND可以接入非MMSE任务；首轮仍走相同Asim→真实事件/性能门禁→指令数值重放→独立参考链路。主要缺口是Softmax/GELU执行语义、工作量/周期口径、共享权重与整图任务契约。

| 实际文件 | 审计证据 | 本阶段判断 |
|---|---|---|
| src/operations/Attention.cc | 已生成GEMM与SOFTMAX，另一融合路径包含EXP/DIV等 | 旧模型代码存在，不能推断已验证真实数值/缩放/布局 |
| src/models/LanguageModel.cc、src/main.cc | 已有language与ONNX入口 | 有模型装配路径，未用当前可信链验收 |
| src/operations/Softmax.cc | 发出opcode8；size按dk×precision，未显式填compute_size | 与当前绑定契约不同，必须核对维度与周期工作量 |
| src/operations/BiasGelu.cc | ADD/GELU的compute_size按元素×precision | 与当前F16 vector_bytes=4口径不同，待独立审计，不能直接套用 |
| src/SystolicWS.cc:get_vector_compute_cycles | SOFTMAX/GELU/EXP已有周期分支；Softmax包含归约/向量估计 | 无需先新增混合精度指令；必须核对行数/元素数是否全部计入 |
| verification/reconstruction/source_views.py、binding.py | opcode白名单尚不含8/22等 | 当前可信重放/物理工作量绑定会拒绝，需补通用语义 |
| verification/profiles/numeric.py | 有GEMM/ADD/MUL/DIV/SQRT，无Softmax/GELU | FP16存储/FP32临时已有；非线性数值模型待定义 |
| verification/reconstruction/stream.py | 单Operation JSON，拒绝尾随记录 | 首轮用显式整图Operation容器；不冒称已有任意ONNX多节点验证 |
| verification/tasks/affine.py、runner.py | task BACKEND、输出shape、独立参考、共用重放已有 | 复用架构，不复制模型专用解释器 |

真实指令必须有typed views、输入/权重绑定及Core事件。仅旧ONNX跑出周期、FormulaLogger自一致或按模型名调用Python网络不算完成。

## 首轮模型与计算规模

命名建议tiny_attention_mlp_v1：单头、自注意力＋GELU＋两层全连接＋残差。称“小型Attention推理块”，不称完整标准Transformer。采用标准scaled dot-product attention公式，参考[原论文](https://arxiv.org/abs/1706.03762)。

默认B=96、序列L=32、隐藏D=32、FFN宽F=64；24核，每核4样本，各矩阵维度16对齐。输入X[B,L,D]，输出Y[B,L,D]，6个共享权重矩阵与6个偏置：

```
Q = X Wq + bq; K = X Wk + bk; V = X Wv + bv
P = softmax_rows((Q K^T) / sqrt(D))
U = X + (P V) Wo + bo
Y = U + GELU(U W1 + b1) W2 + b2
```

所有中间FP16写回点在任务中逐一冻结；GELU先选tanh近似并明确公式/常数/FP32临时运算顺序，不能将erf版与tanh版混用。Softmax采用逐行减最大值→exp→求和→归一化，归约顺序/轴/临时精度/最终FP16存储明确记录。拟议新增模型专用数值策略；不修改旧MMSE精度模型/门限或声称硬件已经校准。

默认含8,416参数（含偏置），每样本327,680个GEMM MAC，B96共31,457,280 MAC；非线性、偏置、残差、DMA另计。每样本逻辑8次GEMM、6次偏置加、2次残差加、1次缩放、1个逐行Softmax及1次GELU；实际tiling后的指令数单列，不强制等于逻辑节点数。

权重固定、输入固定种子且有零/正负/边界值。随机固定权重可用于工程推理一致性测试，无需训练；不能据此宣称分类/语言/CSI恢复效果。第一版不含LayerNorm、dropout、mask、KV cache、embedding或多头切分，避免扩大首轮语义范围。

## 输入、布局与计时契约

- X、全部W/b及缩放常量实际进入外部输入/常量表并绑定SHA；禁止在Python预先算Q/K/V/P作为Asim输入，或从trace之外偷偷提供中间输出。
- X/Y按三维batch组织，W/b按共享二维角色组织；共享权重不应套用样本跨界检查。显式记录广播、K转置、Softmax轴、偏置和残差，禁止凭shape猜测。
- 首轮C++新Model/mode，内部一个整图Operation，保留各层/阶段标签、实际Instruction与物理缓冲；逐batch最后完整STORE一次，符合现有单Operation/单输出STORE限制。后续扩展多Operation时要真正连接外部存储与producer/consumer，不把层分别通过当整网通过。
- 起止为第一笔模型输入/权重DMA至最后输出STORE完成；首轮固定冷权重口径，不外置初始化/恢复。若将来复用权重，新增有版本的热态口径并单列首次加载，不能在不同候选间悄然换口径。
- 首版保守同步，不做融合；运行前列出活跃tensor、SPAD/ACCUM容量与复用周期。未经布局清单不承诺整图驻留片上。

## 分阶段交付与退出条件

| 顺序 | 工作 | 退出条件 |
|---|---|---|
| T0 冻结合同/独立原型 | 新task、输入/权重/维度/种子、参考、精度写回及计时；独立校准 | 稳定Softmax/GELU公式固定，任务专用rtol/atol经校准冻结，独立留出未用于调门限 |
| T1 补通用语义与周期核对 | SOFTMAX/GELU原有opcode的真实地址/轴/形状/工作量/读写/数值语义；检查偏置广播和常量 | 小型原语真实Asim先行后逐值重放；错轴/少算行/漏偏置/未知公式/错误尺寸均拒绝；旧门禁回归 |
| T2 新模型实际映射 | 全部层/权重/常量真实发指令，新Model/mode、整图Operation；独立worktree/build | B96默认形状真实Asim完成，24核均工作，全部指令/事件覆盖、无错误读写或漏层；再数值 |
| T3 整图三路径验收 | 实际指令重放、独立同语义FP16算法、独立FP64数学参考；阶段捕获 | 全部batch/所选中间阶段符合冻结精度要求；全输出与FP16参考位一致，FP64误差达任务专用门限；不是套用MMSE2% |
| T4 泛化与反例 | 形状(L,D,F)=(16,32,64)/(32,32,64)/(32,64,128)，B96，每形状3冻结种子；覆盖特殊输入 | 九场景864batch；同权重不同batch输入可区分；故意错K转置/Softmax轴/缩放、GELU跳过或残差漏加会被拒绝，无分数 |
| T5 交付与扩展 | 阶段/全图周期、逐指令/逐流水/逐核、误差/资源/Agent调用墙钟与版本归档 | 干净环境源码构建和完整首例；专家一条确定性命令运行；旧六类＋affine回归；新无数据版本包 |

T1先核对Softmax的compute_size与tile_m行数含义，不能简单把旧GEMM/ADD字节约定套给归约算子。相同定义下检查随行数、行宽增长的工作量/周期关系；如需修正Asim周期公式，独立版本化并保存旧周期，不升级为硬件级时序认证。不要为绕过Softmax语义而换成非标准attention，却仍沿用标准名称。

T0不把完整IR编译器、ONNX全覆盖、模型训练或任务精度作为前置。完整ONNX/旧LanguageModel可信接入属于首例稳定后的下一里程碑，不能与整图Operation首例混淆。

## 三路径验证与不允许偷懒的门禁

1. 重放仅依据实际指令及typed semantics取数/执行，通用解释器不识别tiny模型名。高层SOFTMAX/GELU可以作为已有抽象指令，但其完整读写、工作量、axis/formula和精度必须显式绑定；不虚构其内部微指令流水细节。
2. 同语义FP16独立Python算法不导入重放数值实现；FP64参考按数学公式独立计算。可增加第三方框架交叉检查，但不能以不相同的FMA/归约顺序强求逐位相等。
3. 不只比较最终Y：捕获Q/K/V、logits、P、U、GELU输出；检查Softmax非负/行和、批次隔离、输出覆盖和权重被实际使用。采用已有captures能力及必要通用扩展，不用人工公式声明作证。
4. 全零、相同logits、正负GELU、受控大幅值、近零输出、换一个权重/样本只改变合理输出等均加入；NaN/Inf或未支持精度明确拒绝，近零输出同时用绝对误差门限。
5. 首轮没有已有模型性能基线，先建正确性基线，只做工作量/周期有意义性检查；之后优化才启用冻结的minimum_speedup等预筛，仍先Asim后数值。不得为了首个模型出分数自造性能优势。

## 工程组织、资源与归档

新task/reference在verification/tasks和verification/reference；新增语义同时覆盖binding、source_views/engine、numeric profile、Core事件/调度与测试，避免只改解释器放宽白名单。候选只能改批准模型源文件；可信验证器/参考/门限由独立变更维护。优先复用src/semantic/SourceViews.h和现有BACKEND，不直接大修历史Attention/MIMO算子。改算子前按CLAUDE/V3/op-flow实际实施并独立审计。

源仓新候选独立分支/worktree/build，2vCPU/约3.8GiB，编译和Asim串行；初步检查磁盘仅约1.1GiB余量，干净冷构建前先估空间，不直接开启全量九场景搜索。先B96单场景，保留代表性trace，完整原件按SHA清单保存/压缩，未估计资源前不批量跑。

每阶段记录输入/权重/模型/源码/binary/config/evaluator/reference/scorer SHA、参数/形状/batch、所有状态、层/流水/总周期、误差、峰值内存和阶段墙钟；纯Agent思考时间不可获取就记null。目录建议experiments/tiny_attention/<版本>与results/tiny_attention/<版本>，论文固定paper_archive索引。失败与旧包保留，不覆盖刚交付的asim-claude-guide-v1。

推荐下一项具体开发：T0独立Python微型Attention块与冻结合同；随后T1真实Softmax/GELU语义及周期工作量验证。以上尚未开发/运行。
