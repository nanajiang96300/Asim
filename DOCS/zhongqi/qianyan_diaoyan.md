前沿调研报告：可探索的研究方向（2026-09）

说明：本报告基于 2026 年 9 月对 2024–2026 年文献的系统检索，覆盖四个方向（低精度/混合精度求逆、LLM 算子代码生成、AI-Native RAN/NPU 基带、深度展开检测与可学习求解器），合计约 120 次检索。标注含义：★＝已读原文或摘要；☆＝仅依据检索摘要；△＝社区/厂商博客。所有条目均来自实际检索；写入论文正式引用前请核对原文与最新版本。

一、最近的相邻工作（必须知道）

1. AI-RAN on NPUs: Baseband Processing Without Baseband Chips（arXiv:2607.04224，2026-07，本课题组）★
   唯一一篇在 COTS NPU（Ascend 310B1）上跑通完整 OFDM 收发链并 3.0 GHz OTA 验证的工作。范围是 SISO：无 MIMO 检测、无预编码、无矩阵求逆、无软解映射；接收端 46.2 ms，比 5G NR 0.5 ms 时隙预算差近三个数量级；其 future work 明确列出混合 NPU-CPU 分区等未做事项。启示：该文确立了"NPU 跑基带"的可行性叙事，但 MIMO 检测、求逆、软解映射整条线仍是空白；本论文 34.36 µs 的全链时延说明张量提升方法论的优势远未被该文触及，其 future work 清单就是本论文后续工作的天然选题。

2. Finite-Precision Conjugate Gradient for Massive MIMO Detection（Fang 等，arXiv:2504.09820，2025-04，2026-04 更新）★
   与本研究混合精度 BJ-PCG 重叠度最高的公开工作：有限精度 CG＋block-Jacobi 预条件＋相关信道，并自承高 SNR 下有轻微损失。差异化必须做足三点：（a）以 SER 而非 BER 为验收口径；（b）以 NPU 周期而非 FLOPs 为成本；（c）六种实现的系统横比（该文只做 CG 一种）。

3. Hierarchical Recursive Precision for Accelerating Symmetric Linear Solves on MXUs（Carrica 等，MIT，SC25 / arXiv:2601.08082）★
   GPU 上 FP16→FP32→FP64 嵌套递归 Cholesky，比 FP64 cuSOLVER 快 5.32 倍。自陈局限：极端动态范围（>1e11）下静态块量化失效、正定性被破坏并传播 NaN——该失效模式与本研究的 BRI 在 CDL-B 上失败同源，且修复在文献中仍是公开问题。

4. 华为生态的 Ascend 算子 Agent 化（2026 年集中出现）☆
   AgenticCANN（arXiv:2607.26661）、CANN Bench（arXiv:2607.20518，提出 HAP 硬件锚定性能解析下界、512 字节对齐压力测试）、AscendKernelGen（arXiv:2601.07160）等。结论："用 LLM agent 优化 Ascend 算子"本身在 2026 年已不是空白。本论文 LLM 框架的差异化必须落在：微架构级约束反馈＋语义 DAG oracle＋周期级仿真器回路，而不是"知识增强的代码生成"。

5. The AI Telco Engineer（NVIDIA，arXiv:2604.19803，2026-04）★
   LLM agent 自主发现无线算法（如 OTFS 均衡器），但其评估器是 Sionna/Python 仿真，无任何硬件成本模型。把周期级 NPU 成本模型接入 agent 算法发现回路，目前无任何工作。

6. Mothra（ETH Zurich，IEEE JSAC 2025）与 ADMIN VLSI（Asilomar 2025）☆
   深度展开检测器落 FPGA/ASIC 的两个标杆（Mothra：22 nm，7.1 Gbps@367 mW）。"展开检测器上 FPGA"不再是新颖性来源；新颖性只能来自新的硬件类别（NPU）或新的算法-硬件协同理论。

二、已被解决的方向（建议不再投入——即老师所说"别人已经解决的问题"）

1. OAMP-Net 类深度展开检测变体：2025–2026 年大量变体（AR-OAMP-Net、OAMP-Net2+ADD、贝叶斯展开、Soft Graph Transformer、In-Context Learning 系列）已把"小参数展开网络在 i.i.d./相关信道下降 SER"的空间填满。开题计划 stage-3 的 OAMP-Net 需要改造论题：从"性能更好"改为"低精度下收敛可证"或"NPU 上能效最优"。
2. GPU 跑 5G NR L1：已产业化（NVIDIA Aerial/ARC、SoftBank 16 层 MU-MIMO 实测、Nokia 商用 AI-RAN 平台）。
3. GPU LDPC 译码：超 100 Gbps，cuPHY/pyAerial 生产实现，DecodeX 跨平台基准。
4. 神经接收机（learned demapping/equalization/channel estimation）：Sionna Research Kit＋TensorRT 已实现 <1 ms 实时多用户 MIMO 接收。
5. 经典混合精度迭代精化（GMRES-IR/LU-IR）收敛界：教科书级，HPL-MxP 基准已纳入。
6. GNN 学习预条件器在通用稀疏系统（SPAI/ILU 生成）：ICLR 2025、NeurIPS 2025 已标准化。
7. Block-Jacobi 预条件器的精度量化（固定 vs 自适应策略、按块条件数选格式）：2025 年已系统做完，且已知"盲目 FP16 多数失败"。
8. Neumann 级数近似求逆＋脉动阵列 VLSI：2013 年起成熟。
9. Chebyshev/SOR/GS 加速线性检测与预编码：成熟，且已知高阶调制下退化。
10. 展开检测器的低比特 PTQ/QAT（KAQ、FCAP-PQ、TAMNet）：已覆盖。
11. 张量等变网络（TENN）预编码：TWC 2025 已建立统一框架。
12. 统一因子图/全消息传递基带 ASIC：BayesBB（JSSC 2026）已占据 framing。
13. 模拟存内计算做 MIMO 检测与矩阵求逆：2025 年集中爆发（Nature Electronics BlockAMC 等）。

三、确认的空白（多来源交叉验证，2025–2026 仍开放）

A. NPU（非 GPU 的 AI 加速器）上的 MIMO 检测与矩阵求逆算子族：零公开论文。CANN ops-solver 只有通用 getri/matinv 算子；北邮"国产 NPU-RAN"项目把 MIMO 检测列为"待探索"。
B. 周期级仿真器放进算子级 LLM 优化回路：无人做过。算子级 LLM kernel 优化全部使用真实硬件（GPU/TPU/Ascend）；仿真器在环的工作全部在架构层 DSE（AgentDSE/Beacon/LLMulator）。原因之一是仿真器评估慢（数十分钟级）；Asim 单次 40–60 秒恰好反转了这一成本结构。
C. 相关信道下块对角预条件失效的补救：deflation/两级预条件在 MIMO 场景 2024–2026 空白；文献对相关信道的标准答复仍是三对角/Fourier 结构化预条件，或直接承认 BJ 失效。
D. FP8 在 Krylov 求解器/矩阵求逆上的收敛界与误差理论：零同行评议工作（检索无命中）。
E. 随机舍入（SR）用于迭代求解器：空白。SR 现有工作集中在 LLM 训练、SGD 收敛、点积误差分析；Krylov/求逆应用为零。
F. 可学习参数 × 低精度算术 × 收敛保证：三者交集为零。深度展开理论综述只谈精确算术；混合精度迭代精化理论只针对无可学习参数的经典算法。
G. 以 SER 为统一验收口径横比直接法与迭代法：数值分析界报前向/后向误差，无线界报 BER，两者从不接合；没有任何工作在被匹配的精度预算下用 SER 横比六种求逆实现。本研究的 SER 评估体系本身就是这个空白的填补，可直接作为方法论贡献写入论文。
H. NPU 上的 FFT 映射与精度恢复：tcFFT 停在 2021 年（仅 FP16、无精度恢复、通用 HPC 库）；5G OFDM 的 NPU/Cube FFT 映射零论文。与本论文实测 FFT 仅 15.2 倍加速（其余模块 >90 倍）互相印证。
I. NPU 向量单元上的软解映射/LLR：零篇（所有软解映射硬件工作都在 FPGA/ASIC）。与本论文软解映射占 98.8% 关键路径的结论完全对应。
J. 6G 极端 MIMO 的 NPU 时延/能耗标度律：sub-100 µs、<1 nJ/bit 目标被多方提出（EurekAlert 2026-01、MDPI 综述 2026），但没有任何工作给出 NPU 视角的外推曲线。
K. LLM 代码生成的微架构 hazard 分类反馈：AutoMegaKernel 做静态接受/拒绝（不修复）；AgenticCANN 只把 UB/queue 当语料稀缺问题；无人把 RAW/WAR/SCALAR 地址约束分类后作为反馈词表回喂 LLM。
L. NPU 侧语义 DAG oracle：ProofWright/VOLTA/Kuiper 全部是 CUDA/PTX；NPU 无等价物；且三者均不使用"代码内嵌语义声明"机制（FormulaLogger 是该机制在文献中的唯一实例）。
M. 学习型预条件器的谱安全：2026 年刚出现神经算子预条件器导致 PCG breakdown 的反例（PDE 场景）；"学习型预条件＋低精度＋MIMO Gram 矩阵"三合一空白。
N. K（层数）×b（比特）联合最优与可达精度界：逐迭代精度分配仅对 AMP 做过（GPU 实测、无理论）；"相关信道下低精度需要额外迭代"目前只有工程经验（NS 的 K=16/K=24），没有理论解释，也没有人把它上升为设计规则。

四、推荐的可探索方向（每条标注：为什么不重复、如何衔接现有资产）

第一类：求逆与检测算法（学位论文核心主线延伸）

方向 1（最稳，直接衔接）：条件数自适应的 SER 校准精度策略
研究问题：在什么 κ/SNR 分档上切换哪种求逆实现（Cholesky/LDL/NS/BRI/INT8-PCG）与哪种精度组合，能在 SER 损失约束下最小化 NPU 周期？输出 SER-周期 Pareto 前沿，并可用 Q-learning/上下文老虎机自动化。
为什么不重复：RL 精度调优刚被占位（Chen 2025、Carson&Chen 2026），但其奖励是范数误差、成本是 FLOPs、验证集是 SuiteSparse；"SER 奖励＋NPU 周期成本＋无线信道验证域"的组合无人做过（空白 G 的直接落点）。
如何衔接：SER 评估体系＝缺失的奖励函数；Asim＝比 FLOPs 更真实的成本项（Cube 利用率、MTE 搬运、SPAD 冲突）。

方向 2（攻痛点，工程量小）：Deflation/两级预条件救 BRI
研究问题：从块对角部分的主特征向量构造低秩 deflation 子空间（纯 GEMM 操作，不新增指令），能否把 CDL-B 下 BRI＋Chebyshev 拉回收敛？
为什么不重复：deflation 预条件从未被移植到相关 MIMO Gram 矩阵（空白 C，检索空结果）；Fang 2025 只做到"高 SNR 轻微损失"，且其预条件仍是块对角。
如何衔接：NS 的谱初始化已经给出特征信息，deflation 子空间几乎零成本；BRI 失效数据（CDL-B 特征值 0.06–3.8）可直接用于有效性预判。
风险：需先离线验证 CDL-B 跨块相关是否低秩（数学风险最高，建议先用已有仿真数据做特征值分解确认）。

方向 3（新颖性最稳，理论贡献）：随机舍入＋动态缩放
研究问题：SR 的无偏性能否推迟低精度残差间隙的饱和，使 FP16/INT8 路径达到目前必须靠 complex64 内部精度才能达到的 SER？与"逐步 scale→低精度 GEMM→unscale"的动态缩放结合效果如何？
为什么不重复：SR 在 Krylov/求逆上零应用（空白 E）；Asim 能对 SR 的 PRNG 周期代价定价——这是所有 GPU 论文无法回答的硬件成本问题。
如何衔接：INT8 PCG＋重启＋NS 是现成载体；SER 体系给出"是否值得"的判据；注意 ARITH 2025 已指出少比特 SR 的有偏性累积，是已知风险＋未验证场景的切入点。

方向 4（延伸已有观测）：低精度损伤的算子级归因
研究问题：NS 路径中量化损伤有多少来自迭代矩阵、残差计算、预条件子施加？是否存在"关键算子必须保精度、其余 INT8/FP16"的最小集合？
为什么不重复：AugGMRES 2025 的"损伤是哪个算子降精度的函数"结论全部来自实数稀疏 SuiteSparse；复数、相关、κ=1e3–1e4、带 NS 非线性预条件的场景无人做，且无人用应用级指标（SER）定价。
如何衔接：complex64 内部＋FP16 I/O 的实现只需参数化改动；产出"复数 MMSE 求逆的低精度算子树"，可直接指导其余五种实现的精度分配。

第二类：LLM 优化框架（方法论主线延伸）

方向 5（独有资产，强烈推荐）：Asim 周期级仿真器进算子级 LLM 优化回路
研究问题：以周期级仿真器替代真实硬件/剖析器作反馈，能否以更低的单次评估代价换来更多搜索迭代与更好的最终 kernel？单元级信号（Cube/Vector/MTE/SCALAR 忙碌率、气泡、队列深度）能否被 LLM 有效利用？
为什么不重复：算子级 LLM 优化 100% 用真实硬件；仿真器在环 100% 是架构 DSE（空白 B）。Asim 40–60 秒/次的成本结构恰好填补这个领域"被迫回避"的空缺。
如何衔接：UOBS 直接取 Asim 计数；可用 CANN Bench 公开的 HAP 解析下界作外部对照口径；FormulaLogger 保证换 kernel 后计算 DAG 不变。
注意划界：与 AgentDSE（架构 DSE）的区别必须写明——它搜 loop order/tiling 的架构参数，本方向做微架构单元级的算子实现搜索。

方向 6（实战升华）：Hazard 类别化反馈与 NPU 失败分类学
研究问题：把失败细分为危险类（UB 上 WAR、ACCUM 上 RAW、SCALAR 地址空间违规、tile 边界越界、队列深度超限），以类别标签而非原始日志回喂 LLM，能否显著降低失败率与到可行解的轮数？不同模型（DeepSeek vs ChatGPT）对哪类最不敏感？
为什么不重复：无人把微架构 hazard 分类学作为 LLM 的反馈词表（空白 K）；本研究 5/9 Agent 撞 SCALAR 死锁正是这个空白的直接观测。
如何衔接：UOBS 黑盒隔离上加一层失败分类器（仍是黑盒输出，只是更结构化）；Agent Swarm 并行提供大样本；自调试 ≤3 轮上限可直接作为被优化指标。

方向 7（反 reward hacking 的结构性防御）：FormulaLogger 语义 DAG oracle
研究问题：以代码内嵌语义声明自动生成金标准，检测"已测 shape 正确、未测 shape 错误"的静默错误（KernelBench-Verified 证明这是 LLM kernel 领域头号可信度问题，73.8% 的优化只有代理收益无真实收益）。
为什么不重复：NPU 侧无任何等价物（空白 L）；ProofWright 从 torch.fx 外部重建图，而 FormulaLogger 是"代码内嵌语义声明"机制的唯一实例。
如何衔接：FormulaLogger 已在每个算子中调用，只需扩展出"导出 DAG→生成 oracle→差分比较"三个工具；UOBS 的 SER 硬约束从黑盒猜测升级为有语义证据的判定。

方向 8（高风险高新颖）：数值算子的算法重表述搜索
研究问题：AlphaEvolve 证明 LLM 在廉价确定评估器下能做算法发现；能否让 agent 在算法层重表述数值算子（NS 的 Gram 空间化、迭代系数/步数重设计、BRI 三相重划分），而不只是重排循环？
为什么不重复：NS 的两大改进（Gram 空间化、分块化）都是人类洞察；LLM kernel 文献中 Richardson 迭代完全缺席（空白确认）。
如何衔接：Asim 天然满足 AlphaEvolve 成功的前提（廉价、确定、难被 game）；FormulaLogger DAG 保证"结构变了但语义不变"的变体能被自动接受——这是其他框架不具备的判定能力。
风险：数值等价性验证最难，建议把"发现是否可复现"作为主要科学问题并保留人工复核。

第三类：基带链完整化（TWC 论文延伸）

方向 9：Cube 对齐 FFT＋精度恢复（补 15.2 倍短板）
研究问题：N 点 OFDM FFT 重构为一批 cube 对齐的小矩阵乘＋混合精度迭代恢复，能否达到与 LMMSE/ZF 同量级（>90 倍）的加速？
为什么不重复：tcFFT（2021）无精度恢复且非无线；5G OFDM 的 NPU FFT 映射零论文（空白 H）；AscendSiPBoost 无公开性能/精度数据。
如何衔接：复用 tensor-lifting 方法论与 4 乘 2 加/3 乘复数分解；310B 实机 OTA 平台可直接测端到端 FFT 时延占比变化。

方向 10：向量单元软解映射（砍 98.8% 关键路径）
研究问题：Max-Log-MAP 的超越函数（exp/log）在向量单元上如何以分段仿射/LUT/多项式近似实现低延迟高吞吐，且 BLER 不损失？
为什么不重复：NPU＋LLR＋软解映射检索零结果（空白 I）；唯一 NPU 基带论文根本未做软解映射（只做标量比特提取）。
如何衔接：已有软解映射瓶颈的实测数据（98.8% 关键路径）与算子开发经验；tensor-lifting 可把 per-bit LLR 提升为 per-symbol-group 张量以消掉符号级分支。

方向 11：6G 极端 MIMO 的 NPU 标度律（34.36 µs 外推 sub-100 µs）
研究问题：天线数×带宽×调制阶数×精度的四维扫描，输出时延-能耗-精度帕累托前沿，并给出"张量提升失效"的规模阈值与达到 sub-100 µs 所需的硬件参数条件。
为什么不重复：sub-100 µs/<1 nJ/bit 目标被多方提出但无人连成 NPU 视角的标度律（空白 J）。
如何衔接：唯一能同时利用"周期级仿真器＋310B 实机＋全链 34.36 µs 基线＋六种求逆算子"四条资产的方向；外推曲线对 NPU 厂商是直接可用的设计需求。

五、对学位论文框架与未来工作的落地建议

1. 论文框架（1.3 第（六）部分）增加"前沿工作洞察"内容：明确写出（a）本赛道的最新态势（NPU 基带仅 SISO 一篇、MIMO 求逆算子族零论文、Ascend 算子 agent 化华为生态已填满）；（b）红海清单与不做的理由（OAMP-Net 变体、GPU 基带可行性、经典混合精度 IR 理论、TENN 预编码等）；（c）空白清单与做的理由（第三节 A–N）。使"未来工作"的每一项都有文献依据说明"别人还没解决"。
2. 下一步研究计划（二-2（四））按选定方向重写，每条注明对应的空白编号与差异化红线（与哪篇最近工作区分）。
3. 开题 stage-3 的 OAMP-Net 建议改造：从"实现 OAMP-Net 深度展开检测变体"改为"低精度可学习迭代求解器的收缩性保持理论"（空白 F）或"NPU 上展开检测的能效最优"——否则会被审稿人判为重复工作（OAMP-Net 变体 2025–2026 已饱和）。
4. 风险提示：（a）arXiv:2607.04224 与本 TWC 投稿同处"NPU 做基带"概念空间且出自本课题组，其 future work 清单就是本论文后续工作的天然选题，时间窗有利但需尽快投出；（b）"tensor lifting"术语在文献中零命中，建议在论文中显式给出形式化定义，并引用 TENN（arXiv:2510.02108）与张量环 NBP（TCCN 2025）作为相邻工作以建立术语谱系；（c）本报告文献信息均来自 2026-09 检索，正式引用前请核对原文与最新版本。
