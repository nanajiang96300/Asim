# Asim 研究与 Codex 自动化计划

更新：2026-11-08（Asia/Hong_Kong）

**当前架构入口：[精度驱动选K与算子工作流](QUALITY_DRIVEN_WORKFLOW.md)。论文资料固定检索：[paper_archive/README.md](paper_archive/README.md)。**

2026-11-08 Q3b当前：新增实际指令I/Q响应与高斯线性SE代理试点，200测试通过。七新入口及NS留出有限样本共375次重放全部与独立精度参考位一致；检查零偏置、预测误差、共轭分量、RHS串扰及叠加性后才计算代理。原周期未改、未新增Asim；不是总体SE或有限调制吞吐认证。见 [协议](../../verification/communication/RESPONSE_PROBES.md)与[证据](../../experiments/response_proxy/2026-11-08-v2/README.md)。下一步扩大H/输入域及失配统计、有限调制/BLER。

2026-11-08 Q3a历史：187测试通过；新增实际重建输出的瑞利/QPSK相对BER/EVM和联合超限置信门禁。七个新入口的10份历史case只作诊断；新NS K10 batch96双噪声6场景576batch真实Asim/重建通过，通信只用4份留出case，每噪声192独立H通过工程试验门禁。SE仍UNSUPPORTED、完整通信未完成。见 [通信规范与证据](COMMUNICATION_VERIFICATION.md)。下一步Q3b实际响应/系数与SE。

2026-11-08 Q1历史：新增独立FP16算法选K工具、冻结数值门限/预算与校准/留出分离；NS/BRI小规模瑞利烟测通过，159测试通过。七个新入口此前的重建验收继续有效；当时选K交接尚未完成，现由Q2补齐受监督入口；正式SE/BER、CDL-B与评分晋级仍待完成。旧重建设计、阶段稿和证据已按版本快照归档，未改动算子或固定运行器。下列日期段为阶段历史，当前优先级以新工作流为准。

2026-11-08 Q2历史：受监督计划冻结/执行交接已验收，绑定筛选证据、参数、任务/输入种子、源码/注册表与固定runtime；171测试、NS/BRI真实Asim 8场景/192batch通过，真实性能筛选负例证明数值NOT_RUN。见 [验收记录](../../experiments/quality_handoff/2026-11-08-v2/README.md)。下一步为真实重建结果的瑞利通信门禁，不等于通信或完整自动化完成。

工程：`/home/nanajiang/Asim/`

状态：系统审计与证据整理v0.3已落地，通用框架实施计划v0.4已归档；通用Asim同源结构重放与局部门禁已实现；完整精度/异步/通信门禁及持久化控制器尚未实现。当前已扩展至七入口真实指令重建诊断，未修改原算术与调度。完整IR与无人值守控制器后置。

最新（2026-10-29）：七入口24核/batch96全部Asim结束并执行重建；五入口语义歧义、BRI边界错误、NS数值/输入绑定错误，可信排行榜BLOCKED。冻结批68、兼容回归69单元测试通过。[七入口重建报告](LEGACY_RECONSTRUCTION_REPORT.md)与[归档清单](../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/run_manifest.json)为最新证据；下文早期21/29/45测试和UNSUPPORTED描述保留为阶段历史。

2026-10-30最新：统一任务/计时边界与共用重建核心已落地；独立NS三seed×96batch三路径结构通过，14,298周期包含Gram/初始化/恢复。旧七入口仍未全过，79测试通过，FP16/G4/通信及评分未验收。见[本轮报告](NS_CORRECTED_VALIDATION.md)及[新增算子指南](ADDING_OPERATOR.md)。

2026-10-30后续进展：逐操作FP16/FP32模型三seed×96batch逐FP16位一致，最大任务误差约0.106%；合成RAW/WAR/WAW基础检查已实现，真实G4仍缺事件。106测试通过。[论文报告入口](PAPER_REPORT.md)已归档阶段稿并拒绝未完成时生成最终稿。

2026-10-30日常稳定性更新：真实Core读写/DMA逐fragment流已接通，正负门禁及同输入重复一致；固定版本日常入口可用，121测试通过。见[稳定性验收与命令](RUNTIME_STABILITY.md)。这是受限仿真模型，不是完整硬件G4、全部算子或评分验收。

2026-10-30其他算子更新：七入口完整语义扫描与只读阶段重建已接入；硬件时序认证由负责人后续上板，软件任务不以前者为前置。见[逐算子阶段结果及修正顺序](LEGACY_SEMANTIC_STAGES.md)。阶段通过不等于整算子通过。

2026-11-06：新增独立Cholesky直接求解，以相同重建核心完成跨算法、双精度与不同尺寸验证，增加Agent工作耗时与分阶段脚本计时。见[普适性与耗时报告](RECONSTRUCTION_GENERALITY.md)。

2026-11-06后续：新增LDL/BRI显式映射及实际中间内存见证，四算法复用核心；144测试通过。见[映射验收](LDL_BRI_MAPPING.md)。

2026-11-08：补齐独立Cholesky-block、LDL-block和合法GEMM归约Merge，七个新入口统一任务重建通过。旧Merge异常已由实际Scalar探针和重建拒绝复现；151测试通过。见[完整报告与七入口周期对照](MERGE_AND_REMAINING_MAPPING.md)。

## 当前目标

利用 Agent 自动迭代 NPU 上的通信算子，在通信质量约束下减少周期，并有效利用 AI Core。当前以 MIMO 均衡为试验任务，开展算法结构的外循环搜索和指令、流水调度的内循环优化。

本阶段先整理历史研究、建立 Git 与实验证据管理、重构可信验证边界，再搭建与 Claude Code 并行的 Codex 自动化流程。

原框架计划（后续已有实现，以最新报告为准）：按 [框架实施计划](FRAMEWORK_PLAN.md) 完成策略协议、受信低层程序、数值重建、统一成本、隔离评估与评分闭环。约束共同任务/原语/精度/计时口径，允许算法产生真实的指令数量差异。先用合成小程序和实际操作变异验收框架，再接入NS等真实算子；旧NS接入任务延期，不等于取消研究方向。

## 阅读顺序

先看本轮交付索引：

| 交付 | 内容与状态 |
|---|---|
| [框架实施计划v0.4](FRAMEWORK_PLAN.md) | FW0–FW7依赖、五份共同契约、防漏算/低报成本约束；Asim预筛后数值，部分已实现 |
| [项目现状总报告](STATUS_AUDIT.md) | 环境、研究结论边界、主要差异及实际诊断 |
| [算子状态卡](operators/INDEX.md) | 六族＋Merge七卡，入口/公式/指令/参数/证据/待复现命令 |
| [历史台账说明](history/README.md)、[NS谱系](history/NS_LINEAGE.md) | 49项记录、其中17项NS；10个旧run，缺证显式保留 |
| [验证与自动化审计](audits/VERIFICATION_AUTOMATION_AUDIT.md) | 25项问题，源码位置与负例，旧Claude流程保留 |
| [NS影子执行实施包](audits/NS_SHADOW_EXECUTION_PLAN.md) | 历史设计起点；后续实现与验收见固定论文档案索引 |
| [制品目录与规则](ARTIFACT_CATALOG.md) | 9,140文件hash、211原始/历史制品原位保留映射 |
| [日常工作流](DAILY_WORKFLOW.md) | 已有工具命令、隔离/预算/失败状态，尚缺门禁不得晋级 |
| [结构化台账](../../experiments/README.md) | 历史、盘点、诊断、模板及依赖任务表 |

研究与架构主文档：

1. [项目目标与研究进展](PROJECT_CHARTER.md)：研究定位、外内循环、已知成果与证据等级。
2. [验证体系设计](VERIFICATION_DESIGN.md)：为什么反推算法不足，以及如何建立独立、可执行的验证。
3. [Codex 自动化架构](CODEX_AUTOMATION.md)：控制器、Agent 集群、仿真器、验证器和评分器的接口。
4. [Git 与实验管理](PROJECT_MANAGEMENT.md)：版本、分支、数据归档及结果晋级规则。
5. [分阶段研究计划](ROADMAP.md)：依赖顺序、任务清单、验收标准与首轮实验。
6. [远程部署操作指南](REMOTE_CODEX_SETUP.md)：Windows SSH 配置、Ubuntu Codex 安装认证、桌面远程项目与子 Agent 使用。
7. [方案变更记录](DECISIONS.md)：逐次记录讨论产生的决定与实施状态。

## 本次决策

- 以用户 2026-10-28 对研究路线的说明作为当前目标来源；9 月中期材料保留为历史计划。
- BRI 是现有研究中性能最好的候选；CDL-B 适用性仍待统一口径验证，不预先判定可用或永久不可用。
- 验证从“Agent 声明什么公式”转向“提交的可执行程序实际上计算什么”。
- 以同一任务契约比较不同算法；内部可以求逆、分解或直接求解。
- 正确性、覆盖和复现门禁通过后才给性能分。利用率作为诊断指标。
- Claude 与 Codex 共享任务契约和可信评估器，使用独立候选工作区与运行目录。
- 先做设计和证据整理；未实现的 API、目录和状态机在本文档中均为拟议方案。

## 证据来源与边界

历史源码基点：`5a86e9fc`（2026-07-14）；本轮系统审计基点：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。核对范围扩展为六族及变体、Model/mode/注册/配置、数值和评分脚本、旧Claude技能、README、历史报告/周报、9月材料和磁盘10个旧run。

2026-10-28首轮审计实际执行五项低成本Python诊断与资料完整性检查；当时没有构建或运行C++算子、通信大样本实验，没有证明历史候选可复现，也没有校准Asim与实机误差。全部算子保持legacy_unverified；局部负例复现不等于完整验证。历史材料中的“首次”“唯一”等文献判断未核验。

历史材料入口：`DOCS/zhongqi/kaiti.md`、`DOCS/zhongqi/zhongqi.md`、`DOCS/zhongqi/qianyan_diaoyan.md`。它们原文归档，不因新判断而回写历史。

当前开发状态（2026-10-29）：FW0–FW2已开始实现Python契约、构建器、结构重放和瑞利独立参考，见 [实现范围](../../verification/contracts/README.md)。21项单元测试通过，真实Asim程序适配尚未接通。候选先Asim及性能预筛，随后才数值；筛掉的候选NOT_RUN且无分。

最新运行证据（2026-10-29）：通用程序已接通真实Asim并从实际Instruction导出重放；29单元测试与7项真实Asim门禁达到预期。旧NS/Cholesky小样例运行成功，但数值语义不足明确UNSUPPORTED。见 [框架实际进度](FRAMEWORK_PROGRESS.md)。FW0–FW2仍in_progress，完整view/精度/G4/通信/晋级未完成。

2026-10-29多核更新：默认24核、batch96（24的倍数），Cube16³；24/96全batch的真实Asim→预筛→三路径数值闭环已运行，45项单元测试通过。计时为给定A/b/X0/C内核，仍非FP16/G4/通信统计验收。详见 [24核验证报告](ASCEND24_VALIDATION.md)。
