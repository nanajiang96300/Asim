# Asim 协作上下文

工程根目录：`/home/nanajiang/Asim/`。当前目标与计划从 `DOCS/research/README.md` 进入。

研究重心是 Agent 自动迭代 NPU 通信算子，充分利用 AI Core；当前任务为 MIMO 均衡。外循环探索 Cholesky、LDL、分块、NS、BRI 等算法；内循环优化各算法的指令与流水。NS 历史实验较详细。BRI 为负责人确认的当前最优候选，CDL-B 仍待统一验证。

区分负责人陈述、历史报告、代码存在、已复现和完整验证，不把旧周期表当成统一排行榜。

Codex 自动化框架已有受限结构实现：统一任务/计时、Asim先行与共用重建核心、独立NS/Cholesky/LDL/BRI及Block/Merge新入口三路径结构及显式FP16/FP32数值模型验证；真实Core事件的保守模型检查与固定日常入口已验收；硬件级认证由负责人后续上板，不作为软件重建前置；通信与完整晋级仍未完成。`DOCS/research/` 中其余拟议接口不代表已有实现，以最新验收报告为准。保留 Claude 流程；不因迁移而覆盖旧结果或批量清理未跟踪文件。

当前质量驱动架构见 `DOCS/research/QUALITY_DRIVEN_WORKFLOW.md`；写算子前已有独立FP16选K/留出筛选，不能把算法筛选当算子或通信验收，候选仍先Asim后重建。受监督的计划冻结/执行绑定已验收，已有Q3a工程试验下瑞利/QPSK相对BER/原始EVM与留出置信门禁；Q3b已有来自实际I/Q响应的高斯线性SE代理有限样本诊断；Q3c七新入口三SNR的完整未编码QPSK链路及独立审计已通过，见experiments/qpsk_full/2026-11-08-v1/README.md；总体SE/BLER、CDL-B及无人值守协调器仍待完成。本阶段求逆/均衡精度不以信道编码为前置；真实CDL-B静态频域输入与K/误差研究见`DOCS/research/CHANNEL_DOMAINS.md`，不得将旧Kronecker近似当CDL-B认证。论文资料固定检索 `DOCS/research/paper_archive/README.md`，按版本保存快照和证据，不覆盖旧归档。

修改算子之前阅读 `CLAUDE.md`、`DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md` 和适用的 `.claude/skills/op-flow/SKILL.md`，落实设计、公式、构建、审计与验证要求。Codex 中不能仅声明已调用不可用的 slash command；实际执行并记录对应步骤。

新候选使用独立工作区、build 和日志；不在共享主工作区用 stash/apply/pop 并行评估。评估器、测试和评分配置与候选修改范围分离。

新验证以任务契约、真实可执行语义和独立参考为准。未知操作、缺失 trace、缺少输入或未覆盖精度语义不能当作通过。Agent 的自报分数不是验收证据。

仅在既定授权范围内修改、提交或运行实验；每次保存候选与结果所依赖的版本。保持远程路径、凭据及生成制品管理清晰，不将密码或私钥写入仓库。

BRI原理与新Python研究见`DOCS/research/BRI_PRINCIPLES.md`：P=D⁻¹与T=PA必须区分，Y迭代以X=YP右恢复；复块/实块、固定步长/Chebyshev和精度存储策略均须显式冻结。该研究不替代原算子Asim/重建验收。

当前BRI后续优先现有FP16张量/原语，不修改Asim来增加混合精度；见`DOCS/research/FP16_FIRST_PLAN.md`。FP32参考/临时数值模型不等于已支持可选FP32算子。混合精度后置，未经新指令不作为前置。
