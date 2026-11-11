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

FP16隐式MMSE早期独立原语/留出研究见`experiments/fp16_implicit/2026-11-08-v3/README.md`，仅证明当时软件样本；实际接入与新增留出的当前状态见下段。显式逆目标通过不代表逆后相乘MMSE通过，原算子状态不升级。

隐式FP16实际接入及重建性能见`experiments/bri_implicit/2026-11-09-v1/README.md`：3档真实Asim模板288batch、新CDL576batch均同算法FP16位一致，但20dB新192-H留出1个样本误差54.3238倍，候选不可晋级。只阻尼β缩小稳定域（坏点谱max7.6388，末步稳定上界7.6052）；下一步联动设计α/β及谱界，先新校准选K再全新留出，坏点只回归。当前runtime固定82文件，七显式入口F16回归通过；新隐式为实验入口，默认日常任务是瑞利，不能据白名单推断CDL认证。op-flow/op-dev/audit-operator已按实际Instruction与数值硬门禁更新。

固定HB隐式新入口bri_stable_hb_fp16_v1已在声明FP16存储/FP32临时/保留次正规模型下完成K16/53/174与三SNR、192新独立CDL-H/576batch实际重建及未编码QPSK门禁；见experiments/fixed_hb_native/2026-11-09-v2/README.md。旧β-only失败不升级；新HB非性能晋级/硬件认证，冷启动跨环境复现未完成。日常新pin codex_runtime_stable_hb_fp16_v1.json；旧82文件pin保留，当前live pin以新清单为准。下一步先解决高K通用重建内存，再新校准研究更紧谱界/指令优化，不在H590留出调K。两轮78旧trace无损gzip回收9.08GB，按storage_cleanup台账显式restore，禁止静默改旧manifest。

M1通用重建内存优化已完成：HB K174/batch96峰值1.88GiB，九入口封存Asim制品回归的统计字节/事件报告/完整输出位一致，249测试；新live runtime为codex_runtime_memory_v1.json（91文件），原90文件pin不覆盖，旧契约按冻结Git版本复现。证据experiments/memory/2026-11-10-v1/README.md。下一步M2使用全新校准集研究更紧谱界/复用/流水，H590只回归不调K；不升级旧CDL失败，不改变K/精度/门限。


M1解析器边界补丁v2：补充小数/指数顶层元数据跨chunk读取测试并修正误拒绝；实际九入口的instructions对象及header整体解码路径不变。新live pin为orchestrator/codex_runtime_memory_v2.json（91文件），v1清单保留。最终250测试及v2默认NS新Asim通过，五类关键文件与v1逐字节一致；HB/九入口完整内存实测属于v1原验收，未冒称在v2重跑全部。见experiments/memory/2026-11-10-v1/parser_v2_manifest.json。

BRI内循环采用同Model/mode+独立candidate分支/worktree/build，只有新算法/契约才新入口。基线bri-hb-base-v1保持主工作区/默认runtime，首轮bri-hb-inner-0001(commit88acdaed)状态槽轮换三档周期17488/44128/131248，比基线减少5.00/6.35/6.93%，公式/K/精度不变且位一致；新H61001/2三SNR与QPSK、基线同输入配对重放通过。见experiments/optimization/bri/inner-0001/README.md与DOCS/research/OPTIMIZATION_TRACKING.md，ledger.jsonl仅追加、comparison.csv从台账重建。H610留出不再调参，M3未晋级，score=null，默认基线不自动切换。


2026-11-11双客户端S0：完整研究已远程归档archive/asim-research-20261111；精简core=67cef8a2，Codex=8b44edb1、Claude=75217b23，工作区/home/nanajiang/asim-worktrees/asim-{core,codex,claude}-v1。原目录保留证据；后续新开发在精简独立工作区。共享数值门禁/监督入口已验收，250测试、实际NS和Claude/DeepSeek无工具API检查通过；不是新clone冷构建或完整模型代码生成闭环。下一步S1整合portable构建，再S2客户端候选闭环；上下文规则尚非OS沙箱。见DOCS/research/DUAL_CLIENT_HANDOFF.md及新core的AGENT_WORKFLOW.md。
