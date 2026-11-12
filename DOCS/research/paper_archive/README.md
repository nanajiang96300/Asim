# 论文资料固定检索入口

**固定位置：`DOCS/research/paper_archive/README.md`。**
后续论文方法、设计取舍、负例与实验引用均从这里检索；机器目录见 [catalog.json](catalog.json)。
当前架构活文档为 [QUALITY_DRIVEN_WORKFLOW](../QUALITY_DRIVEN_WORKFLOW.md)，设计变更写入 [DECISIONS](../DECISIONS.md)。

| 主题/检索词 | 固定归档 | 可支持的论述 | 状态 |
|---|---|---|---|
| 真实指令、影子执行、三路径、FP16、Merge负例、跨算法 | [重建方案与证据索引](2026-11-08-reconstruction-v1/README.md) | 从公式声明到真实指令语义；七个新入口的受限模型验收 | INTERIM_MODEL_VALIDATION |
| 最小K、NS、BRI、MMSE、SE、质量契约、Agent约束 | [精度驱动架构与实施记录](2026-11-08-quality-driven-v1/README.md) | 算法先筛选、Asim先于候选重建、质量与一致性分离 | DESIGN_AND_OFFLINE_SCREEN |
| Q2、计划绑定、版本一致、Asim预筛、数值NOT_RUN | [计划交接验收](2026-11-08-quality-handoff-v1/README.md) | 192batch真实Asim与重建、真实性能拒绝、封存失败修正 | SUPPORTED_PLAN_HANDOFF_GATES_PASSED |
| Q3a、QPSK、BER、原始EVM、MMSE配对、信道置信界 | [通信试验验收](2026-11-08-communication-v1/README.md) | 七入口历史只诊断、新NS batch96双噪声留出通过；SE未支持 | Q3A_TRIAL_GATES_PASSED |
| Q3b初始v1、I/Q响应、共轭分量、SE代理 | [初始响应试点](2026-11-08-response-proxy-v1/README.md) | 历史原件保留；v2进一步补强启动前评估器绑定 | HISTORICAL_BOUNDED_PROXY |
| Q3b当前v2、冻结评估器、I/Q响应、线性SE代理 | [响应与代理v2验收](2026-11-08-response-proxy-v2/README.md) | 200测试；375次正例及100次负例重跑，不是总体SE认证 | BOUNDED_RESPONSE_PROXY_GATES_PASSED |
| Q3c、完整未编码QPSK、BER/SER、真实重放、独立判决 | [完整QPSK验收](2026-11-08-qpsk-full-v1/README.md) | 七新入口三SNR、212测试与412万bit独立复核；有限范围通过 | UNCODED_QPSK_SUITE_CHECKED |
| CDL-B、K曲线、精度平台、BRI谱半径、清理 | [信道与迭代研究](2026-11-08-cdlb-iteration-v1/README.md) | 21组6通过/1数值失败/14拒绝；真实失败及FP16/FP64曲线 | CDLB_QUALITY_NOT_VERIFIED |
| BRI原理、P/T、逆矩阵、Chebyshev谱界、K/精度 | [原理与CDL-B仿真](2026-11-08-bri-principles-v1/README.md) | 正确公式与独立谱审计；有限精度/模型拒绝保留，非算子通过 | BRI_PRINCIPLES_CHECKED_WITH_PRECISION_LIMITATIONS |
| FP16优先、不改Asim、Gram量化、参考误差拆分 | [FP16约束修订](2026-11-08-fp16-first-v1/README.md) | 只验证误差来源，不签发新候选；混合精度后置 | REFERENCE_DIAGNOSTIC_AND_FP16_FIRST_DESIGN |
| FP16、隐式Gram、补偿、独立留出、冻结K | [原语数值可行性](2026-11-08-fp16-implicit-v1/README.md) | 直接MMSE/逆子任务过2%；逆后应用失败，未Asim验收 | FP16_PRIMITIVE_NUMERIC_HOLDOUT_CHECKED |
| 最早论文阶段稿与机器完成门禁 | [2026-10-30阶段稿](../paper_reports/2026-10-30-interim-v1/REPORT.md) | 当时NS验证方法和有限数据 | HISTORICAL_INTERIM |

2026-11-09：[BRI后续方向与状态快照](2026-11-09-bri-direction-v1/README.md)，仅设计与既有证据复核；未新增算子验收。

2026-11-09：[隐式FP16实际接入与质量失败](2026-11-09-bri-implicit-v1/README.md)：真实重建/周期/流水/耗时与门禁完成，新CDL20dB稳定性失败，禁止晋级；240测试与七显式入口回归。

## 固定归档规则

1. 每次里程碑使用 `YYYY-MM-DD-主题-vN/`，已发布版本不覆盖；纠错新增版本，记录 supersedes。
2. 活文档只保留一个当前标准；归档保存当时快照及 SHA-256，明确其历史状态。
3. 文本快照不修改内容。其相对链接可能仍按原目录解析：通过 manifest 的 source 路径定位原件；不能依赖快照内链接找证据。
4. Git 跟踪设计、索引、实验摘要和小型 manifest。大型输入/trace/二进制留在 `results/<独立campaign>/`；记录路径、大小、SHA-256，不批量搬迁或删除。
5. 每项实验引用绑定候选/参考/评估器版本、输入、配置、精度、计时 scope、工具链和运行器；缺项写缺证。
6. 区分负责人陈述、历史报告、代码存在、模型已复现、正式通信通过及硬件验证。禁止把阶段通过写成全部完成。
7. Agent 时间只记录可观测墙钟与工具/脚本阶段。不可观测的纯思考时间为 null，不以残差冒充。

各版本仅归档小型快照与证据清单；大型原始数据保持原位。本次完整QPSK实验不升级未覆盖的历史结论。

2026-11-10：[固定HB原生验收与谱界/选K](2026-11-10-fixed-hb-v1/README.md)：新CDL-B三SNR、192独立H、576batch逐FP16位一致与QPSK门禁；旧β-only失败保留，无硬件/性能晋级。

2026-11-10：[M1通用重建内存优化](2026-11-10-memory-v1/README.md)：K174/batch96峰值1.88GiB、九入口精确回归、新Asim冒烟与QPSK接入，249测试；旧失败与超预算试验保留。

2026-11-10：[BRI内循环首轮与基线/候选台账](2026-11-10-bri-inner-v1/README.md)：三档周期减少5.00–6.93%、精度位一致、新CDL/QPSK和基线配对重放；默认基线保留，未M3晋级。


2026-11-10 工程可复现资料：[专家源码部署包与验收](../../../experiments/distribution/2026-11-10-v1/README.md)，含离线依赖/六类同任务重建/目录迁移/源码与归档SHA；[交付索引](../../../expert_delivery/README.md)。仅规定Rayleigh软件部署验收，非CDL/通信/硬件或跨主机认证。

2026-11-12：[Claude Code＋DeepSeek新算子与通用重建固定检索入口](2026-11-12-agent-codegen-v1/README.md)，真实生成/编译修复/864batch与实际错误指令拒绝；完整证据位于固定精简发行标签。
