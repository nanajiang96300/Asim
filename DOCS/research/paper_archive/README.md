# 论文资料固定检索入口

**固定位置：`DOCS/research/paper_archive/README.md`。**
后续论文方法、设计取舍、负例与实验引用均从这里检索；机器目录见 [catalog.json](catalog.json)。
当前架构活文档为 [QUALITY_DRIVEN_WORKFLOW](../QUALITY_DRIVEN_WORKFLOW.md)，设计变更写入 [DECISIONS](../DECISIONS.md)。

| 主题/检索词 | 固定归档 | 可支持的论述 | 状态 |
|---|---|---|---|
| 真实指令、影子执行、三路径、FP16、Merge负例、跨算法 | [重建方案与证据索引](2026-11-08-reconstruction-v1/README.md) | 从公式声明到真实指令语义；七个新入口的受限模型验收 | INTERIM_MODEL_VALIDATION |
| 最小K、NS、BRI、MMSE、SE、质量契约、Agent约束 | [精度驱动架构与实施记录](2026-11-08-quality-driven-v1/README.md) | 算法先筛选、Asim先于候选重建、质量与一致性分离 | DESIGN_AND_OFFLINE_SCREEN |
| Q2、计划绑定、版本一致、Asim预筛、数值NOT_RUN | [计划交接验收](2026-11-08-quality-handoff-v1/README.md) | 192batch真实Asim与重建、真实性能拒绝、封存失败修正 | SUPPORTED_PLAN_HANDOFF_GATES_PASSED |
| 最早论文阶段稿与机器完成门禁 | [2026-10-30阶段稿](../paper_reports/2026-10-30-interim-v1/REPORT.md) | 当时NS验证方法和有限数据 | HISTORICAL_INTERIM |

## 固定归档规则

1. 每次里程碑使用 `YYYY-MM-DD-主题-vN/`，已发布版本不覆盖；纠错新增版本，记录 supersedes。
2. 活文档只保留一个当前标准；归档保存当时快照及 SHA-256，明确其历史状态。
3. 文本快照不修改内容。其相对链接可能仍按原目录解析：通过 manifest 的 source 路径定位原件；不能依赖快照内链接找证据。
4. Git 跟踪设计、索引、实验摘要和小型 manifest。大型输入/trace/二进制留在 `results/<独立campaign>/`；记录路径、大小、SHA-256，不批量搬迁或删除。
5. 每项实验引用绑定候选/参考/评估器版本、输入、配置、精度、计时 scope、工具链和运行器；缺项写缺证。
6. 区分负责人陈述、历史报告、代码存在、模型已复现、正式通信通过及硬件验证。禁止把阶段通过写成全部完成。
7. Agent 时间只记录可观测墙钟与工具/脚本阶段。不可观测的纯思考时间为 null，不以残差冒充。

本次仅复制小型文档快照，历史报告及原始数据原位保留；没有重跑或升级全部历史结论。
