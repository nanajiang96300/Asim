# 研究证据目录

本目录保存可由Git追踪的摘要与清单，不保存大trace副本，不代表自动化控制器已经实现。

| 路径 | 内容 |
|---|---|
| [history/ledger.jsonl](history/ledger.jsonl) | 49项历史记录，NS17项；真实缺证为null，trusted_metrics/score均null |
| [history/legacy_runs.json](history/legacy_runs.json) | 10个既有run的配置、摘要解析状态、版本与86个原制品的内容摘要 |
| [history/sources.json](history/sources.json) | 19个关键来源与历史Git线索 |
| [history/audit_check.json](history/audit_check.json) | 子Agent进行的来源/制品结构校验；不是算子数值验证 |
| [inventory/2026-10-28-baseline/manifest.json](inventory/2026-10-28-baseline/manifest.json) | 9,140文件清单及211项保留映射的自校验摘要 |
| [audit/2026-10-28/diagnostics.json](audit/2026-10-28/diagnostics.json) | 5项已实际执行的Python诊断；明确NOT_PERFORMED算子验收 |
| [audit/2026-10-28/metadata_check.json](audit/2026-10-28/metadata_check.json)、[tool_checks.json](audit/2026-10-28/tool_checks.json) | 资料交叉检查及清单工具的损坏/覆盖拒绝负例 |
| [tasks.json](tasks.json) | 依赖顺序、状态、允许范围和验收标准 |
| [templates/campaign.json](templates/campaign.json) | 固定任务、作用域与预算模板，未补齐不能执行 |
| [templates/run_manifest.json](templates/run_manifest.json) | 源码/程序/输入/工具链/门禁绑定模板；默认NOT_RUN，score=null |

历史cycle_kind包括reported_measured、estimated等，以逐项字段为准；复现命令null表示无法忠实重现，不生成“近似等同”实验冒充原run。历史score保留原值但不作为当前受信分数。

2026-10-29新增FW0–FW7通用框架计划（全部planned），原S*和真实算法pilot延期；当前不修改具体算子。台账中的旧审计check是当时快照，不因任务数量变化重写历史结果。最新计划见 [FRAMEWORK_PLAN](../DOCS/research/FRAMEWORK_PLAN.md)。

现阶段工具：`python3 scripts/research_inventory.py create/check ...`、`.venv/bin/python scripts/research_audit_checks.py --output <new-file>`、`python3 scripts/research_validate.py`。具体命令与失败边界见 [日常工作流](../DOCS/research/DAILY_WORKFLOW.md)。

原制品继续保留在results等路径；清单与Git都不能恢复一个缺失的原始输入。将来归档迁移按 [制品规则](../DOCS/research/ARTIFACT_CATALOG.md) 校验副本，不自动删除原件。候选提交与结果提交分开，清单引用完整来源SHA，禁止自引用自身提交。
