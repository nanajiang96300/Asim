# 重建方案归档 v1

主题：真实指令语义、独立三路径参考、逐操作精度、跨算法复用及性能异常负例。
状态：**INTERIM_MODEL_VALIDATION**，不是全部通信验证完成。

[manifest.json](manifest.json) 保存捕获版本、15份小型文本快照与历史 campaign 清单的 SHA-256。
快照按原始工程路径保存于 `snapshots/`；内容逐字保留，不回写历史报告。
阅读原件请按下表进入；某些早期“待实现”描述仅代表当时状态。

| 问题与解决路线 | 原始材料 | 证据边界 |
|---|---|---|
| 人工公式不能证明真实指令；需要索引/切片/常量/初始化/读写语义 | [NS最小实施计划](../../audits/NS_SHADOW_EXECUTION_PLAN.md)、[验证体系](../../VERIFICATION_DESIGN.md) | 设计起点，不把拟议接口当实现 |
| 统一 H/Y→xhat 和完整成本、独立NS候选 | [NS三路径](../../NS_CORRECTED_VALIDATION.md) | 早期结构/精度逐阶段验收 |
| 实际Core事件、保守读写检查、固定日常入口 | [稳定性](../../RUNTIME_STABILITY.md) | 软件保守模型，板级认证外置 |
| 重建核心跨算法复用 | [Cholesky普适性](../../RECONSTRUCTION_GENERALITY.md)、[LDL/BRI](../../LDL_BRI_MAPPING.md) | 独立新候选；旧程序失败未抹除 |
| 全部七个新入口；Merge异常快的漏计工作量 | [Block/Merge及周期报告](../../MERGE_AND_REMAINING_MAPPING.md) | 模型周期、统一身份；不是旧2.34倍加速背书 |
| 历史七入口问题和论文完成保护 | [旧入口报告](../../LEGACY_RECONSTRUCTION_REPORT.md)、[论文入口](../../PAPER_REPORT.md) | 旧失败保留，最终报告门禁不能绕过 |

可写入论文的方法结论：实际序列而非人工公式作为重放事实来源；分离实现一致性与算法任务质量；欠定义语义失败关闭；不同算法共享解释核心。
不可据此宣称：历史全部算子正确、SE/BER/CDL-B已通过、实机周期准确、无人值守闭环或正式评分已完成。
