# Asim：Claude Code / DeepSeek 入口

先读取AGENTS.md、DOCS/research/README.md、DOCS/research/AGENT_WORKFLOW.md。与Codex共用Asim、注册表、任务/精度/计时契约、重建器、独立参考与评分规则。用户在机器上配置Claude Code兼容API；仓库不保存认证、模型账户或settings.local.json。

算子任务执行.claude/skills/op-flow/SKILL.md、op-dev、audit-operator及V3标准。先设计/固定算法参数与校准/留出，再候选源码、实际指令语义、独立构建和审计，先Asim后数值。未知语义UNSUPPORTED，不为过门禁改参考/门限；FormulaLogger不是数值证据。2-vCPU上编译/Asim串行。

`python scripts/agent_entry.py doctor`检查CLI；context准备冻结任务；evaluate转发到共用重建runner。具体命令/阶段验收见AGENT_WORKFLOW。当前受监督，尚未实现持久无人值守协调器。任务路径约束尚非OS沙箱。

新clone使用独立build和日志，不复用共享可写目录。旧pin只按其冻结版本复现，不能复制服务器凭据来部署。便携源码构建S1已完成两个全新Linux clone验收，入口见BUILDING.md；WSL2仍待实测。

历史Claude iterate、eval-patch/opt-round/verify技能仍保留，仅作旧流程，不自动使用旧分数晋级。精简前原CLAUDE文档可从archive/asim-research-20261111恢复。

新算子受监督示例：先读DOCS/research/CLAUDE_NEW_OPERATOR.md；scripts/claude_codegen.py只收三文件提议，scripts/affine_acceptance.py负责真实Asim/重建/独立参考/任务内评分。生成成功不等于验收，不能改参考或门限。

专家完整使用说明：DOCS/CLAUDE_CODE_USER_GUIDE.md，覆盖源码包初始化、六类运行、新算子、内循环优化及平台扩展。
