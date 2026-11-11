# 精简共享基线 / 双客户端准备 S0

2026-11-11（香港）。完整版本dde9e866及五个独立候选/基础设施分支已推送origin；归档标签archive/asim-research-20261111，GitHub main未改。HTTPS无凭据，使用原有SSH身份推送；未复制/记录认证内容。

## 本轮完成

- 在独立worktree建立codex/asim-core-v1，不修改原研究目录。移除12,630个文件，逻辑字节75,135,973（约71.66MiB）：PPT工具链、未启用的ONNX backend测试数据、旧Timeloop生成输出。恢复清单removal_manifest.json；全部可从远程归档恢复。此数字是精简树的内容减少，不是原服务器磁盘回收量；Git历史仍保留。
- 六族及变体、src/verification完整保留；264个核心/验证文件对照原目录SHA全相同。旧Claude iterate/skills、研究报告、论文快照和实验台账保留；原目录results及未跟踪资料未清理。
- 新共享入口scripts/agent_entry.py：doctor、context、evaluate。客户端共用策略/门禁，evaluate只转发到原trusted runner。任务准备不是OS沙箱或完整计划冻结器；受监督门禁仍由原planning/daily和独立审查执行。
- 新CLAUDE.md对齐当前标准，撤销旧文档把FormulaLogger/旧Score当默认路径的引导；旧版本在归档。AGENT_WORKFLOW规定共享core、两客户端分支与S1–S4验收。

## 验证

250项原验证测试全部通过，无跳过。NS实际Asim→事件检查→重建→独立FP16/FP64 MMSE通过：batch96、Nr64/Nt16/RHS8、λ.1、Rayleigh seed42、K16、7,950周期；完整输出逐位一致，最大误差0.105726%，低于2%。使用原已封存binary（SHA见ns_shared_entry.json），未声称在精简树重新构建二进制或跨环境验收。

负例：未知算子UNSUPPORTED，返回码1，numeric NOT_RUN且未启动Asim；未知客户端拒绝、已有任务文件不覆盖。Codex/Claude上下文的policy SHA/门禁顺序相同。

CLI：Codex0.165.0、Claude Code2.1.272。Claude实际单次API短请求通过（配置模型deepseek-v4-pro，约4.28秒，响应精确校验），禁用全部工具/持久会话；只记录不含凭据的摘要。此项是连通性检查，不是模型编写算子或执行工具的验收。Codex本轮仅检查CLI/共用上下文，不额外调用其API。原始API响应/本轮trace/输入只在本worktree results/maintenance中保留，不推送。

## 后续依赖

1. S1将专家包已验收portable构建整合到Git精简core，解决原根CMake的Conan/build路径假设，再在两个全新clone各构建/跑六类；当前企业部署继续用既有离线包。
2. S2在两个客户端分支各完成一次候选闭环；限制候选路径、保护受信评估器/参考/门限、预算/超时、日志/补丁/模型计量和负例必须真实验收。现有上下文规则不冒称强隔离。
3. S3完成适用通信/统一评分/干净环境晋级；持久跨天控制器S4后置。

分支初始只设置客户端profile，共享算法与门禁不得分叉。旧CDL失败及BRI内循环基线/未晋级状态不升级。完整命令见DOCS/research/AGENT_WORKFLOW.md。
