# 2026-11-12 最新交接：便携构建与真实新算子已验收

2026-11-12补充交付：Claude完整中文使用手册已覆盖部署/新算子/优化迭代/OpenCode等平台接入；本目录expert_delivery/asim-claude-guide-v1.tar.gz为无数据离线源码包，已完成解压校验和重建回归，见expert_delivery/README.md。固定标签asim-claude-guide-v1-20261112，源码675cd233；最新手册及验收报告在精简分支。临时验证数据已清，旧研究不动。OpenCode尚未实测，本轮没有再冷编译，不改算法。


以下S0记录保留为历史快照，当前状态以本节及精简分支验收报告为准；原研究源码/BRI默认基线没有切换。

- core `28d3ea7a`；Codex `fae35ed2`；Claude `e62e9b72`，三个分支已推送，客户端相对core仅agent_profiles/active.json不同。
- 固定发行标签：`asim-claude-demo-v1-20261112`、`asim-codex-demo-v1-20261112`。Linux双全新clone冷构建/六类门禁通过；WSL2仍未实测。
- Claude Code＋DeepSeek真实生成新的仿射GEMM；审计/编译错误均反馈模型修复。九场景864batch真实Asim/重建与独立FP16逐位一致，最大FP64任务误差0.0353%；任务内配对评分1.0，不声明优化晋级或无线认证。
- 真实误读输入负例Asim685周期/事件通过但96batch数值全部拒绝、无评分；原六类在新二进制/共用runner下全部再次通过，周期不变。254测试中252通过、2可选依赖跳过，另8项框架测试通过。
- 新算子仍是受监督闭环；正式跨路径/跨candidate基线、新runtime冻结、Codex同任务真实生成及持久控制器待后续。调用墙钟及失败保留，纯思考时间未知记null。

专家使用[Claude固定版本使用/原理](https://github.com/nanajiang96300/Asim/blob/asim-claude-demo-v1-20261112/DOCS/research/CLAUDE_NEW_OPERATOR.md)、[构建说明](https://github.com/nanajiang96300/Asim/blob/asim-claude-demo-v1-20261112/BUILDING.md)、[全部验收证据](https://github.com/nanajiang96300/Asim/blob/asim-claude-demo-v1-20261112/experiments/claude_operator/2026-11-11-v1/README.md)。本机对应/home/nanajiang/asim-worktrees/asim-core-v1；原件在本工程results/claude_operator_demo/2026-11-11-v1/，SHA清单在精简分支报告中。只清理本次新clone可再生成构建缓存；研究原件与未跟踪文件保留。

```bash
git clone --depth 1 --single-branch --branch codex/asim-claude-v1 https://github.com/nanajiang96300/Asim.git
cd Asim
python3 scripts/bootstrap.py --jobs 1
# 后续生成新候选/评估按CLAUDE_NEW_OPERATOR.md，不以CLI回答OK代替新算子验收。
```

# 2026-11-11 精简基线与双客户端交接

本目录仍是保留完整资料的原研究工作区，不在此切换精简分支或批量删除原数据。新开发在独立工作区进行。

| 版本 | GitHub分支 | 提交 | 本机工作区 |
|---|---|---|---|
| 完整研究归档 | research/codex-transition-20261028，冻结标签archive/asim-research-20261111 | 归档dde9e866；此分支后续仅追加交接索引 | /home/nanajiang/Asim |
| 精简共享核心 | codex/asim-core-v1 | 67cef8a2 | /home/nanajiang/asim-worktrees/asim-core-v1 |
| Codex开发入口 | codex/asim-codex-v1 | 8b44edb1 | /home/nanajiang/asim-worktrees/asim-codex-v1 |
| Claude Code开发入口 | codex/asim-claude-v1 | 75217b23 | /home/nanajiang/asim-worktrees/asim-claude-v1 |

仓库：[nanajiang96300/Asim](https://github.com/nanajiang96300/Asim)。三个新分支和两个标签已推送并核对远程SHA；main保持5a86e9fc，未强推、未改写历史。原五个候选/基础设施分支也已推送，避免台账引用的commit只存在本机。HTTPS缺凭据，使用原有SSH身份，origin保留HTTPS fetch并设置SSH push；没有将用户认证写入仓库。

## 核心设计与当前边界

两客户端共用Asim、任务/精度/计时、注册表、数值重放、独立算法/MMSE参考、事件检查和评分。当前两个分支只比共享核心多一个agent_profiles/active.json，默认选择各自客户端，不复制一套算法或门禁。

当前是S0受监督入口：scripts/agent_entry.py的doctor/context/evaluate；context只是任务准备、不是OS沙箱或完整冻结器；evaluate转发原重建runner，Asim/事件/性能预筛后才数值。Claude Code通过现有用户配置DeepSeek API；不复制settings、密钥或账户，不把CLI自报评分视为验收。

可执行用法、S1–S4依赖、预算/失败与清理规则，以新核心[AGENT_WORKFLOW.md](https://github.com/nanajiang96300/Asim/blob/codex/asim-core-v1/DOCS/research/AGENT_WORKFLOW.md)为准。本轮验收、清理manifest和负例在该分支experiments/maintenance/2026-11-11-slim-v1/。本目录experiments/maintenance/2026-11-11-handoff-v1/profiles.json保存分支默认选择/差异复核。

## 清理与验证

只从新精简树移除12,630文件/75,135,973逻辑字节（约71.66MiB）：无关PPT工具链、未启用的ONNX上游测试数据及旧Timeloop输出；逐文件SHA和恢复commit已记录。原研究目录、全部结果大制品与未跟踪资料不动，不能将逻辑瘦身数字称原服务器空间回收。旧Claude自动循环/六族变体/研究报告/论文证据保留。Git历史不裁剪，企业端建议浅克隆选定分支。

- 原250项验证测试全部通过，无跳过；264个src/verification文件SHA与完整树一致。
- 共用新入口真实NS：Rayleigh seed42、batch96、Nr64/Nt16/RHS8、λ.1、K16，7950周期、逐FP16位一致、最大MMSE误差0.105726%。复用原已封存binary，不宣称精简树新冷构建。
- 未知算子返回UNSUPPORTED/NOT_RUN，不启动Asim；未知客户端拒绝、已有任务不覆盖；两客户端策略/门禁顺序一致。
- Claude Code2.1.272使用配置模型deepseek-v4-pro，无工具/无持久会话的短API请求4.28秒通过且响应精确校验。只证明连接可用，不证明代码生成闭环；Codex0.165.0本轮只查CLI和共用入口，未另调API。

## 下一步：先S1，再S2

S1将已验收专家包的portable CMake/离线依赖整合进Git core，去掉原根CMake的Conan/build固定假设；两个全新clone/独立build均跑六类门禁后才叫独立部署完成。当前专家离线部署继续使用expert_delivery/asim-expert-v1.tar.gz，不能称Git精简分支已经替代该包。

S2再分别在Codex和Claude分支完成一次“协调策略→受限候选→冻结→构建→Asim→重建/同算法/MMSE→适用通信→评分/归档”监督闭环，补候选写入范围保护和预算/超时负例。S3统一晋级/干净环境复现；持久无人值守控制器S4后置。旧CDL失败、BRI内循环基线和未晋级状态不改变。

专家取分支示例：

```bash
git clone --depth 1 --single-branch --branch codex/asim-claude-v1 https://github.com/nanajiang96300/Asim.git
cd Asim
python3 scripts/agent_entry.py doctor
claude
```

此示例选择源码/Agent入口，不自动安装依赖或声称构建通过；完整部署按S1后更新的说明执行。
