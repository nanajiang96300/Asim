# Codex / Claude Code 共享工作流与版本计划

2026-11-11，第一阶段：精简共享基线与受监督入口。Claude Code使用负责人机器上配置的DeepSeek兼容API，Codex使用其现有环境。API密钥/地址/模型配置属于用户环境，不进入源码、任务或制品。

## 分支与职责

- `research/codex-transition-20261028`及`archive/asim-research-20261111`：精简前完整研究快照；GitHub main不自动更新。
- `codex/asim-core-v1`：共享核心。Asim、任务/精度/计时契约、注册表、数值重建、独立参考、事件检查、质量与评分规则均只维护一套。
- `codex/asim-codex-v1`、`codex/asim-claude-v1`：从同一精简基线分支，初始只通过`agent_profiles/active.json`选择使用入口，不复制算法或评估器。后续只在各自分支开发客户端适配；共用修复先回core，再合入两分支。
- 算子候选另建独立candidate/worktree/build；客户端分支不作为算法身份。每轮记录client、实际配置模型、源码commit、任务/K/精度、周期/流水、质量与失败证据；客户端自报分数不验收。

## 现在可执行的入口

```bash
python3 scripts/agent_entry.py doctor
# task.txt由负责人写明目标、允许路径、预算、参数和验收范围
python3 scripts/agent_entry.py context --client claude --task-file task.txt --output work/task-001.json
# Codex用同一命令，仅替换--client codex；output必须是新文件。
claude
# 在对话中要求读取AGENTS、CLAUDE和生成的任务JSON；执行op-flow实际步骤。
```

context只准备本地任务，不会调用模型或授权修改受信代码；doctor只检查CLI，不声称API连通。Claude的slash skill可以读取，但真正验收由共享脚本完成，不能用旧`/opt-round`的Score替代数值硬门禁。Codex按相同文件执行，不声明不可用slash命令已运行。

统一评估：`python scripts/agent_entry.py evaluate -- <verification.reconstruction.runner的原参数>`。必须明确binary/config/cwd/output/profile/operator/K；runner负责先Asim、事件和性能预筛，再实际重放/独立同算法/FP64 MMSE。中间状态及UNSUPPORTED不改成PASS，失败不自动加K/重试。`context`中的预算/路径约束是受监督任务协议，尚非OS沙箱，不能宣称已阻止恶意Agent修改评估器；冻结和独立评审仍必须执行。

完整受监督计划冻结/执行继续使用verification/planning和既有daily入口。旧runtime pin是原研究环境的冻结证据，不因分支/路径迁移改写，不声称在新clone直接可用。旧Claude iterate/skills原样保留作为历史流程，默认不运行其自动循环。FormulaLogger/DAG自一致不代替实际Instruction数值。

## 分阶段实施与验收

| 阶段 | 改动 | 退出条件 |
|---|---|---|
| S0 本轮 | 远程保存、清理清单、共享受监督入口、两个开发分支 | Git远程SHA匹配；src/验证核心字节不变；250测试和实际NS共用门禁；CLI/API测试分别记录 |
| S1 构建统一 | 将已验收专家包的离线依赖/portable CMake整合为版本化依赖获取，不依赖服务器缓存路径 | 两个全新clone、独立build都能构建并跑六类；用户密钥不复制；WSL另测 |
| S2 客户端适配 | coordinator冻结任务，Claude/Codex在独立候选工作区按方向生成patch，确定性脚本负责构建/门禁 | 两客户端各完成一次同任务候选闭环；禁止改受信路径的检查及超时/调用/磁盘预算有负例；构建/Asim串行 |
| S3 晋级归档 | 固定运行时版本、同输入基线、通信、评分、干净环境复现 | 数值/适用通信硬门禁全部通过，append-only台账可重建；不以Cube利用率奖励冗余工作 |
| S4 后置 | 持久协调器/恢复/跨天预算 | S2/S3监督闭环稳定后再做，不以自动调用CLI冒称无人值守系统完成 |

S1前企业部署仍使用expert_delivery已验收源码包；精简Git分支不是替代该包的已认证离线构建发行版。不会因本轮精简升级旧CDL失败或性能晋级。

## 清理规则

本轮只在新worktree移除PPT工具链、ONNX上游非启用测试数据、旧Timeloop生成结果。逐文件SHA/大小/恢复commit见experiments/maintenance/2026-11-11-slim-v1/removal_manifest.json。六族及变体、脚本、原始报告/论文归档、结构化实验台账、核心测试保留。原研究目录及results大制品未删除，未跟踪文件不动。

历史文件可在原研究分支读取，或`git restore --source archive/asim-research-20261111 -- <path>`恢复到新临时分支；不要在新核心中无审计地恢复无关依赖。Git历史仍保留，想减少企业clone体积可用`--depth 1 --single-branch --branch codex/asim-claude-v1`，不改写远程历史。后续每次删代码先做引用/构建检查并保存清单，不能只按文件名认定过时。
