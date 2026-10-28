# Git、研究证据与实验版本管理

日期：2026-10-28。以下规则用于后续研究，工具实现仍需按路线图落地。

## 1. 当前仓库与本次整理

工程现有主分支为 `master`，不是部分旧报告所称的 main。本次从既有 HEAD `5a86e9fc` 建立 `research/codex-transition-20261028`，在该分支整理，保留既有主分支。

历史开题、中期与调研 Markdown 原文加入 Git。其余未跟踪的 PDF、PPT 与生成目录仍保留原位；后续先盘点大小、来源和许可，再选择 Git LFS 或外部制品存储。尚未确认是否推送到远程，因此不能把本地提交称为异地备份。

## 2. 单一信息入口

- `DOCS/research/`：当前目标、进展、设计和任务依赖。
- `DOCS/operators/`：算法公式、适用域及实现映射。
- `DOCS/reports/`：不可覆盖的阶段性历史报告。
- `experiments/`（拟建）：Git 跟踪的小型证据清单与结果摘要。
- `results/`：大体积 trace、输入、运行日志；当前被 `.gitignore` 忽略。

**不能只把关键结果写入 results/**，否则 Git 无法追踪。新实验应将摘要和内容摘要清单写入 experiments，原始制品另存可校验位置。

## 3. 版本对象

一项结论必须能够追溯到：

`research_question → algorithm_id → candidate_id → candidate_commit → run_id → result → claim`。

同时固定 `contract_version`、`suite_version`、`precision_profile`、`simulator_commit`、`evaluator_commit`、`scorer_version`、工具链与硬件配置 hash。更改任何影响结果的对象都新建版本；不同版本结果不能直接参与同榜比较。

保存任务文本、模型标识、CLI/SDK 版本、已知推理设置、调用次数和可获取的 token 用量。模型生成具有随机性，重放已提交候选和重现整个搜索过程是两种不同可复现目标。

## 4. 分支、提交与晋级

| 类型 | 建议命名 | 用途 |
|---|---|---|
| 研究设计 | research/topic-date | 目标、设计与实验计划 |
| 可信基础设施 | infra/verification-v1 | 解释器、模拟器、评估器；独立审查与版本升级 |
| 算法搜索 | algo/campaign/idea | 新算法结构 |
| 实现候选 | opt/campaign/candidate | 单一候选的调度／指令优化 |
| 已验证基线标签 | baseline/task-vN | 完整验收后的稳定参照 |
| 论文实验快照 | paper/experiment-vN | 冻结代码、清单和制品引用 |

候选从固定 base 创建独立工作区；只提交允许路径。基础设施修改不混入候选补丁。候选完成后控制器检查完整 diff，再创建不可变提交并评估；发生任何后续修改都使旧验证失效。

晋级流程：验证通过 → 同环境干净重跑 → 当前主分支上的集成检查 → 更新冠军清单 → 合并。合并若改变源或依赖需重跑。负结果保留原因与候选引用，不用强制重写历史清理研究轨迹。

模型可自动产生候选提交，研究里程碑由负责人审阅。共享远程推送和分支保护按已配置的协作策略执行；本次仅建立本地版本记录。

## 5. 实验清单最小内容

下面是字段模板，不是已完成实验数据：

```json
{
  "schema_version": "1.0-draft",
  "run_id": "<unique-id>",
  "campaign_id": "<campaign>",
  "candidate_id": "<candidate>",
  "parent_candidate_id": null,
  "candidate_commit": "<full-sha>",
  "source_tree_hash": "<sha256>",
  "contract_hash": "<sha256>",
  "suite_hash": "<sha256>",
  "precision_profile": "<explicit-profile>",
  "simulator_commit": "<full-sha>",
  "evaluator_commit": "<full-sha>",
  "scorer_version": "<version>",
  "toolchain_manifest": "<path-and-hash>",
  "measurement_scope": "equalizer_end_to_end",
  "status": "PROPOSED",
  "gates": {},
  "metrics": null,
  "score": null,
  "artifacts": [],
  "limitations": []
}
```

每个 artifact 包含路径或 URI、字节数、SHA-256、类型和生成命令。最终验收时必须验证制品可访问且 hash 一致。Git 中不能只留下一个已过期 `/tmp/` 路径。结果清单与候选提交分开提交，避免自引用 commit。

## 6. 历史研究整理任务

首先建立只读清单：每个目录／报告的文件、时间、hash、算法、参数与代码引用。文件 mtime 仅作线索，不能当作实验发生时间。

对每个算法建立结果卡：动机、基线、外循环变化、内循环变化、原始配置、原始结果、质量标准、失败记录和待复核项。NS 逐轮记录；BRI 单独记录 Rayleigh 最优配置和 CDL-B 的已尝试配置。

缺少源码或配置的实验标记 evidence_missing，不补造数据。旧报告不得为匹配新结果而覆盖；新报告给出差异及原因。

## 7. 研究节奏与决策记录

每轮 campaign 先写假设和验收，再运行。每周更新：新增证据、已否定假设、前沿变化、失败簇、预算、下周依赖。每个重要改变写 ADR（架构决策记录），记录背景、选择、替代方案、代价、状态与证据。

首批 ADR 建议：任务输出采用 x_hat；IR 双后端；评分质量硬门禁；Claude/Codex 平行适配；可信评估器隔离；统一预处理成本。

首版评分和门禁不因候选表现临时调整。必要修改需新 suite/scorer 版本，并重跑旧冠军，保留新旧结果。
