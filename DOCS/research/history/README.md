# 历史实验台账与使用规则

审计日期：2026-10-28（Asia/Hong_Kong）。源码／资料读取基点：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。本目录是对历史记录的整理，不回写原报告；本轮未构建或运行历史算法实验。

## 交付与阅读入口

| 文件 | 用途 |
|---|---|
| [NS_LINEAGE.md](NS_LINEAGE.md) | NS 四轮探索、v3 基线与验证器演变分开复盘 |
| [EVIDENCE_GAPS.md](EVIDENCE_GAPS.md) | 数据冲突、缺失来源、后续补证任务 |
| [ledger.jsonl](../../../experiments/history/ledger.jsonl) | 49 个统一结构条目：NS、六种实现、Merge、BRI 优化、验证变化与旧运行 |
| [legacy_runs.json](../../../experiments/history/legacy_runs.json) | 10 个现存旧运行包的摘要、配置、版本、逐制品大小与 SHA-256 |
| [sources.json](../../../experiments/history/sources.json) | 19 个关键历史来源的大小、内容摘要及最后修改提交 |

入口返回 [研究索引](../README.md)。仓库完整文件清单和归档映射由 `experiments/inventory/2026-10-28-baseline/` 保存；本目录只提取研究意义与运行元信息，不搬迁原件。

## 证据标准

`owner_reported` 为负责人陈述，`documented` 为历史报告／提交消息／旧运行摘要可见，`code_present` 为相应代码或代码变更可见，`reproduced` 要求本轮按冻结条件重新执行，`validated` 要求完整验收门禁。本台账没有将任何历史条目升级为 `reproduced` 或 `validated`。

`implementation_commit` 只表示发生相关代码变更的提交，不能代替产生某个周期或质量结果的 `candidate_commit`。`sources.json` 的 `last_change_commit` 也只是资料版本；例如文件名写 2026-07-08 的综合报告实际在 7 月 14 日提交并继续修改，不能把文件名当作所有实验的发生日期。文件 mtime 不用作实验时间。

负责人确认 BRI 是当前已有研究中的性能最优候选，这一陈述保留在项目目标中。台账不把混合口径的历史周期表升级为统一排行榜，也不把有限 CDL-B 负结果推广为算法永久不可用。

## 结构化字段

所有 `ledger.jsonl` 条目具有相同字段：

| 字段 | 含义 |
|---|---|
| `id/algorithm/title/baseline_id` | 稳定条目标识、算法族、改动及对应对照 |
| `change_kind` | `algorithm`、`parameters_precision`、`instruction_pipeline`、`validation_scoring`；缺资料为 `unknown`，可多选 |
| `hypothesis/parameters` | 当时假设、初值、K/L/B、精度、尺寸、batch 与统计条件；未知字段不推断 |
| `implementation_commit/candidate_commit` | 已定位的相关代码变更／实际运行版本；无法完整绑定时后者为 null |
| `experiment_date` | 只有原运行包明确记录 timestamp 时填写；提交日期不冒充运行日期 |
| `reported_metrics` | 原文／摘要中的周期、质量和旧分数，未经本轮重新计算 |
| `trusted_metrics/trusted_score` | 本轮受信重跑的结果；本次全部 null |
| `failure_reason/conclusion` | 负结果、失效归因及当前解释边界 |
| `sources/raw_artifacts` | 可定位的来源路径／行号和原始制品路径 |
| `reproduction_command` | 已有脚本入口，仅代表可供审查的历史入口；不是已执行记录，也不保证当前默认参数复现原表 |
| `missing_evidence` | 尚缺候选、输入、配置、工具链、原日志、评估器或评分器版本 |
| `evidence_levels/reproduction_status/promotion_status` | 证据等级、本轮是否重跑、是否可参与新榜；历史项均为 `legacy_unverified` |

`cycle_kind` 的区分必须保留：`reported_measured` 是报告声称的实测，`stored_summary` 是现存运行摘要，`estimate_linear_2x_2591` 是推算；它们均不是本轮实测。遇到不合法原始 JSON，记录 `INVALID` 和解析错误，原件保持不变，不用 0 或 null 伪装成成功周期。

## 现存旧运行包

| 目录 | 模型记录 | 周期摘要 | 版本边界 |
|---|---|---:|---|
| `results/cholesky_block/run_001` | 64×16，batch96，B2 | 3,326 | unversioned，旧 mode |
| `results/cholesky_noblock/run_001` | 64×16，batch96，strict lowering | 6,142 | unversioned，旧 mode |
| `results/cholesky_noblock_v2/run_001` | 64×16，batch96 | 15,261 | e66a5d7 ＋未冻结修改 |
| `results/cholesky_noblock_v2/run_002` | 64×16，batch96 | 15,261 | e66a5d7 ＋未冻结修改 |
| `results/ldl_block/run_001` | 64×16，batch96，B2 | 1,682 | unversioned，旧 mode |
| `results/ldl_noblock/run_001` | 64×16，batch96 | 3,673 | unversioned，旧 mode |
| `results/ldl_noblock_v2/run_001` | 配置另见清单 | **INVALID** | `finish_cycle: N/A` 为非法 JSON，未完成证据 |
| `results/ldl_noblock_v2/run_002` | 64×16，batch96 | 20,733 | 版本文本见清单，不能绑定干净候选 |
| `results/newton_schulz/run_001` | 32×32，batch96，配置 K10 | 38,806 | unversioned，`newton_schulz_opt_test` |
| `results/block_richardson/run_001` | 64×16，batch96，L8/B2 | 1,979 | unversioned，旧 mode |

配置均记录 24 核及 `precision=2`；后者是周期模型的元素字节数，不能证明已执行 FP16 数值算术。这些旧包与后来的 23,439／9,999／6,440／4,959／2,591／1,993 表不是同一次运行。原 mode 的当前可用性、数值语义和干净构建仍需另验。

## 日常追加与复核

1. 新历史发现先追加条目，保留原假设及失败；没有日志则填写 `missing_evidence`。
2. 读取原始数据前按 inventory 核对路径、大小、SHA-256。文件变化时生成新清单，不覆盖旧摘要。
3. 将来源行号与 `audit_base_commit` 一起使用，避免未来源码变化导致行号漂移。
4. 新实验按项目工作流冻结候选、契约、输入及工具版本，生成新 run；不要复用旧 `run_001` 目录。
5. 质量和周期必须来自同一候选及同一计时范围，先过数值／调度／通信／复现门禁，再记录受信分数。

可低成本检查台账格式：

```bash
python3 - <<'PY'
import json
from pathlib import Path
rows = [json.loads(x) for x in Path('experiments/history/ledger.jsonl').read_text().splitlines()]
assert len({x['id'] for x in rows}) == len(rows)
assert all(x['trusted_score'] is None for x in rows)
assert not any(set(x['evidence_levels']) & {'reproduced', 'validated'} for x in rows)
print('历史台账解析通过，条目数:', len(rows))
PY
```
