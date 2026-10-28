# Asim 研究与 Codex 自动化计划

更新：2026-10-28（Asia/Hong_Kong）

工程：`/home/nanajiang/Asim/`

状态：研究整理与框架设计 v0.1；新执行框架尚未实现。

## 当前目标

利用 Agent 自动迭代 NPU 上的通信算子，在通信质量约束下减少周期，并有效利用 AI Core。当前以 MIMO 均衡为试验任务，开展算法结构的外循环搜索和指令、流水调度的内循环优化。

本阶段先整理历史研究、建立 Git 与实验证据管理、重构可信验证边界，再搭建与 Claude Code 并行的 Codex 自动化流程。

## 阅读顺序

1. [项目目标与研究进展](PROJECT_CHARTER.md)：研究定位、外内循环、已知成果与证据等级。
2. [验证体系设计](VERIFICATION_DESIGN.md)：为什么反推算法不足，以及如何建立独立、可执行的验证。
3. [Codex 自动化架构](CODEX_AUTOMATION.md)：控制器、Agent 集群、仿真器、验证器和评分器的接口。
4. [Git 与实验管理](PROJECT_MANAGEMENT.md)：版本、分支、数据归档及结果晋级规则。
5. [分阶段研究计划](ROADMAP.md)：依赖顺序、任务清单、验收标准与首轮实验。

## 本次决策

- 以用户 2026-10-28 对研究路线的说明作为当前目标来源；9 月中期材料保留为历史计划。
- BRI 是现有研究中性能最好的候选；CDL-B 适用性仍待统一口径验证，不预先判定可用或永久不可用。
- 验证从“Agent 声明什么公式”转向“提交的可执行程序实际上计算什么”。
- 以同一任务契约比较不同算法；内部可以求逆、分解或直接求解。
- 正确性、覆盖和复现门禁通过后才给性能分。利用率作为诊断指标。
- Claude 与 Codex 共享任务契约和可信评估器，使用独立候选工作区与运行目录。
- 先做设计和证据整理；未实现的 API、目录和状态机在本文档中均为拟议方案。

## 证据来源与边界

工程源码基点：`5a86e9fc`（2026-07-14）。核对范围包括 `src/Common.h`、NS 算子、`scripts/trace_replay.py`、`scripts/uobs_scorer.py`、`scripts/evaluate_operator.sh`、`orchestrator/pipeline.json`、旧 `opt-round` 技能，以及 7 月周报和 9 月材料。

本次没有重新运行算法实验，没有证明历史数值结果可复现，也没有校准 Asim 与实机误差。历史材料中的“首次”“唯一”等文献判断不作为本计划的已核验结论。

历史材料入口：`DOCS/zhongqi/kaiti.md`、`DOCS/zhongqi/zhongqi.md`、`DOCS/zhongqi/qianyan_diaoyan.md`。它们原文归档，不因新判断而回写历史。
