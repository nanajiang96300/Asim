# 数值验证、周期、评分与自动化审计

日期：2026-10-28（Asia/Hong_Kong）。审计源码基点：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。

子 Agent 先完成静态源码审计；主 Agent 另运行了五项隔离低成本诊断，见 [diagnostics.json](../../../experiments/audit/2026-10-28/diagnostics.json)。没有构建或运行 C++ 算子、通信仿真或自动搜索。多数问题证据等级为 `code_present`；标记 `reproduced` 的条目仅表示特定 Python 负例已经复现，不能升级为算子完整验证。问题结构化台账见 [issues.json](issues.json)，NS 首个改造包见 [影子执行计划](NS_SHADOW_EXECUTION_PLAN.md)。旧脚本、技能和结果保持原样。

## 1. 当前可信边界

Asim 提供模型预测周期、资源及依赖状态；当前不计算完整中间数值。`src/Core.cc:305` 的计算完成逻辑更新 SRAM 状态并记录周期；`src/TraceLogger.h:14` 的事件只有名称、执行单元、起止周期。`src/Common.h:78` 的实际 Instruction 含地址和尺寸，但没有完整元素 view、共轭、常量与逐步精度语义。`src/Instruction.h` 中另一个旧定义不应被误当成当前执行接口。

FormulaLogger 是与 Instruction 分开编写的数学声明。现有 DAG 可检查声明的连通性、自一致性及部分算法质量，但不能证明实际 C++ 指令数值等价。周期 CSV 缺少地址和唯一实例绑定，也不足以补出这些信息。当前六族及优化变体不能仅凭历史 PASS、代码存在或 DAG 一致标记为完整 `validated`。

## 2. 现有入口实际做什么

| 入口 | 已有能力 | 边界及证据 |
|---|---|---|
| `scripts/trace_replay.py` | CSV 解析、单元统计；匹配公式后进行部分 GEMM | `:121` 只读四列；`:329` 搬运和 Scalar/Vector 均跳过；无公式时只计数 |
| `scripts/uobs_dag_executor.py` | 按 FormulaLogger 执行数学 DAG；未知 op/缺输入会抛错 | `:292` 以逻辑名存 tensor；`:336` 按名称补常量、共轭；没有真实地址、搬运或异步执行 |
| `scripts/verify/*.py` | 每算子公式检查与多种子误差 | `_base.py:24` 过滤 batch 0；NS reference 与 DAG 共享原语并主动平方，不能证明求逆输出正确 |
| `scripts/newton_schulz_numeric_test.py` | 独立实数 NS 算法小测试 | `:18` 为转置范数初值，`:22` 返回 X；未执行实际算子 |
| `scripts/evaluate_ns_se.py` | 独立 NS 通信算法仿真 | `:34` 谱范数初值、complex64 中间运算，`:40` 返回 X；不是上述初值或 C++ 入口 |
| `scripts/check_inverse_replay_consistency.py` | 公式 CSV 与同算法参考比较 | `:101` 无 NS；`:109` 缺文件跳过；残差只打印、验收只比较同算法误差 |
| `scripts/trace_audit.py` | 按名称和单元检查覆盖数量 | `:52` 从名称猜 Scalar opcode；缺公式时仅统计，不能验证运算数与数值 |
| `scripts/uobs_scorer.py` | 周期统计、公式识别、旧 UOBS 分数 | `:433` 从新随机信道调用注册器获得“候选逆”，不是实际算子输出 |
| `scripts/evaluate_operator.sh` | 构建、运行、检查产物存在、调用 scorer | `:31` 共享 stash；`:51` 共享 build；`:46` 起使用全局临时文件；不适合并行新候选 |
| `scripts/run_benchmark_suite.sh` | 保存日志、配置副本、版本短串和摘要 | `:11` 固定 build；`:12` 固定硬件配置；`:64` 文本提取失败可生成非法 JSON；无输入/二进制/评估器摘要 |
| `orchestrator/pipeline.json` | 声明阶段与 shell 检查，供技能执行 | `:65` 数值阶段可选；`:67` 仅 import 注册器、无公式重放并截取五行 |
| `orchestrator/iterate.sh` | 多轮 Claude 调用、收集结果与停滞规则 | `:119` 从跨会话 `/tmp` glob 采纳 Agent 自报分数，缺受信验收和 run 绑定 |

`scripts/experiment_flow_skill.py` 是已有确定性导出/重建辅助（`:32` 起执行外部命令，`:78` 起导出周期表），并非本轮拟议的数值影子执行器或持久化控制器。`automation/`、`verification/` 的新执行链设计仍待实现。

## 3. 必须优先处理的问题

稳定 ID、触发条件、负例与依赖任务保存在 JSON 中。严重度：CRITICAL 表示可能让不可信结果被接受或污染候选环境；HIGH 表示会使质量、语义或成本判断失真；MEDIUM 表示入口、归档或诊断不足。

| ID | 严重度 | 证据及影响 |
|---|---|---|
| VA-001 | CRITICAL | `trace_replay.py:342` MOVIN/MOVOUT 为 pass，`:363` Scalar/Vector 跳过；缺少 NS 残差运算仍可计数 |
| VA-002 | CRITICAL | `trace_replay.py:197` 地址由 name/step/batch hash 生成，`:213` 缺输入随机生成；真实跨步数据流没有建立 |
| VA-003 | CRITICAL | `reference_inverse_registry.py:131` 吞 DAG 错误，`:191` 回退按算法名计算；候选缺失可被另一算法掩盖 |
| VA-004 | HIGH | `reference_inverse_registry.py:125` 强制 NS 至少 15 轮；不再与候选迭代数相同 |
| VA-005 | CRITICAL | `verify/newton_schulz_v3.py:28` reference 也做 X@X，并共享 GEMM/SUB；自一致不约束 A⁻¹ |
| VA-006 | HIGH | `uobs_scorer.py:353` trace 一致性函数读取失败返回通过；`:364` 只需 Cube 数量达到公式 50% |
| VA-007 | CRITICAL | `uobs_scorer.py:441` 数值候选是注册器输出，不是仿真实际程序输出 |
| VA-008 | HIGH | `uobs_scorer.py:404` 分数含 10% Cube、10% 并行度，可奖励无效工作；基线只按 Nt 查表 |
| VA-009 | HIGH | `uobs_scorer.py:391` 无显式 finite 检查，NaN 不满足 `>threshold`；`:397` 零周期按 1 处理，没有拒绝 |
| VA-010 | HIGH | `uobs_dag_executor.py:21` FP16 helper 回到 FP64；`:36` GEMM 以复数高精度计算后量化；`:69` Cholesky 可修改矩阵稳定化 |
| VA-011 | CRITICAL | `evaluate_operator.sh:33` stash 仅保护旧目录；`:39` pop 失败回退 checkout；与并行候选或用户改动冲突 |
| VA-012 | HIGH | `evaluate_operator.sh:46` 全局 `/tmp/_uobs_*`、共享 build_asim；`ci_gate.sh:30` 固定临时目录 |
| VA-013 | HIGH | `ci_gate.sh:239`、`:271`、`:283` 将解释器与脚本路径拼成一个词；`:283` 还错误使用 scorer 位置参数 |
| VA-014 | HIGH | `ci_gate.sh:242` 按文本 PASS 验收；`verify/newton_schulz_v3.py:48` 只打印单次状态及多种子最大误差，没有失败退出码 |
| VA-015 | HIGH | `pipeline.json:65` 数值验证可选；`:67` 无公式、仅截日志；`:76` 未实现扩展可输出提示后成功 |
| VA-016 | CRITICAL | `iterate.sh:119` 读取 `/tmp/agent_*_rN_result.json`，`:127` 使用自报 score；无 campaign/hash/gates 绑定 |
| VA-017 | HIGH | 旧 opt-round `:63`、`:94` 只允许 src/operations，与当前注册表 src/inverse 冲突；`:43` 把评分维度与模型配置割裂 |
| VA-018 | MEDIUM | `run_benchmark_suite.sh:8` 接收 config 却未使用，`:12` 固定配置；`:75` 把 N/A 裸插 JSON；`:20` 编号非原子 |
| VA-019 | HIGH | `channel/cdl.py:21`、`:62`、`:89` 实际是角扩展参数控制的指数相关 Kronecker 平坦信道代理，不能代表完整 CDL-B |
| VA-020 | HIGH | `unified_verify.py:48` lambda=Nt/SNR，`:101` 又令 noise_power=lambda；其他脚本用 sigma² 或 sigma²Nt，成本和归一化未统一 |
| VA-021 | HIGH | `check_inverse_replay_consistency.py:109` 缺文件跳过，全部缺失亦可打印通过；`verify/_base.py:24` 默认仅 batch0 |
| VA-022 | HIGH | `Common.h:78` 缺 view/复数/常量/精度；`TraceLogger.h:14` 无唯一实例读写信息；按 start 排序不能证明 RAW/WAR/WAW |
| VA-023 | HIGH | `uobs_dag_executor.py:145`、`:192` LDL 为列缩放，再 YᴴY；一般 L,D 不满足正确重建（状态卡另有细审） |
| VA-024 | HIGH | `uobs_dag_executor.py:157` 按维度猜 BRI block size；`:384` SCALE 依赖 aux 默认 omega；不是实际参数绑定 |
| VA-025 | HIGH | `OPERATOR_DEVELOPMENT_STANDARD_V3.md:75` 把 ADD(Reg,Reg)称复制；`:129` 合成1，`:152` 却用它作0取负；`:159` LDL列缩放；规范本身不能作真值 |

VA-006 的边界：`score_operator` 在一致性函数之前先调用 `extract_trace_stats`（`uobs_scorer.py:423`），因此本报告没有声称整个 scorer 对任意缺失 trace 都必然返回通过。问题是该函数本身的跳过行为及数量门槛不足。

VA-010 的边界：高精度中间计算后显式量化可以定义一种近似数值模型，但必须明示。它不是已校准的 Ascend FP16 乘法/FP32 累加/存储语义。`scripts/eval_pure_fp16.py:21` 同样返回 FP64/complex128，虽在许多算术步骤后量化，仍需逐项检查复数展开、归约顺序、融合和溢出规则，不能仅凭名称“pure FP16”验收。

VA-019 只描述当前仓库生成器；本轮没有核对 3GPP 标准，也没有修改类名或历史报告。后续新报告应称其为“现有 CDL-B 参数相关信道代理”，保留历史名称作来源信息。失败结论只适用于具体脚本、参数、种子和预算。

VA-025 表明“符合旧开发规范”与“数值正确”可能冲突。旧规范原文保留；后续迁移必须显式定义 COPY/FILL/zero 常量及正确缩放，并独立测试，不能把模型中的区域分配成功当成数学初始化正确。规范修订应单独记录版本及受影响结果范围。

主 Agent 的诊断确认：旧 `compute_score` 接受 NaN 误差及零周期；缺 trace 的一致性 helper 返回 true；一个 2×2 复 HPD 样本的 LDL DAG 列缩放相对误差约0.212498，而正确 Python 行缩放误差约1.89e-16；在显式减法和正确输入假设下，NS末尾平方数学见证的未归一化 Frobenius 残差约0.901388。前四项属于实际 Python 函数负例，第五项是数学见证，均不是 C++ 指令数值重放。

## 4. Claude 流程保留与迁移边界

已阅读 `.claude/skills/op-flow/SKILL.md`、`opt-round/SKILL.md`、`eval-patch/SKILL.md`、`audit-operator/SKILL.md`，本轮仅审计这些工作流，没有运行 slash command。

- op-flow 的设计先行、公式推导、构建、审查、归档意图继续沿用；实际执行并保存相应命令结果。数值阶段当前可选，与新质量硬门禁要求不一致。
- audit-operator 可以发现静态公式/地址疑点；Python 参考“权威表达”（技能 `:26`）仍可能有共同错误。Agent 审查不能成为最终真值来源。
- opt-round 的环境清理段仍展示 checkout/drop stash 命令；不执行。即使排版有删除线，也不应复制为新流程。它的并行策略与共享 eval 脚本相冲突。
- eval-patch/iterate 保留为历史入口。新候选通过独立工作区、独立 build/TMPDIR、固定可信评估器执行；不在共享工程运行旧 patch evaluator。
- 本轮形成文档与台账，不创建或宣称完成 Codex runner、影子执行器、硬件精度模型、异步冒险验证或无人值守控制器。

## 5. 新链必须保持的证据与验收

每次运行绑定：候选完整 commit/源码摘要、任务契约、输入及种子、实际低层程序、编译器与依赖、构建参数及二进制、硬件配置、评估器、评分器、输出与事件记录。缺任一必需对象返回 INVALID/UNSUPPORTED/INFRA_ERROR，不回退算法。

数值功能、异步调度、周期和通信质量分别给状态。只有所需门禁全部通过、周期有限且大于零、成本口径一致、干净环境复现成功，才允许晋级。先统计有效指令覆盖，再检查阶段值和最终输出；覆盖率不能按 GEMM 数量近似。

通信任务固定 H、符号、噪声并成对比较；保留分组误差、归一化残差、SER/BER 样本量与置信界。旧 SE/逆矩阵平均误差不是这些门禁的替代品。预处理、谱界估计、初值、预条件构造、输出恢复、Xb 应逐项计入相应测量范围。

后续首版只用冻结权重的周期几何加速比；Cube/并行度用于解释，不进入奖励。历史 score 保留原 scorer 版本，不重标为新榜成绩。

本轮可做的低成本核验是 JSON/路径/行号、脚本语法及隔离的原语反例。它们只能升级对应检查的复现状态；完整算子仍需 NS 实际指令影子执行与后续调度门禁。
