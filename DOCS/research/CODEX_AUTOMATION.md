当前实现进度（2026-11-08）：七个新入口已有真实Asim与三路径逐操作精度验证；已有离线选K、独立留出与受监督计划冻结/执行交接，Q2验收通过；Q3a受限瑞利/QPSK相对BER/EVM和留出置信门禁已接通，Q3b有限I/Q响应与线性SE代理已接通，200测试通过。当前阶段契约以 [QUALITY_DRIVEN_WORKFLOW](QUALITY_DRIVEN_WORKFLOW.md) 为准；总体SE质量/BLER/CDL-B、无人值守协调器、完整评分与晋级仍待实现，Q3a不是完整通信认证。下文日期版本保留研究演进背景，旧“尚未创建”不代表当前源码状态。

# Codex 平行自动化框架设计

状态：完整控制器/adapter/可信执行链仍为拟议架构。本轮已经使用当前服务器上的原生主/子Agent完成系统审计，盘点/诊断/资料校验脚本落地；未启动无人值守搜索，也未独立核验CLI安装、认证和SDK接口。当前操作规程以 [DAILY_WORKFLOW](DAILY_WORKFLOW.md) 为准。

## v0.4 当前实施优先级

2026-10-29按负责人要求先完善通用框架，不修改具体算子。具体模块、约束、阶段验收与预算以 [FRAMEWORK_PLAN](FRAMEWORK_PLAN.md) 为准：策略Agent提出方向，确定性协调器校验/分派；候选程序由受信builder生成最终低层对象，Asim和数值重建消费同一程序；全部门禁/复现后由独立评分器签发结果。FW0–FW2数值基础先行，首批使用合成fixture；真实NS迁移在FW7框架验收之后。下文NS/pilot及完整持久化状态机仍为后续设计。

不同算法不强制相同指令数量；统一原语粒度、精度与硬件成本。候选不得修改计时字段、评估器或profile；高层求解、SCALAR批量融合、复数与转置的真实工作必须共同展开/计费。缺语义UNSUPPORTED，禁止靠FormulaLogger补齐。完整约束矩阵见框架计划第6节。

## v0.2 实施调整

近期采用桌面 Codex 原生 SSH 远程项目，Ubuntu 上运行 Codex 与主／子 Agent，工程直接位于 `/home/nanajiang/Asim/`。原生子 Agent 负责研究与生成；先用确定性脚本完成构建、验证和评分。后续需要跨天运行、预算控制与断点恢复时，再引入本文完整持久化控制器。

远端安装不会自动产生新框架，也不会自动实现工作区隔离。每个写代码候选仍需独立 worktree 和 build；子 Agent 需要明确分工和允许路径。近期操作命令见 [部署指南](REMOTE_CODEX_SETUP.md)。

本文第 4 节的 exec/SDK adapter 是无人值守阶段的实现选择，不是使用原生子 Agent 的前提。验证模块近期采用数值影子执行器，详见 VERIFICATION_DESIGN.md 的 v0.2 更新。

## 1. 设计原则

把长期闭环状态放在持久化控制器中，Agent 只负责提出假设、生成候选和解释失败。编译、验证、评分、资源调度和晋级由确定性程序执行，不依赖某次聊天的上下文。

保留 `.claude/`、旧脚本与旧结果。`experiments/` 已建立历史/盘点/诊断台账与模板；`automation/`、`verification/` 的新执行链尚未创建。两种平台最终通过适配器产生相同候选包，使用同一新版受信评估链，目前尚未验收该能力。

## 2. 角色与边界

| 模块 | 输入 | 输出与责任 |
|---|---|---|
| 均衡任务管理器 | 契约、信道、形状、质量要求 | 不可变 campaign 配置与数据集合 |
| 外循环规划 Agent | 已验证前沿、失败归因、预算 | 算法假设，适用域，预期成本，验证需求 |
| 内循环实现 Agent | 固定算法与契约、性能诊断 | IR／允许路径内补丁、参数和变更说明 |
| 审查 Agent | 候选与规范 | 风险与假设检查；没有最终通过权 |
| 控制器 | 候选任务队列 | 工作区、超时、预算、重试、状态与日志 |
| 构建与仿真 worker | 候选快照、固定工具链 | 二进制摘要、trace、周期、资源统计 |
| 受信验证器 | 同一程序、输入、独立参考 | 分层门禁、数值误差、质量置信界 |
| 受信评分器 | 验证通过证据与周期 | score、Pareto 状态、对比表 |
| 晋级与归档器 | 复核后的候选 | 新冠军引用、可复现清单与 Git 版本 |

“Agent 集群”可由多个独立 Codex 进程或 SDK 会话构成，不必依赖某个平台私有的 subagent 命令。

## 3. 闭环与状态机

```mermaid
flowchart LR
    Q[任务契约与预算] --> P[独立FP16算法筛选与选K]
    P --> H[独立留出验收与参数冻结]
    H --> A[多个候选 Agent]
    A --> F[冻结候选与摘要]
    F --> C[静态检查与构建]
    C --> S[Asim 与事件及性能预筛]
    S --> V[实际指令重建 同算法及MMSE比较]
    V --> E[通信质量验收]
    E --> R[评分与前沿]
    R --> P
    R --> M[复现后晋级／归档]
```

目标候选状态机为 `ALGORITHM_SCREENED → GENERATING → FROZEN → BUILDING → SIMULATING → RECONSTRUCTING → QUALITY_CHECK → SCORED → CONFIRMED → PROMOTED/ARCHIVED`；这是目标集成接口，不是完整控制器已实现。独立筛选失败为 `NO_FEASIBLE_K/HOLDOUT_FAILED`。其余失败包括 `BUILD_FAILED`、`NUMERIC_FAILED`、`HAZARD_FAILED`、`QUALITY_FAILED`、`UNSUPPORTED`、`INCONCLUSIVE`、`TIMEOUT`、`INFRA_ERROR`、`CANCELLED`；实际执行器仍使用其已验收的具体状态名。

原始假设保留不变；修复产生新的 attempt，记录 parent_attempt。相同基础设施故障有限重试；数值或调度错误回给 Agent 形成新候选。每个候选最多 2 次修复为首版建议值，启动 campaign 时冻结。

控制器持久化 job ID、阶段、租约、进程组、attempt、输出路径。建议 SQLite WAL 保存事务状态，JSONL 保存追加事件，Git 保存最终小型清单。重启按最后完成阶段恢复；部分文件写到临时位置后原子改名，不将半份结果当成成功。

缓存键至少包含候选源码／IR hash、契约与数据 hash、编译器、构建选项、硬件配置、模拟器和评估器版本。只按算法名或 commit 缓存会产生错误复用。

## 4. Codex 接入方案

首版推荐 Python 控制器调用 `codex exec`，易于管理子进程、超时、退出码和 JSONL。需要更细会话控制时再使用 Codex SDK；保持 `AgentAdapter` 边界不变。

截至 2026-10-28，官方非交互模式支持 `--json` 事件、`--output-schema` 结构化最终输出和 `-o` 保存结果；这些仅保证接口格式，不保证候选正确。[官方非交互文档](https://learn.chatgpt.com/docs/non-interactive-mode)

Codex SDK 可用于程序化启动与恢复会话。选定部署平台后固定 SDK／CLI 版本，并记录模型配置；不在设计中指定尚未确认账户可用性的模型。[官方 SDK 文档](https://learn.chatgpt.com/docs/codex-sdk)

下例是未来 adapter 的调用形态，变量和 schema 文件由控制器创建；当前不是已部署脚本：

```bash
codex exec -C "$CANDIDATE_WORKTREE" \
  --sandbox workspace-write --json \
  --output-schema "$TRUSTED_SCHEMA" \
  -o "$RUN_DIR/agent_result.json" - \
  < "$RUN_DIR/task.txt" > "$RUN_DIR/agent_events.jsonl" \
  2> "$RUN_DIR/agent_stderr.log"
```

结构化输出只包含 `candidate_id`、`hypothesis`、`changed_paths`、`assumptions`、`claimed_effect`、`self_reported_tests` 和 `limitations`。最终数值、pass 和 score 必须由 runner 重新计算。CLI 参数升级时重新做 adapter 兼容检查，不依赖终端文本关键词判断成功。

先前会话检测记录：Windows控制端codex-cli版本与“Ubuntu PATH未发现Codex”只代表当时状态，不能当作当前检测结果。本轮实际主/子Agent直接访问nanajiangserver工程，nproc仍为2；未单独检查CLI/SDK版本或认证。未来控制器部署仍需固定版本并做最小测试，不复制登录令牌。上面的官方接口示例是历史设计引用，本轮没有重新核验产品API。

过渡期可以 Windows 生成候选、Linux 评估，但需以相同 base commit 导出补丁和完整清单，服务器验证 hash 后重建；不可依赖 Windows 绝对路径或未经记录的工作区文件。

## 5. 并发与隔离

当前只读审计允许3个子Agent；候选生成仍建议 `agent_concurrency=2`、`build_jobs=1`、`sim_jobs=1`，编译与仿真避免同时争抢2核。资源/超时/修复次数已写入 [campaign模板](../../experiments/templates/campaign.json)，需要campaign启动时冻结。不要让每个worker执行-j$(nproc)。

每候选独立源码、build、TMPDIR、日志和结果目录。共享依赖缓存只读或带锁；不共享可写 CMake build。旧 `evaluate_operator.sh` 的 stash/apply/pop 和 `/tmp/_uobs_*` 不能直接并行复用。

由控制器从固定 commit 创建 worker worktree 或干净克隆；候选 Agent 不获得评估器写权限。评估 worker 在独立环境应用允许路径补丁，拒绝路径穿越、符号链接逃逸、构建入口或可信模块的越界修改。worktree 本身不能防止同用户进程读写其他目录，正式黑盒运行需权限／容器边界。

生成阶段需要模型网络访问；编译和仿真阶段通常无需模型凭据。资源管理需覆盖 CPU、内存、墙钟、磁盘以及模型调用预算。超时结束本次 worker 进程组，不匹配名称终止其他任务。

## 6. 外内循环协议

外循环修改算法结构、初值、预条件器、收敛策略和精度分配，输出新的 `algorithm_id` 与适用域假设。内循环从该算法的 verified 基线出发，固定数学语义，优化实现和调度。

第一版不自动嵌套无限内循环：每个外循环候选先过小规模功能筛选，再分配固定数量内循环尝试；最终按同一质量门禁、完整成本口径比较。失败样本反馈包含输入规模、失败层、指令 ID、首个坏值、读写 buffer、残差或 ΔSER；不能只给原始几千行日志。

停止条件包括：预算用尽、连续若干轮无经确认收益、重复失败簇、验证能力缺失。停止后归档 incumbent 与未决问题。阈值、预算和重试次数是 campaign 配置，不由 Agent 临时改写。

## 7. 旧流程映射与逐步替换

| Claude 组件 | Codex 平行组件 | 迁移方式 |
|---|---|---|
| opt-round | campaign controller＋planner adapter | 抽取算法策略，重写状态与并发 |
| eval-patch | isolated evaluation runner | 保留输入意图，取消共享 stash 与全局临时文件 |
| op-flow | 确定性 gate pipeline | 质量验证必需；不能只 grep 日志 |
| audit-operator | reviewer adapter＋静态／动态检查 | Agent 提示风险，机器检查决定门禁 |
| FormulaLogger | 从 IR 派生的解释日志 | 过渡期保留，最终退出 oracle 角色 |
| UOBS | scorer v2 | 旧分数只保留历史；新分数要求独立验证证据 |

迁移按 shadow → pilot → default：先新链旁路记录，不改历史结论；再用少量候选试运行；完整复现与负例验收后才成为新实验默认。Claude 和 Codex 若比较效率，应使用同一任务、模型预算计量方法、候选预算、评估器与留出集；区分模型差异与编排差异。

当前实施更新（2026-10-29）：FW0–FW2初始库已开始开发，详见 [实现范围](../../verification/contracts/README.md)。候选执行顺序以最新决定为准：静态检查 → Asim运行与性能预筛 → 数值重建；不竞争/异常/运行失败时数值NOT_RUN、score=null。先瑞利，后续扩展CDL-B。完整Asim接入和框架验收尚未完成。

## 2026-10-30统一重建更新

[本轮报告](NS_CORRECTED_VALIDATION.md)为最新覆盖：共同H/Y/λ及计时scope、共享view/初始化/算术核心、新NS三seed×96batch三路径结构通过。79测试通过，旧七入口保留原失败。FP16/G4/通信/G6及评分尚未验收。按[接入指南](ADDING_OPERATOR.md)增加候选与独立参考，不在重建核心添加算法名分支。prepared内核与完整任务不得直接比较；旧generic SUB→raw ADD现显式UNSUPPORTED。

2026-10-30稳定入口更新：实际Core读写/DMA事件已完成受限保守模型验收，固定runtime日常命令可用，121测试通过。详情及支持/拒绝边界见[稳定性报告](RUNTIME_STABILITY.md)。旧报告中的缺真实Core事件是当时状态；硬件G4/通信/全部旧算子/最终评分仍未通过，不能由该模型证书升级。
