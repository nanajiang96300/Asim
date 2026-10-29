# 有人监督的日常研究工作流

生效日期：2026-10-28（Asia/Hong_Kong）。资源基线：2 vCPU、约3.8 GiB内存。

本文是当前操作规程；只读盘点与小诊断工具已经落地，受信 builder、影子执行、异步门禁、新评分器和持久化控制器尚未实现。当前可完成证据整理和历史诊断，不能将旧流程结果晋级为 verified 基线。不得把下文“待实现命令接口”当作已安装程序。

2026-10-29当前工作范围：按 [框架实施计划](FRAMEWORK_PLAN.md) 先建设通用验证和协调设施，用合成程序验收；暂不执行本文后面的具体算子候选模板。受信builder推导成本字段，候选不得单独填compute_size/tile/延迟；框架完成后再开放真实算子接入。

## 1. 一次任务先冻结什么

先写 `experiments/<campaign>/task.json`，从 [campaign 模板](../../experiments/templates/campaign.json) 复制并补齐：研究问题、外/内循环、算法/候选/父候选、基础commit、允许路径、契约与suite版本/摘要、硬件配置、输入制品、精度、成本范围、预算、停止条件。未知字段保留null并使状态保持PROPOSED；不要填“默认已通过”。

内循环固定算法、X0、迭代数、预条件器、精度和适用域。改其中任何一项必须标注参数/算法变化。外循环提出新的 algorithm_id，先过小规模结构筛选，再安排内循环。

MMSE任务统一为 A=HᴴH+λI、b=Hᴴy、solve(A,b)，交付x_hat。影子执行首批单独使用“已给A/X0/C的求逆内核”契约，参考solve(A,I)，不能与均衡全流程混榜。需明确complex布局、共轭、batch和Nrhs。噪声/星座/信道归一化及λ成套冻结。

## 2. 阶段及硬退出条件

| 阶段 | 输入 → 输出 | 命令/执行者 | 退出条件与失败状态 | 允许修改范围 |
|---|---|---|---|---|
| 盘点 | 工作区 → 环境与内容清单 | 下节inventory命令 | 无读取错误；否则INFRA_ERROR | 新清单目录 |
| 设计 | 契约/基线 → 假设、推导、等价性、指令预算 | Agent＋负责人检查；按op-flow实际阶段执行 | 输入/输出/精度/成本明确，否则INVALID | task与新设计文档 |
| 生成 | 固定base → 候选补丁 | 子Agent，独立worktree | diff仅命中允许路径；否则INVALID | 候选算子/调度参数 |
| 冻结 | diff → 不可变commit/清单 | git diff/commit，run模板 | 工作区干净、所有版本可定位；否则INVALID | 候选提交，随后独立清单提交 |
| 构建 | 冻结源码/工具链 → 二进制与构建日志摘要 | 独立build，-j1 | 退出码0且二进制存在；否则BUILD_FAILED/TIMEOUT | build与本次日志 |
| 数值 | 实际低层操作/固定输入 → 三路径快照/覆盖 | 受信影子链（待实现） | G0–G3全部通过；缺语义UNSUPPORTED；数值不符NUMERIC_FAILED | 只写run结果，评估器只读 |
| 周期/调度 | 同程序/同输入 → 事件/读写冲突/周期 | Asim＋G4（G4待实现） | 正周期、完整trace、无hazard；否则HAZARD_FAILED/INVALID | 本次run目录 |
| 通信 | 同H/符号/噪声 → 分组SER/BER/残差/区间 | 固定suite受信程序（待实现） | 各组过冻结阈值；失败QUALITY_FAILED，样本不足INCONCLUSIVE | 本次run目录 |
| 评分 | 全部门禁证据 → score/Pareto | 独立评分进程（待实现） | 全必测项通过才有分，否则score=null | 受信评分结果 |
| 复现 | 同提交/制品/工具链 → 干净运行 | 新worktree、build、TMPDIR，串行 | G0–G6与周期口径一致；否则INFRA_ERROR/不可复现 | 新复现目录 |
| 晋级归档 | 复现证据 → 基线引用/结果卡 | 主Agent检查完整diff；负责人审阅里程碑 | 引用与hash全可访问才晋级 | 台账/卡片；不重写旧结果 |

审查Agent指出风险，不能签发最终通过；Agent自报分数不作为门禁。数值功能顺序执行不代替异步完成语义，G4未完成时不得整体评分。

## 3. 现阶段可以实际执行的命令

从工程根目录执行。快照目录必须新建，工具拒绝覆盖。

```bash
cd /home/nanajiang/Asim
git status --short --branch
nproc
free -h
df -h .
python3 scripts/research_inventory.py create experiments/inventory/<new-snapshot-id>
python3 scripts/research_inventory.py check experiments/inventory/<new-snapshot-id>
# 更新当前文档后，只核对原始资料是否未变；全量检查会正确报告文档变化
python3 scripts/research_inventory.py check experiments/inventory/2026-10-28-baseline --raw-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/research_audit_checks.py \
  --output experiments/audit/<new-run-id>/diagnostics.json
python3 scripts/research_validate.py
```

尖括号是必须替换的参数；诊断脚本退出0表示已重现列出的旧问题，不表示求逆算子通过。新实验修好旧函数后，旧诊断可能返回UNEXPECTED_RESULT，应解释差异并保留旧诊断。

## 4. 候选隔离与构建命令

未来开始算子开发时，先读 `CLAUDE.md`、V3标准及 `.claude/skills/op-flow/SKILL.md`，实际记录设计、推导、构建、审查与归档；不能宣称调用Codex中不存在的slash command。V3数学/初始化冲突见审计，受信数值契约优先，不以“满足旧标准”代替正确性。

下面为可执行命令模板，先替换任务特定参数。当前没有创建这些候选，也未编译。

```bash
ASIM_ROOT=/home/nanajiang/Asim
RUN_ID=<unique-run-id>
CANDIDATE_BASE=<full-frozen-commit>
CANDIDATE_TREE=/home/nanajiang/asim-worktrees/$RUN_ID
RUN_DIR=$ASIM_ROOT/results/codex/$RUN_ID
mkdir -p /home/nanajiang/asim-worktrees
git -C "$ASIM_ROOT" worktree add -b "codex/$RUN_ID" "$CANDIDATE_TREE" "$CANDIDATE_BASE"
mkdir -p "$RUN_DIR/tmp"
export TMPDIR="$RUN_DIR/tmp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# 按允许路径完成候选并提交后，冻结候选commit；评估期间禁止再改源码
git -C "$CANDIDATE_TREE" status --porcelain
git -C "$CANDIDATE_TREE" rev-parse HEAD

# 若依赖未就绪，先记录INFRA_ERROR；不能借用共享build假装干净构建
timeout --signal=TERM --kill-after=10s 10m cmake -S "$CANDIDATE_TREE" \
  -B "$CANDIDATE_TREE/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > "$RUN_DIR/configure.log" 2>&1
timeout --signal=TERM --kill-after=10s 20m cmake --build "$CANDIDATE_TREE/build" \
  --target Simulator -j1 > "$RUN_DIR/build.log" 2>&1
```

命令逐条执行并检查退出码；configure失败立即停止。Conan包缺失时只在独立build内按固定conanfile、Conan版本和离线/受控缓存安装，记录profile/包revision；本轮系统PATH未找到conan，不假设依赖已可干净重建。受信评估器在固定版本的另一个checkout运行，不允许候选修改测试/评分/契约/构建入口。当前worktree仅隔离版本，不提供权限边界。

历史周期诊断可使用下面命令，但只能写 `legacy_unverified`，不是新数值门禁。先通过实际版本的 `--help` 确认参数；主程序历史 `--mode` 行为及入口以状态卡为准。

```bash
export ONNXIM_TRACE_CSV="$RUN_DIR/trace.csv"
export ONNXIM_FORMULA_JSON="$RUN_DIR/formula_steps.json"
cd "$CANDIDATE_TREE/build"
timeout --signal=TERM --kill-after=10s 2m "$CANDIDATE_TREE/build/bin/Simulator" \
  --config "$CANDIDATE_TREE/configs/ascend_910b_quiet.json" \
  --models_list "$CANDIDATE_TREE/example/newton_schulz_v3_test.json" \
  --mode newton_schulz_v3_test --log_level info > "$RUN_DIR/simulator.log" 2>&1
cd "$ASIM_ROOT"
```

根CMakeLists.txt将二进制输出硬编码为源码目录的build/bin，因此仅换-B目录并不能隔离二进制，必须使用独立worktree。Ramulator2配置路径还依赖cwd；上述示例从worktree/build运行以匹配配置中的 `../configs/...`。若生成绝对路径配置，必须另存并hash，不能回写共享原配置。

新数值链的计划接口及文件边界见 [NS影子执行计划](audits/NS_SHADOW_EXECUTION_PLAN.md)。它尚无可以运行的CLI；当前缺门禁时必须停止在UNSUPPORTED，不能用 `scripts/verify/*`、`ci_gate.sh` 或 `uobs_scorer.py` 替代受信验收。

## 5. 预算、重试和晋级

- 当前只读审计最多3个子Agent并发；生成候选最多2个；编译1个、每次-j1；仿真1个。编译与仿真避免同时争抢2核。
- 初始campaign建议：1个复现候选＋3个优化候选；每候选1次初试＋最多2次修复，修复产生新commit和parent_attempt。冻结后不改预算。
- configure 10分钟、build 20分钟、小规模sim 2分钟、单次小验证2分钟；超时记TIMEOUT，不自动扩大规模。更大样本先估算并记录预算，再启动。
- 同样的基础设施瞬时故障最多重试1次，保留失败日志；缺实现/未知语义不可通过重试绕过。数值或调度失败回到新候选，不能只修改pass字段。
- 正确性通过后才比较同contract/suite/precision/hardware/cost范围的正周期。Cube利用率、并行度和指令数只作诊断。几何平均加速比评分见VERIFICATION_DESIGN。
- 总成本包含Gram/正则化、初值/谱估计、预条件构建、迭代、输出恢复及搬运；已给A/b的内核单独建组。多右端项/复用按显式R记录setup+R×apply，不给免费预处理。
- 当验证能力缺失、预算耗尽或重复失败时保留incumbent，记录停止原因。缺完整验证时incumbent可为null，不能从历史最快周期强行选冠军。

## 6. 每日收尾

更新结构化台账、失败原因、制品路径/bytes/SHA-256与任务依赖；`git diff --check`，运行资料校验工具，审查 `git diff --cached --stat` 后按明确路径提交。源码与结果清单分开提交避免自引用commit。不执行共享stash/apply/pop、不自动推送、不改写旧历史。

下一轮输入只引用冻结证据与未决问题。每周更新研究索引与决定；跨天控制器、SQLite租约、SDK适配器在最小受信闭环完成后建设。

当前实施更新（2026-10-29）：FW0–FW2初始库已开始开发，详见 [实现范围](../../verification/contracts/README.md)。候选执行顺序以最新决定为准：静态检查 → Asim运行与性能预筛 → 数值重建；不竞争/异常/运行失败时数值NOT_RUN、score=null。先瑞利，后续扩展CDL-B。完整Asim接入和框架验收尚未完成。

最新可执行结构门禁已实现：参见 [FRAMEWORK_PROGRESS](FRAMEWORK_PROGRESS.md) 的真实Asim重跑命令；尚不能替代本文完整候选晋级链。

2026-10-29多核更新：默认24核、batch96（24的倍数），Cube16³；24/96全batch的真实Asim→预筛→三路径数值闭环已运行，45项单元测试通过。计时为给定A/b/X0/C内核，仍非FP16/G4/通信统计验收。详见 [24核验证报告](ASCEND24_VALIDATION.md)。

## 2026-10-29七入口重建更新

[七入口重建报告](LEGACY_RECONSTRUCTION_REPORT.md)记录最新实际覆盖：真实Instruction附加views、逐元素初始化/部分写、DMA batch范围、完整输出与事件覆盖检查已落地；七入口正确性未全通过，score=null。独立同算法参考、精度/G4/通信仍缺证，不能用周期正常结束替代验收。专用原件目录`results/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/`，Git小清单在`experiments/reconstruction_campaigns/`；重跑命令与hash检查见报告，旧数据不移动。
