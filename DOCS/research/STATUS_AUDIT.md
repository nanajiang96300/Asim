# 项目现状与系统审计（2026-10-28）

研究目标：Agent自动迭代NPU通信算子，在通信质量约束下减少模型预测周期并发挥AI Core有效算力。载体为MIMO均衡/求逆；算法外循环与固定算法的指令/流水内循环分别记账。BRI是负责人确认的现有最佳候选，CDL-B适用域仍开放；探索先后顺序不构成普遍性能排名。

审计源码基点：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`；分支 `research/codex-transition-20261028`。本报告及台账由后续独立提交保存，避免把报告自己的commit写入被哈希的源码快照。

## 1. 本轮实际完成与未完成

已完成：仓库/资源/文件盘点，六族及Merge七张状态卡，历史实验与NS谱系，验证/评分/自动化问题清单，原件内容摘要与保留映射，NS影子执行的文件级实施设计，日常规程、模板和依赖任务表。三个子Agent先并行只读，再分别只写 operators/、history/、audits/及相应历史台账；主Agent统一口径、低成本检查、索引和提交。

本轮没有修改算子/评估器/旧Claude流程，没有构建或运行C++算子，没有通信大样本复测、自动搜索或实机校准，也未实现builder/影子执行/G4/新评分器/持久化控制器。旧报告与原始资料没有移动、删除或覆盖。代码可见、公式自一致、报告PASS与统一完整验证分开记录；本轮没有晋级任何算子为validated。

证据等级可组合：owner_reported、documented、code_present、reproduced、validated。reproduced须指明复现对象，本轮复现的是Python函数负例/小原语差异，不是历史算子run；validated必须完成固定契约和全部适用门禁。

## 2. 环境与仓库边界

| 项 | 本轮检查 |
|---|---|
| 目录/主机 | `/home/nanajiang/Asim` / `nanajiangserver` |
| 实际资源 | nproc=2；内存4,009,860 KiB（约3.8GiB）；盘点时可用磁盘28,001,783,808 bytes |
| 初始Git状态 | 已跟踪文件干净；仅未跟踪 `DOCS/zhongqi/Communication_Operators_on_NPU_edit(6).pdf` 与 `DOCS/zhongqi/projects/` |
| 当前构建物 | `build/bin/Simulator`、`Simulator_test`存在；`build_asim/`不存在，与CLAUDE历史描述不同 |
| 工具链观察 | Python3.12.3、NumPy2.5.0、CMake3.28.3、G++13.3.0；系统PATH无Conan；不推断干净构建已可用 |
| build缓存 | Release、USE_RAMULATOR=ON、BRI_ENABLE_*均OFF；现存二进制源版本未绑定，只有内容摘要 |
| 依赖/远程 | `git submodule status`为空；未将其解释成依赖可重建。未读取凭据、未推送或声称异地备份 |
| 并发 | 只读审计3个Agent；候选生成至多2，编译-j1/仿真串行，避免共享build/stash |

初始命令为pwd、hostname、git status、git log、nproc、free、df；机器可读环境在 [environment.json](../../experiments/inventory/2026-10-28-baseline/environment.json)。环境快照是在清单目录开始创建后采集，故其git_status还包含本轮新增清单/脚本，不等于上表初始状态。既有README/CLAUDE及9月材料保留为各自时点上下文。

## 3. 算子现状

完整字段见 [统一状态卡索引](operators/INDEX.md)。下面是最先阻碍可信比较的问题，不是算法优劣判决。

| 实现 | 当前代码/文档主要差异 | 当前结论 |
|---|---|---|
| Cholesky NoBlock | 低层元素、共轭与初始化不完整；Model输出声明M×U而写回U×U | 代码存在，需真实语义迁移 |
| Cholesky Merge | 合并SCALAR compute_size=j，但时序模型固定标量延迟；公式Gram方向仍有差异 | 9,999历史周期收益需重审计时口径 |
| LDL NoBlock | Python行缩放，DAG列缩放；C++缺元素view；输出shape不一致 | 小样例已证实Python/DAG差异，不能归因算法固有退化 |
| Cholesky Block | 高层DAG分解掩盖低层块内标量与共轭缺口 | 6,440仅历史报告，未统一验证 |
| LDL Block | 缩放/重建及块内语义未对齐；B参数/尾块需审查 | 4,959仅历史报告，未统一验证 |
| NS | Model把H/RegI/X_init/C接给A/X0/C；忽略U按M²算；ADD代SUB；末尾真实X@X；默认K16未透传 | 当前入口先拒绝，不能由脚本默认输入补齐 |
| BRI | 预条件器与预条件系统含义不同；Ynext未交换回Yk；恢复/输出与DAG/Python不同；输出batch范围不足 | 最优候选的负责人结论与当前v3可执行正确性分开 |

另有ONNX NS和legacy优化实现，入口/算法/参数与当前Baseline不同；不将同名视为相同程序。状态卡记录mode、Model、registry、配置、原始run及待重跑命令。

## 4. 历史与数据结论

NS详细谱系见 [NS_LINEAGE](history/NS_LINEAGE.md)，台账见 [ledger.jsonl](../../experiments/history/ledger.jsonl)。历史四轮材料记录基线5,352/score7.13、R1块逆预热2,992/9.06，R3高阶Neumann、R4 BRI+NS等；涉及初值、K和算法变化，不能合计为纯流水收益。关键原始 `results/COMPLETE_SUMMARY.md` 与候选/输入绑定未找到，缺项显式null。

当前磁盘保留10个旧run包；NS旧run是32×32、batch96、K10的38,806周期，与README的2,591（N32,K8）及5,182估算不是同一次实验。Cholesky/LDL v2 run也与7月汇总不一致，部分版本带dirty diff描述。`results/ldl_noblock_v2/run_001/summary.json`含裸N/A，不是合法JSON；保留原件并记INVALID。

7月周报同文对NS K16在CDL-B给出0与0.0318，其他报告又有不同数字。台账保留矛盾及各自来源，不择优覆盖。现有CDL-B生成器是指数相关Kronecker平坦信道代理；本轮仅审计工程实现，未核对外部标准。历史有限失败只约束当时的实现、参数和预算。

完整文件清单包含9,140个文件（其中8,339项来自已跟踪.claude依赖）；211个原始/历史制品共67,488,854 bytes，原位保留。范围、逐文件SHA-256与搬迁前规则见 [制品目录](ARTIFACT_CATALOG.md)。清单是完整性证据，不能补出缺失的输入/源码版本，更不是远程备份。

## 5. 验证与评分可信边界

问题清单见 [审计报告](audits/VERIFICATION_AUTOMATION_AUDIT.md) 与 [issues.json](audits/issues.json)。FormulaLogger与实际Instruction分离；trace重放跳过搬运/Scalar/Vector并随机补输入；DAG失败后可能回退按算法名执行。旧scorer评价的是注册器计算出的逆，且奖励Cube/并行度；不能据此认证真实算子。旧CI存在命令拼接、PASS文本判定、数值可选等问题；共享stash与全局/tmp也阻碍并行隔离。

低成本检查命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/research_audit_checks.py \
  --output experiments/audit/<new-id>/diagnostics.json
```

本轮实际 [diagnostics.json](../../experiments/audit/2026-10-28/diagnostics.json) 的五项结果：

| 检查 | 观察 | 能证明的边界 |
|---|---|---|
| NaN误差 | compute_score接受，返回有效分数 | 特定函数缺finite门禁 |
| cycle=0 | compute_score接受并给高分 | 特定函数缺正周期门禁 |
| 缺trace | 一致性helper返回true | 不等于整个scorer必然通过，顶层可能先读trace失败 |
| LDL 2×2复HPD | DAG相对误差0.212498；行缩放Python约1.89e-16 | 实际Python原语/算法差异，非C++输出验证 |
| NS平方见证 | X保持真逆，X²的||AX²−I||F约0.901388 | 显式减法与合法输入假设下的数学反例，非实际指令重放 |

逐步量化后使用complex128/float64 matmul并不等于原生FP16累加。开发标准本身的“ADD作为复制”“unity作为零取负”“LDL列缩放”也存在冲突；保留历史原文，后续规范升级应单独版本化。

资料交叉检查结果见 [metadata_check.json](../../experiments/audit/2026-10-28/metadata_check.json)：49条历史、25项问题、16项依赖任务、10个run及86个原制品，当前研究文档链接与来源行号范围均通过。三个新脚本通过Python语法检查；清单工具在损坏清单和覆盖已有快照两个负例中都返回非零，记录见 [tool_checks.json](../../experiments/audit/2026-10-28/tool_checks.json)。211项原始/历史制品再次核对完整性通过。这些都是资料/工具检查，不是G0–G6算子验收。

## 6. 后续执行顺序

近期只建设“真实操作对象＋NS数值影子执行”，详细文件边界、schema、最小原语、三路径、负例和退出条件见 [实施包](audits/NS_SHADOW_EXECUTION_PLAN.md)。首个开发任务是冻结小内核契约及固定样例，建立输入绑定/初始化/范围的拒绝检查，然后接入受信builder和顺序执行器。当前错误入口必须作为失败基线保存；算法修复单独候选提交。

日常操作、预算、失败状态、隔离和构建命令见 [DAILY_WORKFLOW](DAILY_WORKFLOW.md)。依赖任务表见 [tasks.json](../../experiments/tasks.json) 与 [ROADMAP](ROADMAP.md)。新数值/调度/通信门禁未实现时停止在UNSUPPORTED、score=null；完成可复现最小链后才安排1个复现候选＋3个优化候选。完整IR、跨天无人值守控制器和大范围外循环搜索后置。
