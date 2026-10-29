# 原有七入口的真实指令重建与周期诊断

2026-10-29。最终运行 `2026-10-29-legacy7-b96-v2-frozen`。**Asim→实际指令重放→独立任务参考→失败归档已经打通；七入口尚未全部通过正确性门禁，可信周期排行榜仍 BLOCKED，score=null。**

本次按 `orchestrator/operator_registry.json` 完整覆盖六族与 Cholesky NoBlock Merge 共七入口。每入口 96 个不同瑞利输入，24 核每核4个tile，Cube16³。全部 Asim 正常结束、计算与屏障事件按dispatch/core/unit/name的多重集合完整匹配；全部96个batch分别执行重建，未平均掩盖错误。测试是有限代表性配置，不覆盖注册表全部尺寸。

## 1. 改动和证据边界

在真实 Instruction 构造位置附加 `source-views-v1` 元数据，记录本地区域、真实 Model 输入输出、行列切片与共轭/转置；没有更改原算术、地址、指令数量、初始化或调度来使算法通过。[设计](audits/LEGACY_SEMANTICS_DESIGN.md) 与 [136个构造点对照](../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/source_equivalence.json) 保存有序原始 Instruction initializer 的精确文本一致性检查，构建工作区源码hash匹配。

重建器只读取实际导出的 Instruction，逐条绑定opcode/操作数基址，执行初始化掩码、DMA槽位/起止batch/范围、单个完整输出、区域重叠和配置容量检查；不读取 FormulaLogger 作为执行依据。不唯一的索引/归约标记 `SEMANTIC_AMBIGUITY` 后停止该batch，属于语义不足，不能改报为数值正确或算法永久无效。元数据是待审查的源码解释，逻辑complex128切片/共轭不等于真实硬件FP16指令已经实现这些语义。

模型成本保持原配置precision=2，重建用complex128以排除舍入干扰；不是FP16验证或原生复数成本认证。独立高精度任务参考为 `solve(HᴴH+0.1I,I)`，BRI为 `solve(HᴴH+0.1I,HᴴY)`；H按 `1/sqrt(2M)` 归一化，seed42，原输入和参考均归档。七入口的独立同算法参考尚未接入，不宣称完整三路径通过。

本轮是历史正确性诊断，所有入口在Asim成功后进入重建，明确不按性能淘汰。这不改变候选优化链的“冻结基线→Asim→性能预筛→重建”；没有同任务的已验证基线，本批不得性能晋级。

## 2. 实际结果：按入口列示，未按速度排名

| 入口 | M/U，算法参数 | Asim总周期 | 实际指令数 | 96个batch重建结果 | 主要阻断 |
|---|---|---:|---:|---|---|
| Cholesky NoBlock | 64/16 | 23,444 | 239,232 | 96 SEMANTIC_AMBIGUITY | FWD_NEG零源下标不唯一；输出声明96×64×16，与逆矩阵96×16×16不符 |
| Cholesky NoBlock Merge | 64/16 | 10,004 | 77,952 | 96 SEMANTIC_AMBIGUITY | 融合SCALAR缺显式归约；同上输出声明不符 |
| LDL NoBlock | 64/16 | 25,633 | 297,792 | 96 SEMANTIC_AMBIGUITY | FWD_NEG零源下标不唯一；同上输出声明不符 |
| Cholesky Block | 64/16，B2 | 6,445 | 43,584 | 96 SEMANTIC_AMBIGUITY | TRSM缺完整列索引，不能从base猜测 |
| LDL Block | 64/16，B2 | 4,965 | 35,904 | 96 SEMANTIC_AMBIGUITY | LAPPLY缺加权／归约索引 |
| Newton–Schulz | 32/32，K16 | 4,647 | 9,792 | batch0 NUMERIC_FAILED；batch1 TENSOR_BOUNDS_ERROR；其余94 INPUT_BINDING_MISMATCH | Model提供H/RegI/X_init/C，Op按A/X0/C读取前三槽 |
| Block-Richardson | 64/16，B2，L8 | 1,998 | 10,656 | 96 BOUNDS_ERROR | W需要16×64，声明仅16×16；输出缺batch维 |

这里的周期只可作为源码诊断。NS尺寸、逆矩阵与BRI均衡结果的范围不同；precision=2的成本与complex128结构也不同，不能计算统一加速比。BRI当前入口的实现问题不能推广为BRI算法永久不适用，更不能据此否定负责人确认的其他候选。

## 3. 源码问题与下一步

完整机器台账：[issues.json](../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/issues.json)，各失败类型的首例包含实际ID、batch、索引和构造点；每batch结果在原件 `reconstruction.json`。当前为首错停止，未到达部分仍未验证。

- **NS输入与输出**：[Model](../../src/inverse/newton_schulz/NewtonSchulzBaselineModel.cc) 第18–28行给H/RegI/X_init/C；[Op](../../src/inverse/newton_schulz/NewtonSchulzBaselineOp.cc) 第42–44行按A/X0/C取前三槽，RegI又被按batch加偏移，未消费实际C。第105–109行残差是实际ADD，第137–140行最终输出是X@X。本轮保留两者；batch0完整重放相对任务参考误差1.0，不能归因成只有一个缺陷。
- **BRI**：[Op](../../src/inverse/block_richardson/BlockRichardsonBaselineOp.cc) 第70行W只分配U²，第211–218行写U×M，重建在BRI_W拒绝；[Model](../../src/inverse/block_richardson/BlockRichardsonBaselineModel.cc) 第25行输出为无batch的U×U。另有源码可见疑点：2×2逆的非对角分子缺负号；Y0实际为RegI+RegI，残差实际ADD；注释声称swap而第202–205行没有swap/copy。这些未被后续任务输出验证，须分别设计新候选与独立算法对照。
- **NoBlock三入口**：Model用shape_MU声明Ainv，而Op按U²批步长写回。Cholesky/LDL NoBlock的FWD_NEG仅aReg基址，原指令不能区分对角λ和非对角0；明确歧义，不选择一个值再判算法错误。其后的逆矩阵重建／LDL缩放方向尚未由本轮完整重放认证。
- **Merge与Block**：Merge的CHOL_NB_POTRF_SQ等“批量SCALAR”缺归约定义；Block的CHOL_BLK_TRSM_DIV、LDL_BLK_LAPPLY等缺列／加权归约。需要在新的版本中把真实工作量和访问定义补齐，才能继续，而不是让Python按标准Cholesky/LDL求解。新增FILL/COPY/NEG/REDUCE若必要，应在共同原语与成本模型中定义并测试。

另有 `NewtonSchulzOp.cc` 的 ONNX factory 路径，不属于当前七入口注册表；没有本轮冻结的ONNX任务/模型输入，**未测试**。旧VERSION中的opt mode/文件与当前main入口不一致，不能把七入口覆盖称为所有历史版本覆盖。

最值得执行的下一项：先冻结统一H/λ/Y与A输入的边界及计时范围，建立NS独立新候选，修复其Model/Op槽位、明示残差和最终输出，接入同初值/K的独立算法路径；随后消除NoBlock零源/Block归约歧义，重测BRI的预条件、迭代与输出恢复。每项使用独立源码/build/result，不覆盖本轮失败证据。本轮没有擅自实施这些算法修复。

## 4. 流水周期统计与归档

[Git摘要与完整hash清单](../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/README.md)，[48项流水摘要CSV](../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/pipeline_summary.csv)。原件专用目录：`results/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/`，与旧run分开。

每入口包括：实际instructions/dispatch、trace、FormulaLogger诊断、Asim日志/receipt、原输入npz、高精度任务参考npz、96batch重建结果、`pipeline_statistics.json`、`instruction_pipeline_detail.csv`。最后两项逐core/unit/opcode/完整instruction name统计事件数、时长和、区间并集、min/max/mean、首尾周期；另统计每核busy/idle和多单元重叠。保留冻结二进制与环境，98件原件约635MB，逐件bytes/SHA-256；不将大trace入Git。

流水包括Cube、Vector、Scalar、MTE2、MTE3、Barrier、Wait；NS无Scalar事件，不能杜撰时长。`duration_sum`是事件驻留时长之和，跨指令／核重叠；不能相加得到总周期。顶层active_union是跨核时间区间并集，不是24核有效算力利用率。MTE事件是请求fragment，MTE1无独立观测；Wait只有已记录CubeWait，旧ID重复，因此不是完整G4时序证明。

先前v1原位保留，标记SUPERSEDED_EXPLORATORY：独立审查指出零源对角解释不唯一，最终v2已改为歧义，不能拿v1的假定索引结论当算法定论。

## 5. 验收与复现

隔离工作区串行 `-j1` 构建成功；68项单元测试通过，包含未知schema、缺容量／重叠区域、起止batch越界、单完整STORE、空/缺core trace、计算事件缺漏、缺七入口等拒绝负例。独立只读审查发现的缺口已修复并复审通过；[审查记录](../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/review.md)仅认可结构诊断边界。[门禁记录](../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/acceptance.json)不代表算子正确。

冻结runner `a0b059c0`；C++二进制源码 `add6ba38`（隔离工作区）；本轮功能提交32bfb901、62608ed2、527c895e、388ceca1、f3063715。全部score=null。FW0–FW2仍in_progress；同算法独立参考、硬件精度、复数成本、G4、通信统计、G6和CDL-B均未验收。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s verification/tests -v
python3 scripts/check_reconstruction_archive.py experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/run_manifest.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.legacy.campaign \
  --binary /home/nanajiang/Asim/results/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/bin/Simulator \
  --config configs/ascend_910b_quiet.json --registry orchestrator/operator_registry.json \
  --cwd /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build \
  --output results/reconstruction_campaigns/<new-run-id>
```

复跑output必须不存在。该批预期退出1并留下CORRECTNESS_BLOCKED，不将非零退出改成PASS。固定7入口、Asim每入口120秒/1,000,000周期看门狗，输入和配置不得替换；未来代码变化应使用所列runner提交再复现。构建依赖用了本地副本，完整干净构建尚未认证。

追加兼容性回归（runner `0ee63e50`）：当前69单元测试通过，7项真实Asim框架探针得到预期结果。通用adapter对legacy schema明确转为UNSUPPORTED，legacy专用重建结果见本报告，不把两者混用。回归原件另存`results/reconstruction_campaigns/2026-10-29-framework-regression/`，[冻结清单](../../experiments/reconstruction_campaigns/2026-10-29-framework-regression/run_manifest.json)。七入口v2证据及原68测试日志未覆盖。
