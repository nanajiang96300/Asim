# 六种算子与优化变体状态卡索引

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
本目录是当前审计状态；`DOCS/operators/`、各算子`VERSION.md`、周报和results保留历史原文。新卡不回写或覆盖旧结论。

## 范围与结论

| 算子 / 变体 | 卡片 | 当前CLI mode | 输出意图 / 实际证据边界 |
|---|---|---|---|
| Cholesky NoBlock v2 | [状态卡](CHOLESKY_NOBLOCK.md) | cholesky_noblock_v2_test | Ainv；实际元素/共轭语义不完整，Model输出shape不一致 |
| Cholesky NoBlock Merge | [状态卡](CHOLESKY_MERGE.md) | cholesky_noblock_merge_test | Ainv；标量合并成本及数学工作未证明、DAG链也有差异 |
| LDL NoBlock v2 | [状态卡](LDL_NOBLOCK.md) | ldl_noblock_v2_test | Ainv；DAG列缩放与正确行缩放不一致 |
| Cholesky Block v3 | [状态卡](CHOLESKY_BLOCK.md) | cholesky_block_v3_test | Ainv；子块view/Schur减法/尾块未编码完整 |
| LDL Block v3 | [状态卡](LDL_BLOCK.md) | ldl_block_v3_test | Ainv；缩放方向、D含义及子块数据流待核实 |
| Newton–Schulz v3 | [状态卡](NEWTON_SCHULZ.md) | newton_schulz_v3_test | 实际生成最终X@X，且输入绑定错位、SUB身份缺失 |
| Block-Richardson v3 | [状态卡](BLOCK_RICHARDSON.md) | bri_v3_test | C++意图Xhat、DAG却Y²；无轮间swap、预条件系统不一致 |

全部七个入口为`code_present`，整体数值验收状态均为`legacy_unverified`。本轮没有任何算子获得`validated`。源码问题与历史数学算法的优劣必须分开：尤其不能根据现存LDL缩放错误或BRI不同路径的失败，断言算法固有不可用。

统一卡片字段为：数学与适用域；输入输出；源码/Model/mode/config/registry/验证入口；参数和精度；实际指令对照；优化与历史结果；版本/制品；待执行诊断与缺证任务。卡片中的`Op.cc`、`Model.cc`是其入口表所列类文件的简称；`unknown`/`evidence_missing`表示尚无可追溯证据，不补造参数。

证据等级遵循研究总入口：`owner_reported`、`documented`、`code_present`、`reproduced`、`validated`。它们应附在具体结论上，例如“小矩阵Python反例已复现”不升级整个C++算子。

## 入口、注册与变体边界

- 当前七个入口由`src/main.cc:144`–`:181`直接实例化对应Model；`orchestrator/operator_registry.json`可定位路径和默认配置，注册表描述不是数学验收。
- ONNX `NewtonSchulz`额外通过`src/operations/OperationFactory.cc:60`绑定`src/inverse/newton_schulz/NewtonSchulzOp.cc`；它与CLI Baseline v3不同。
- `src/main.cc:159`的`ldl_noblock_merge_test`显式disabled。仅有mode名称与示例JSON不等于可执行算子。
- `src/CMakeLists.txt:8`只递归收集`src/`，`legacy_operators/`不被自动编入当前二进制；恢复旧实现须在独立工作区显式移植/注册并冻结版本。

| 历史族 | 保留的实现 / 配置意图 | 当前边界 |
|---|---|---|
| Cholesky NoBlock | `legacy_operators/inverse/cholesky_noblock/CholeskyInvNoBlockOp`；strict_iso_lowering等 | 旧`cholesky_noblock_test`不是当前有效mode |
| LDL NoBlock | `LDLDecompNoBlockOp`（包装block类）、`LDLDecompNoBlockAlignedOp`（独立aligned） | 均在legacy；不等于当前Baseline |
| Cholesky Block | `CholeskyInvOp`、`CholeskyInvChainOp`及ChainModel | block/chain/solve_steps等历史入口单独恢复 |
| LDL Block | `LDLDecompOp`；pack_blocks、bwd_steps | 当前Baseline不实现旧打包属性 |
| NS | legacy原版和`NewtonSchulzOptOp`；预热/初值/K/分块/调度 | 旧opt mode不在当前main；当前ONNX原版另有覆盖问题 |
| BRI | legacy `BlockRichardsonOp`：2×2、direct、Cholesky、fallback预条件，group_sync、fused_by、权重/谱界等 | 与当前232行Baseline是不同实现；编译选项也须按源码路径核实 |

## 共通语义与精度缺口

`src/Common.h:78`的Instruction主要保存基地址、尺寸、时序；缺少显式复数layout、转置/共轭、元素view、立即数、逐步dtype及融合表达式。因此有SCALAR_SUB的指令仍无法定位哪一个元素；没有SUB的ADD更不能由日志猜成减法。`src/Tensor.cc:21`和`:105`只分配地址/shape；`set_produced()`不提供输入值。

`src/operations/Operation.cc:200`的make_address按DRAM transaction向下对齐；恢复数值时必须另存逻辑元素offset、有效范围和尾块mask，不能按请求地址猜内容。输出、SPAD、ACCUM必须用真实view和core/buffer区分。

`configs/ascend_910b_quiet.json`设precision=2、24模拟核、Cube16×16×16、Ramulator2，precision是元素字节/成本设置，不能证明硬件FP16数值。`scripts/uobs_dag_executor.py:21`把值经float16量化后转回float64，`:36`的matmul在高精度复数组上执行，`:69`的Cholesky用complex128；LDL分解同样用complex128/float64（`:104`）。部分算法脚本使用complex64，其他使用complex128。乘法、累加、存储格式必须分别记录。

旧开发标准自身也有冲突：`DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md:149`用aUnity作为零源，注释却写0−sum；`:157`给出LDL列缩放。审计保留原规范，后续基础设施修订应配独立数学反例，不照抄规范来证明实现正确。

## 历史数据如何使用

每卡分别记录：①本地raw run（有log/trace/config但可能无commit）；②7月报告观察；③当前源码。报告的周期与SER不是同一实际程序结果，不能组合成统一排行榜。

本地run通用硬件条件可由相应`config.json`读取：precision2、24模拟核、Cube16³、Ramulator2；各卡列出batch和算法配置。`unversioned-*`不是commit，短commit后带“1 file changed”的版本也缺少diff快照。原始输入种子、二进制、完整工具链和评分器hash大多未绑定。报告的64×16、20dB、10 trials、DAG5seeds不自动适用于所有周期记录；NS还存在N32与U16标题混用。

BRI 1,993既被记录为L4也被记录为L8；NS K16约5,182是估算；Merge的SCALAR合并可能改变有效工作和模型成本。所有此类条目只能保留`documented`，不作为新冠军。

## 待执行的周期诊断模板

本轮未执行下列构建或仿真。它用于从当前commit重新检查入口；历史mode恢复和质量验收需要另外的版本化任务。先按[项目管理](../PROJECT_MANAGEMENT.md)准备独立worktree。2 vCPU服务器一次只构建/仿真一个任务，`-j1`；失败保留日志，不直接扩大搜索。

```bash
# WORKTREE、RUN_DIR必须是绝对路径；MODE、MODEL由状态卡选择。
# 示例：MODE=newton_schulz_v3_test
# MODEL=example/newton_schulz_v3_test.json
cmake -S "$WORKTREE" -B "$WORKTREE/build_audit" > "$RUN_DIR/configure.log" 2>&1
cmake --build "$WORKTREE/build_audit" --target Simulator -j1 > "$RUN_DIR/build.log" 2>&1
cd "$WORKTREE/build"
timeout 120s env ONNXIM_HOME="$WORKTREE"   ONNXIM_TRACE_CSV="$RUN_DIR/trace.csv"   ONNXIM_FORMULA_JSON="$RUN_DIR/formula.json"   "$WORKTREE/build/bin/Simulator"   --config "$WORKTREE/configs/ascend_910b_quiet.json"   --models_list "$WORKTREE/$MODEL" --mode "$MODE" --log_level info   > "$RUN_DIR/simulation.log" 2>&1
```

使用前创建唯一RUN_DIR，并在每个命令后检查退出码；构建失败不得继续仿真。根`CMakeLists.txt:20`硬编码二进制输出为源码工作区`build/bin`，所以`-B build_audit`不会把二进制放在`build_audit/bin`。选择独立worktree仍能隔离该输出。运行cwd设为worktree/build，是因为配置中Ramulator路径为`../configs/...`并由`src/Dram.cc:143`直接使用。若改为绝对路径配置，保存派生配置与hash，不混用旧周期。

上述默认batch96只是历史入口诊断；最小语义测试应使用独立归档的batch1小矩阵配置，不能改共享示例或依靠默认值。周期仿真完成只标记`TIMING_OBSERVED`，FormulaLogger诊断完成只标记`DAG_DIAGNOSTIC`；在可信影子执行、调度和任务质量门禁完成前score=null。

## 近期交叉验证证据

主Agent的[小矩阵诊断制品](../../../experiments/audit/2026-10-28/diagnostics.json)复现了Python/DAG层的LDL方向差异，并给出NS最终平方的数学见证；详情见对应卡。它没有执行C++指令，不能称为算子重放通过。待办依赖与验收以[路线图](../ROADMAP.md)为准。

## 最新实际重建（2026-10-29）

当前七个注册入口已执行24核/batch96真实Asim及重建，详细结果见[统一报告](../LEGACY_RECONSTRUCTION_REPORT.md)。五入口因原索引/归约歧义停止，BRI越界、NS数值/输入绑定错误；没有入口完成全部正确性验收。历史状态卡不覆盖这些新事实，也不能据旧DAG PASS晋级。
