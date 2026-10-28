# 算子状态卡：Cholesky Block v3

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
证据等级：`code_present`＋`documented`；本轮只读源码和历史制品，未构建、未运行算子、未完成数值复现或统一验证。当前状态：`legacy_unverified`。
路径与行号相对工程根目录，均指上述源码快照。通用精度／运行／证据规则见 [INDEX](INDEX.md)。

## 数学与输入输出

目标 `A=H^H H+lambda I=L L^H`。块Cholesky按对角块POTRF、非对角TRSM、trailing rank-k更新，再解 `Y=L^{-1}`、重建 `Ainv=Y^H Y`。适用于HPD矩阵；首版源码还隐含 `U%B==0`，未实现有效尾块。

Model声明H `[batch,M,U]`、RegI `[U,U]`、Ainv `[batch,U,U]`；输出意图是逆矩阵，不含均衡乘法。证据：`src/inverse/cholesky_block/CholeskyBlockBaselineModel.cc:17`。

## 实现、入口与参数

| 项 | 当前入口 |
|---|---|
| C++ / Model | `src/inverse/cholesky_block/CholeskyBlockBaselineOp.{h,cc}` / `CholeskyBlockBaselineModel.{h,cc}` |
| mode / main | `cholesky_block_v3_test`，`src/main.cc:162` |
| config / registry | `example/cholesky_block_v3_test.json`；registry key `cholesky_block` |
| 默认 | M64,U16,batch96,B2；无迭代/预条件；precision2周期配置 |
| 参数透传 | Op解析block_size并强制>=2，但Model仅传batch_size，JSON的block_size没有传入 |
| 参考 / 文档 | `scripts/verify/cholesky_block_v3.py`；`DOCS/operators/03_cholesky_block_v3.md` |

参数证据：`Op.cc:16`、`:46`（nB=U/B向下取整）；`Model.cc:21`。当前入口不能把配置中的非默认B当作实际运行B。

## 实际指令与公式对应

`Op.cc:88`：LOAD/GRAM/REG；`:115`：每块POTRF使用B×B GEMM与SCALAR_MUL/SUB/SQRT；`:155`：TRSM的GEMM与SCALAR_DIV；`:177`：RK GEMM写aA；`:190`：块回代；`:233`：最终U×U GEMM写ACCUM并MOVOUT。

实际块编号 j/i/k 只进入字符串ID，所有子块仍用aL/aA/aTmp整块基地址；未携带offset/stride、共轭、Schur负号与覆盖/累加规则。POTRF的GEMM产物aTmp没有显式减回对角aA；RK也无显式减法身份。nB向下取整使尾块不被分解，但最终仍按U×U输出。FormulaLogger反复对完整A调用CHOLESKY，以高层原语覆盖真实分块步骤，不能证明对应关系。证据：`Op.cc:120`、`:129`、`:152`、`:180`、`:229`；通用字段缺失见INDEX。

## 优化、历史结果与版本

- 当前基线不等于 legacy 的 `CholeskyInvOp` 或 `CholeskyInvChainOp`。旧Chain、solve_steps等选项保存在`legacy_operators/inverse/cholesky_block/`，不在当前CLI mode列表和src递归构建目录。
- 本地 `results/cholesky_block/run_001`：旧mode=`cholesky_test`，3,326周期、96tiles；64×16,batch96,B2,solve_steps1,precision2,24模拟核,Ramulator2；version=`unversioned-20260629-130229`。完整log/trace/config在该目录，但缺不可变候选版本，不能作为v3复现。
- `DOCS/reports/2026-07-08-7operator-summary.md:15`：v3周期6,440，Rayleigh/CDL-B ΔSER均0。报告头条件64×16,20dB,10trials,DAG5seeds；与6,440精确绑定的输入/二进制/完整工具链缺失。
- `08fff993`增补FWD_SOLVE日志、`ddd89fb9`后续DAG改动；这些只是日志语义维护，不等于低层指令已数值验证。

## 复现入口与缺证任务

INDEX模板：`MODE=cholesky_block_v3_test`，`MODEL=example/cholesky_block_v3_test.json`。历史DAG诊断：`python3 scripts/verify/cholesky_block_v3.py "$RUN_DIR/formula.json"`。

优先明确每块view与Schur减法/累加、B透传、有效尾块和Gram/重建共轭；以B=2及U=15/17反例检查覆盖。在支持尾块前应明确UNSUPPORTED。6,440对应原始run包、真实精度/质量统计仍缺证。
