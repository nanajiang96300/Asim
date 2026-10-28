# 算子状态卡：Cholesky NoBlock v2

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
证据等级：`code_present`＋`documented`；本轮只读源码和历史制品，未构建、未运行算子、未完成数值复现或统一验证。当前状态：`legacy_unverified`。
路径与行号相对工程根目录，均指上述源码快照。通用精度／运行／证据规则见 [INDEX](INDEX.md)。

## 数学与输入输出

目标为 `A=H^H H+lambda I=L L^H`，`Y=L^{-1}`，`Ainv=Y^H Y`。适用于 Hermitian 正定 A；复数必须使用共轭转置。输入 H 为 `[batch,M,U]`、RegI 为 `[U,U]`，意图输出 `[batch,U,U]` 的逆矩阵，不含接收信号或均衡乘法。

实际 Model 将 Ainv 声明成与 H 相同的 `[batch,M,U]`，Op 按每批 U×U 写回；M≠U 时逻辑 shape 与批次 stride 不一致。证据：`src/inverse/cholesky_noblock/CholeskyNoBlockBaselineModel.cc:15`、`:30`、`:32`；`src/inverse/cholesky_noblock/CholeskyNoBlockBaselineOp.cc:50`。

## 实现、入口与参数

| 项 | 当前入口 |
|---|---|
| C++ / Model | `src/inverse/cholesky_noblock/CholeskyNoBlockBaselineOp.{h,cc}` / `CholeskyNoBlockBaselineModel.{h,cc}` |
| mode / main | `cholesky_noblock_v2_test`，`src/main.cc:144` |
| config / registry | `example/cholesky_noblock_v2_test.json`；`orchestrator/operator_registry.json` 的 `cholesky_noblock` |
| 默认 | M=64,U=16,batch=96；B=1；无迭代与预条件器；每 batch 一个 tile，多核 round-robin |
| 精度 | 硬件配置 precision=2 只决定字节/计时；未有逐步 FP16 数值执行 |
| Python / DAG 检查 | `scripts/algo/cholesky_noblock.py`；`scripts/verify/cholesky_noblock_v2.py`；`scripts/test_all_operators.py` |
| 历史说明 | `DOCS/operators/01_cholesky_noblock_v2.md`；目录 `VERSION.md` 同时保留旧版信息 |

## 实际指令与公式对应

`Op.cc:87`：2×MOVIN(H,RegI) → barrier → GEMM_PRELOAD(Gram) → ADD(Reg)。`Op.cc:123`：每列 SCALAR_MUL/SUB/SQRT，随后 SCALAR_MUL/SUB/DIV 更新列、barrier。`Op.cc:178`：前向回代 SCALAR_DIV/MUL/SUB。`Op.cc:221`：GEMM_PRELOAD(aY,aY) → ACCUM，随后 MOVOUT。

以下字段尚不足以恢复数学语义：Gram 两个源均为 aH，无共轭/转置；所有 SCALAR 指令使用整块基地址，未带 `(i,j,k)` view；内积循环多次写 aTmp，却无显式求和对象；初始化 ADD(RegI,RegI) 不是明确的零填充；`RegI−tmp` 被解释为负和需要明确零源。最终 aY,aY 也无 `Y^H` 标志。FormulaLogger 的每列 CHOLESKY 实际调用全矩阵原语，不能证实这些低层步骤。证据：`Op.cc:94`、`:108`、`:127`、`:145`、`:196`、`:204`、`:225`（此节 Op 为上表 C++ 文件）。

## 优化、历史结果与版本

- 本卡为未合并标量操作的基线，Merge 见 [独立卡](CHOLESKY_MERGE.md)。旧 `legacy_operators/inverse/cholesky_noblock/` 包含原版／iso 等变体，不能把其周期直接归入 v2。
- 本地 `results/cholesky_noblock_v2/run_001` 和 `run_002` 保存 log、trace、formula、config、models_list、summary。summary 均记录 15,261 周期、96 tiles；64×16、batch=96、precision=2、24 模拟核、Ramulator2；version.txt 为 `e66a5d7 (1 file changed, 18 insertions(+))`，不是完整不可变源码包，diff 内容及二进制 hash 未归档。
- `results/cholesky_noblock/run_001/summary.json` 为旧 mode `cholesky_noblock_test`，6,142 周期，version=`unversioned-20260629-130250`，独立保留。
- 7 月汇总报告记录 23,439 周期和 Rayleigh/CDL-B 的 ΔSER=0，但周期与质量来自不同路径。报告条件为 64×16、SNR=20dB、10 trials、DAG 5 seeds；batch、输入、二进制和评分器完整绑定缺失。证据：`DOCS/reports/2026-07-08-7operator-summary.md:5`、`:13`、`:45`、`:54`。
- `ddd89fb9` 为后期 DAG 链修正相关历史提交；当前审计源码绑定首行 full SHA。报告中的 PASS 不升级为 validated。

## 复现入口与缺证任务

使用 INDEX 的独立工作区模板，设置 `MODE=cholesky_noblock_v2_test`、`MODEL=example/cholesky_noblock_v2_test.json`。可用 `python3 scripts/verify/cholesky_noblock_v2.py "$RUN_DIR/formula.json"` 做历史 DAG 诊断；该命令不验证真实指令。

下一步：补 Model 输出 shape、共轭/元素索引、显式填充及内积归约的契约，先记录当前失败再以独立变更修复；增加复数非对称已知解与未初始化读负例。原始输入、工具链、每步精度和 23,439 的精确 run 包均 `evidence_missing`。
