# 算子状态卡：LDL Block v3

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
证据等级：`code_present`＋`documented`；本轮只读源码和历史制品，未构建、未运行算子、未完成数值复现或统一验证。当前状态：`legacy_unverified`。
路径与行号相对工程根目录，均指上述源码快照。通用精度／运行／证据规则见 [INDEX](INDEX.md)。

## 数学与输入输出

意图块化 `A=L D L^H`，求逆应为 `L^{-H}D^{-1}L^{-1}`。若D为实对角，令 `Z=L^{-1}`、`Y=D^{-1/2}Z`按行缩放，再 `Y^H Y`；若D为完整HPD对角块，需明确每块逆/平方根或等价加权乘法，不能直接使用逐元素sqrt。

Model输入H `[batch,M,U]`、RegI `[U,U]`，输出Ainv `[batch,U,U]`。源码的块循环内仍逐标量构造D，没有证明完整块D分解语义；记录为实现待澄清。证据：`src/inverse/ldl_block/LDLBlockBaselineModel.cc:17`、`src/inverse/ldl_block/LDLBlockBaselineOp.cc:132`。

## 实现、入口与参数

| 项 | 当前入口 |
|---|---|
| C++ / Model | `src/inverse/ldl_block/LDLBlockBaselineOp.{h,cc}` / `LDLBlockBaselineModel.{h,cc}` |
| mode / main | `ldl_block_v3_test`，`src/main.cc:167` |
| config / registry | `example/ldl_block_v3_test.json`；registry key `ldl_block` |
| 默认 | M64,U16,batch96,B2；无迭代/预条件；precision2周期模型 |
| 参数透传 | Op可解析block_size>=2；Model仅透传batch_size，所以配置block_size不起作用 |
| Python / DAG / 文档 | `scripts/verify/ldl_block_v3.py`使用`prim_ldl_decompose`；`scripts/validate_ldl_block_operator_python.py`为独立算法检查；`DOCS/operators/04_ldl_block_v3.md` |

证据：`Op.cc:16`、`:47`；`Model.cc:21`。nB=U/B向下取整，没有尾块分解或整除拒绝。

## 实际指令与公式对应

LOAD/GRAM/REG和初始化见`Op.cc:90`；块D更新GEMM+SCALAR_MUL/SUB/DIV见`:132`，非对角L更新见`:173`，前向回代见`:199`，sqrt缩放见`:227`，最终U×U GEMM(aY,aY)和STORE见`:243`。

所有块运算仍用相同aL/aD/aA基地址，缺子块view、共轭、累加与Schur减法数据流。D更新GEMM写aTmp后没有明确完整的 `L D L^H`乘积与减回；不能仅看ID推断正确。

缩放方向与 [LDL NoBlock](LDL_NOBLOCK.md) 具有同样风险：`Op.cc:201`声称Y=Z*sqrt(Dinv)，DAG `scripts/uobs_dag_executor.py:145`按列缩放，再Y^H Y；独立Python按行才得到正确重建。`:230`实际生成SCALAR_SQRT，头文件“无SQRT”只能用于分解阶段，不能描述整个内核。`LDL_DECOMPOSE`高层原语同时代替分解/回代/缩放（`:194`），会掩盖真实分块错误。

## 优化、历史结果与版本

- 旧版 `legacy_operators/inverse/ldl_block/LDLDecompOp` 支持pack_blocks/bwd_steps等；`VERSION.md`提到2×2打包到16×16、2869周期等，属于旧实现报告，当前Baseline没有这些属性，不能沿用为当前优化。
- `results/ldl_block/run_001`：旧mode=`ldl_test`、1,682周期、96tiles；64×16,batch96,B2,bwd_steps1,precision2,24模拟核,Ramulator2；version=`unversioned-20260629-130334`。
- `DOCS/reports/2026-07-08-7operator-summary.md:17`：v3 4,959周期、Rayleigh ΔSER=0.0136、CDL-B=0.9503；头条件64×16,20dB,10trials,DAG5seeds。该报告还将误差归于算法/精度，但缩放方向与低层指令未排除，结论需重验。
- `d0bf27e8`将DAG改为单一LDL_DECOMPOSE、`ddd89fb9`合并覆盖声明；不是完整指令数值验收。

## 本轮窄范围诊断证据

与NoBlock共用的DAG原语在[诊断制品](../../../experiments/audit/2026-10-28/diagnostics.json)的2×2复HPD例子中出现0.2124982367相对误差，正确行缩放Python为1.89e-16。这只复现共享Python/DAG缩放问题，不证明C++ Block已经重放或全部LDL输入均失败。

## 复现入口与缺证任务

INDEX模板：`MODE=ldl_block_v3_test`、`MODEL=example/ldl_block_v3_test.json`。历史DAG诊断：`python3 scripts/verify/ldl_block_v3.py "$RUN_DIR/formula.json"`；“algorithm-equivalence”标签不证明C++等价。

任务：区分标量D与块D契约；证明正确重建/缩放方向；明确块view、累加、尾块及B透传；保留当前错误为负例。4,959原始run、精度分步语义、输入和同程序SER证据均待补。
