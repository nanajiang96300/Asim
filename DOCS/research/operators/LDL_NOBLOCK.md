# 算子状态卡：LDL NoBlock v2

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
证据等级：`code_present`＋`documented`；本轮只读源码和历史制品，未构建、未运行算子、未完成数值复现或统一验证。当前状态：`legacy_unverified`。
路径与行号相对工程根目录，均指上述源码快照。通用精度／运行／证据规则见 [INDEX](INDEX.md)。

## 数学与输入输出

对 Hermitian 正定 A：`A=L D L^H`，L 为单位下三角，D 为正实对角。令 `Z=L^{-1}`，正确重建为 `Ainv=Z^H D^{-1} Z`。若用一次对称 GEMM，应令 **`Y=D^{-1/2}Z`（按行缩放）**，再计算 `Y^H Y`。分解本身可避免平方根；当前重建路径仍生成 SQRT。

输入 H `[batch,M,U]`、RegI `[U,U]`，意图输出逆矩阵 `[batch,U,U]`。Model 实际把输出声明为 `[batch,M,U]`，Op 的批次偏移为 U²。证据：`src/inverse/ldl_noblock/LDLNoBlockBaselineModel.cc:15`、`:32`；`src/inverse/ldl_noblock/LDLNoBlockBaselineOp.cc:50`。

## 实现、入口与参数

| 项 | 当前入口 |
|---|---|
| C++ / Model | `src/inverse/ldl_noblock/LDLNoBlockBaselineOp.{h,cc}` / `LDLNoBlockBaselineModel.{h,cc}` |
| mode / main | `ldl_noblock_v2_test`，`src/main.cc:154` |
| config / registry | `example/ldl_noblock_v2_test.json`；registry key `ldl_noblock` |
| 默认 | 64×16,batch=96；B=1；无迭代/预条件；每batch一tile |
| 精度 / 参考 | 配置precision=2；`scripts/algo/ldl_noblock.py` 使用complex128/float64；`scripts/verify/ldl_noblock_v2.py` 使用量化高层DAG |
| 文档 | `DOCS/operators/02_ldl_noblock_v2.md` |

## 实际指令与公式对应

`Op.cc:90`：LOAD → GRAM GEMM → REG ADD → 初始化 L/D/Dinv/Tmp/Y。`:139`：逐列 SCALAR_MUL/SUB/DIV 构建D和L。`:229`：前向回代；`:258`：每列 SCALAR_SQRT(Dinv) + SCALAR_MUL；`:278`：最终 GEMM_PRELOAD(aY,aY) → ACCUM → STORE。

缩放方向存在三方冲突：

| 路径 | 当前行为 | 证据 |
|---|---|---|
| 独立Python算法 | `Y=Z*sqrt(Dinv)[:,None]`，按行 | `scripts/algo/ldl_noblock.py:55` |
| DAG原语 | `Y=Z*sqrt_d[None,:]`，按列，随后Y^H Y | `scripts/uobs_dag_executor.py:143` |
| C++意图 | 注释写“sqrt(Dinv)*Z”，又写“scale column”；实际只有基地址，无法确定元素方向 | `Op.cc:258` |

按列缩放会得到 `D^{-1/2}Z^H Z D^{-1/2}`，一般不等于 `Z^H D^{-1}Z`。历史误差不能直接归因于 LDL 算法或 FP16：结构错误尚未排除。DAG 的 `LDL_DECOMPOSE`一次完成分解、回代与缩放，无法证明C++对应低层过程；验证脚本重复使用同一个错误原语。证据：`Op.cc:221`；`scripts/verify/ldl_noblock_v2.py:17`。

其他语义缺口：所有列/元素SCALAR地址重复，无切片/共轭；初始化ADD(RegI,RegI)并非零；`RegI−tmp`并非显式取负；回代乘法多次覆盖临时区而无明确归约。`:263`实际生成SQRT，与头文件“无SQRT”口径需区分分解与重建。

## 优化、历史结果与版本

- `legacy_operators/inverse/ldl_noblock/`有basic包装与aligned变体；`src/main.cc:159` 虽识别`ldl_noblock_merge_test`，随后直接报disabled。没有当前可用Merge算子，不把名称识别当成实现完成。
- `results/ldl_noblock_v2/run_002/summary.json`：20,733周期、96 tiles；64×16,batch96,precision2,24模拟核,Ramulator2；version=`e66a5d7 (1 file changed,18 insertions(+))`，缺差异快照。`run_001/summary.json` 含裸 `N/A`，不是合法JSON，且仅48tiles，不能当完成结果。
- `results/ldl_noblock/run_001`旧mode记录3,673周期，unversioned；不等于v2。
- `DOCS/reports/2026-07-08-7operator-summary.md:16`记录25,628周期、ΔSER(Rayleigh)=0.0139、CDL-B=0.9494；头部为64×16,20dB,10trials,DAG5seeds。报告数值属于历史记录，不能排除缩放方向/验证路径共同错误。
- `ddd89fb9`修改DAG合并覆盖；首行SHA为本轮静态源码版本。上述历史周期/质量尚未建立同run绑定。

## 本轮窄范围诊断证据

主Agent运行`experiments/audit/2026-10-28/diagnostics.json`中的2×2复HPD案例：`L=[[1,0],[0.5+0.25j,1]]`、`D=[2,5]`。实际DAG原语重建的相对误差为0.2124982367，而独立Python按行缩放为1.89e-16。该差异已在Python函数层复现，不能称为C++指令验证。开发标准`:157`的列缩放也需独立修订；旧报告“算法固有误差”的判断暂不成立为已验证结论。

## 复现入口与缺证任务

INDEX模板：`MODE=ldl_noblock_v2_test`，`MODEL=example/ldl_noblock_v2_test.json`。历史DAG诊断：`python3 scripts/verify/ldl_noblock_v2.py "$RUN_DIR/formula.json"`；阈值0.10及自比较不能用于新硬门禁。

优先建立非对角L、非均匀D的小矩阵，用独立solve验证按行/按列重建；再补齐低层view语义。不要先放宽精度阈值。原始输入/工具链/25,628 run包、实际FP16累加语义及缩放实现均待补证。
