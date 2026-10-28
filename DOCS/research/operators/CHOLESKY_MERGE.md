# 算子状态卡：Cholesky NoBlock Merge

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
证据等级：`code_present`＋`documented`；本轮只读源码和历史制品，未构建、未运行算子、未完成数值复现或统一验证。当前状态：`legacy_unverified`。
路径与行号相对工程根目录，均指上述源码快照。通用精度／运行／证据规则见 [INDEX](INDEX.md)。

## 数学与输入输出

意图保持 `A=H^H H+lambda I=L L^H`、`Y=L^{-1}`、`Ainv=Y^H Y`；仅合并标量内积实现，适用域与 NoBlock 相同。H 为 `[batch,M,U]`、RegI 为 `[U,U]`；输出意图为逆矩阵。Model 仍声明 Ainv 为 `[batch,M,U]`，实际写回 U×U。证据：`src/inverse/cholesky_noblock/CholeskyNoBlockMergeModel.cc:30`、`:32`；`src/inverse/cholesky_noblock/CholeskyNoBlockMergeOp.cc:50`。

## 实现、入口与参数

| 项 | 当前入口 |
|---|---|
| C++ / Model | `src/inverse/cholesky_noblock/CholeskyNoBlockMergeOp.{h,cc}` / `CholeskyNoBlockMergeModel.{h,cc}` |
| mode / main | `cholesky_noblock_merge_test`，`src/main.cc:149` |
| config / registry | `example/cholesky_noblock_merge_test.json`；registry key `cholesky_noblock_merge` |
| 默认 | M=64,U=16,batch=96；B=1；无迭代/预条件；按 batch 分 tile |
| 精度 / 验证 | precision=2 计时；无 Merge 专用 `scripts/verify` 入口；`scripts/test_all_operators.py` 有 synthetic 公式比较 |
| 设计文档 | `DOCS/operators/01b_cholesky_noblock_merge.md` |

## 实际指令与公式对应

骨架与 NoBlock 相同。`MergeOp.cc:125` 将 j 次 MUL(1)+SUB(1) 改成 MUL/SUB(compute_size=j)；`:149` 合并 TRSM 内积；`:192` 合并回代内积。指令仍是 SCALAR，没有显式向量归约算子，也没有元素 view。

关键问题：`src/SystolicWS.cc:234` 的 SCALAR 延迟按 opcode 固定返回，与 compute_size 无关。合并后减少发射数会自然减少模型周期，但未证明一条标量指令能完成 j 项乘法、求和和减法。需要把“有效数学工作保留”和“时延/吞吐模型”分别验证，不能把历史 2.34× 直接解释为有效算力提升。

公式日志也与 Baseline 有差异：`MergeOp.cc:98` 声明 `H @ H^H`，却声称输出 U×U；对于 M×U 的 H，该乘积为 M×M。`:144` 的 CHOLESKY 输出是 `L_diag`，后续 TRSM 又读 `L`；`:169` 的 TRSM(A,L) 被赋予逐元素输出，但现有原语是求完整 L 的逆；最后`:221`声明Y^H Y，缺少完整 FWD_SOLVE 公式链。这些属于声明本身待核实，不能以相同高层算法名代替。数值字段还继承 NoBlock 的索引、共轭、初始化和求和缺失。

## 优化、历史结果与版本

- 该变体属于内循环优化意图；在补齐算术与成本证据前，不能确认只改变调度。
- `DOCS/reports/2026-07-08-7operator-summary.md:14` 报告 9,999 周期，Rayleigh/CDL-B ΔSER=0；相比23,439约2.34×。报告头条件为64×16、20dB、10 trials、DAG 5 seeds；原始输入、同commit二进制、硬件计时配置、评分版本绑定缺失。
- `DOCS/operators/01b_cholesky_noblock_merge.md:7` 声称 SCALAR 向量化能力，`:52` 用 ceil(j/128) 估计延迟；当前标量时延实现不采用这一公式，保留为文档/源码冲突。
- 本地 results 常规 run 包未定位到该 9,999 专属完整包；raw_run=`unknown`，candidate_commit=`unknown`。当前可见源码以首行 SHA 为准。

## 复现入口与缺证任务

使用 INDEX 模板设置 `MODE=cholesky_noblock_merge_test`、`MODEL=example/cholesky_noblock_merge_test.json`。这是待重跑周期诊断命令；不能借用 NoBlock 的数学输出作为本变体验收。

优先任务：明确合并原语的读取view、乘法、归约和输出更新；校准 compute_size 对延迟影响；修复/重新派生公式链；与未合并实现比较同输入阶段值和实际工作量。全部修复应独立候选，不回写历史周期。
