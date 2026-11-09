# 算子状态卡：Block-Richardson

2026-11-09状态补充：以下主体保留2026-10-28旧入口审计。旧入口缺陷未因新候选通过而修复；`bri_explicit_v1`瑞利已验收、CDL-B失败；新隐式FP16候选仅独立数值留出通过，待原生映射。当前分类、证据与下一步见[优化计划](../BRI_OPTIMIZATION_PLAN.md)。

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
证据等级：`code_present`＋`documented`；本轮只读源码和历史制品，未构建、未运行算子、未完成数值复现或统一验证。当前状态：`legacy_unverified`。
路径与行号相对工程根目录，均指上述源码快照。通用精度／运行／证据规则见 [INDEX](INDEX.md)。

## 数学与输入输出

负责人确认BRI为已有研究最优候选（`owner_reported`）；此结论不等于当前v3代码已统一验证。用无歧义符号P表示块对角逆，`P=blockdiag(A_ii^{-1})`、`B=P A`。一种合法逆迭代为 `Y0=0`、`Y_{l+1}=Y_l+omega_l(I−B Y_l)`、`Ainv≈Y_L P`。固定omega时需rho(I−omega B)<1；Chebyshev需要可靠谱界；A为HPD时可考察与PA相似的Hermitian系统。

MMSE全流程还要 `W=Ainv H^H`、`Xhat=W Yin`。当前源码、DAG和Python使用不同初始化、B含义及输出恢复，不能拼接成同一算法的证据。

## 实现、入口与参数

| 项 | 当前入口 |
|---|---|
| C++ / Model | `src/inverse/block_richardson/BlockRichardsonBaselineOp.{h,cc}` / `BlockRichardsonBaselineModel.{h,cc}` |
| mode / main | `bri_v3_test`，`src/main.cc:177` |
| config / registry | `example/bri_v3_test.json`；registry key `block_richardson` |
| 默认 | M64,U16,batch96,B2,L8；每batch一tile；precision2计时配置 |
| 参数透传 | Op解析B/L（`Op.cc:16`），Model仅传batch_size（`Model.cc:27`），故JSON layers/block_size被忽略 |
| 输入 / 输出声明 | H和Yin均`[batch,M,U]`，RegI `[U,U]`；输出名Ainv却只分配`[U,U]`（`Model.cc:22`），Op按每batch U²偏移写出 |
| Python / DAG | `scripts/reconstruct_formula_se_compare.py:274`；`scripts/uobs_dag_executor.py:149`；`scripts/verify/bri_v3.py` |
| 历史说明 / 变体 | `DOCS/operators/06_block_richardson_v3.md`；`legacy_operators/inverse/block_richardson/`保存原多预条件器/调度实现 |

输出声明无batch，但bs默认96，`:58`的写回偏移会超出该tensor逻辑分配。中间aW按U²分配，但W维度U×M，M>U时也需要范围检查（`Op.cc:69`、`:210`）。nB=U/B没有尾块覆盖。

## 实际指令、Python与DAG对照

| 阶段 | 当前C++ Baseline | 独立Python算法 | DAG / verify |
|---|---|---|---|
| 预条件 | B=块A逆的标量2×2骨架；不生成B@A | P=块逆，B=P@A | prim_bri_precond只返回P，偶数U固定选B2 |
| 初值 | 注释Y0=I，但发ADD(RegI,RegI) | 全零Y0 | C++日志首轮名I；verify却从P.copy起 |
| 迭代 | BY=B@aYk，残差ADD(RegI,BY)，更新ADD→aYnext | R=I−B@Y；Y+=omega*R | 按公式名执行减法/加法，可与低层不符 |
| 轮间状态 | **没有实际swap或copy回aYk** | Y每轮更新 | 按Y_l逻辑名自然传递，与C++不同 |
| 恢复 | 未有Y@P；输出仍读取aYk | Ainv=Y@P | 最终GEMM(Y,Y)名Ainv |
| 最终结果 | 2次GEMM：(aYk@H^H)@Yin，意图Xhat | 返回Ainv | 不含输入Yin的Xhat，返回Y² |

C++证据：`Op.cc:124`(预条件标量都用aA/aB基地址，缺2×2元素/负号)、`:162`(初值)、`:168`–`:202`(aYk/aYnext为const地址且无交换)、`:204`–`:219`(输出与日志不同)。Python证据：`scripts/reconstruct_formula_se_compare.py:225`–`:242`、`:280`–`:302`。DAG证据：`scripts/uobs_dag_executor.py:157`(按尺寸猜block_size)、`scripts/verify/bri_v3.py:39`–`:45`(P初值与末尾平方)。因此当前Baseline无法声称实现了Python的同参数BRI。

当前Baseline对所有B都生成同一2×2标量骨架，不能把B>=4当作已有正确求逆路径。Gram/恢复缺共轭字段，ADD代SUB/常量I未编码，实际多核读写事件也未完整记录。

## 优化、历史结果与版本

- legacy `BlockRichardsonOp.cc`保留group_sync、fused_by_gemm、by_preload_period、kernel_fuse、iter_weight、adaptive_bounds和不同precond_solver；`VERSION.md`仍描述该旧实现。编译选项BRI_ENABLE_*不能据名称推断当前Baseline启用对应逻辑。
- `results/block_richardson/run_001`旧mode=`block_richardson_test`，64×16,batch96,B2,L8,group_sync2,adaptive_bounds1,iter_weight1,fused_by_gemm1,preload_period4；precision2,24模拟核,Ramulator2；summary=1,979周期、96tiles，version=`unversioned-20260629-130348`。
- `DOCS/operators/README.md:25`将1,993对应L8；`DOCS/reports/2026-07-08-7operator-summary.md:19`却对应L4。不可把1,993同时赋予两个L，当前Model又不透传L。
- 汇总`:47`记录Rayleigh B2,L4 Chebyshev ΔSER=0.0001；头条件64×16,20dB,10trials；`:58`、`:85`对CDL-B给失败解释，但未排除算法三路径差异及有限参数范围。旧失败不能证明永久不可用；应在可信语义下重新分组验证。
- 当前源码快照见首行；对应1,993及质量的候选commit、同run输入/二进制/评分器摘要缺失。BRI不同预条件器的历史负例继续保留，由台账记录尝试范围。

## 复现入口与缺证任务

INDEX模板：`MODE=bri_v3_test`、`MODEL=example/bri_v3_test.json`。历史DAG诊断：`python3 scripts/verify/bri_v3.py "$RUN_DIR/formula.json"`；脚本已注明self-consistency而非逆矩阵正确性。

先冻结P、B、Y、omega及恢复顺序，选定真正要迁移的legacy或新Baseline路线；核对初值/真实swap/PA构建/右乘P/最终Xhat，与solve(A,H^Hy)分开比对；校正tensor范围、参数透传与view语义必须另立变更。在此之前不存在可晋级的统一BRI基线。CDL-B重验需先审核信道模型，再固定条件数/相关度、精度、层数和全成本范围。
