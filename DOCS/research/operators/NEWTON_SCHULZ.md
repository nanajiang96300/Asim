# 算子状态卡：Newton–Schulz

审计日期：2026-10-28（Asia/Hong_Kong）；源码快照：`7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。
证据等级：`code_present`＋`documented`；本轮只读源码和历史制品，未构建、未运行算子、未完成数值复现或统一验证。当前状态：`legacy_unverified`。
路径与行号相对工程根目录，均指上述源码快照。通用精度／运行／证据规则见 [INDEX](INDEX.md)。

## 数学与输入输出

标准求逆NS：`T_k=A X_k`、`R_k=2I−T_k`、`X_{k+1}=X_k R_k`；在 `rho(I−A X0)<1` 时收敛到A^{-1}。对HPD A，`X0=I/||A||2`是文档采用的初值；初值/谱估计有成本，不能免费省略。内核输入应为 `[batch,N,N]` 的A和X0，加显式常量2I；目标输出X_K。

当前实际v3 Model却连接H `[batch,M,U]`、RegI `[U,U]`、X_init `[batch,U,U]`、C `[U,U]`。Op仅消费前三项当A/X0/C，取N=M并忽略U。RegI被按batch偏移加载，X_init首批被当共享C，第四输入不读；即使M=U也不成立。证据：`src/inverse/newton_schulz/NewtonSchulzBaselineModel.cc:17`、`:27`；`src/inverse/newton_schulz/NewtonSchulzBaselineOp.cc:33`、`:41`、`:78`。Tensor只有地址/shape，未初始化真实X0与2I（`src/Tensor.cc:21`）。

## 实现、入口与参数

| 版本 | 代码 / 入口 | 当前状态 |
|---|---|---|
| Baseline v3 | `src/inverse/newton_schulz/NewtonSchulzBaselineOp.{h,cc}`、`NewtonSchulzBaselineModel.{h,cc}`；main:172 mode=`newton_schulz_v3_test` | registry `newton_schulz`，示例`example/newton_schulz_v3_test.json`为N32,batch96 |
| 当前ONNX NS | `src/inverse/newton_schulz/NewtonSchulzOp.{h,cc}`；`OperationFactory.cc:60`注册ONNX op_type=`NewtonSchulz` | 与v3是不同代码路径；默认10轮 |
| 历史Opt / Model | `legacy_operators/inverse/newton_schulz/NewtonSchulzOptOp.{h,cc}`及Model，旧mode=`newton_schulz_opt_test` | 不在当前CLI modes；不在src递归构建 |

Baseline默认K=16（`BaselineOp.h:20`）；Op能解析iterations，但Model只传batch_size（`BaselineModel.cc:23`），JSON iterations不会生效。precision2为周期字节设置。Python参考有`newton_schulz_numeric_test.py`（转置范数初值，返回X）、`evaluate_ns_se.py:22`（谱范数初值，complex64，返回X）；`verify/newton_schulz_v3.py:19`固定I/10并在末尾平方。三者不是同初值、同K的统一验证。

## 实际指令与公式对应

Baseline：3×MOVIN → barrier；每轮 `GEMM_PRELOAD(aA,aX)→aT` → barrier → **`ADD(aC,aT)→aR`** → barrier → `GEMM(aX,aR)→aX` →非末轮barrier；最后**额外 `GEMM_PRELOAD(aX,aX)→ACCUM`** →barrier→MOVOUT。证据：`BaselineOp.cc:76`、`:88`、`:101`、`:114`、`:127`。

默认K16每批33条GEMM、16条ADD、49条barrier、3条MOVIN、1条MOVOUT，共102条，这是源码循环静态计数，未运行验证。C−T的SUB身份未编码；原地aX更新需明确先读后写；常量/初值没有数值来源。若修正输入并按减法解释迭代，X趋向A^{-1}时，最终平方趋向A^{-2}；不能把X@X当求逆结果。现阶段应忠实保留最终平方并报告契约失败。

ONNX路径第二轮起旧X与T复用`addr_T`：T=A@X覆盖旧X后，后续X@R仍读该地址；属于跨指令覆盖风险（`NewtonSchulzOp.cc:217`–`:265`）。旧Opt分块GEMM对m0/n0/k0未编码view/累加（`legacy_operators/inverse/newton_schulz/NewtonSchulzOptOp.cc:257`），需独立迁移。

## 优化、历史结果与版本

- `results/newton_schulz/run_001`是旧Opt入口、N32,batch96,iterations10,precision2,24模拟核,Ramulator2，summary=38,806周期但仅24tiles；version=`unversioned-20260629-130336`。log/trace/formula存在，尚无固定commit/输入/二进制绑定，不算完整96batch基线。
- `DOCS/operators/README.md:24`与`05_newton_schulz_v3.md:82`记N32,K8=2,591；7月汇总把它放入U16标题表，且K16≈5,182是线性估算。证据：`DOCS/reports/2026-07-08-7operator-summary.md:18`、`:21`、`:68`。
- 同报告`:46`记谱初值K16 Rayleigh ΔSER=0；`:55`–`:57`记CDL-B K32=0、K16=0.021、K8=0.9409。这是独立Python质量路径，不能绑定到C++最终平方指令。`verify/newton_schulz_v3.py:28`–`:36`参考同样计算X@X，只证明DAG自一致。
- `10fff833`增加BWD_ASSEMBLE与2I日志；`02f7f518`修改默认K16；历史初值/预热/流水谱系由研究台账单列。本卡不把这些变化当纯内循环收益。

## 本轮数学见证

[诊断制品](../../../experiments/audit/2026-10-28/diagnostics.json)选`A=diag(2,4)`、`X0=diag(1/2,1/4)`、`C=2I`、K2。明确使用减法的NS迭代保持X0；输出X残差为0，输出X²残差为0.9013878。该见证说明平方改变求逆契约；它假定有效输入和显式SUB，没有执行当前不完整的C++指令。

## 复现入口与缺证任务

INDEX模板：`MODE=newton_schulz_v3_test`、`MODEL=example/newton_schulz_v3_test.json`。历史诊断：`python3 scripts/verify/newton_schulz_v3.py "$RUN_DIR/formula.json"`；PASS不满足求逆任务。

首要开发：在真实构造入口接入typed builder和shape/输入绑定检查，先拒绝当前不完整语义；补显式SUB、内存view和固定X0/2I，重放T/R/X/最终STORE，分别对比同参数NS和solve。当前算子错误的修复另立候选，不在解释器中修补。支持前明确UNSUPPORTED；FP64结构通过不代表FP16或异步调度通过。
