# NS corrected v1：共同任务与结构候选

2026-10-30（Asia/Hong_Kong）。新增候选，不覆盖NewtonSchulzBaseline或历史结果。op-flow在Codex中按实际步骤执行，不声称调用slash命令。

## 1. 文件与任务范围

候选：src/inverse/ns_corrected/，mode ns_corrected_test。受信共同任务：verification/contracts/mmse.py；统一重放核心verification/reconstruction/，旧source-views和通用structure-program仅负责格式适配，不按算法名执行标准算法。独立NS参考verification/reference/ns.py不得调用指令重建器。

输入冻结为96个独立瑞利H(64×16)、Y(64×8)、λ=0.1；共同常量I、λI由任务提供并计入DMA。输出MMSE均衡结果16×8，接口编码为R(x)32×16。外部decode只做所有算法共享的表示解码，并检查四象限一致性；这里的输出恢复指X@b，不宣称原生复数封装成本。转为显式实块R(Z)=[[Re,-Im],[Im,Re]]，实际Asim尺寸加倍，成本precision8。只认证FP64结构，不称为真实910B FP16吞吐或通信验收。CDL-B未知时拒绝。

完整计时范围mmse_h_y_to_xhat_v1：输入搬入、HᴴH、正则化、HᴴY、初始化、迭代、输出恢复和写回全部计入。外部只做共享信道/噪声生成和实块编码，不生成A/b/X0供候选。既有given_A_b_X0_C测试为prepared_solver_kernel_v1，二者禁止直接排名。

## 2. 公式推导

A=HᴴH+λI为Hermitian正定，b=HᴴY，目标x=solve(A,b)。实块a=R(A)，q=R(b)，τ=trace(a)=2trace(A)。因0<λ_max(a)≤trace(a)，X0=I/τ满足ρ(I-aX0)<1。

X(k+1)=Xk(2I-aXk)。独立参考同一τ/初值/K，直接复数实现；参考输出XK b，另以complex128 solve(A,b)为任务真值。有限K时算法误差与指令错误分开：actual↔algorithm和algorithm↔task分别报告，不凭算法名自动通过。默认K16；低K应actual一致但任务失败。

因Asim没有独立Vector SUB，实际执行T=aX；N=T*(-1)；R=2I+N；X=X R，不能把ADD解释成SUB。-1由标量SUB构造：z=I00-I00，neg=z-I00。τ用逐对角SCALAR_MUL/ADD累计；X0用有显式scalar_rhs广播语义的DIV，MUL及DIV的成本均按完整物理矩阵元素×8byte计。输出XK q，不输出XK²。

## 3. SPAD布局

H 128×32，Y 128×16；Reg/I/A/X/T/negT/R各32×32；b32×16；τ/z/neg各标量；output位于ACCUM32×16。独立64byte对齐区域，按precision8计算。所有被读元素必须先写；输出槽声明[96,32,16]。禁止aW类似越界、隐藏swap或免费归约。

## 4. 指令映射与同步

LOAD H/Y/Reg/I→GEMM HᵀH→ADD Reg→GEMM HᵀY→SCALAR trace及负一→ADD I+I→DIV I/τ→K次GEMM/MUL/ADD/GEMM→GEMM Xb→STORE。每个计算/DMA后显式PIPE_BARRIER；源views与同一Instruction绑定，DMA base_addr=0，MOVOUT src_from_accum及last_inst。shape/工作量按物理实块计算。

## 5. FormulaLogger与可信证据

set_algorithm(ns_corrected_v1)及逐算术emit_step仅为旧工具诊断，重建依赖实际Instruction及source-views，忽略FormulaLogger。新算法只注册生成端与独立参考，不得在重建核心添加算法分支。

## 6. 验证计划

先独立build -j1，再Asim smoke，独立子Agent审计，固定96batch三路径验证及流水归档。至少三个Rayleigh种子；actual/algorithm误差≤1e-10，task误差≤1e-10，逐batch不得平均。另实际序列变异：残差符号/最终平方/跨batch输入；低K区分算法与任务失败。初次正确性建立基线，非竞争候选仍跳过数值，离谱加速隔离。完整FP16/G4/G6/通信统计未验收不发分。

状态：实现与受限结构门禁完成，见[验收报告](../research/NS_CORRECTED_VALIDATION.md)。

| op-flow阶段 | 实际执行证据 | 状态 |
|---|---|---|
| 设计/推导 | 本文编码前创建，τ=2trace(A)、残差展开及计时范围固定 | 完成 |
| v3源码检查 | base_addr0、barrier、初始化检查、正确MOVOUT；I为共同契约提供并计DMA，非将Reg当单位元 | 结构范围完成 |
| 构建 | 独立工作区-j1，[100%] Built target Simulator | 完成 |
| smoke | 96batch实际Asim 14,298周期，三路径通过 | 完成 |
| 独立审查 | legacy_semantic_review只读检查并复审Scalar修复 | 结构范围通过 |
| benchmark/归档 | 共同runner替代缺task_contract的旧benchmark脚本；三seed/trace/结果JSON归档 | 完成 |
| 数值 | 实际Instruction/独立NS/任务三个路径，全部96batch | 结构范围通过 |
| 精度/异步/通信/G6 | 完整门禁待开发，score=null | 未通过 |

FormulaLogger以ELEMENTWISE/GEMM记录算术构造；旧DAG不支持的op不作为新验收依据，不以旧可选数值规则替代新硬门禁。

## 2026-10-30精度模型扩展设计

同一NS公式新增modeled_fp16_fp32_v1：P=2的真实Asim请求/Vector工作量，与P=8结构模式分开。输入实块编码后一次FP16 RNE；GEMM内部按k升序非融合FP32乘后加，整条写回FP16；每条标量/Vector也单独写回FP16，保留subnormal，非有限拒绝。实际header绑定完整precision_policy，源视图执行器使用该冻结策略，独立参考自行实现同顺序，硬门禁逐FP16位一致（含signed zero）。任务参考仍为原始H/Y/λ的complex128 solve；2%任务目标只为模型工程目标，非硬件/通信验收。详见verification/profiles/README.md。
