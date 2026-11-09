# 隐式FP16点Jacobi阻尼Richardson MMSE候选

2026-11-09。op-flow new实际执行记录：先设计、再代码、串行构建、独立审计、真实Asim先行、实际指令重建及独立参考。数值来源experiments/fp16_implicit/2026-11-08-v3，已完成实际接入，CDL质量未全部通过。任务是H/Y/λ→xhat，不输出A逆，不沿用历史BRI周期。

## 1. 文件清单

新Model/Operation：src/inverse/bri_implicit/BRIImplicitModel.{h,cc}；mode bri_implicit_test；受信适配器/参考另行维护。旧算子不修改。独立worktree/build/log，-j1；批96/核24，Nr64/Nt16/RHS8。精度仅FP16存储/既有FP32临时算术模型。

## 2. 公式推导

A=HᴴH+λI，MMSE解Ax=Hᴴy。使用R(H)完整实块、单侧堆叠Y=[ReY;ImY]，状态X为32×8。P的每个复流对角p_j=1/(Q(h_jᵀh_j)+λ)，复制至实虚对应行；不保存完整Gram。初值X=previous=carry=0。

g=Q(Q(Hᵀ Q(Y−Q(HX))))−Q(λX))。lo=λ/4、hi=8、d=Q((hi+lo)/2)、c=Q((hi−lo)/2)。首步rho=Q(c/d)、alpha=Q(1/d)、beta=0；后续den=Q(Q(2d)−Q(c rho))，rho_new=Q(c/den)，alpha=Q(2/den)，beta=Q(Q(rho_new rho) Q(0.9))。区间是冻结试验常量，不宣称通用包谱。

δ=Q(Q(alpha Q(Pg))+Q(beta Q(X−previous)))；u=Q(δ−carry)；new=Q(X+u)；carry_new=Q(Q(new−X)−u)，previous←X，X←new。所有显式操作后FP16存储。最终从左侧Re/Im构造完整R(xhat)，四象限全部验收。

点P采用逐行MUL替代研究版含零GEMM；实数非零输出应等价，但有符号零也须与冻结独立参考位比较。若不一致须报告，不可修改参考掩盖。K13/33/76分别绑定λ1/.1/.01；先回归，再用预冻结新留出，不能调K自验。

## 3. SPAD布局

所有区域64B对齐：H128×32、Y128×16、RegI/Identity32×32；packedY128×8、X/previous/carry/new与向量临时32×8；HX/data_res128×8；expandedP32×8；scalar常量与系数各1×1。专用barrier区域按实际指令上界分配。输出ACCUM32×16。所有局部读之前有LOAD或实际算术写入，无免费FILL/COPY。

## 4. 指令映射

| 公式/布局 | 实际指令 |
|---|---|
| 共同输入/输出 | MOVIN四输入，MOVOUT完整batched输出，base_addr=0 |
| 常量0/1/2/8/.9/λ/4 | 由Identity与RegI的SCALAR算术构造；无外部算法专用输入 |
| 初值、复制、单侧Y/象限恢复 | Vector MUL(0或1)，负号用MUL(-1)+ADD(0) |
| 点P能量 | 每流1×128与128×1 GEMM，再SCALAR_ADD/DIV；实/虚p显式存储 |
| 每轮HX/Hᵀres | 两条实际GEMM，m/k/n与view相同 |
| Vector减法 | MUL(-1)+ADD，两个结果分别落FP16；不能raw ADD伪称SUB |
| P乘g | 预条件阶段32条逐行广播展开系数；迭代前ADD(+0)规范化gradient有符号零，再一条完整Vector MUL，按完整元素工作量计费 |
| 系数 | 单元素SCALAR_ADD/SUB/MUL/DIV |
| 前态/补偿 | 实际Vector复制/加减；每条写后PIPE_BARRIER |

## 5. FormulaLogger与泛式约束

FormulaLogger仅辅助索引；真实Instruction的src/dst view、切片、转置、标量广播、大小/计时为重建依据。公共engine/source_views/binding/runtime不加算法名分支。未知语义拒绝。全局硬件/精度模型不改。

## 6. 验证数据

待运行：真实Asim、事件/流水、同算法逐FP16位、MMSE≤2%、未编码QPSK、重复/负例/干净复现、分阶段墙钟与制品hash。性能预筛前不执行候选重建；代码存在不算通过。硬件时序后续上板。

## 7. 实际验收结果与问题

2026-11-09：独立-j1完整构建、静态审查、240测试通过。3档真实Asim模板288batch、新CDL576batch均与独立算法完整FP16输出位一致；周期16079/33999/72527，K13/33/76。新CDL0/10dB最大误差0.8743%/0.9376%通过；20dB H56002/b95误差5432.38%，不通过，score=null。

失败属于算法稳定域：原名义谱区间包住坏点，但β单独阻尼后的末步稳定上界7.6052低于谱max7.6388，完整76步独立误差多项式放大192.92。不得仅加K或更新日志来修复。当前候选为实验入口，不是全域可用算子；下一版需联动设计α/β、重新校准、新留出。

完整指令/流水/计时、重复与拒绝门禁、七显式入口回归和独立诊断见[实验报告](../../experiments/bri_implicit/2026-11-09-v1/README.md)。当前实际规范和技能为op-flow/op-dev/audit-operator；FormulaLogger只做辅助。
