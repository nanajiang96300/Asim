# Block Cholesky / Block LDL / 合法Merge显式映射设计

2026-11-08（香港）。独立候选，保留旧源码与旧失败。遵循op-flow实际设计/公式/构建/独立审查/归档流程。

## 1. 文件

src/inverse/structured_solve/StructuredSolveModel.{h,cc}，main三个独立mode；受信catalog注册三ID。verification/reference/structured.py独立模型。共用engine/source_views/binding/runtime不按算法名补值。

## 2. 公式推导

共同任务A=R(H)ᵀR(H)+λI，b=R(H)ᵀR(Y)，输出x=A⁻¹b，不额外生成全逆。分块大小4个实维度，全部预处理和恢复计时。

Cholesky-block：逐面板A11=L11L11ᵀ，块内逐列DIV/SQRT与真实秩一Schur更新；列递推解L21L11ᵀ=A21；A22←A22−L21L21ᵀ，一次GEMM内维4。先Lz=b，再Lᵀx=z。

LDL-block：块内LDLᵀ，L单位对角；Dj=Ajj，Lij=Aij/Dj，真实Dj加权秩一更新。面板递推L21D11L11ᵀ=A21，W=L21D11，A22←A22−WL21ᵀ。Lz=b后按行w_j=z_j/Dj，再Lᵀx=w。

Merge：左看Cholesky，Ajj←Ajj−L[j,:j]L[j,:j]ᵀ；列尾Aij←Aij−L[i,:j]L[j,:j]ᵀ，然后DIV。所有内积真实GEMM、内维j；三角解的行归约也用GEMM内维j或n-j-1。SCALAR只处理一个已完成的标量结果，绝不把compute_size=j解释为免费dot/sum。

## 3. 内存

共同H/Y/RegI/Identity；A/L方阵，D为1×2Nt（仅LDL），W为2Nt×4，rhs/forward/backwork为2Nt×2Nrhs，T/negativeT覆盖所有临时形状。L初始化分别zero/I；one=Reg00/Reg00、zero=I00−I00、neg1=zero−one。所有读前初始化；ACCUM整块xhat单次完整STORE。同步token按静态上界预留。

## 4. 指令/FormulaLogger

面板内逐元素SCALAR DIV/SQRT、整列/行真实Vector；所有负号、Schur减法与权重有实际MUL+ADD。跨列归约只能GEMM，明确m/k/n，按元素/算术字节计Vector。每数学指令source-views与emit_step；每步保守barrier；MOVIN base0、MOVOUT last/src_from_accum。

## 5. 独立参考和门禁

F64独立高精度Cholesky/LDL参考与任务真值；F16独立实块实现对应分组、逐指令RNE和逐k FP32累加，不导入重放核心/策略。默认24核、96batch、64×16、8RHS；第二组48batch、32×8、4RHS；瑞利seed42/123。先Asim/真实事件与成本再数值。未知归约、欠定义操作、非法SCALAR工作量都拒绝。

实际L/D阶段快照比对独立F16位模式，F64检查分解残差；输出三路径。负例删除Schur、改变归约等在离线记录中明确标注，不宣称运行错误候选二进制。编译/仿真串行，独立workspace/build；硬件认证外置。结果：主批次16case/1392batch、6case/576因素见证、4次日常384batch全部通过，151测试及独立审查通过。默认FP16 Cholesky-block21338、LDL-block21665、Merge19028周期；见DOCS/research/MERGE_AND_REMAINING_MAPPING.md与experiments/remaining_mapping/2026-11-08-block-merge-v1。
