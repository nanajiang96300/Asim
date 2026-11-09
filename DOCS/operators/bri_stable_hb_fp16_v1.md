# 固定HB隐式FP16 MMSE原生候选

2026-11-09。op-flow new：设计先于代码，独立worktree/build/log，先Asim后实际重建。前置数字方案及冻结K见experiments/fixed_hb/2026-11-09-v1，数值通过不替代本算子验收。

## 1. 文件与任务

src/inverse/bri_stable_hb/BRIStableHBModel.{h,cc}；mode bri_stable_hb_test；codex adapter bri_stable_hb_fp16_v1。共同H/Y/RegI/Identity→完整batched xhat，Nr64/Nt16/RHS8，24核/batch96，FP16存储/现有FP32临时算术。旧候选不修改。

## 2. 公式推导

A=HᴴH+λI，目标Ax=HᴴY。复点P的p_j=Q(1/Q(e_j+λq))，e_j为真实实块H首16列的有序FP32 dot→FP16。每矩阵计算一次固定α/β：Eu=Q(9/8*Q(8*sqrt(sqrt(sum_Q Q(Q(e/8)^2)^2))))；Pu=Q(9/8*Q(128*sqrt(sum_Q Q(p/128)^2)))；l=Q(15/16*Q(λq/Q(Eu+λq)))；u=Q(9/8*Q(sum_Q Q(pe)+Q(λq Pu)))。每个运算分别FP16写回。

rU=Q(sqrt(u)),rL=Q(sqrt(l)),den=Q(rU+rL)；α=Q(15/16*Q(4/Q(den²)))，β=Q(Q(Q(rU−rL)/den)²)。全程固定；不单独阻尼β。支持域e_j∈[2^-8,8]、sum_Q e∈[8,32]，λ1/.1/.01。域与实际系数稳定条件由独立参考/验证器拒绝检查，不能裁剪输入；不是免费oracle谱估计。

g=Q(Q(HᵀQ(Y−Q(HX)))−Q(λqX))；Pg用展开对角逐元素MUL，先ADD(+0)明确零语义。δ=Q(Q(αPg)+Q(βQ(X−previous)))。u_step=Q(δ−carry)，new=Q(X+u_step)，carry=Q(Q(new−X)−u_step)，previous←X，X←new。最终显式构造全部R(xhat)象限。K按校准冻结为16/53/174，不从算子留出调参。

固定线性模态条件0≤β<1、α>0、αu<2(1+β)覆盖完整迭代；谱界舍入证明见DOCS/research/F16_STABLE_HB_DESIGN.md与数学审查。此证明不替代逐操作FP16/补偿误差验收。

## 3. SPAD/ACCUM

沿用已验收单侧隐式候选布局：H128×32/Y128×16；packedY128×8；X/previous/carry/new/delta/P/向量临时32×8；HX与残差128×8；所有标量独立64B对齐。P和完整ACCUM输出32×16在局部写之前完整分配。辅助barrier区域按本K上限充分预留。所有首次读必须有真实LOAD或算术初始化。

## 4. 实际指令映射

能量dot用1×128与128×1真实GEMM；范数/迹归约逐项SCALAR_MUL/ADD，SQRT用单元素SCALAR_SQRT，常量由Identity/RegI经SCALAR合成。Vector SUB明确MUL(-1)+ADD，两次写回；复制/打包用实际MUL(1)，计费全部状态与恢复。无新增MIN/MAX或隐藏solve/eig。每写后barrier，实际事件作保守检查。输入/预处理/系数/预条件/迭代/恢复/所有DMA均计时。

## 5. 泛式与独立参考

Instruction同一构造helper同步派生raw尺寸/工作量及typed views；共用engine/source_views/binding/runtime，不加算法分支。FormulaLogger只作诊断。参考由独立逐样本fixed HB实现提供完整FP16输出与系数证据。

## 6. 退出条件

构建→真实Asim/实际事件/性能预筛→实际重建→独立同算法逐位→FP64 MMSE≤2%→未编码QPSK相对门禁。额外算子留出预留CDL H59001/59002、payload59101/59102，各96；不用于K选择。错误语义/反馈/系数等负例、重复与新runtime另验。未完成不晋级，score=null，硬件认证由负责人上板。
