# 显式Block-Richardson映射

2026-11-06。独立新候选，不覆盖旧BRI或否定其他历史候选。

## 1. 文件

src/inverse/bri_explicit/BRIExplicitModel.{h,cc}；verification/reference/bri.py；受信catalog参数。共用H/Y→xhat边界与重建核心。

## 2. 公式推导

A=R(H)ᵀR(H)+λI，b=R(H)ᵀR(Y)。B为A各2×2实主对角块的逆，块外为零。逆块[a b;c d]⁻¹=[d -b;-c a]/(ad-bc)，负号必须真实生成。ω=1/2，P=ωB，S=I−PA，c=Pb。固定点迭代x0=0，x_{k+1}=Sx_k+c，等价于x_k+ωB(b−Ax_k)。预计算P/S/c全部计时。默认K192，允许1..256；不保证所有信道收敛，仍由任务真值门禁拒绝。足够多迭代用于验证结构，非最佳性能参数。

## 3. 内存与初值

H/Y/RegI/Identity共同输入；A/B/P/S/临时矩阵为2Nt方阵，rhs/c/x/t为2Nt×2Nrhs；B块外=0，x0=rhs*zero；one/zero/negative/half真实标量算出。输出完整batched xhat，真实x更新写回，禁止只有注释swap。B2指实块大小，不冒充复数2×2块。

## 4. 指令与FormulaLogger

每逆块ad/bc/det/负号/DIV逐条SCALAR，固定延迟只处理单元素；矩阵P及减法Vector；GEMM产生PA/c/Sx，ADD更新x，MUL(one)复制最后输出。全部实际指令附SourceViews，emit_step只作诊断。每步barrier，DMA无变换，完整输出恢复计时。

## 5. 验证与界限

独立F64实块算法仿真（逐块solve）及独立FP16/FP32参考（显式2×2操作与舍入）；任务真值复数MMSE。先Asim与事件门禁再数值，低K允许算法质量失败且不能白名单。记录实际trace、配置、input与时间；硬件认证外置，CDL-B后续。结果待补。
