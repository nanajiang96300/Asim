# Cholesky显式直接求解：重建普适性候选

2026-11-06。新候选保留全部旧实现，非旧周期优化声明。

## 1. 文件

`src/inverse/cholesky_explicit/CholeskyExplicitModel.{h,cc}`；独立`verification/reference/cholesky.py`；共同catalog注册适配，不修改engine/source_views/binding/runtime的算法分支。采用共同full-MMSE任务与真实Core observer。

## 2. 公式推导

A=HᴴH+λI，b=HᴴY。实块R(A)为对称正定矩阵，Cholesky A=LLᵀ。逐列右看式：Ljj=sqrt(Ajj)，Lij=Aij/Ljj；对尾部A←A−l lᵀ。先解LZ=b：第j行Z_j=b_j/Ljj，尾部b_i←b_i−Lij Z_j。再解Lᵀx=Z：反序行求解，前部Z_i←Z_i−Lji x_j。对应逆矩阵为L⁻ᵀL⁻¹，但任务直接求解，无需制造完整逆的额外工作。

## 3. SPAD

H/Y/RegI/Identity完整输入；A/L为2Nt方阵；rhs/forward/backwork为2Nt×2Nrhs；tmp/negative以2Nt×max(2Nt,2Nrhs)容纳秩一更新；常数zero/one/neg1显式初始化；同步token按上界预留。ACCUM输出完整2Nt×2Nrhs，单次STORE。row/col切片包含真实步长；不允许DMA变换/隐式归约。

## 4. 指令与FormulaLogger

Gram/rhs与秩一更新用GEMM；分解单元素DIV/SQRT用真实SCALAR，compute_size=1；行解为Vector DIV并显式scalar_rhs；减法用MUL(-1)+ADD，工作量按实际视图计费。L初始化为eye*zero，不用2λI伪装zero。所有数学操作emit_step，source-views直接附在实际Instruction。每次操作后保守barrier，仅建立正确性基线，不以此宣称优化性能。

## 5. 独立参考与适配

结构参考采用独立复数Cholesky及三角求解；FP16/FP32模型参考独立实现实块右看式与逐指令存储舍入，不导入engine/precision策略。精度条件沿用共同2%任务门限，逐位校验不能放松。直接法算法参数为factorization/initialization，不携带虚构迭代次数。catalog通过受信adapter function提供参数；参考返回通用evidence而非强迫所有算法有X0。旧NS接口兼容。

## 6. 验收

默认24核/batch96/Nr64/Nt16/RHS8，瑞利seed42/123；先Asim及指令/事件/预筛，再数值三路径。用改变标量索引/未知指令的负例验证拒绝，不改核心迁就候选。独立审查、串行-j1构建、数值/工具耗时、制品SHA-256均归档到results/generality/2026-11-06-cholesky-v1及experiments对应目录。硬件级认证外置。已完成：135单元测试、独立审查、串行构建；默认F64 34154周期，FP16模型28006周期（seed42/123均逐位一致），第二组尺寸F64 5677周期；NS回归7950周期。5样本432batch与3次日常288batch均通过，详情见DOCS/research/RECONSTRUCTION_GENERALITY.md。首次CMake源码注册链接失败及重试均留日志。
