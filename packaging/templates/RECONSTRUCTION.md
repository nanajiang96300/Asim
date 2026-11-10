# 关注点二：算子重建验证的原理

Asim在这里主要模拟周期、存储、发射与流水事件，**不执行完整的中间矩阵数值运算**。因此“仿真正常结束”只说明程序在该模拟器中走完，不能证明输出正确。数值影子执行器补上明确语义，独立执行真实导出操作序列。

## 三条分开的路径

```text
C++ Model/Operation真正生成并发射的Instruction
    ├─ Asim周期、dispatch、trace、真实Core读写/DMA事件
    └─ 从同一Instruction构造处导出typed views
             ↓ 参数/地址/工作量/覆盖绑定
          通用CPU数值重放 ──比较── 独立同算法精度模型
             ↓                      同初值/K/预条件/写回
          MMSE任务精度 ────比较── FP64任务参考
```

1. **实际操作重放**：读取`instructions.jsonl`，不能根据算子名直接运行标准Cholesky/求逆。`engine.py`和`source_views.py`只实现LOAD/STORE、视图、GEMM、元素/标量操作及精度规则。物理区域内的row/col、转置、共轭、广播、常量和写回均显式。
2. **同算法参考**：独立Python代码按相同参数和精度顺序求值，区分“指令/布局实现错误”与“算法本身有限精度不达标”。FP16 profile要求完整输出逐bit一致，不能只看误差不超过2%。
3. **任务参考**：独立FP64求解(HᴴH+λI)x=HᴴY。检查实际重放结果的相对Frobenius误差。即使实际指令和同算法模型一致，如果算法不收敛/精度不够，也不能通过任务门禁。

例如实际程序误把最终X写为X@X，应先被同算法比较发现；若两个实现都抄错，高精度任务参考仍提供不同来源的检查。不能用另一份手写FormulaLogger再次声明公式来证明真实程序正确。

## 先Asim后数值的顺序

- 冻结任务、实际请求的Model、源码/binary/库/配置/输入/精度；校验输出目录为新目录。
- 真正启动Asim子进程，确认正常结束与周期receipt，核对制品SHA。
- 检查实际Instruction与Model/参数、24核dispatch、全部compute/barrier及DMA分片覆盖；m/n/k、元素/标量工作量和allocation不能伪报。
- 检查真实Core读/写可见、DMA请求/响应、buffer epoch及保守依赖约束。
- 性能预筛：既有同任务/输入/配置/精度/计时identity的基线下，不竞争或异常离谱的结果可以数值NOT_RUN；没有references/outputs/numerics_started。首次建立正确性基线是显式baseline_creation，不自动称优化。
- 只有前面通过才启动数值独立参考和实际重建，随后比较FP64任务与适用的BER/SER/EVM等通信指标。

## 精度、输出及计时

默认profile是FP16持久存储、FP32临时算术/累加、固定k递增GEMM乘加、每个原语FP16 RNE写回、次正规保留、不融合。它不是全FP16 MAC，也不是已上板的硬件精度认证。改变FTZ、累加或融合需要新profile和验证。

共同输入H/Y/λ，实块R(Z)=[[Re,-Im],[Im,Re]]。输出xhat的四象限须一致，不能只取左上块掩盖共轭/符号错误。默认Nr64/Nt16/RHS8、batch96（24核各4片），矩阵核按16³成本建模。

从输入DMA到输出DMA计时；Gram/HᴴY（采用时）、初始化、预条件、系数、迭代/分解、缩放和恢复都计入。信道生成和共同实块编码/解码在边界外。直接均衡与显式逆输出不是同一任务，不能混比周期。

## 拒绝状态与证据

- Asim失败/性能拒绝：数值NOT_RUN，不给分。
- 未实现原语、缺输入/切片/精度/事件：UNSUPPORTED或INVALID，不能随机补齐或跳过。
- 实际重放≠独立算法：REPLAY_MISMATCH/精度不一致。
- 二者一致但MMSE不达标：ALGORITHM_QUALITY_FAILED。
- 默认smoke全部通过只说明规定Rayleigh样本/λ.1/尺寸/精度/软件模型内通过，不代表全部CDL-B、全SE/BLER、全部旧入口或正式晋级。

核心文件：`verification/reconstruction/runner.py`（门禁顺序），`binding.py`（实际成本/地址绑定），`engine.py`/`source_views.py`（通用执行），`verification/schedule/runtime.py`（真实事件模型），`verification/reference/`（独立算法），`verification/contracts/mmse.py`（共同任务及FP64参考）。

## 信任边界

typed views补齐了Asim没有实现的详细数值索引语义，并与原始Instruction地址/形状/成本做一致性绑定。仍须审查生成点的语义声明与公式；现有基地址周期模型不能独自证明这些切片将来在真实NPU微码中的实现。保守事件检查是软件模型证书，硬件级精度和时序以后上板验证，不作为本软件流程前置。
