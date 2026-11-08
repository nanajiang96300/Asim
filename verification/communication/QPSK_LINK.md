# 未编码QPSK完整复基带链路测试

当前入口：`qpsk_sources`准备实际Asim源程序，`qpsk_link`进行新输入的bit-to-bit重放，
`verification.reporting.qpsk_full`独立审计并导出论文图表。源代码、评估器、数据/参数和预算分别冻结。

## 明确的测试范围

随机bits → 单位能量Gray QPSK → IID瑞利平坦MIMO＋复AWGN → 实际指令重建的MMSE均衡实现 → QPSK判决 → bits。
已知准确H、理想同步，无信道编码；不把未指定的FEC、射频同步、CDL-B或BLER加入“完整”结论。
本入口完整覆盖声明的未编码复基带链路，不依赖Q3b的Gaussian线性SE代理。

调制定义：I/Q各一bit，0→−1/√2，1→+1/√2；轴上零值判为bit1。
四点环绕相邻标签相差一bit。符号能量Es=1，输入bit必须为二进制整数；避免uint8减一回绕。
同时使用符号正负判决和独立最小距离判决，二者必须完全一致。
SER直接数一个符号是否至少有一bit错误，不能将MIMO SER硬套成2BER−BER²。

## 噪声与信噪比口径

H_ij~CN(0,1/Nr)，E[xxᴴ]=I，接收噪声每复元素方差λ，I/Q分量方差λ/2，正则化也为λ。

\[
E_s/N_0=1/\lambda,\quad (E_b/N_0)_{dB}=(E_s/N_0)_{dB}-10\log_{10}2.
\]

平均接收端总信号/噪声比为Nt/(Nrλ)，与前两者不同。默认Nr64/Nt16时，比Es/N0低约6.02 dB。
曲线横轴明确标为每流单位符号能量的Es/N0；不混用旧脚本的未归一化H或噪声乘Nt规则。

AWGN控制使用H=1和精确接收器，理论为

\[
P_b=\tfrac12\operatorname{erfc}(1/\sqrt{2\lambda}),\quad P_s=2P_b-P_b^2.
\]

参考：[MathWorks BER/SER解析式](https://www.mathworks.com/help/comm/ug/analytical-expressions-used-in-berawgn-function-and-bit-error-rate-analysis-app.html)。
本轮控制点−2/0/3/6/10 dB，每点200,000个独立QPSK符号，用已知均值Bernoulli的双尾Bernstein界检查，family_alpha=0.001。
这只用于调制/噪声标定，不是瑞利MIMO性能参考；后者始终使用相同H/Y的高精度MMSE。

## Asim先行与实际指令重放

每个算法、任务形状、精度、迭代配置和λ都先有成功的真实Asim/事件/三路径数值验收。
不同λ生成新的实际程序，保持header任务契约准确；匹配的旧源程序可只读复用。
本次固定已验收NS K16、BRI K192及直接法配置，验证已有实现，不宣称迭代最小或性能最优。

随后对完全相同的冻结指令，输入新H/Y、同λ的RegI及Identity进行多帧数值重放。
当前序列无数据相关控制分支，输入变化不改变地址、形状、布局或执行序列；逐帧重新做实际语义/工作量绑定。
这不是为每帧重跑Asim，也不改变原源程序周期。发生未知语义、位不符或缺少输入时拒绝，不能随机填补。
每一batch完整执行并与独立同算法FP16参考逐位一致，再检查对原始H/Y的FP64 MMSE任务误差≤2%。
实际输出来自`execute_batch`，不能用参考输出替代。原始输入、oracle、成本profile不允许候选修改。

普通CPU的调制、信道生成和统计不计入NPU周期；周期边界仍为完整H/Y→xhat，包括Gram/初始化/预条件/恢复及DMA。
多帧即使复用H，原序列仍完整执行，不免费跳过预处理或按同一H摊销硬件成本。

## 当前冻结套件与验收

默认24核、batch96、Nr64/Nt16/RHS8，七个显式新入口，Es/N0=0/10/20 dB（λ1/0.1/0.01）。
源程序验证seed19001；新留出H seeds22001/22002，每H四帧seeds33001..33004。
相同H、发送bits及基础噪声随机数跨算法/SNR配对；Y随λ变化，同一λ的实际输入在算法间共享，不能因此增加独立H数。
每算法/SNR有192独立H、98,304符号、196,608bit；21配置累计约413万bit判决，但只有192个不同H。

按独立H把四帧合并后检查ΔBER≤0.01、ΔSER≤0.02、Δ原始软EVM≤0.01，任一超限的联合概率≤5%。
Clopper–Pearson区间同时覆盖21组、BER/SER/EVM/联合四事件、双尾，family_alpha=0.05；最少192独立H。
同H的bits/多帧不是独立信道试验。事件针对**声明的四帧有限块**的观测指标，并非无限长条件BER的证明。
另保存每流BER/SER和4×4混淆矩阵。当前硬门禁是每H聚合指标，不假装已经定义逐流业务可靠性规范。

UNCODED_QPSK_SUITE_CHECKED表示整个冻结未编码套件完成且通过试验门限；
uncoded_qpsk_suite_complete=true并不自动授权全部通信、编码/硬件或评分晋级，score仍null。
相对MMSE通过也不表示所有SNR下达到某个尚未定义的绝对BER业务目标。

## 命令和资源

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.communication.qpsk_sources \
  --contract experiments/qpsk_full/2026-11-08-v1/source-contract.json \
  --contract-sha256 <冻结SHA-256> --output results/qpsk_full/<new-id>/sources
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.communication.qpsk_link \
  --contract experiments/qpsk_full/2026-11-08-v1/link-contract.json \
  --contract-sha256 <冻结SHA-256> --output results/qpsk_full/<new-id>/link
```

重跑时重新冻结新路径/源码/源索引摘要，不复用旧批准hash；始终拒绝覆盖现有目录。
源准备限制14次新Asim、4.2GB新制品；链路限制16,128次batch重放、300MB结果，磁盘保留至少1.5GB，两个阶段各限90分钟。
当前2vCPU，构建/Asim串行。原算子/runtime不修改，原始资料不移动或删除。
帧进度逐帧原子落盘；这是受监督任务，不是已经实现自动恢复/重试控制器。
预算、源码或批准摘要不匹配时，在前置阶段拒绝。过程失败保留已有制品，不能删掉坏帧后重报通过。

## 报告

报告器以独立最小距离判决重新计数全部bits、symbols、每H误差和混淆矩阵；
从H/Y重新求解FP64 MMSE，并用SciPy独立核对CP界。它不调用候选算法生成结果。
输出quality/per_stream/pipeline CSV、BER/SER/AWGN对照和星座图，保存PNG及PDF。
零计数在对数图上仅放在1/N位置便于展示，不是BER上界；置信界另按H报告。
流水统计保留事件时长和并集覆盖，重叠事件时长不能相加当成总周期或Cube有效算力。

固定论文索引：[paper_archive](../../DOCS/research/paper_archive/README.md)。

已完成实验报告：[2026-11-08-v1](../../experiments/qpsk_full/2026-11-08-v1/README.md)。
独立审计/图表的只读输入重跑命令（输出必须用新目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.reporting.qpsk_full \
  --result results/qpsk_full/2026-11-08-v1/link/result.json \
  --sha256 44cead815803949862aaebfff2b8fe8963e2be9272a78d1d65ed877ea42a4cee \
  --output results/qpsk_full/<new-report-id>
```
