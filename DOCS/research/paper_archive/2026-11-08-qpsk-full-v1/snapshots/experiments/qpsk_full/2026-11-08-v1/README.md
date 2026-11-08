# 未编码QPSK完整链路验收 v1

日期：2026-11-08。状态：**UNCODED_QPSK_SUITE_CHECKED**；独立审计通过，212项单元测试通过。完整范围是理想同步、完美CSI、IID平坦瑞利信道下的未编码复基带bit-to-bit测试，不包含FEC、BLER、CDL-B或实机认证。全研究通信认证仍为false，评分为空。

## 方法与冻结条件

随机bits → 单位能量Gray QPSK → Hx+n → 实际指令数值重建 → 硬判决bits；另运行独立同算法逐操作FP16/FP32参考及高精度MMSE，不用SE代理替代判决。
详见[可执行协议](../../../verification/communication/QPSK_LINK.md)。实际模板必须先通过真实Asim、事件及原数值门禁，才允许新输入重放；缺语义、参数不符、hash变化均拒绝。

固定24核、batch96、Nr64/Nt16、RHS8；H~CN(0,1/Nr)，每流Es=1，复噪声方差及MMSE正则化λ一致。Es/N0=0/10/20dB对应λ=1/0.1/0.01；Eb/N0比横轴低3.0103dB，总接收信噪比低6.0206dB。CPU调制、生成信道/噪声和统计不计入NPU周期；Asim仍计完整H/Y→xhat的Gram、初始化、预条件、恢复与DMA。

覆盖七个已修正显式新入口，非历史旧入口升级声明。NS K16；BRI K192、实块2、omega0.5、x0=0、预计算定常迭代；Cholesky/LDL直接与实块4，以及合法GEMM归约Merge。K为冻结正确性基线，不是最小可行K或性能最优结果，尤其不等于负责人所指历史最优BRI实现。

14次新增真实Asim补齐λ1/0.01，λ0.1复用7份已封存源。源校验seed19001，留出H种子22001/22002、载荷帧种子33001–33004。数据无关模板在同几何/λ下复用，每帧仍验证语义、地址并执行完整程序，未为每个载荷另跑Asim。

21配置×8帧=168帧，16,128次batch重放，4,128,768次bit和2,064,384次符号判决。每组192个独立H、196,608bit/98,304符号。算法与SNR配对复用H及发送bits；全实验只有192个不同H、196,608个独特发送bit，不把多算法重放计成新增独立样本。

## 结果

所有21配置通过；全部重建输出与同算法精度参考逐FP16位一致，最大相对MMSE任务误差0.2744181%，低于2%数值门限。每组候选相对MMSE的误判bit计数最多差6/196,608，不作为算法质量改善证据。

| 每流Es/N0 | 七入口BER范围 | 配对FP64 MMSE BER |
|---|---|---|
| 0dB | 17.55371%–17.55829% | 17.55676% |
| 10dB | 0.304667%–0.305684% | 0.305176% |
| 20dB | 每组196,608bit未观测到误码 | 同样未观测到误码 |

20dB零观测不等于真实BER为零。精确计数、SER、原始软符号EVM及置信界见[quality.csv](quality.csv)；逐流见[per_stream.csv](per_stream.csv)，4×4符号混淆矩阵见[summary.json](summary.json)。

按H汇合四帧，冻结ΔBER≤0.01、ΔSER≤0.02、Δraw EVM≤0.01，联合超限概率≤0.05。Clopper–Pearson与Bonferroni覆盖21组×4事件×双尾，family alpha0.05；全部联合超限为0，各组上界4.140833%通过。这是固定四帧有限块的相对MMSE事件，不是无限条件BER、绝对业务质量或各配置相互独立的声明。

调制器另经−2/0/3/6/10dB、每点200,000符号的AWGN理论BER/SER控制，均通过已知均值Bernoulli界，见[modem_controls.json](modem_controls.json)。独立报告器用最小距离重新判决全部4,128,768bit，从H/Y重新求FP64 MMSE，精确复核bit/符号/混淆矩阵/每H EVM及SciPy置信界，见[audit.json](audit.json)。

![BER](figures/BER.png)

[BER PDF](figures/BER.pdf)、[SER曲线](figures/SER.png)、[AWGN控制](figures/AWGN_controls.png)；完整21面板星座图原位保存于`results/qpsk_full/2026-11-08-v1/report/constellations.pdf`及PNG，点数/绘图范围见协议。零误码曲线绘图点1/N只是显示约定，不是置信上界。

## 周期与流水

| 新入口 | 完整任务周期（所有三λ相同） |
|---|---:|
| NS | 7950 |
| Cholesky NoBlock | 28006 |
| LDL NoBlock | 18697 |
| BRI | 23047 |
| Cholesky Block | 21338 |
| LDL Block | 21665 |
| Cholesky Merge | 19028 |

[逐流水统计](pipeline_cycles.csv)含事件数、时长和、全局/每核活跃区间并集、均值和最大值。average_core_event_coverage为每核并集之和/(24×任务周期)，不等于MAC有效算力利用率；并发流水、DMA时长和不可相加当总周期。三λ周期相同来自相同固定指令结构，不表示对任意条件的性能普适性；K未统一调优，本表不能作为最终算法优劣排行榜。

## 负例、资源与复现

三条真实入口负例均在数值前拒绝：batch预算不足、冻结评估器hash错误、批准契约SHA错误。见各`link-*-reject-result.json`，没有生成输入/输出NPZ，不是算子故障。212项单元测试通过；未修改算子、固定runtime或既有原始制品。

新增Asim源准备672.25秒，完整链路1278.32秒；2vCPU服务器串行执行，BLAS单线程。不可观测的模型纯思考时间为null。源trace约3.51GB，仅保留原位与校验清单，未提交大型数据。具体复现命令、预算、数据口径见协议；必须使用新的输出目录，不能覆盖本轮原件。

版本：源准备be404cb0；链路8bdbb001；独立报告器131a6ffc；工具链见[toolchain.json](toolchain.json)。契约见[source-contract.json](source-contract.json)及[link-contract.json](link-contract.json)；原始源/每帧数据/输出/报告的大小与SHA见[run_manifest.json](run_manifest.json)，报告自身清单见[report_manifest.json](report_manifest.json)。固定论文检索为[paper_archive](../../../DOCS/research/paper_archive/README.md)。

## 后续验收

1. 在冻结业务质量门限和预算后，将完整链路接入质量驱动K校准/独立留出闭环，避免将K16/K192当最优；不使用本次留出结果继续调参再自验。
2. 新版本增加独立H和维度/病态信道覆盖，声明目标BER及误码数/置信区间的采样停止规则。
3. 单独定义CDL-B归一化、采样、CSI与任务边界后复用相同链路；FEC/BLER、GMI或有限调制SE分别设计，不以当前Gaussian代理代替。
