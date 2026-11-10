# BRI inner-0001 独立动态复核（三档模板与新CDL完成）

2026-11-10（Asia/Hong_Kong）。状态：INDEPENDENT_DYNAMIC_AND_FRESH_CDL_REVIEW_PASSED。以下声明范围内未发现阻塞；不是硬件认证、冷构建复现或完整晋级。

冻结manifest：experiments/optimization/bri/inner-0001/candidate.json，SHA `0c4cc1f8b4fd5402370d0f2257a6eebfb719b1a2718f6a8c4699fa0443b4119e`。本次核对84可信源码/10依赖SHA均匹配，候选二进制为manifest指定80e822ee…，基线仍原封存18409/47121/141017。完整静态审计原件未修改，其SHA也是冻结依赖。

审查脚本：results/bri-inner-0001-dynamic-check.py。三档机器结果：results/optimization/bri/inner-0001/independent_dynamic_audit.json（前两档阶段原件另存independent_dynamic_audit-partial2.json）。独立子Agent未编译或启动Asim，未载入完整巨大Instruction JSON/events，仅流式读取header及batch0实际指令，并检查小型NPZ/result/receipt/config/model/dispatch等制品。原完整事件认证依赖主评估器封存结果，不冒称本审计再次检查全部事件。

| λ/K | 实际周期 | 全96batch最大MMSE误差 | batch0实际指令 |
|---|---:|---:|---:|
| 1/16 | 17488 | 0.6191060434% | 1263 |
| .1/53 | 44128 | 0.5461240318% | 2817 |
| .01/174 | 131248 | 0.5114429902% | 7899 |

三档完成：

1. 完整保存96batch输出与封存独立算法参考及原基线逐FP16位一致；任务/算法/seed/comparison_identity与各原基线一致。独立从原H/Y重算全部FP64 MMSE及逐batch误差，与封存数值一致。
2. 每档流式抽batch0全部实际指令，逐轮核对HX、lambdaX、previous差分、new/difference_new地址，最终四项输出src均为K mod3对应的当前逻辑x。未根据region名字“X”猜最终逻辑状态。
3. 实际重放与新鲜独立FP16算法参考逐位一致；在最后carry.add处只读捕获逻辑current与previous，分别与独立K和K−1状态逐位一致。没有向回放内存填入参考数据。
4. 在同一实际指令上测试正/负零RHS，全部输出与新鲜独立参考逐FP16位一致。该操作只改测试输入，并非修改候选/契约/参数或计时结果。

三档脚本墙钟3.3759秒（此前两档阶段1.4203秒），纯Agent思考时间null。K16/53/174实际指令和输出覆盖三物理槽余数；随后完成的新CDL-B/QPSK独立复核见下节；完整冷构建/性能晋级仍由项目主工作流判断。


## 新CDL-B完整任务与QPSK独立复核

主评估器已封存`holdout/result.json`，状态INNER_CDLB_RECONSTRUCTION_CHECKED、三组均QPSK_GROUP_CHECKED。本子Agent的独立脚本为 `results/bri-inner-0001-cdl-independent-check.py`，机器结果为 `results/optimization/bri/inner-0001/independent_cdl_audit.json`。只读取六个小NPZ和相关dataset/index/契约，不重放第二遍完整576batch指令，不读取巨大events。

按冻结H61001/61002各96与载荷61101/61102，独立生成QPSK bits/symbols、AWGN noise及Y，逐值等于封存帧；dataset各文件SHA与冻结index一致。实际192个不同H在三SNR复用，总576个任务，不能记为576独立H。

从每个原H/Y独立求解FP64 MMSE，576个任务参考及误差全部与封存数据一致；候选和MMSE各147456个bit使用独立最近星座点判决重新计数，不调用主验证器的demodulator或group_metrics。按每H聚合BER/SER/原始EVM的超限指示，并用独立SciPy beta分位数重算Clopper–Pearson界。

| λ/K | 新192-H最大MMSE误差 | 候选/MMSE错误bit（每档各49152bit） | 候选/MMSE错误symbol |
|---|---:|---:|---:|
| 1/16 | 0.8408121548% | 13361 / 13373 | 11599 / 11603 |
| .1/53 | 0.9716081215% | 8139 / 8156 | 7456 / 7470 |
| .01/174 | 0.8172673830% | 4023 / 4018 | 3824 / 3818 |

三组BER/SER/EVM及joint超限均0/192；按冻结family alpha .05、8×3修正，置信上界3.1643656463%<5%，与封存主门禁数值一致。没有相位/增益拟合、调K、参考代填、裁剪或修改门限。

另对六帧各batch0，从原H/Y新执行独立同算法FP16模型，完整输出逐位一致。完整576batch同算法位一致来自主评估器的原始完整实际重建；本次独立审计不将6个抽样称为又重放了全部576batch。

本CDL独立复核脚本墙钟0.9876秒，纯思考时间null。此前三档模板脚本3.3759秒，二者只记录脚本运行，不把等待Asim、写报告或时间残差当作纯思考时间。

## 基线配对及证据来源边界

三档原始瑞利模板的同task/algorithm/input/config/measurement comparison_identity、96batch基线输出及参考已由本子Agent核对一致；周期取各自实际Asim receipt而非指令数估计。

主Agent另行运行paired_baseline.py，对相同新H610/载荷重放基线576batch，并将写`paired_baseline.json`。该全量基线配对由主Agent脚本提供，本独立审计没有第二次加载基线大指令重跑，不能冒称独立执行了它。本报告完成时该配对结果尚未纳入本审查；后续报告引用应注明其来源。

静态审计原件SHA仍保持冻结值；候选C++、评估器和台账均未由本子Agent修改。软件数值/事件模型、固定形状/信道域/精度范围之外不自动继承本结论；score应在完整晋级前保持null。
