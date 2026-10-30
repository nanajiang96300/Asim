# FP16存储/FP32算术模型诊断

三seed×96batch全部FP16位模式与独立参考一致，任务最大相对误差约0.106%；实际Asim使用P2存储、P4 Vector工作量，7,950周期。此为显式数值模型，未校准实机/转换成本，不能与P8的14,298周期当同口径优化收益。score=null。

原件：`/home/nanajiang/Asim/results/precision_campaigns/2026-10-30-fp16-v2-frozen`。G4、通信、CDL-B、干净复现均未验收；论文报告必须为阶段稿。
