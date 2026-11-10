# 固定HB v2最终动态复核：主Agent确定性脚本

封存实验版本2026-11-09-v2，最终整理2026-11-10（Asia/Hong_Kong）。状态PRIMARY_DETERMINISTIC_RECHECK_PASSED。

本文件明确区分审计来源。独立子Agent已完成原生静态审计及v1前两档动态落盘报告，另在会话反馈了v2三档及六CDL帧的复核结果，但其最终v2报告因执行额度中断未落盘。那些会话反馈不充当可执行审计制品。本文件由主Agent运行确定性脚本生成并汇总，不能称为第二份独立Agent证书。

可执行脚本：`experiments/fixed_hb_native/2026-11-09-v2/tools/audit.py`。实际结果：`results/fixed_hb_native/2026-11-09-v2/final_recheck.json`，本轮墙钟1.7435秒；纯Agent思考时间null。主Agent独立编写任务参考、信号/噪声重建、最近点解调和零事件二项置信计算；FP16同算法对照使用既有独立reference，未导入候选数值实现，不修改重建内存。

## 完成范围

- 核对83源码及4项活动binary/config/library/DRAM依赖SHA，当前90文件runtime加载校验通过。
- 三档288个瑞利batch的完整保存输出与封存独立参考逐FP16位比较全部一致；从原H/Y独立重算全部FP64 MMSE与逐batch任务误差。周期18409/47121/141017，最大误差0.619106%/0.546124%/0.511443%。
- 增量读取真实Instruction文件，只保留K174/batch0共8595条实际指令，使用共同execute_batch完整重放。只读capture设置结束及最后iteration173.state之后的α/β、lower/upper、Eu/Pu、完整展开P，均与重新执行的独立FP16 reference逐位一致。检查全部iteration目的地址不覆盖这些常量区。没有加载完整巨大Instruction JSON或重读全部events，不冒称完整事件二次认证。
- 新CDL六个NPZ从封存dataset H及[fseed,hseed]独立重新生成bits/symbols/noise/Y，全部逐值一致。独立重算576个任务的FP64 MMSE/误差，以及候选和MMSE各147456个最近点bit判决。六帧各batch0重新执行独立FP16算法，完整输出位一致；全576batch同算法一致性依据原始完整重建记录，不冒称本脚本又重放了576batch。

| λ / K | 新CDL最大误差 | 候选/MMSE错误bit（各49152bit） |
|---|---:|---:|
| 1 / 16 | 0.739478% | 13326 / 13327 |
| .1 / 53 | 0.806329% | 8071 / 8064 |
| .01 / 174 | 0.799617% | 4090 / 4090 |

每档192个独立H，跨SNR复用不增加独立H数量。按BER增加.01/SER增加.02/原始EVM增加.01检查，四项（含joint）均0/192超限；独立零事件二项置信上界3.1643656463%小于冻结5%。无增益/相位拟合，无参考代填，无调K。

本审计不覆盖硬件时序、次正规FTZ、任意信道/形状、全域保证或性能晋级；score仍null。首次v1中断记录、旧β-only的CDL质量失败均保留。
