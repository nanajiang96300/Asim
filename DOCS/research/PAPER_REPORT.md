# 面向论文的重建验证报告与完成门禁

2026-11-08 CDL-B/K曲线更新：编码不是求逆/均衡精度的前置。新增真正Sionna CDL-B静态频域输入，21组中6组通过、BRI 1组数值失败、14组象限/溢出拒绝，整体未通过；独立复核7个完整组688,128bit。FP16/FP64的K—误差曲线已归档，显示NS精度平台及固定BRI的收敛问题；不能只加K。216测试通过，仅清理约3.02GB可再下载缓存，旧研究原件保留。见[报告与图](../../experiments/channel_validation/2026-11-08-v1/README.md)及[输入域/精度协议](CHANNEL_DOMAINS.md)。


2026-11-08 Q3c当前：完成声明范围内未编码QPSK完整bit-to-bit链路。七新入口×Es/N0 0/10/20dB共21配置，14次新Asim、168帧、16,128次batch重放通过；212测试与独立4,128,768bit复核通过，全部同算法精度参考位一致，最大MMSE任务误差0.2744%。每组192独立H，全实验配对复用同192个H。该试验通过相对BER/SER/raw EVM门禁，不签发FEC/BLER、CDL-B、总体SE或最终晋级。见[验收报告](../../experiments/qpsk_full/2026-11-08-v1/README.md)及[可执行协议](../../verification/communication/QPSK_LINK.md)。


2026-11-08 Q3b：新增 [实际响应与线性SE代理有限样本档案](paper_archive/2026-11-08-response-proxy-v2/README.md)，200测试和375次正例实际重放通过。没有新增Asim周期、总体SE置信或有限调制吞吐结论；完整完成门禁不变。

2026-11-08 Q3a：固定论文档案新增 [实际输出的QPSK通信试验](paper_archive/2026-11-08-communication-v1/README.md)，包括历史七新入口只读诊断及NS batch96双噪声新留出置信门禁。187测试通过，但SE/BLER/CDL-B与完整晋级未完成；旧最终报告生成器不能因此签发全部完成。

2026-11-08固定检索位置迁入 [paper_archive/README.md](paper_archive/README.md)：原报告原位保留，重建设计/验收材料另存SHA-256快照，新增精度驱动选K架构及小规模烟测。本文以下2026-10-30生成器与完成门禁是历史阶段能力，不自动覆盖七新入口或新通信方案。新架构未完成正式通信验收，仍不能生成“全部完成”的最终结论。

2026-10-30最新归档：[阶段报告](paper_reports/2026-10-30-interim-v1/REPORT.md)、[机器完成状态](paper_reports/2026-10-30-interim-v1/readiness.json)、[论文小表CSV](paper_reports/2026-10-30-interim-v1/numerical_results.csv)、[证据索引](paper_reports/2026-10-30-interim-v1/evidence_index.json)。状态INTERIM，不能称为全部验证完成。

已通过：实际指令结构重放、NS独立同算法与任务参考；三seed各96batch的FP16存储/FP32累加模型与独立参考全部FP16位模式一致，最大任务相对误差约0.106%。实际P2存储/P4 Vector工作量模型周期7,950；与P8的14,298为不同profile，不是同精度优化加速比。

本轮继续实现FW3逐操作精度模型（含RNE、subnormal、逐指令写回与非有限拒绝）；FW4新增保守RAW/WAR/WAW检查器及合法重叠正例。最终106项单元测试通过。FW3仍缺硬件/转换成本校准，FW4缺真实Core读完成/写可见/DMA实例绑定，不能用合成检查器或pipeline CSV充当G4通过。旧七入口仍有歧义/失败；通信统计、CDL-B、干净复现未完成。

报告可用于论文方法、实验设置、阶段性数值结果与失败案例；完整框架、全部算子正确、实机性能、通信质量及CDL-B结论仍缺证。

## 可复现生成

运行器是`verification.reporting.paper`。它逐件复核三批归档清单的SHA-256，再从实际gate summary生成报告、状态JSON、CSV和证据索引。每次使用新目录；不覆盖旧报告。

```bash
.venv/bin/python -m verification.reporting.paper \
  --structure-manifest experiments/unified_reconstruction/2026-10-30-ns-v1-frozen/run_manifest.json \
  --precision-manifest experiments/precision_campaigns/2026-10-30-fp16-v2-frozen/run_manifest.json \
  --legacy-manifest experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/run_manifest.json \
  --output DOCS/research/paper_reports/<new-report-id>
```

`--final`要求全部必需门禁PASS；缺门禁会在写入前拒绝。[实际拒绝探针](paper_reports/2026-10-30-interim-v1/FINAL_REPORT_GUARD.json)已经验证。后续门禁模块落地后，将其真实验收清单接入生成器，不能手填PASS或删除困难门禁生成最终稿。

## 接下来按依赖执行

1. 在Core实际发射/响应路径记录唯一op/DMA fragment与core/buffer绑定、读完成、写可见事件，接入调度检查器；验证删barrier/提前覆盖会拒绝，合法重叠能通过。
2. 完成混合精度转换/成本模型闭合，校准实机声明的适用边界；当前模型周期保持诊断。
3. 同契约修正其余独立算法候选，建立通信suite、CDL-B、干净环境复现，再开放评分与完整自动迭代。
4. 全部必需门禁真实验收后，生成新版本最终报告与论文图表清单，继续保留所有阶段证据。

2026-10-30稳定入口更新：实际Core读写/DMA事件已完成受限保守模型验收，固定runtime日常命令可用，121测试通过。详情及支持/拒绝边界见[稳定性报告](RUNTIME_STABILITY.md)。旧报告中的缺真实Core事件是当时状态；硬件G4/通信/全部旧算子/最终评分仍未通过，不能由该模型证书升级。
