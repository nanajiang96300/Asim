# 隐式FP16实际接入：重建通过、CDL质量失败

2026-11-09（Asia/Hong_Kong）。状态：ACTUAL_REPLAY_CHECKED_CDLB_QUALITY_FAILED。新候选实际Asim/共同重建映射有效，新CDL20dB 1/192样本严重失败，不晋级，score=null。

[完整实验报告](../../../../experiments/bri_implicit/2026-11-09-v1/README.md)记录周期、逐核/指令流水、重建耗时、240测试、七显式回归、重复及拒绝门禁。只阻尼β导致稳定域缩小已由独立谱/76步多项式与bit/任务审计复现；原192-H软件成功档案保留，新失败不覆盖。

[manifest.json](manifest.json)固定本次方案/结果/三个skill快照、工具链与原件清单。大型原件仍位于results/bri_implicit/2026-11-09-v1/，无需从快照相对链接推断路径。capture_commit是归档前基点，快照本身SHA固定新增文档。

下一步联动设计α/β与保守谱界，先新校准选K，再冻结新候选和全新留出；本次坏点仅回归。不以增加K、改门限或混合精度自动修复该稳定性缺陷。
