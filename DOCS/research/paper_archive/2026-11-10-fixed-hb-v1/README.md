# 固定HB：FP16隐式MMSE的新原生重建验收

状态FIXED_HB_CDLB_RECONSTRUCTION_CHECKED。校准→冻结K16/53/174→新算法留出→真实Asim→实际指令重建→另192新CDL-B与完整未编码QPSK，均为既定软件数值模型；无硬件认证或性能晋级。

[原生报告](../../../../experiments/fixed_hb_native/2026-11-09-v2/README.md)记录周期、流水、重建时间与门禁；[算法研究](../../../../experiments/fixed_hb/2026-11-09-v1/README.md)含K/误差曲线、保守谱界、旧失败回归。新方案不覆盖旧β-only失败；首次原生v1中断也保留，不当作通过。

[manifest](manifest.json)固定快照与原件SHA。大型制品保持results/fixed_hb/与results/fixed_hb_native/的独立版本目录；旧制品空间回收采用可恢复gzip台账，不能静默修改旧manifest。

下一步先优化通用重建器大trace内存，再研究不使用留出调参的更紧谱界/更低指令成本；所有优化重新冻结和重验。
