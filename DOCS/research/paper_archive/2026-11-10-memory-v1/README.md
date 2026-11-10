# M1通用重建内存优化（2026-11-10）

状态M1_MEMORY_ACCEPTED_BOUNDED_SCOPE。九入口封存真实Asim制品全量回归，HB K174/batch96完整事件检查与重建峰值1.88GiB，流水JSON/逐指令CSV字节一致、schedule报告相等、完整输出逐FP16位一致。249测试，新Asim日常NS/HB及QPSK Replayer接入通过。

[完整报告](../../../../experiments/memory/2026-11-10-v1/README.md)包含三轮内存测量、受限范围、脚本、清理台账、新91文件runtime和后续计划；[manifest](manifest.json)固定本次快照。旧档案不覆盖，原始结果仍在results/memory/独立版本目录。

这是软件验证器资源优化，不改变算子/K/精度/质量阈值，不增加独立留出样本，不升级旧CDL失败，也不是硬件时序、任意规模常数内存或性能晋级。下一步M2在新校准集研究更紧谱界/复用/流水。
