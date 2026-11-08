# CDL-B与K/精度研究 v1

状态：CDLB_QUALITY_NOT_VERIFIED。七新入口三SNR共21组，6通过、1数值失败、14拒绝；保留实际失败，不升级为全信道认证。编码不是本阶段求逆/均衡精度的前置。

[报告快照](snapshots/experiments/channel_validation/2026-11-08-v1/README.md)、[协议快照](snapshots/DOCS/research/CHANNEL_DOMAINS.md)、[manifest](manifest.json)。当前报告与图为experiments/channel_validation/2026-11-08-v1/。快照链接按原source路径解释。

可用于论文讨论FP16平台、固定BRI谱条件、实际指令与独立算法一致但任务质量仍失败、Rayleigh到真实CDL输入域泛化。不能支持最优K、逆矩阵逐元素精度、全部CDL或硬件通过。旧Kronecker报告不等于真实CDL-B。旧原始数据保持原位，仅清理可再下载缓存并保留台账。
