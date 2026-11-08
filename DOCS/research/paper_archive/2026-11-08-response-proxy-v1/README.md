# Q3b：实际响应与Gaussian线性速率代理

状态：**BOUNDED_RESPONSE_PROXY_GATES_PASSED**。有限样本诊断，不是实际FP16可达率、总体SE或硬件认证。

- [冻结协议](RESPONSE_PROBES.md)：I/Q探针、独立精度检查、复线性适用条件、代理公式和预算。
- [实验报告](../../../../experiments/response_proxy/2026-11-08-v1/README.md)：七新入口和NS留出、精确线性/预算拒绝、计时与边界。
- [manifest.json](manifest.json)：小型快照、摘要和原件索引的SHA-256。

方法论证据：从实际指令获取响应，而非从算法名/人工公式另建候选W；
先验证模型适用性才产生代理，并区分数值一致性、有限响应模型和通信统计保证。
375次正例与独立精度参考位一致，未给出全域误差或总体SE保证；Q3a真实符号指标继续保留。
旧快照不覆盖；快照内相对链接按manifest的source位置检索。
