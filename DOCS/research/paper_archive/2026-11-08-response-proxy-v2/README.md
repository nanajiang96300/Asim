# Q3b v2：启动前绑定评估器的实际响应代理

状态：**BOUNDED_RESPONSE_PROXY_GATES_PASSED**。当前版本替代v1的入口契约，保留v1全部原件。

- [冻结协议](RESPONSE_PROBES.md)：源码预绑定、实际I/Q响应、复线性筛查及Gaussian线性代理定义。
- [实验报告](../../../../experiments/response_proxy/2026-11-08-v2/README.md)：200测试、375次正例及100次严格线性负例重跑。
- [manifest.json](manifest.json)：版本化小型快照、摘要和原件清单SHA-256。

v2增加批准契约中的评估器源码hash，拒绝启动前版本变化；与v1使用同一批冻结输入，测量完全复现，不能将重跑计为额外独立H。
这仍是有限样本的响应模型/线性SE代理，不认证实际FP16总体可达率、有限调制吞吐或硬件周期，不替代Q3a真实符号指标。
快照内相对链接按manifest的source路径检索。
