# Q2：从质量计划到实际指令重建

状态：**SUPPORTED_PLAN_HANDOFF_GATES_PASSED**。本档案是受监督软件交接验收，不是通信/硬件认证。

- [当前架构冻结快照](QUALITY_DRIVEN_WORKFLOW.md)：QW2，Q1选K与Q2受信交接的阶段边界。
- [验收报告](../../../../experiments/quality_handoff/2026-11-08-v2/README.md)：NS/BRI真实Asim、192batch逐位一致、真实预筛拒绝、初次封存失败及修正。
- [manifest.json](manifest.json)：本轮文档/摘要快照和原始制品清单的SHA-256。

方法贡献的候选描述：算法预筛后将任务、精度、参数、独立参考与执行版本绑定；
以外部批准摘要校验计划，在Asim预筛后才执行真实指令重建，区分实现一致性和任务数值质量。
本轮未对比Agent搜索效率收益，没有正式SE/BER、CDL-B或实机结论。
前版Q1资料继续保留，新增档案不覆盖旧快照；快照内相对链接按manifest的source路径查找。
