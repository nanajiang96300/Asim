# Q3a：实际输出的瑞利/QPSK通信检查

状态：**Q3A_TRIAL_GATES_PASSED**，不是完整SE/BLER/CDL-B认证。

- [方案冻结快照](COMMUNICATION_VERIFICATION.md)：指标、角色、精确置信界及后续SE协议。
- [实验报告](../../../../experiments/communication/2026-11-08-v1/README.md)：七入口历史诊断、NS默认batch96新留出与真实入口负例。
- [manifest.json](manifest.json)：小型方案/报告/结果快照及原件索引SHA-256。

可支持的方法论：实际指令重建输出连接通信指标，独立MMSE仅作配对参考；
区分校准、留出和历史数据；按独立H而非相关bits构造置信门禁；缺少SE语义时明确拒绝。
工程试验门限未作为最终论文/业务阈值批准，不宣称算法优越性、Agent搜索效率提升或硬件性能。
旧快照保留；快照内相对链接按manifest的source位置检索。
