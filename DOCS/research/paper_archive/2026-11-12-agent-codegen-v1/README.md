# 受监督 Agent 源码生成与通用重建快照

固定关键词：Claude Code、DeepSeek、Python到算子、真实Instruction、typed views、仿射GEMM、负例、思考时间。

本轮实际新算子并非求逆：Z=half(half(A@B)+C)，同一个真实指令重放核心由MMSE扩展到独立实矩阵任务。九组864batch逐FP16位一致，FP64任务误差≤0.0353%；源生成→审计与编译反馈→Asim→重建→双参考→任务内评分已真实执行。人为误读输入负例仍有685周期且事件正常，但数值全部失败并拒绝评分，验证公式声明不能代替实际指令。

[完整报告与证据](../../../../experiments/claude_operator/2026-11-11-v1/README.md)、[操作/原理](../../CLAUDE_NEW_OPERATOR.md)、[S1两个全新Linux构建](../../../../experiments/deployment/2026-11-11-s1-v1/README.md)。原始模型提议、两轮模型修复及authorship链保留，不发布内部思考流。成功调用墙钟278.55+8.98+13.07秒，不等于纯思考时间；前期失败/超时单列保留。

范围：单个受监督非无线示例，不是任意任务普遍正确性证明；无MMSE/信道、无线通信或性能晋级声明。demo score=1.0只证明串接，research_score=null。正式跨平台候选比较身份/新runtime/预算控制、Codex同任务真实生成对照、无人值守仍待后续。旧报告、pin与失败保留。
