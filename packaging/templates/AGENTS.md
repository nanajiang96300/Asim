# 部署包协作规则

从README.md与docs/开始。写算子前读取CLAUDE.md、DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md及.claude/skills/op-flow/SKILL.md；不能只声明调用不可用命令。

候选使用独立源码工作区/build/日志。冻结H/Y/λ、K、精度、预条件、初始化、恢复与计时边界。先Asim，后实际指令重建、同算法精度比较、任务精度比较。未知语义UNSUPPORTED，不随机补输入、不静默跳过。不更改评估器或门限来放行候选，不以Agent自报分数为证据。先校准选K，再全新留出；不在留出调K。

保留基线和失败记录，不覆盖历史；不提交账号凭据。没有硬件/通信验收证据不得宣称通过。包不含原服务器实验资料，不能将文档中的流程说明当作已完成实验。
