# Claude Code＋DeepSeek新算子与通用重建归档入口

本研究目录保留原源码；新实现/完整证据位于精简共享分支，避免复制不同版本说明。

固定版本：[完整报告](https://github.com/nanajiang96300/Asim/blob/asim-claude-demo-v1-20261112/experiments/claude_operator/2026-11-11-v1/README.md)、[工作流与原理](https://github.com/nanajiang96300/Asim/blob/asim-claude-demo-v1-20261112/DOCS/research/CLAUDE_NEW_OPERATOR.md)。本机/home/nanajiang/asim-worktrees/asim-core-v1/下同路径；原件results/claude_operator_demo/2026-11-11-v1/。分支SHA/部署命令见../../DUAL_CLIENT_HANDOFF.md。

新affine算子九场景864batch逐FP16位一致、FP64任务误差最大0.0353%；真实语义错误负例Asim正常而重建拒绝；六类新二进制回归通过。成功生成及两轮模型修复调用墙钟约300.61秒，失败/超时另外保留；纯思考时间null。属于受监督非无线示例、任务内评分；不签发无线或M3晋级，不声明S2全部完成。
