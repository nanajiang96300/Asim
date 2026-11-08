# FP16隐式Gram原语模型与留出证据

状态：FP16_PRIMITIVE_NUMERIC_HOLDOUT_CHECKED。192独立CDL-B留出上直接MMSE与逆子任务数值通过；逆后应用MMSE仍失败。未改Asim，无新算子/实际指令/周期验收，score=null。

[报告快照](snapshots/experiments/fp16_implicit/2026-11-08-v3/README.md)、[计划快照](snapshots/DOCS/research/FP16_FIRST_PLAN.md)、[manifest](manifest.json)。source路径用于解析快照相对链接；原输入/日志/失败路径均原位保留。可支持FP16张量的软件补偿及任务区分结论，不支持全FP16硬件算术、全部CDL-B或周期优势。
