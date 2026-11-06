# LDL/BRI显式重建映射冻结验收

2026-11-06（香港时间）。SUPPORTED_MAPPING_GATES_PASSED。新增LDL与实B2/ω0.5/K192 BRI；共同四核心与92bb3383逐字节相同。旧算子失败不覆盖，不输出可信历史排行榜，score=null。

主批次11场景：864正例batch通过、24低K质量负例拒绝；严格stages-v2的672batch实际L/D及B/X0见证通过；日常4次384batch通过，BRI基线复用和输出/指令/事件流重复一致。144单元测试、独立代码与严格门禁复审通过。

[完整报告](../../../DOCS/research/LDL_BRI_MAPPING.md)，[时间口径](effort_summary.md)，[清单](run_manifest.json)。

- case_summary.csv/json：尺寸、参数、实际周期、误差与分阶段秒数。
- pipeline_summary.csv：每流水时长/并集等摘要；各core/unit/opcode/name细表留原件。
- common_core_identity.json：共同解释/绑定/事件核心未改。
- offline_negatives.json：LDL错误D行、BRI末次ADD换MUL的离线数值拒绝，没有错误候选周期声明。
- review.json：代码PASS、门禁初审问题和修复后PASS；模型内部思考时间不可观测。
- runtime_spec.json：四算法固定包；previous_runtime保留旧包。新旧基线不混用。
- effort.json：任务墙钟1645.40秒，工具并集806.45秒，Agent+未埋点残差838.94秒；纯思考null，截止快照后的归档/Git收尾另计。

原件results/operator_mapping/2026-11-06-ldl-bri-v1。最终阶段证据**stages-v2**；旧stages仅探索，入口失败及各重试保留。tool_sources封存严格审计脚本与清单工具。全部Instruction/trace/Core流、input/reference/output、实际阶段快照、库/配置/binary、日志和时间台账都有SHA-256。

验收命令：`.venv/bin/python scripts/check_solver_mapping_archive.py experiments/operator_mapping/2026-11-06-ldl-bri-v1/run_manifest.json`。

范围：声明的逻辑实块/精度与周期模型、瑞利、两组尺寸。硬件认证外置，CDL-B/通信suite和旧Block/Merge入口仍未完成。K192为正确性参数，不是历史最优BRI候选。
