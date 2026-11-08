# 剩余Block/Merge映射与旧Merge异常冻结验收

2026-11-08（香港时间）。REMAINING_MAPPING_GATES_PASSED：新增实块4 Cholesky-block、LDL-block及合法GEMM归约Merge，统一七新入口。四个重建/绑定/事件核心未改。151单元测试、独立代码/证据复审通过。

主批次16case/1392batch通过，6case/576完整L/D因素见证通过；三个日常入口与Merge同包基线重复共384batch通过。Merge重复输出/指令/Core事件流逐字节一致。旧NoBlock/Merge各96batch继续SEMANTIC_AMBIGUITY，不改报PASS。

[完整报告](../../../DOCS/research/MERGE_AND_REMAINING_MAPPING.md)，[耗时](effort_summary.md)，[七入口周期对照](latest_cycle_comparison.csv)，[清单](run_manifest.json)。

- 旧Merge：相同235008声明Scalar元素却压成73728条指令（30240条多元素）；Scalar计费不看compute_size/k。诊断周期10004对23444不是已验证2.34×收益。
- 两Scalar成本探针仅声明宽度/compute_size差异，实际均51cycle/Scalar1cycle；width16明确非法，不是数值/优化通过。
- 新Merge真实k归约：默认FP16 19028；Cholesky-block21338、LDL-block21665。两seed/两组尺寸与FP64/F16模型都通过。
- case_summary.csv/json含每case误差、参数、周期及Asim/检查/独立参考/重放秒数。
- latest_cycle_comparison.json/csv只用统一comparison_identity的七新入口，score=null，不跨旧数据/精度/任务排名。
- pipeline_summary.csv为流水摘要；逐core/unit/opcode/name细表保留原件。不能相加duration_sum为总周期。
- forensics_summary.json、campaign_summary.json、review.json、common_core_identity.json记录拒绝证据、门禁、审查和源码身份。
- runtime_spec.json是七入口固定包，previous_runtime保留旧包；探针及旧Merge不进入新白名单。
- effort.json：任务墙钟1886.29秒，已观测工具并集1037.23秒，Agent/未埋点残差849.06秒；纯内部思考null，收尾归档/Git在快照后另计。

原件专用results/remaining_mapping/2026-11-08-block-merge-v1。forensics与campaign/daily分开；包含全部实际指令/事件/trace、输入/参考/输出、因子快照、源码/库/配置/binary、失败构建/修复日志与时间台账，均按SHA-256索引。新候选不同分组/Vector布局，不是分块单因素性能消融。通信/CDL-B/晋级仍待后续，硬件认证由上板承担。

校验命令：`.venv/bin/python scripts/check_remaining_archive.py experiments/remaining_mapping/2026-11-08-block-merge-v1/run_manifest.json`。
