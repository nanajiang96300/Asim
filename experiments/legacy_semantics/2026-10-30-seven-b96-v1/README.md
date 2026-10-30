# 旧七入口完整语义覆盖与阶段重建冻结验收

2026-10-30。实际24核/batch96、瑞利seed42，七入口Asim均结束。714,912条实际Instruction描述完整扫描；首错前数值阶段观察；全部原始指令与此前一致，只有LDL 1,536条显式广播元数据改变。完整旧算子仍BLOCKED，无排行榜、score=null。

Cholesky NoBlock L通过；LDL D通过/L单位对角失败；BRI预条件与初值失败。Block/Merge缺定义，NOT_REACHED不能当数值通过。参见[完整报告](../../../DOCS/research/LEGACY_SEMANTIC_STAGES.md)。

- stage_summary.json：各入口全96batch摘要；逐batch及快照在原件目录。
- source_equivalence.json：此前冻结批→首轮原始字段一致，首轮→冻结v2完整Instruction文件逐字节相同。
- pipeline_summary.csv：流水概览；177,048条逐core/unit/opcode/完整name统计在原件CSV，不能相加duration_sum为总周期。
- review.md：独立审查与修复。
- runtime_spec.json：本轮固定生产入口；previous_runtime.json保留旧包定义，旧基线不跨包复用。
- run_manifest.json：raw/stages/双精度NS回归、库/配置/日志/测试及摘要大小/SHA-256。

校验：`.venv/bin/python scripts/check_legacy_semantic_archive.py experiments/legacy_semantics/2026-10-30-seven-b96-v1/run_manifest.json`。

130单元测试通过。硬件级时序认证由负责人后续上板，本轮不做。NS双精度回归已通过；日常白名单仍仅NS。旧算法缺陷需要独立新候选真实生成完整运算，禁止靠标准算法替代回放。
