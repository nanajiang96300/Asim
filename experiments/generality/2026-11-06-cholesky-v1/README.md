# 跨算法重建普适性冻结验收

2026-11-06（香港时间）。SUPPORTED_SCOPE_PASSED：NS迭代法和独立Cholesky直接法，共同任务/重建核心。5个跨算法样本432batch、3次日常入口288batch，共720次batch验证；4项离线拒绝负例和完整参数基线拒绝通过。135单元测试通过，独立代码审查通过。score=null。

[完整报告](../../../DOCS/research/RECONSTRUCTION_GENERALITY.md)，[Agent耗时](effort_summary.md)，[清单](run_manifest.json)。

- `case_summary.csv/json`：全部样本、误差、周期及Asim/检查/独立参考/重放时间。
- `pipeline_summary.csv`：流水汇总；详细逐core/unit/opcode/name统计在原件。
- `common_core_identity.json`：四个核心文件与7ea3405e逐字节相同；新增算法不修改解释器求解逻辑。
- `negative_cases.json`：离线指令记录变异，不是实际错误候选运行。
- `review.json`：审查范围及66秒墙钟，不是模型内部思考时间。
- `runtime_spec.json`：新固定运行包，已接通Cholesky及NS日常入口；previous_runtime保留旧包。
- `effort.json`：本轮任务墙钟992.48秒，已观测工具并集445.79秒，Agent及未埋点残差546.69秒。纯思考时间不可观测/null；并行审查不重复相加。截止快照后的归档/Git收尾不包含。

原件：results/generality/2026-11-06-cholesky-v1，包括失败构建、串行重试、全部Instruction/observer/trace/输入/参考/输出、库/配置、工具时间台账及日常基线重复结果。两个Cholesky日常run的输出/指令/事件流均逐字节相同。

校验：`.venv/bin/python scripts/check_generality_archive.py experiments/generality/2026-11-06-cholesky-v1/run_manifest.json`。

范围：两类算法、两组尺寸、声明FP16/FP32与FP64模型、瑞利。旧七入口失败保留；没有任意算法、CDL-B或完整通信验收声明。FP64 observer约251MiB接近256MiB预算，下一步需改善流式检查的内存占用。硬件认证由上板承担。
