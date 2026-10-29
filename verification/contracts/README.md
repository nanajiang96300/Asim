# FW0–FW2初始实现范围

2026-10-29：`core.py`提供冻结契约/Asim性能筛选规则；`shadow/program.py`提供严格schema与构建器、派生逻辑工作量；`shadow/execute.py`实现实数FP64整块内存结构重放；`fixtures/rayleigh.py`提供固定种子归一化复瑞利H/y/λ及独立solve参考。

这些是基础设施第一批代码，不是完整FW0–FW2验收。尚缺五契约完整schema、完整异步事件/读写绑定、切片/stride/mask/部分初始化、归约/COPY之外的常量填充、精度/复数低层展开和独立算法比较器。复杂操作一律不支持，不猜测。工作量统计不是周期模型。

生产顺序：schema完整性检查 → 受信runner运行Asim并认证程序/二进制/config/trace摘要 → 性能筛选 → 数值重建。低性能、运行失败或离谱加速不做数值执行，score=null。性能筛选只节省评估成本，不能判断数值对错。阈值应按campaign冻结，有效未优化对照仍需数值验收；不同算法全流程比较不直接套内循环阈值。

`execute_after_asim`保留为内部原语测试接口；实际本地评估改用 verification.asim.runner：它启动真实Asim并校验本次Instruction导出和compute/barrier事件，随后筛选与重放，不接受外部Agent提供的JSON作为可信仿真证明。通用程序已接通Asim，但完整异步事件认证和OS隔离未实现。`_replay`仅用于原语测试，测试不受候选性能筛选限制。

瑞利复数fixture只用于任务参考；当前结构执行器只接受实数float64，不能据此声称复瑞利端到端算子已通过。CDL-B接口保留为显式UNSUPPORTED，后续先审核模型语义再实现，禁止静默回退瑞利。

运行：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s verification/tests -v`。

最新门禁：29项单元测试、7项真实Asim门禁得到预期结果；旧NS/Cholesky成功运行后因缺数值语义返回UNSUPPORTED。详见 [实际进度](../../DOCS/research/FRAMEWORK_PROGRESS.md)。

多batch接口：`verification.asim.batch.evaluate_batch`及`python -m verification.asim.batch_gates`已接通24核/batch24与96。每batch输入不同、实际dispatch/DRAM绑定、全部batch三路径比较；默认96、Cube16³对齐。45单元测试与真实门禁见 [24核报告](../../DOCS/research/ASCEND24_VALIDATION.md)。原始矩阵32×32/32×16为复数实块表示，仍结构FP64而非校准FP16。
