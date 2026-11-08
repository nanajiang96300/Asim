# Q2 受监督计划交接验收

日期：2026-11-08。实现：`9fcd328e`，封存修正：`ac25b987`。
固定论文检索：[paper_archive](../../../DOCS/research/paper_archive/README.md)。
状态：**SUPPORTED_PLAN_HANDOFF_GATES_PASSED**，通信NOT_RUN、score=null。

## 解决的问题

选K结果以前只能人工把参数填入执行器，默认daily还固定了任务尺寸，容易混用K、精度、输入或版本。
现在handoff重新核算冻结契约的全部校准/留出曲线，再绑定独立模型、注册表、运行器、参数和逐场景任务。
execution只接受批准计划及其外部SHA-256，串行复用同一已验收Asim与重建核心，不提供参数覆盖。
每次运行前后核验版本，保持Asim/事件/性能预筛先于候选数值；计划的更严格数值阈值另外执行。

这不是签名系统或OS隔离，也不是无人值守控制器。当前接入的是固定已验收二进制，
新C++候选仍需要独立工作区/build和新的受信runtime发布。
原算子、同算法参考、旧daily和runtime pin未修改，53个运行器依赖文件校验通过。

## 真实Asim结果

复用Q1小规模配置：24核/batch24，Nr8/Nt4/RHS2，瑞利/QPSK；λ=0.1、1.0，seed42校准、seed123留出。
NS K7、BRI K28；FP16存储/逐操作FP32算术，任务误差阈值2%。两个算法各4场景，共192个batch。
本次重放既有留出集是交接一致性检查，不是新增独立质量验证集。

| 算法 | 场景数/batch数 | 完整scope周期 | 全场景最差MMSE任务误差 | 重建与同算法 |
|---|---|---:|---:|---|
| NS K7 | 4/96 | 1,344 | 1.8011011% | 全部FP16位一致，algorithm_error=0 |
| BRI K28 | 4/96 | 2,535 | 1.9230541% | 全部FP16位一致，algorithm_error=0 |

所有场景实际Core事件检查及计划数值门禁通过。周期是本次小尺寸完整H/Y→xhat模型结果，
不能与默认batch96/Nr64/Nt16旧表混用，也没有证明最终通信质量或算法性能前沿。
两个配置均接近2%数值门限；扩大适用域时必须重新筛选和用新留出集验收。

## 门禁负例与失败保留

- **真实Asim性能拒绝**：引用上述NS逐场景封存基线，冻结minimum_speedup=1.01，重复相同实现。
  第一个场景仍为1,344周期，得到SKIPPED_NOT_COMPETITIVE；无numerics_started.json、references.npz、outputs.npy。
  其余三个场景NOT_RUN，计划PLAN_GATE_FAILED，不改K、不重试。
- **真实入口的错误SHA拒绝**：PREFLIGHT_REJECTED，数值NOT_RUN，未启动Asim。
- **Asim失败路径**：通过单元mock确认独立数值参考和实际重建均未调用；本轮不把它宣称为真实故障二进制实验。
- **首轮实际失败**：`results/quality_handoff/2026-11-08-v1/ns/`的首场景重建通过，但封存器把JSON整数键/元组转换误判为篡改。
  原件及计划保留，未覆盖。修正后使用全新v2计划重跑；新增测试确认规范化后真正计数变化仍拒绝。

全部171单元测试通过；本轮新增12项，覆盖伪造筛选结果、参数/种子/精度/源码/覆盖变化、
缺基线、阈值丢失、版本在运行中变化、批准hash、JSON规范化及Asim失败跳过数值。

## 命令与证据

[入口命令](../../../verification/planning/README.md)；重跑时使用新目录，禁止覆盖。
[summary.json](summary.json)记录逐场景结果、冻结参数、计划hash、绑定源码和边界；
[run_manifest.json](run_manifest.json)绑定217份原件的路径、字节数与SHA-256，包括失败v1。
原始数据目录：`results/quality_handoff/2026-11-08-v2/`；不向Git提交大型trace。
原始计划`ns-plan.json`、`bri-plan.json`及性能拒绝计划均保留，可核对批准SHA。
每场景保留核心evaluation_result.json、runtime_pin.json及plan_binding.json。

脚本elapsed_seconds记录执行墙钟，不包含Agent纯思考时间；后者不可观测，为null。
冻结阶段重新核算还会额外消耗一次完整筛选预算，尚未做前缀缓存加速。

下一步：Q3从实际重建输出构建统一瑞利通信指标，明确SE代理/BER口径、实际输出恢复、
配对MMSE及统计门限；之后扩大batch96、多SNR和新留出集，再扩CDL-B。暂不开放评分晋级。
