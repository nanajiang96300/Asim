# Asim数值重建流程验证报告（论文资料阶段稿）

日期：2026-10-30（Asia/Hong_Kong）；状态：INTERIM。当前不具备生成“全部验证完成”最终报告的条件。

## 研究问题与方法

目标是在通信质量约束下减少NPU通信算子周期。验证对象是实际生成的Instruction，而不是按算法名调用标准求解，也不是仅重放FormulaLogger声明。共同任务输入H/Y/λ，输出MMSE均衡结果；完整计时包括Gram、初始化、迭代、输出恢复和DMA。

三路径分别为实际指令重建、同初值/参数的独立算法参考、独立高精度solve任务参考。Asim成功、成本/事件绑定与性能预筛之后才执行数值；不竞争或异常加速保持NOT_RUN。

## 与早期流程的区别

| 环节 | 早期证据边界 | 当前新增约束 |
|---|---|---|
| 数值来源 | 公式日志或算法名不能证明实际Instruction | 源构造点视图、opcode/地址/shape/DMA/成本绑定 |
| 多核与输入 | 单样例、batch/输出绑定可能混用 | 24核、batch96、每batch独立输入与输出检查 |
| 数学真值 | DAG自一致不等于MMSE正确 | 独立同算法与高精度任务两种参考 |
| 精度 | FP64结构通过不等于FP16 | 逐指令FP16写回、FP32逐k累加模型与完整位比较 |
| 周期 | 不同输入、范围或估算可能混排 | task/input/config/scope绑定；profile不同不算优化收益 |
| 报告 | PASS陈述可能缺原件 | commit/输入/工具链/验证器/制品hash与最终报告硬门禁 |

## 实验设置

共同配置：瑞利H(64×16)、Y(64×8)、λ=0.1，batch96，24核，Cube16³；种子42/123/456。采用显式实块编码，物理H128×32、输出32×16。NS K16，X0=I/trace(R(A))；Gram、trace初始化、负号MUL与最终X@b均实际计时。

精度模型自定义为：FP16存储与每指令写回，FP32标量/Vector运算，GEMM按k升序非融合FP32乘后加。输入编码后RNE量化一次；subnormal保留、非有限拒绝。独立参考自行实现同操作顺序，任务oracle使用原始H/Y/λ。Vector工作量按FP32宽度计，存储按FP16宽度计。

这不是原生910B精度/转换成本校准。2%任务误差目标是本轮工程目标，不是硬件误差界或正式通信质量阈值。

## 数值结果

| profile | seed | batch | 周期 | FP16逐位相同batch数 | 最大任务相对误差 |
|---|---|---:|---:|---:|---:|
| structure_f64 | seed42 | 96 | 14298 | 不适用 | 1.13572e-15 |
| structure_f64 | seed123 | 96 | 14298 | 不适用 | 1.12108e-15 |
| structure_f64 | seed456 | 96 | 14298 | 不适用 | 1.10357e-15 |
| modeled_fp16_fp32 | seed42 | 96 | 7950 | 96 | 0.00105726 |
| modeled_fp16_fp32 | seed123 | 96 | 7950 | 96 | 0.000968335 |
| modeled_fp16_fp32 | seed456 | 96 | 7950 | 96 | 0.00102787 |

FP64与FP16模型的周期属于不同profile，不能把差值写成同精度优化收益。旧NS4,647周期与新完整任务14,298周期也不可直接比较。

## 拒绝错误与剩余问题

低K候选能与算法参考一致而任务质量失败；漏LOAD、错误残差、最终平方、跨batch读取、工作量低报和不匹配基线均有拒绝证据。旧七入口仍有五类语义歧义入口、BRI越界及NS输入/数值失败，不可声称全部算法正确。

RAW/WAR/WAW检查器已测试部分写、初始化、core/buffer别名及合法重叠；这些是合成事件模型测试。实际Asim缺少一一绑定的读完成/写可见事件，pipeline寿命不能替代G4。

## 最终报告门禁

| 门禁 | 当前状态 |
|---|---|
| actual_instruction_replay | PASS |
| independent_algorithm | PASS |
| high_precision_task | PASS |
| declared_precision_model | PASS |
| hardware_precision_cost_calibration | PENDING |
| actual_async_G4 | UNSUPPORTED_MISSING_RUNTIME_RW_EVENTS |
| formal_communication | PENDING |
| CDL_B | UNSUPPORTED |
| all_legacy_operators | FAILED_OR_UNSUPPORTED |
| clean_reproduction | PENDING |

只有全部必需门禁PASS才允许--final。当前可用于论文方法、实验设置与阶段性数值结果；不得写成完整框架、全部算子、硬件精度、CDL-B、通信质量或实机性能均已验证。

## 证据与复现

- [unified_reconstruction/2026-10-30-ns-v1-frozen](../../../../experiments/unified_reconstruction/2026-10-30-ns-v1-frozen/run_manifest.json)
- [precision_campaigns/2026-10-30-fp16-v2-frozen](../../../../experiments/precision_campaigns/2026-10-30-fp16-v2-frozen/run_manifest.json)
- [reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen](../../../../experiments/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/run_manifest.json)

readiness.json为机器可读完成状态，evidence_index.json绑定原清单hash与提交；numerical_results.csv为可用于论文作图的原始小表。大trace、输入、参考、输出和二进制保留原目录，旧报告不覆盖。
