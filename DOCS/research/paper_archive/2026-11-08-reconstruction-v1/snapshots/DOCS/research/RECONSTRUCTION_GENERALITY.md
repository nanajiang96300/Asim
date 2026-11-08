# 重建流程普适性与Agent耗时记录

2026-11-06，香港时间。本轮以独立Cholesky直接求解接入现有NS流程，检验“不同算法、相同任务、同一解释器”是否成立。硬件级认证由后续上板承担。

## 结论与范围

共同数值engine、实际Instruction/source-view适配、物理工作量绑定、Core事件检查四个模块与7ea3405e逐字节相同，见归档`common_core_identity.json`。没有新增“看到Cholesky名字就调用标准分解”的回放分支。新增内容仅是候选真实指令、独立参考、算法参数适配，以及公共证据/耗时字段。

NS迭代法与Cholesky直接法均已通过真实Asim→事件/成本检查→数值重放→独立算法→高精度MMSE参考。这是两类算法、两组尺寸、两种精度模型上的复用证据，不是任意算法或所有指令的普适性证明。旧七入口仍保留原失败状态，不能用新候选替其改报PASS。

## 接入时发现与修正

原catalog强制所有算法使用NS的iterations与I/trace初值，参考归档强制X0。这两项已移出公共假设：受信参数工厂声明算法参数；参考返回通用evidence，NS仍保留X0，Cholesky提供factor_L等证据。直接法当前以iterations=1作为CLI兼容占位，冻结algorithm里不记录虚构迭代数。基线比较完整参数和封存制品。

增量构建还发现CMake GLOB不自动发现新源码，首次链接失败；增加CONFIGURE_DEPENDS后构建成功。失败日志及重试耗时保留。所有编译-j1，仿真串行；独占隔离工作区/build增量构建，使用已存在依赖，不宣称全新机器干净重建。

## 实验与判据

共同full-MMSE边界为H/Y/λ输入至xhat写回，包含Gram、正则化、初始化/分解、求解/恢复和DMA。默认24核、batch96、Nr64/Nt16/RHS8；第二组24核、batch48、Nr32/Nt8/RHS4。瑞利seed42/123，λ=0.1。实块表示保持四象限检查。

| 样本 | 模型 | 周期 | 最大任务相对误差 | 结果 |
|---|---|---:|---:|---|
| Cholesky，默认尺寸，seed42 | FP64结构 | 34154 | 1.07e-15以内 | 三路径通过 |
| Cholesky，默认尺寸，seed42 | FP16存储/FP32算术模型 | 28006 | 0.2541%以内 | 所有FP16位一致、任务通过 |
| Cholesky，默认尺寸，seed123 | 同上 | 28006 | 0.2733%以内 | 所有FP16位一致、任务通过 |
| Cholesky，第二组尺寸，seed42 | FP64结构 | 5677 | 8.67e-16以内 | 三路径通过 |
| NS，默认尺寸，seed42 | FP16存储/FP32算术模型 | 7950 | 0.1058%以内 | 回归通过 |

这是保守串行Cholesky正确性基线，没有优化其周期。不同尺寸/精度不能混排；所有score=null。逐core/unit/opcode/name流水统计留在各run的pipeline_statistics.json和instruction_pipeline_detail.csv；Git保存摘要CSV与hash。

4项离线负例全部被拒绝：两个尺寸的错误对角索引均数值失败，未知opcode变异在实际操作数绑定检查处被拒绝。负例是实际记录的离线变异，没有运行错误候选二进制，不使用其周期。新独立候选原始指令与旧算法不同是实际算法工作量差异；禁止以额外无效矩阵乘获取奖励。

## 时间口径

需要区分三个量：

1. **一次性Agent接入工作**：本轮从环境读取后的首个时钟起，涵盖理解、设计、写候选/参考/适配、审查、验证和归档。记录任务墙钟；不把全部墙钟称作思考。
2. **工具运行时间**：timed_command以monotonic记录命令、开始/结束和退出码；runner以perf_counter分出Asim、制品/调度检查、独立参考、实际逐指令重放与比较。异步命令与工具调用重叠时取区间并集，不重复相加。
3. **Agent处理代理值**：总墙钟减去已观测工具区间并集。它仍含代码输出、调度、未埋点工具和等待，仅供粗略预算；`pure_model_thinking_seconds=null`。当前接口无法取得模型内部纯思考计时或可靠思考token计数。子Agent审查墙钟66秒（含读取/工具/判断），与主Agent并行，不相加成总耗时。

接入完成后，重建脚本本身不调用LLM；不需要Agent逐条重新解释指令。新增语义/算法接入的开发成本与重复执行同一解释器的运行成本分别比较。

默认Cholesky seed42实测：FP64数值重放与比较约11.35秒、独立参考0.014秒、Asim30.60秒、制品与调度45.06秒；FP16数值重放与比较约6.43秒、独立参考1.40秒、Asim22.37秒、制品与调度37.37秒。精确字段在case_summary.csv/result.json，不等于NPU周期。整组5样本命令约254.38秒（包含归档）。各次日常回归另计。

本轮任务总耗时及Agent代理值见同目录归档的effort.json/effort_summary.md；计时截止于生成该证据，后续文档收尾和Git提交不包含在快照里。首次环境工具及部分协调器调用未全面埋点，故代理值不可冒充精确“思考秒数”。

## 资源与稳定入口

FP64默认Cholesky实际Core观察日志约251MiB，接近256MiB预算；观察到检查进程RSS约2.35GiB（采样值，不是完整峰值测量）。未放宽资源门禁；更大batch/尺寸需要先评估日志和内存。下一步优先流式绑定/检查以减少内存，同时保留相同拒绝语义。

新runtime发布后可运行：

```bash
.venv/bin/python -m verification.reconstruction.daily --operator cholesky_explicit_v1 --iterations 1
# 默认NS保持兼容
.venv/bin/python -m verification.reconstruction.daily
```

目录自动唯一，旧基线不能跨runtime复用。当前只支持已验收的声明精度模型、瑞利和明确source-view语义；未知归约、DMA变换、tail和CDL-B仍明确拒绝。

## 证据入口与后续

[冻结清单](../../experiments/generality/2026-11-06-cholesky-v1/run_manifest.json)；[摘要与计时](../../experiments/generality/2026-11-06-cholesky-v1/README.md)。原件单独保存在results/generality/2026-11-06-cholesky-v1，旧数据不覆盖。

下一步用独立LDL候选验证D因子与行权重恢复，并处理已实测的日志/内存瓶颈。继续沿用真实指令→共同解释器、独立算法和任务真值，不要求每加算法改一次核心。随后再接Block/BRI和通信/CDL-B suite。

本轮计时快照：任务墙钟约16.54分钟，已观测工具区间并集7.43分钟，Agent处理与未埋点残差9.11分钟；内部纯思考时长不可观测。Cholesky日常首次/基线复用与NS默认入口均已通过，重复输出、指令和事件流字节一致。
