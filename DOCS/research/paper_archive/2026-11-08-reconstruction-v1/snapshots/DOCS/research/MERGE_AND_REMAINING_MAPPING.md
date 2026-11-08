# 剩余分块/合并算子重建与旧Merge异常复盘

2026-11-08（香港时间）。本轮保留旧实现与历史报告，新增Cholesky-block、LDL-block和合法GEMM归约Merge候选；三者复用既有四个重建/绑定/事件核心。

## 旧Merge：已证实的问题

最新二进制重新运行旧NoBlock与Merge，各24核/batch96、M64/U16、precision2周期模型；原件独立保存。两者全部96batch仍在欠定义操作处拒绝，未获得数学正确/优化通过证书。

| 实际观测 | 旧NoBlock | 旧Merge |
|---|---:|---:|
| Asim周期（诊断） | 23444 | 10004 |
| 总Instruction | 239232 | 77952 |
| SCALAR条数 | 235008 | 73728 |
| SCALAR声明元素总量 | 235008 | 235008 |
| compute_size>1的SCALAR | 0 | 30240 |
| Scalar驻留时长之和 | 317952 | 156672 |

`CholeskyNoBlockMergeOp.cc`把j条乘法/累减合并为`SCALAR_MUL/SUB compute_size=j`，却没有显式归约结果/索引。`SystolicWS::get_scalar_compute_cycles`只按opcode返回固定延迟，不读compute_size/tile_k。历史文档声称SCALAR约ceil(j/128)计费，与当前实现不符。

因此可以确定：**声明工作量没有减少，被固定延迟计费的指令数却大幅减少，且真实归约语义未定义。** 重建首错为batch0 index26、CHOL_NB_POTRF_SQ_1、SEMANTIC_AMBIGUITY；不能依据注释替其执行dot/sum。另有Model输出[batch,M,U]和Op实际[batch,U,U]步长不符。旧NoBlock也有前代漏累加/零源歧义，不能将其当已验证数学真值。

这足以拒绝旧“2.34×优化”结论，但不足以给一个欠定义程序计算精确数值误差；不能把缺失的161280个标量工作单位直接加到总周期，多核/依赖/重叠都要重新模拟。旧源码、旧报告与失败证据不覆盖。

## 成本探针：实际验证固定延迟

相同程序LOAD→Vector初始化→单元素SCALAR_MUL→STORE，仅修改SCALAR.compute_size为1或16。整套导出记录比较只允许该字段及header声明宽度不同。两次实际结果均**51周期，Scalar事件1周期**。

width16故意违反单元素SCALAR契约，明确INVALID_SCALAR_WORKLOAD，不是合格MMSE程序，不获得数值或性能分。新候选真实记录中把SCALAR宽度改16后，共同成本绑定器也拒绝，证明此低报方式不能进入正常候选数值验收。探针不与batch96算子比较性能。

## 三个新候选如何补齐真实工作

- Cholesky-block：实块4，块内POTRF逐条展开；off-panel列递推TRSM；真实k=4的A22-=L21L21ᵀ。前后解/输入输出恢复全计时。
- LDL-block：单位L、D加权块内更新；W=L21D11及A22-=WL21ᵀ；前解后逐行除D再后解，显式所有权重。
- 新Merge：左看分解的对角/列尾归约和三角解归约全部用真实GEMM，k=j或n-j-1；SCALAR只处理一个已经完成的标量结果。不存在“把多项归约声明成一条SCALAR”的捷径。

共同完整MMSE任务H/Y/λ→xhat，默认24核/batch96、Nr64/Nt16/RHS8、瑞利、λ0.1；第二组batch48/Nr32/Nt8/RHS4。实块表示R(Z)，三角解直接求任务结果，不额外制造全逆。新旧程序边界/表示/输出不同，不用10004与新Merge周期作修正倍率或优化比。

## 数值/事件/阶段验收

三个新入口均按先Asim、实际操作/成本/事件检查，再独立算法与任务参考、逐batch重放的流程测试。FP16存储/FP32算术按每指令RNE写回、GEMM逐k累加；新参考不导入engine/precision策略。实际L/D只读快照与独立完整因子比较，精确shape/finite、FP16位模式；不合成STORE、不向内存写参考、不越过首错。

主批次实际16个场景（3新入口各4组＋4已验收入口回归），1392个batch；其中6场景576个batch额外做完整L/D见证。新运行包另做三个入口日常与Merge基线重复，384batch全部通过。151项单元测试、独立代码与严格证据复审通过。精确状态、误差、时间与流水在冻结清单case_summary、pipeline_summary中，以实际门禁结果为准。

| 新入口 | 默认F64结构周期 | 默认FP16模型周期 | 第二尺寸F64周期 |
|---|---:|---:|---:|
| Cholesky-block | 26037 | 21338 | 5371 |
| LDL-block | 26942 | 21665 | 5391 |
| 合法Merge | 19852 | 19028 | 5005 |

这三者采用不同分组与列Vector安排，不能把周期差异都归因于“分块”单一因素；本轮建立可重建基线，尚未做性能消融、通信suite/CDL-B或晋级评分。所有score=null。硬件级认证由负责人后续上板。

## 统一周期对照与归档

七个新入口已全部通过本轮受限门禁，已生成相同任务/输入/配置/精度/计时identity的模型周期对照CSV，同时保留逐core/unit/opcode/name、事件数量、duration_sum/union、min/max/mean等详细流水。驻留时长不能直接相加为总周期，MTE按fragment观测。该对照不混入旧周期或跨精度数据，也不替代通信质量验收。

本轮原件：`results/remaining_mapping/2026-11-08-block-merge-v1/`；Git证据：[冻结清单](../../experiments/remaining_mapping/2026-11-08-block-merge-v1/run_manifest.json)及[摘要](../../experiments/remaining_mapping/2026-11-08-block-merge-v1/README.md)。包含实际Instruction/dispatch/trace/Core流、输入/参考/输出、因子快照、库/配置/binary、源码版本、失败构建/重试、审查与耗时。

耗时延续既定口径：任务墙钟、工具区间并集、Agent+未埋点残差，纯内部思考时间null。实际数值重放是确定性脚本，不需要Agent逐条重新推理。

## 后续

七个新入口的受限结构与精度验证完成后，优先减小真实Core检查的日志/内存占用，再接正式通信指标与CDL-B，最后开放有正确性硬门禁的评分/优化闭环。旧Merge保持拒绝；未来合法融合必须提交真实原语和可核查成本，不能通过自报FormulaLogger或compute_size改变来绕过工作量门禁。


## 本轮七入口FP16模型周期对照

以下同为24核/batch96、Nr64/Nt16/RHS8、瑞利seed42、完整H/Y→xhat边界；comparison_identity在全部七行完全相同。误差是对高精度MMSE任务的相对误差，不是通信BER/EVM。

| 新入口 | Asim周期 | 最大任务误差 |
|---|---:|---:|
| Cholesky NoBlock | 28006 | 0.2541%以内 |
| LDL NoBlock | 18697 | 0.2543%以内 |
| Cholesky Block | 21338 | 0.2184%以内 |
| LDL Block | 21665 | 0.1940%以内 |
| Newton–Schulz，K16 | 7950 | 0.1058%以内 |
| Block-Richardson，K192 | 23047 | 0.1498%以内 |
| Cholesky Merge（真实GEMM归约） | 19028 | 0.1356%以内 |

这是当前受限模型和正确性参数的对照，不是性能晋级榜；BRI K192也不是负责人历史最优候选。详细case_summary含两个seed和第二尺寸，latest_cycle_comparison保留统一身份，pipeline_summary及原件细表保留流水统计。不同算法的合法指令数量可以不同；本框架拒绝的是未定义归约/虚构工作量。

本轮构建曾因成本探针Tensor构造参数需要lvalue shape而失败，修复后成功构建；两次失败与最终成功日志均保留，没有使用失败构建产物验收。证据门禁复审还补了case身份、非空因素角色、legacy原件绑定、daily版本与完整probe差分的拒绝测试。
