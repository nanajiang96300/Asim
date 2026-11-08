# Scalar工作量计费诊断探针

2026-11-08。仅诊断Asim固定Scalar延迟，不是MMSE候选，不进入白名单/评分。

## 1. 文件与公式推导

src/semantic/ScalarCostProbeModel.{h,cc}。相同LOAD、Vector初始化、单元素SCALAR_MUL与STORE；仅raw compute_size分别1/16，实际视图始终1×1。宽度16故意违反已有SCALAR工作量契约。若周期相同，说明改变compute_size不能实现/计价16项运算，不能据此声称融合归约正确。

## 2. 内存/指令

64B完整输入及输出，Vector先初始化完整输出，再单元素MUL修改首值。source-view-probe-v1记录与实际Instruction绑定；完整DMA，PIPE_BARRIER，FormulaLogger诊断步骤。运行24核配置但仅1tile，不把其周期与batch96算子比较。

## 3. 验证

两探针真实Asim/Scalar trace对比，width16明确INVALID_SCALAR_WORKLOAD，无数值/性能通过声明；它与旧Merge源码/重建拒绝一起提供异常来源证据。不得给旧Merge猜测dot语义。构建与只读审查后归档。实际两个probe均51周期，Scalar事件均1周期；整套record只差声明宽度和scalar.compute_size。旧Merge仍全96batch拒绝，正常入口宽度16也被共同成本绑定器拒绝；详见MERGE_AND_REMAINING_MAPPING.md。
