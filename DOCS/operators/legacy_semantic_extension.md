# 旧七入口语义覆盖与阶段重建扩展

2026-10-30。保留原算法与调度；不将硬件级时序认证设为本轮门槛。

## 1. 文件与范围

`verification/reconstruction/source_views.py`增加只读阶段快照；`verification/legacy/coverage.py`检查完整指令列表；`verification/legacy/stages.py`声明观察点与独立阶段参考；`verification/legacy/extended.py`串行执行七入口并归档。共享engine仍是唯一算术执行器。原先欠定义的操作仍停止，不跳过、不补数值。

## 2. 公式推导

A=HᴴH+λI，逆矩阵任务为solve(A,I)，BRI均衡为solve(A,HᴴY)。Cholesky阶段应满足A=LLᴴ；LDL阶段A=LDLᴴ，L单位下三角。Z=L⁻¹时逆为ZᴴD⁻¹Z，权重应作用于Z的行。BRI块预条件参考是逐2×2主对角块的逆，块外必须为零。参考只用来比较快照，绝不填充实际执行内存。

## 3. 内存与语义

沿用原Model与regions，不改分配、shape、地址或opcode。唯一允许的元数据补充：LDL_NB_SCALE明确scalar_rhs，原U元素SCALAR成本问题继续作为拒绝证据。未知归约/零源不猜测。快照只在原指令执行成功后读取已初始化视图，快照失败独立记录，不改变主重建状态。

## 4. 指令映射与覆盖

逐条审查原始opcode、基址绑定、视图范围、GEMM尺寸、广播和SCALAR工作量；包括首错以后的未执行操作。静态可解释不等于数值已执行，分别统计。REG后观察A；最后COL后观察L及D；BRI初始化与预条件后观察B/Y0。不得添加合成STORE或虚构已执行事件。

## 5. FormulaLogger与验证

原FormulaLogger保持诊断用途。真实Instruction→共用engine→阶段快照，对比独立数学参考。全输出仍走原execute_batch与高精度任务参考，任何阶段PASS不覆盖整算子失败。

## 6. 验证数据

待实现后填写真实批次、测试、审查和结果路径。不覆盖旧报告；固定24核/batch96、瑞利seed42，按注册配置运行。编译与Asim串行，build -j1。
