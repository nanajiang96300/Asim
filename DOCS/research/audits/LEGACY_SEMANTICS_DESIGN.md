# 七入口源码语义迁移与重建测试设计

2026-10-29。用户明确授权补语义后重建测试；本轮不修改指令算术、数量、地址或调度来使算法通过。

## 2. 公式推导与真值边界

目标契约为给H和λ的A=HᴴH+λI求逆，BRI另有Y输入时检查solve(A,HᴴY)。高精度solve独立于候选。结构精度complex128，保持实际ADD/MUL/SUB/DIV/SQRT等算术；不将ADD解释为SUB、不把TMP覆盖改成累加、不补漏写的单位元或负号、不省略NS最后X²。物理complex/FP16成本仍未认证。

在实际Instruction构造点附加views，view指定base、row/col、rows/cols、transpose/conjugate；源base须与该Instruction实际操作数相符。标量逐元素、全矩阵、列缩放分别明确。无法从已有循环/数据流唯一确定的block行列位置使用显式invalid reason，不猜索引。未知/歧义是问题，不是通过。

Operation记录真实Model输入输出tensor的地址、shape、name、size及本地region分配。DMA读取先验证源tensor范围及请求覆盖，写回先验证声明输出范围。每个region有逐元素初始化掩码；partial写不初始化整矩阵。新语义绑定到同一Instruction，FormulaLogger只作历史诊断。

输入采用固定独立瑞利H，RegI=λI、X_init=I/||A||₂、C=2I，依据实际Model槽位与shape提供，不将NS的H改成A来掩盖输入绑定。批次间输入不同，24核、batch96，规模按各注册入口配置保留；NS矩阵输入/输出契约单列，不混合排行榜。

矩阵块POTRF内部源A而非L、block TRSM缺完整列下标等原意不唯一之处必须报告SEMANTIC_AMBIGUITY。Merge的批量SCALAR若没有显式归约，不能默认为dot/sum；可执行元素乘积但减法结果shape不合时失败，并标记成本模型不一致。

门禁顺序：静态schema→真实Asim→实际typed Instruction重放→独立任务参考。正确性审计覆盖全部入口，即使性能无竞争力也执行（诊断campaign固定无性能淘汰，不能拿该结果晋级）。发现错误或歧义则记录，不能把“全部正确”作为预设结论。全部门禁通过才生成verified对比，未通过仅归档unverified诊断周期。

统计包含按core/unit/opcode/完整instruction name的计数、总事件时长、区间并集、最小/最大/平均时长、重叠和每核空闲。总时长之和不是端到端周期；wait/barrier单独列示。MTE trace按fragment计数，旧重复ID不冒充唯一instance时序认证。结果独立目录results/reconstruction_campaigns/<id>与Git小清单experiments/reconstruction_campaigns/<id>。
