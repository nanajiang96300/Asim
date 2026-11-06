# LDL与BRI的真实指令重建映射

2026-11-06（香港时间）。本轮新增独立LDL、Block-Richardson候选，沿用现有NS/Cholesky的任务、解释器、事件与数值门禁。旧源码与历史失败均保留。硬件认证由负责人后续上板。

## 实现范围

共同输入仍为H/Y/λ与公共Identity，输出完整batched xhat；Gram、正则化、初始化、预条件、迭代和恢复全部计时。默认24核/batch96，Nr64/Nt16/RHS8，瑞利，λ=0.1；第二组batch48/Nr32/Nt8/RHS4。四个核心engine/source_views/binding/runtime与92bb3383逐字节相同，没有按算法名替代真实重放的分支。

- LDL：右看式LDLᵀ，L=I；显式D加权Schur更新；Lz=b后按**行**除D，再解Lᵀx。整列DIV/MUL真实Vector工作量，单元素才用SCALAR。没有旧实现的2λI单位对角和列权重恢复问题。
- BRI：2×2**实块**对角逆，块外零，逆块非对角负号真实生成；ω=0.5，x0=0。P=ωB、S=I−PA、c=Pb全部真实计算，K192次x←Sx+c，每次写回。与x←x+ωB(b−Ax)等价，但有限精度以所生成的预计算形式为准。

本BRI用于验证语义，K192不是性能最优参数，也不代表负责人所指历史最优候选。旧BRI的范围、代码问题和新候选均分别保留；不能据本轮周期否定其他候选或CDL-B适用性。

## 实际数值结果

| 新候选 | 模型、参数 | 默认周期 | 最大任务相对误差（seed42/123） |
|---|---|---:|---:|
| LDL | FP64结构，直接法 | 24647 | 见冻结逐batch结果，全部≤1e-10 |
| LDL | FP16存储/FP32算术模型 | 18697 | 0.2543%以内 |
| BRI | FP64结构，K192 | 42037 | 见冻结逐batch结果，全部≤1e-10 |
| BRI | FP16存储/FP32算术模型，K192 | 23047 | 0.1598%以内 |

默认两seed的FP16输出均与独立逐指令精度模型**全部位一致**，任务误差门限统一2%，未为新算法放宽。F64参考与复数MMSE真值独立；各阶段F16参考不导入重放器或共同精度策略。

第二尺寸F64：LDL4689周期、BRI9588周期；旧已验收NS7950、Cholesky28006周期FP16回归通过。上述不是跨尺寸/精度或历史源码的性能排名，score=null；详细逐core/unit/opcode/name流水统计保留在每次run原件及Git摘要CSV。

主批次11场景：864个正例batch通过，24个低K质量负例按预期拒绝。**BRI K1只有874周期，重放与算法一致，但任务误差最高约57.9%，ALGORITHM_QUALITY_FAILED**，不获得性能分。两个离线算术变异也被拒绝：LDL错误D行下标误差约3.91%，BRI末次ADD换MUL误差约100.57%。离线变异没有错误候选仿真周期声明。

## 实际中间内存见证

`scripts/audit_solver_stages.py`复核已完成主批次的receipt/制品hash，再从实际Instruction做只读capture；不创建合成STORE，不越过首错，不向执行内存写参考值。另做672个batch阶段复核：

- LDL：所有实际L对角为1、D正；F64满足A=LDLᵀ；FP16实际L/D与独立参考的完整位模式一致。
- BRI：实际B与独立逐块参考一致，实际x0为零。

这使映射检查既覆盖最终任务，也检查分解/预条件阶段。参考永远不能作为未知指令、遗漏归约或未初始化内存的补值。

## 构建、门禁与日常入口

144项单元测试通过，独立代码审查通过；审查墙钟88秒，仅表示阅读/工具/判断总时间，纯思考不可观测。独立工作区/build串行独占、-j1构建成功；主批次生产验证器源码前后指纹相同。阶段审计脚本首次因直接执行的import路径失败，修复入口后复跑成功。复审进一步收紧阶段精确shape/finite、完整制品清单及八场景/672batch覆盖，新增4个拒绝回归测试；严格stages-v2复跑通过，初版stages仅留探索证据。失败与重试日志均保留。

日常LDL、BRI、BRI基线复用、NS四次验收通过，384batch全部通过；BRI重复输出/指令/事件流字节一致。新固定包支持4个独立候选：

```bash
.venv/bin/python -m verification.reconstruction.daily --operator ldl_explicit_v1 --iterations 1
.venv/bin/python -m verification.reconstruction.daily --operator bri_explicit_v1 --iterations 192
# NS默认入口继续可用
.venv/bin/python -m verification.reconstruction.daily
```

BRI的迭代预算作为受信adapter约束1..256；NS仍为1..32，直接法仍使用1占位。白名单不代表所有参数自动通过；每次仍需实际Asim/事件与完整数值验收。低K失败就是此约束的实际负例。新旧runtime基线禁止混用。

## 证据与后续

本轮原件单独存储于`results/operator_mapping/2026-11-06-ldl-bri-v1/`，Git索引为[冻结清单](../../experiments/operator_mapping/2026-11-06-ldl-bri-v1/run_manifest.json)及[摘要](../../experiments/operator_mapping/2026-11-06-ldl-bri-v1/README.md)。包含Instruction/trace/Core事件、输入/参考/输出、实际中间内存、配置/库/二进制、流水、失败重试与时间台账。

耗时继续按任务墙钟、已观测工具并集、Agent+未埋点残差记录，纯模型思考时间null；重放/参考/Asim/检查秒数分别归档。见effort.json与effort_summary.md，不能把本轮开发耗时当成重复执行重建的耗时。

当前形成四类算法的同核重建映射；Cholesky-block、LDL-block和Merge仍需独立真实展开块内解/Schur更新/归约，不能把旧入口改报通过。下一步优先建立这些块算法的显式映射，同时按已有256MiB/百万事件预算处理观察日志的内存瓶颈，再扩通信与CDL-B。声明的逻辑view与周期模型仍不等于原生910B布局/搬运成本校准。
