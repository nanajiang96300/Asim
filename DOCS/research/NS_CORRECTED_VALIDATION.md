# 统一任务、共用重建核心与独立NS候选验收

2026-10-30（Asia/Hong_Kong）。本轮完成R02的结构范围：统一H/Y/λ输入和完整均衡计时范围，新增独立NS候选，同一实际指令重建核心连接旧七入口与通用程序；NS三种瑞利种子各96个batch三路径比较通过。**这是FP64实块结构验收，完整FP16/G4/通信质量/G6与评分仍未通过，score=null。**

## 实现

- [共同任务与边界](../../verification/contracts/mmse.py)：MMSETask默认batch96、Nr64、Nt16、Nrhs8、λ0.1；固定归一化瑞利/QPSK及共同I/λI常量。完整任务输入不给A/b/X0，候选指令内生成；给定A/b/X0/C的旧内核scope单列，禁止直接比较。
- [独立候选](../../src/inverse/ns_corrected/NSCorrectedModel.cc)：新增ns_corrected_test，保留原NS与七入口注册表。HᴴH+λI、HᴴY、逐对角trace、初始化、16轮迭代、X@b和DMA均进入Asim周期。由X0=I/trace(R(A))初始化；实际MUL(-1)+ADD执行残差，最终输出X@b。没有隐含X²或buffer swap。每batch唯一ID，24核每核4tile。
- [共用重建核心](../../verification/reconstruction/engine.py)：统一ArrayMemory初始化掩码、切片/共轭/转置、read/write、原语算术与显式广播。source_views适配实际源视图指令，shadow适配通用程序；legacy/replay只保留兼容导入及历史fixture。核心不根据算法名求解，不读FormulaLogger补答案。
- [受信运行器](../../verification/reconstruction/runner.py)：实际header/input/shape/成本/dispatch绑定，Cube16³强制启用，全部Cube/Vector/Scalar与DMA事件检查，Asim后才计算参考和候选数值。预筛绑定task/input/hash/config/scope的comparison_identity，防止跨输入或计时范围复用基线。
- [独立算法参考](../../verification/reference/ns.py)：直接复数NS，使用同K和τ=2trace(complex A)，不调用重建器。独立任务路径另外solve(A,HᴴY)。原始H/Y、X0、两个参考和实际输出均归档。

实际计时是precision8、显式实块H128×32/Y128×16/xhat32×16；矩阵负号与除法都按完整物理元素计Vector工作量。decode仅为各算法共用的表示解码并检查四象限，不宣称原生复数或FP16硬件成本。Asim SRAM配置的KB单位/容量尚未校准实机。

## 实测门禁

冻结runner `c9f6edeb`，独立工作区C++ `5fd2ab81`（主仓库候选提交`a531bc91`）。串行-j1构建成功，二进制SHA及源码/工具链/原件摘要见[清单](../../experiments/unified_reconstruction/2026-10-30-ns-v1-frozen/run_manifest.json)。

| 场景 | 实际结果 | Asim周期 | 数值 |
|---|---|---:|---|
| seed42，K16，96batch | STRUCTURE_CHECKED | 14,298 | 最大actual↔algorithm 1.10e-15；actual↔task 1.14e-15 |
| seed123，K16，96batch | STRUCTURE_CHECKED | 14,298 | 1.18e-15；1.13e-15以内 |
| seed456，K16，96batch | STRUCTURE_CHECKED | 14,298 | 1.17e-15；1.11e-15以内 |
| seed42，K1 | ALGORITHM_QUALITY_FAILED | 6,907 | 实际序列符合独立算法，任务精度不够；不冒充重建错误 |
| 不竞争门槛（冻结基线0.1倍） | SKIPPED_NOT_COMPETITIVE | 14,298 | NOT_RUN |
| 离谱加速门槛（冻结基线100倍） | QUARANTINED_IMPLAUSIBLE_SPEEDUP | 14,298 | NOT_RUN |

后两项是预先定义的门禁探针，不是优化收益。全部按同task/input/config的基线identity匹配。

79项单元测试通过，包括共享初始化/广播、未知原语、decode四象限、同初值低K参考、scope/CDL-B拒绝及提前Scalar事件。独立只读审查发现的τ/decode/广播成本与Scalar dispatch问题均已处理并复审通过，见[审查记录](../../experiments/unified_reconstruction/2026-10-30-ns-v1-frozen/review.md)。

四个离线实际记录变异被拒绝：错误残差算术、漏LOAD、把最终应用改成平方、跨batch读取；两个工作量低报变异被成本绑定拒绝；跨seed基线与关闭Cube模型在Asim前被拒绝。离线变异明确不是新候选的实际Asim运行，未报告变异性能。

旧七入口使用上轮已校验hash的真实Asim导出/输入重新通过统一核心执行，每入口96batch的失败分类与原记录一致：五入口仍歧义、BRI仍越界、NS仍数值/输入绑定失败。没有重跑旧周期，也没有修复旧算法。它们仍不能构成可信排行榜。

## 兼容性变化和剩余边界

旧通用schema将SUB直接映射成raw ADD，不能证明实际减法。当前投影明确UNSUPPORTED；只有显式展开MUL(-1)+ADD的新候选可走可信数值路径。旧given-inputs原型的历史PASS原位保存，不能用其旧证据绕过当前更强的投影检查。CPU纯原语测试仍可执行抽象SUB。

本轮“统一所有算子的重建逻辑”指共享数值/内存核心及一致的拒绝语义；不表示旧算子全部补齐数学语义，也不表示已统一改造所有旧Model为H/Y完整均衡任务。真实数值字段仍是源码可执行语义描述，未实现硬件精度、完整异步读写可见性证明或OS级隔离。

[新增算子接入指南](ADDING_OPERATOR.md)说明契约、最小插件协议、成本检查和独立参考接口。算法生成参数可以由适配层扩展，不能把分解/求逆硬编码进重建核心。下一项优先补精度profile与异步读写门禁；并在独立候选中消除NoBlock/Block归约歧义、修复BRI初始化/迭代/输出恢复，再做同口径比较。CDL-B仍需单独定义信道模型后接入。

## 归档与复现

专用原件：`results/unified_reconstruction/2026-10-30-ns-v1-frozen/`，103件约209MB，包含六个新Asim run、原输入/参考/输出/实际Instructions/trace、逐core/unit/opcode/name流水统计、旧七入口重放、负例、冻结binary和环境。Git只保存[小型摘要与清单](../../experiments/unified_reconstruction/2026-10-30-ns-v1-frozen/README.md)；旧资料不移动、不覆盖。

```bash
python3 scripts/check_reconstruction_archive.py experiments/unified_reconstruction/2026-10-30-ns-v1-frozen/run_manifest.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.reconstruction.gates \
  --binary /home/nanajiang/Asim/results/unified_reconstruction/2026-10-30-ns-v1-frozen/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --cwd /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build \
  --output results/unified_reconstruction/<new-run-id>
```

output必须不存在；使用冻结runner版本复现。首次smoke另存`2026-10-30-ns-v1-smoke`，不覆盖最终证据。14,298周期包含比旧NS更多任务，不能与旧4,647周期直接算加速比。

追加通用适配回归：共用核心下原有7项真实Asim框架探针均得到预期结果（包括正例Gram、数值错误拒绝、漏LOAD及性能预筛）。原件`results/unified_reconstruction/2026-10-30-generic-regression/`，summary及全部hash收录同一清单的supporting_artifacts；它不使用已禁用的SUB投影。
