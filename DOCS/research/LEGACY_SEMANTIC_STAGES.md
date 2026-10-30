# 七入口语义覆盖与阶段数值重建

2026-10-30。本轮落实负责人要求：硬件级时序认证由后续上板完成，不作为软件重建工作的前置；优先其他算子的真实语义和重建测试。**测试框架完成，旧七入口整体验证仍阻断。** 不输出可信周期排名和分数。

## 范围与方法

24核/batch96，瑞利seed42，λ=0.1；除旧NS32×32/K16，其他入口64×16，块大小2，BRI8层。保持旧Model/Op的算术、指令数量、地址、布局和同步。唯一C++修改为LDL NoBlock列缩放的显式`scalar_rhs`元数据；仍单独标记批量SCALAR成本不支持。原始指令全部字段与前次冻结批逐条相同，1,536条LDL元数据补充的证据见归档source_equivalence。

所有入口先真实Asim完成，再做数值。此轮是正确性诊断，没有同任务已通过基线，不做性能淘汰。不是调优候选的性能准入。旧算子逻辑complex128与precision2周期模型分开，不是FP16正确性或原生复数成本认证。

新增完整静态扫描覆盖714,912条实际指令，包括首错后的描述；按实际opcode/基址/切片/GEMM尺寸/DMA槽位与batch/显式广播/SCALAR工作量检查。同一指令只报告首先发现的问题；静态合法不代表已经数值执行。未知零源和归约仍UNSUPPORTED/SEMANTIC_AMBIGUITY，不猜测。

原execute_batch新增只读快照，读取实际成功执行的内存，不创建合成STORE，不越过首错继续执行。阶段参考只比较真实值，不写回；未到达为NOT_REACHED。形状与有限值严格检查，防止NumPy广播；观察点错误单独记录，不改变主执行结果。共同engine仍是唯一逐指令算术实现。

## 实际阶段结果（各96 batch）

| 入口 | Gram+正则化 | 分解/预条件阶段 | 完整输出 |
|---|---|---|---|
| Cholesky NoBlock | 96通过 | L全部通过，最大相对误差1.60e-16以内 | 前代零源歧义；循环覆盖Tmp且无累加；输出shape/批步长不符 |
| Cholesky NoBlock Merge | 96通过 | 遇到未定义融合SCALAR归约，NOT_REACHED | 阻断 |
| LDL NoBlock | 96通过 | D全部通过，最大误差2.61e-16以内；L全部失败 | L对角保留0.2，前代歧义/漏归约、列缩放与批量SCALAR问题 |
| Cholesky Block | 96通过 | TRSM索引欠定义，NOT_REACHED | 阻断；Schur减法/三角解需真实程序补全 |
| LDL Block | 96通过 | LAPPLY/加权索引欠定义，NOT_REACHED | 阻断 |
| BRI | 96通过 | 块预条件96失败，相对误差0.119–0.335；初值96失败，误差0.8 | W区域越界；输出缺batch，尚未完整验证 |
| 旧NS | 输入槽位错误，无共同Gram阶段 | 不适用 | 1数值失败、1tensor越界、94输入槽位失败 |

LDL的D正确不证明当前L正确：实际aL初始化为2λI，程序未将对角改成1。后续公式A⁻¹=ZᴴD⁻¹Z要求D权重作用于Z的行；旧代码按列缩放，且一条固定延迟SCALAR处理U元素。BRI快照显示块外初始化污染与非对角缺负号；初值实际2λI而非声明I。迭代残差是ADD且没有swap/copy，不能把注释当执行语义。上述问题针对当前注册入口，不否定其他历史候选或算法本身。

阶段参考是独立数学阶段目标，不声称这些旧代码已完成同初值/同参数完整三路径验证。尤其Block在首错后从未数值执行，不能据静态索引检查宣称分解正确。

## 本轮周期与流水制品

依注册入口列示诊断周期，不是排行榜：Cholesky NoBlock23444、Merge10004、LDL NoBlock25633、Cholesky Block6445、LDL Block4965、旧NS4647、BRI1998。与上次相同，符合未改原始算术与调度。不同尺寸、输入边界和输出任务不得计算加速比。

每入口保留`pipeline_statistics.json`及`instruction_pipeline_detail.csv`，逐core/unit/opcode/完整instruction name统计count、duration_sum/union、min/max/mean、首尾周期；另有每核busy/idle和重叠。总计177,048条细分统计行（不含七个表头）。MTE是fragment观测，不能把驻留时间之和当总周期。详细表体保存在原件目录，Git保存摘要与大小/SHA-256。

同一新验证器与二进制下，修正NS公共入口回归：FP16存储/FP32算术模型7950周期、FP64结构14298周期，两者96batch、Core保守检查和数值均通过。不跨精度算收益。新runtime只白名单该NS，不因其他算子运行结束就将其加入日常白名单。

## 运行与归档

当前冻结批：`results/legacy_semantics/2026-10-30-seven-b96-v2-raw/`；阶段原件：`results/legacy_semantics/2026-10-30-seven-b96-v2-stages/`。首次探索批保留，其运行期间仅验证器的可选观察实现继续完善，故不将首次批次作为冻结runner的最终验收。原始模拟器二进制始终固定。

小型证据入口：[本轮清单](../../experiments/legacy_semantics/2026-10-30-seven-b96-v1/run_manifest.json)。包括实际Instruction一致性、分阶段结果、130项单元测试、独立审查、固定二进制/库/配置、新runtime、两种NS回归及原件hash。旧报告与数据均不覆盖。

```bash
# 先Asim，预期退出1：完整正确性仍被阻断
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.legacy.campaign \
  --binary results/legacy_semantics/2026-10-30-runtime/bin/Simulator \
  --config configs/ascend_910b_quiet.json --registry orchestrator/operator_registry.json \
  --cwd /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build \
  --output results/legacy_semantics/<fresh-raw>

# 再检查完整指令描述与实际阶段值；0只表示诊断工具运行完成
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.legacy.extended \
  --campaign results/legacy_semantics/<fresh-raw> \
  --output results/legacy_semantics/<fresh-stages>

.venv/bin/python scripts/check_legacy_semantic_archive.py \
  experiments/legacy_semantics/2026-10-30-seven-b96-v1/run_manifest.json
```

目录必须不存在；不自动重试、不随机补输入、不批量清理。检查完整原件清单、实际receipt、全部生产verification源码前后指纹，变化时拒绝。stage工具退出0绝不代表算子PASS，报告状态固定DIAGNOSTICS_COMPLETE_OPERATORS_BLOCKED。

## 后续实施顺序

1. 独立新Cholesky NoBlock候选：保持已通过分解，明确零/单位元，实际逐项累加，完整输出shape，共同H/Y→xhat边界；通过同算法参考与任务真值后加入白名单。
2. 独立LDL候选：显式单位对角、行权重恢复，拆分真实标量或采用计成本Vector；同上三路径。
3. 独立Block候选：实际展开块内三角解与Schur减法，不将一条SCALAR声明成归约。
4. 独立BRI候选：冻结初值、预条件、迭代反馈与恢复公式，正确区域与batch；在统一任务下验证质量，不能用旧报告预判永久失败。
5. 同一参考/重建协议下再加入多seed、通信suite与CDL-B。硬件认证交由上板，不阻塞上述软件任务。

本轮没有为让测试通过而实施这些算法修复；它们需要在独立候选中真实生成缺失指令，不能仅给重建器加“标准算法”分支。
