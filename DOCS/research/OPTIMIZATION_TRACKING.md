# 内循环优化的候选版本与统计规则

2026-11-10，当前首例BRI固定HB。统一链路：family_id → algorithm_id → baseline_id → candidate_id → run_id → evidence。BRI性能优化归内循环，保留子类型区分instruction/layout/schedule与algorithm_parameters；若递推/预条件/精度/输入输出目标实质变化，则建立新的algorithm_id或task_id，不能伪装纯指令优化。

## 实现与Git

默认同一算子族/同一Model和mode，以独立分支+worktree+build隔离候选。candidate_id与Git commit共同标识一次冻结实现；分支名可以移动，不能单独作为证据。主工作区保留基线源码/固定binary/runtime及封存结果，候选不直接覆盖。每轮不复制一套C++类，不在共享源码添加逐轮if开关，也不让候选修改评估器/参考/评分门限。新算法或契约变化才新建入口。晋级后仍保留最初baseline和历次champion，另用有版本的选择记录指定当前推荐，不改旧行。

## 目录与单一事实来源

- experiments/optimization/bri/baseline-v1.json：不可变基线快照，原结果路径/大小/SHA与算法/任务参数/精度/周期/质量/工具链版本。
- experiments/optimization/bri/inner-NNNN/：设计、冻结candidate/contract、run证据、失败原因、汇总及差异。
- experiments/optimization/bri/ledger.jsonl：仅追加、前向hash链的候选/运行事实；不手工改旧记录，不以Agent自报分数为门禁。
- experiments/optimization/bri/comparison.csv：从台账确定性生成的视图，可重建；不能反向改台账。
- results/optimization/bri/inner-NNNN/<run_id>/：大trace/输入/输出/事件/详细流水；manifest记录原始字节和SHA。结果目录不覆盖。

一行run记录一个候选×场景，不用均值掩盖失败。固定保存family/algorithm/candidate/baseline/parent、change_kind、hypothesis、source/binary/evaluator/reference/config/input/contract/scorer版本或hash；Nr/Nt/RHS/batch/24核、K、λ/SNR、信道/种子、初值/预条件/谱界/阻尼/精度/舍入/输出恢复和计时边界；实际Asim周期、指令数/逐流水/逐核统计路径、MMSE误差分位数与最大值、同算法位一致、BER/SER/EVM和置信门禁、阶段墙钟/峰值内存、失败/跳过/不支持状态。缺项显式null，score在未完成晋级时为null。

## 比较与约束

只有共同task/input/config/precision/measurement的comparison_identity相同，且适用正确性门禁通过，才计算speedup=baseline_cycles/candidate_cycles；验证器版本差异必须记录，并有重验/兼容证据。历史周期不是统一榜。Asim失败/性能拒绝则数值NOT_RUN，不借改K或门限补救；全部失败尝试也追加台账。

纯指令优化冻结全部算法参数/K，要求和同算法独立参考及基线位一致，再检查MMSE/QPSK；无需为不变算法重新选K。若改变界/预条件/参数，独立新校准→冻结K→全新留出，单列parameters子类，不在旧H590上选参。每轮只提出有限假设，首轮只有state rotation一个。

新候选必须先真实Asim/事件/性能预筛后数值；旧已验收Asim可以作为明确封存的对照，不能声称重新运行或新增独立H。硬件认证外置，M3冷构建/完整晋级及无人值守仍待。
