# BRI内循环首轮：状态槽轮换与不可变基线

2026-11-10。状态INNER_LOOP_SOFTWARE_CANDIDATE_CHECKED。同Model/mode，独立candidate分支88acdaed；主工作区/默认runtime保持原基线，M3未晋级，score=null。

[完整报告](../../../../experiments/optimization/bri/inner-0001/README.md)记录三档真实周期18409→17488、47121→44128、141017→131248，周期减少5.003%/6.352%/6.928%；公式/K/精度不变，Rayleigh、新192-H CDL三SNR576batch与原基线配对重放均位一致，QPSK门禁通过。不是全部历史BRI/全算子统一排行榜或硬件认证。

[manifest](manifest.json)固定基线、候选、追加台账快照、CSV、参数、精度分布、流水、审计、控制和图。原始大制品仍在results/optimization/bri/inner-0001/；同一算法的后续优化以candidate_id+commit记录，不每轮复制算子类。下一轮独立预声明假设；改谱界/参数必须新校准/留出，当前H610只能回归。
