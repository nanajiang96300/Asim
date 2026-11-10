# M1：通用重建内存优化与九入口回归

2026-11-10（Asia/Hong_Kong）。设计见DOCS/research/M1_MEMORY_DESIGN.md。研究基点14b23db1，独立工作树codex/m1-memory-20261110；不修改任何C++算子、binary、K、输入、精度模型、质量阈值或实际Core观察覆盖。

## 改动与通用性

- 增量读取单operation JSON，不同时保留整行原始文本/副本；严格检查单operation、完整EOF、截断、重复键、非有限JSON常量。
- 完全相同的七字段物理view通过有上限的池共享只读描述；不共享数值张量。独立编辑应deepcopy单条指令，不能原地修改共享描述。
- trace覆盖检查直接迭代CSV；流水结果逐块写JSON，避免额外完整缩进字符串。时间区间tuple复用，输出明细消费旧分组时释放对象，检查完后释放流水统计再进入事件检查。
- 实际事件的固定三阶段时间戳采用有界64位整数数组，trace跨度采用slots对象，不删事件、batch或DMA。时间戳超容量明确拒绝，不截断。原所有时序/读写/布局门禁保留。
- 日常evaluate与QPSK Replayer共用读取器，没有添加算法名分支或按HB特殊跳过验证。

本轮并未将所有指令/统计完全磁盘化。目标是在声明范围稳定降低峰值，不承诺任意大K/矩阵的常数内存。

## 实测与边界

K174/batch96全量核验：原始Asim所有制品先按封存SHA检查，然后检查真实指令绑定、完整trace、流水、完整Core事件与96batch实际数值重建。峰值1,971,172KiB（约1.88GiB），≤2GiB目标；193.74秒，周期仍141017。流水JSON及逐指令CSV与原件字节相同，schedule报告精确相等，输出及封存同算法参考逐FP16位一致。

首轮/第二轮结果保留为超预算诊断：3,181,332KiB与2,251,492KiB，数值/统计均一致，但内存不算通过；最后版本没有通过降低K/batch/覆盖实现预算。首轮已经包含最初增量读取改动，不能把它当未经优化旧版的严格基线。原始Asim进程内存单列，本轮不优化模拟器本体。

[summary.json](summary.json)与[cases.csv](cases.csv)覆盖九入口：新HB K174、NS、Cholesky、LDL、BRI、Cholesky-block、LDL-block、Merge和旧隐式。其余八例复用各自封存真实Asim，不重新产生周期，不增加独立信道样本，不升级旧候选CDL质量状态。每例必须统计字节一致、事件报告相等、完整输出位一致且≤2GiB。249项测试包括新解析负例与原事件/质量/性能门禁测试。

新runtime下另跑真实Asim日常NS K16与HB K53，结果见[daily_ns.json](daily_ns.json)、[daily_hb.json](daily_hb.json)。完整独立算法参考和FP64 MMSE均由日常入口执行；它们为瑞利默认任务，不是新增CDL认证。

## 版本、复现与归档

新固定runtime：orchestrator/codex_runtime_memory_v1.json，旧stable-HB runtime JSON不覆盖。binary仍为原已验收C++版本；Python验证器身份改变，需要新pin。旧契约/旧pin应配合其冻结Git checkout复现，不能改旧SHA声称连续。

```bash
# 在M1提交对应checkout；使用新的输出目录，拒绝覆盖既有结果。
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
 experiments/memory/2026-11-10-v1/check.py \
 /home/nanajiang/Asim/results/fixed_hb_native/2026-11-09-v2/asim-lambda0.01 \
 /home/nanajiang/Asim/results/memory/<全新目录>
# 新Asim日常验证
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
 -m verification.reconstruction.daily --operator bri_stable_hb_fp16_v1 \
 --iterations 53 --output results/memory/<全新日常目录>
```

check.py读取已有完整Asim证据，不是绕过Asim的新候选入口。该脚本使用封存的同算法参考，不声称重算了全部独立算法；新日常入口补执行独立参考。纯Agent思考时间null，RSS为各独立Python进程ru_maxrss，单位KiB，各阶段为累计墙钟；不把脚本耗时当思考时间。

原始结果results/memory/2026-11-10-v1、v2、v3，日志清单见[run_manifest.json](run_manifest.json)。前两次重复生成且字节与封存原件相同的4份统计被按SHA核验、先写台账后删除，释放696,501,876字节；原统计、输入、数值输出及失败预算结果保留，见[清理台账](redundant-statistics-retirement.json)。无用户未跟踪文件、凭据或旧失败删除。

下一步M2：在全新校准集研究更紧谱界/复用及流水，保留2%与QPSK质量硬门禁。H590留出只回归不选K。冷构建跨环境复现、完整性能评分/晋级及持久协调器仍未完成；硬件时序由负责人上板另验。

QPSK Replayer集成检查：使用封存真实K16模板和既有H59001完整96batch重放，新读取器输出与旧CDL封存输出逐位一致，独立任务参考相等；墙钟13.58秒，参考3.00秒、实际重建6.81秒。见[结果](replayer_smoke.json)。复用旧输入，不增加独立留出数量。


M1解析器边界补丁v2：补充小数/指数顶层元数据跨chunk读取测试并修正误拒绝；实际九入口的instructions对象及header整体解码路径不变。新live pin为orchestrator/codex_runtime_memory_v2.json（91文件），v1清单保留。最终250测试及v2默认NS新Asim通过，五类关键文件与v1逐字节一致；HB/九入口完整内存实测属于v1原验收，未冒称在v2重跑全部。见experiments/memory/2026-11-10-v1/parser_v2_manifest.json。
