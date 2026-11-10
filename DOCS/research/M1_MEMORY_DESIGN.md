# M1 通用重建大制品内存优化

2026-11-10，实施于独立worktree codex/m1-memory-20261110，基点14b23db1。不修改算子、binary、参数K、数值模型、输入或质量阈值。主工作区固定90文件runtime在验收前保持不变。

第一阶段消除重复文本/对象驻留：增量解析单个actual-operation JSON的instructions数组，普通dict/list对象与原语义不变；拒绝额外操作、尾随数据、截断、重复键和非有限JSON常量；通用trace覆盖检查直接迭代CSV；流水JSON用json.dump写入，避免完整缩进字符串；流水明细写完且TRACE_VALID后释放统计，再进行完整实际事件检查。QPSK Replayer复用相同读取器。

本阶段并未承诺指令对象完全按batch磁盘化，流水统计仍保留明细；若K174实际峰值仍超过2GiB，继续优化而非提高预算或减少覆盖。事件stream本已流式读取，但跨事件索引仍驻内存。未知语义/缺batch/截断trace必须继续拒绝。

验收先用已封存真实Asim的完整K174/batch96制品核SHA，再实际验证绑定、trace、流水、事件及全96batch重建。新统计JSON和逐指令CSV必须与旧文件字节一致，schedule精确相等、输出逐FP16位相等、原周期不变。此路径只是可信冻结Asim制品再验证，不声称再次运行Asim；同算法参考使用原已封存独立参考。随后九入口代表性回归、负例、必要新Asim日常入口，再发布新的runtime pin。纯思考时间null，脚本阶段墙钟与峰值RSS分别记录。

首轮测量（results/memory/2026-11-10-v1/hb174）完整数值/流水/事件均保持一致，但峰值3,181,332KiB，未达到2GiB；保留失败预算证据。第二轮增加：精确七字段物理view以有上限的池共享只读描述，深复制单条指令可生成可编辑副本；不合并数值张量或改变算术。运行时compute时间戳及trace跨度改为slot结构，保持原字段/检查；流水统计共用不可变区间tuple、用单opcode/歧义状态替代大量单元素set，覆盖完成后及时释放索引。原JSON/CSV统计字节一致为硬门禁。


最终M1验收：九入口封存制品全量回归均通过，HB K174/batch96峰值1,971,172KiB（1.88GiB），193.74秒；流水JSON/逐指令CSV字节一致、事件报告相等、完整输出逐FP16位一致。原生NS K16/HB K53新Asim日常入口通过且五类关键制品与旧版字节一致。249测试，live pin为codex_runtime_memory_v1.json（91文件）。详见experiments/memory/2026-11-10-v1/README.md。


M1解析器边界补丁v2：补充小数/指数顶层元数据跨chunk读取测试并修正误拒绝；实际九入口的instructions对象及header整体解码路径不变。新live pin为orchestrator/codex_runtime_memory_v2.json（91文件），v1清单保留。最终250测试及v2默认NS新Asim通过，五类关键文件与v1逐字节一致；HB/九入口完整内存实测属于v1原验收，未冒称在v2重跑全部。见experiments/memory/2026-11-10-v1/parser_v2_manifest.json。
