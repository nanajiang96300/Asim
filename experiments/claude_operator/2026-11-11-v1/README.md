# Claude Code + DeepSeek 新算子端到端验收

运行ID始于2026-11-11，完成于2026-11-12 UTC。状态：新算子受监督闭环与原六类新二进制回归全部通过。本报告针对一个受监督的新任务示例，不宣称S2/S3或无人值守协调器全部完成。

## 实际生成与反馈

Claude Code2.1.272使用用户现有DeepSeek API配置。主代码生成调用由CLI报告deepseek-v4-pro，另有内部flash调用元数据；不根据CLI的provider/cost标签推定第三方实际账单。无密钥、API环境或用户配置入库。

负责人先冻结独立Python原型、输入输出/精度/计时、可写范围与门禁。模型实际生成`AffineGemmModel.h/.cc`及设计文档三个文件，未由主Agent手写C++冒充生成；`authorship.json`验证原始响应→两轮模型补丁→当前源码逐字节相同。主Agent另外编写可信注册/任务/参考集成。源码版本：作者初稿b8c06994、审计修复b5db3a36、编译反馈修复b1ae63f0；执行目录最终e8e8bc7b。共享源码与候选的156项src/CMake/lock文件相同，见source_match.json。

保留失败：v1/v2各600秒预算超时，插件/只读探索及输出限额使之未生成源码；v3请求Write但bare实际不提供，259.58秒由协调器停止。改为bare、无文件工具、限定JSON三文件提议协议后v4成功，墙钟278.55秒。独立审计后v5补batch预算与辅助元数据，8.98秒；真实编译发现Tensor维度要求可写vector，原始错误见logs/claude-affine-build-v2.log，交回DeepSeek修复v6，13.07秒，之后编译通过。初轮源审计未发现该编译问题，不能以审计代替构建。

成功生成及两轮修复合计调用墙钟约300.61秒，不含前述失败、人工协调、编译或仿真。纯模型思考时间没有可靠接口，保持null；reported thinking_tokens=0不证明第三方未内部思考。原始提议、用量、时间及逐轮补丁见generation/。未发布内部思考流。

## 正例

任务为实矩阵仿射乘加`Z=half(half(A@B)+C)`。24核、batch96、Cube16×16×16；FP16存储、逐k FP32乘加、中间GEMM先写回FP16再ADD写回，不融合、不增加Asim混合精度指令。统一计时含三个输入DMA、GEMM、ADD、同步及输出DMA，输入生成不计。种子42/123/20261111，各包含零A、零B、零C及符号零边界。

| m/k/n | 场景数 | batch合计 | 每场周期 | 最大FP64任务相对误差 |
|---|---:|---:|---:|---:|
| 16/16/16 | 3 | 288 | 367 | 0.033170% |
| 32/32/32 | 3 | 288 | 685 | 0.034890% |
| 32/16/48 | 3 | 288 | 681 | 0.035295% |

精确值以cases.csv为准。九场景864batch全部真实Asim完成、trace工作量检查、Core保守读写模型通过，实际指令数值重放与独立FP16算法逐位一致，FP64任务误差全部小于2%。该示例不使用信道，不冒称MMSE/通信通过；只实际测试表中尺寸和batch96。runner的原MMSE路径与本任务复用同一指令重放核心；本轮没有另造affine专用解释器。

三条路径：①从实际指令与typed views重放；②不导入重放引擎的独立Python FP16算法；③从原始double输入计算FP64任务参考。FormulaLogger只留诊断。每场实际源指令1056条：288MOVIN、96GEMM_PRELOAD、96ADD、96MOVOUT、480barrier。各场完整逐指令流水CSV及保留所有core/unit统计的pipeline_summary.json在cases/；汇总pipelines.csv。流水duration_sum跨核/事件可重叠，不能相加当makespan；未独立观测MTE1。

九场景加32方阵相同输入重复基线共19.13秒（不含编译/代码生成），重复运行亦通过全部数值/事件检查。可信评分器在核对所有原件SHA及负责人冻结result SHA后计算685/685=1.0；它仅演示此任务评分串接，research_score=null、promotion=false、optimization_claimed=false，没有优化收益或M3晋级声明。

## 原六类回归与框架测试

恢复的正确新二进制经当前共享runner再次执行原六类：Cholesky28006、LDL18697、Cholesky-block21338、LDL-block21665、NS K16为7950、BRI K192为23047周期，与S1旧二进制一致。各96batch、Rayleigh seed42、Nr64/Nt16/RHS8、λ0.1；同算法FP16逐位一致及2% FP64 MMSE门禁全部通过。此回归是新增真实执行，不复用S1结果冒称新二进制通过；同96个H跨算法配对复用，不是576独立信道。细节与删除前原件SHA见six-regression.json。

当前可信Python测试254项中252通过、2项缺可选依赖明确跳过；另8项bootstrap/提议导入测试通过，原始日志在logs/。三项实际评分证据破坏检查与真实错误指令负例另列，不冒称单位测试。未改旧六类的数学实现。

## 必须拒绝的真实负例

独立测试分支`codex/claude-affine-negative-v1` commit a9d7cee2，人为将ADD的真实src地址及typed src同时由C改为B，FormulaLogger仍写C。此变异不是DeepSeek提交的正确候选。真实重新编译/运行后Asim仍完成685周期、Core读写检查仍通过；96batch全部REPLAY_MISMATCH，评分器拒绝且score=null。表明正确的公式声明与看似正常的周期不能替代重建。见negative-summary.json和cases/negative-actual/。

恢复正确源码重新编译后binary SHA与正例完全相同，见restored-runtime.json。负例二进制未另保留，按冻结commit可重建；日志、全部实际指令/输出原件及SHA保留。另三项实际证据拒绝检查通过：错误的负责人result SHA、缺少batch结果、输出文件被改字节；原正例未改，见score-negative-tests.json。

## 复现与可追溯性

操作步骤/原理：[CLAUDE_NEW_OPERATOR.md](../../../DOCS/research/CLAUDE_NEW_OPERATOR.md)；便携构建：[BUILDING.md](../../../BUILDING.md)。`scripts/claude_codegen.py`可真实调用CLI或显式离线验证封存响应；后者receipt.agent_executed=false，不能冒称再调用模型。`scripts/affine_acceptance.py`完成九场景与配对评分，`--single`不签发九场景通过。

原件根目录：`results/claude_operator_demo/2026-11-11-v1/`（原研究工作区）。raw_artifacts.json记录全路径/大小/SHA；Git的cases/result.json是原件副本，所列输入、trace等大原件在raw根目录，不声称都存在Git副本目录。原六类回归默认哈希归档后删除本次raw，摘要保留明确raw_retained=false。只清理本次新建build的PCH/对象缓存255,260,072字节，清单与恢复方法见build-cache-cleanup.json；未动原研究资料或旧manifest。

## 边界与后续

本次是实际CLI生成、审计、反馈修复、冻版本、构建、Asim、事件/性能预筛、重建、独立参考、任务内评分和归档的受监督示例。可信路径由白名单导入器保护，模型无文件工具；不是任意C++的OS沙箱。旧daily/runtime pin保留，不用新实验改写旧冻结证据。

S2仍需规范化跨路径物理配置身份、跨合法candidate binary的正式基线绑定、新平台完整runtime冻结/预算控制、Codex同任务真实生成对照；完整研究晋级仍须相应通信与干净环境复现。本affine任务不是无线算法，无需伪造通信指标。详见deployment报告的S2_DEPENDENCIES.md。未在本轮更改旧六类算法或BRI默认基线。
