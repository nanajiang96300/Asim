# 独立只读审计记录

2026-11-12，op-flow规定的独立审计，由原生子Agent nn_primitive_review执行；只读，不并行编译或Asim。主Agent实际落地修复、构建、验证和封存。

源码终审HEAD16c28f7a：未发现阻断四个受信图族交付的问题。确认构造器稳定root id、先注册输入Tensor再连边、输出producer指向实际Operation；node未知属性/前向依赖/重复张量/输出形状预检；双bank及sync容量；新Softmax完整行成本和DIV计费、GELU工作量、写回与事件/阶段门禁。

审计者另逐个核对suite-v2的36个存档SHA、内层result SHA、96batch阶段/最终位一致、binary及周期与summary相符。该审计发生在重复中间存档按清单退休前；最终suite-v3由独立只读脚本再次核对，见archive-audit.json。

提出并已处理的边界：最终skip补丁改变verification/tests指纹，不能将新源码与旧suite-v2混封；停止并保存六类v2部分运行，重新跑suite-v3/六类v3，不改旧指纹。六类旧quiet配置回归不能和新NN拼统一排行榜；node属性校验不代表任意JSON/ONNX；固定权重工程一致性不代表训练/无线/硬件认证，score保持null。

剩余加固建议：将独立archive/result与summary的一致性校核内建到freeze。当前freeze依赖可信summary写入方，本轮以额外逐包审计防止误封；不宣称该加固已经自动实现。任意ONNX、多Operation和硬件级时序不属于本次验收范围。
