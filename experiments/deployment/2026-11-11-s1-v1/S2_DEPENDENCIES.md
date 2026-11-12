# S1检查中确认的S2前置边界

本轮S1不改重建核心和旧pin语义；以下是下一阶段必须显式实现/验收的基础设施任务，不宣称已解决。

1. `verification/reconstruction/runner.py`的comparison digest将完整hardware_config纳入，其中包括实际dram_config_path/icnt_config_path。同内容文件换位置将改变identity。S1比较周期/门禁输出，不用跨路径identity冒充同身份性能晋级。S2应以规范化物理配置+外部配置文件内容SHA定义比较身份，路径作为locator另记；修改YAML内容、精度、时钟、预算、任务/输入等仍须拒绝。不能简单删掉配置校验。
2. `verification/reconstruction/daily.py:baseline`要求基线binary SHA及旧runtime spec相同，这是原固定日常复现约束。不能原样用于新binary的候选晋级，也不能放宽旧baseline函数来伪造历史通过。S2建立独立baseline/candidate身份及允许的source diff关系，记录双方binary/runtime/evaluator，统一任务/输入/硬件/精度/计时后比较；受信代码/门限改动仍拒绝，旧daily行为保留。
3. 新clone没有历史results下的旧pin目标。应针对新本地build生成新运行时冻结清单，源/锁/二进制/依赖/解释器/config内容均绑定，不能改旧pin或复制服务器认证。S1便捷smoke创建正确性基线但默认清原始制品，不能用其summary绕过baseline要求；S2重新生成带完整原件的目标基线。
4. context只准备监督任务。候选路径保护、调用/时间/磁盘预算、超时/失败恢复、patch审查与模型调用记录需要确定性执行器与负例；不能靠提示词宣称强隔离。每个客户端至少真实产生一个受限候选，走完整冻版本/构建/Asim/重建/参考/质量/适用评分归档。

验收最少包含：等内容异路径可比较；物理配置或输入内容变化拒绝；不同合法candidate binary可绑定到冻结baseline；未授权source/evaluator/reference/threshold变更拒绝；旧daily回归不变；缺原始证据/未知语义/性能拒绝均无数值启动与无分数。此处是计划与源码证据，不是已实现能力。
