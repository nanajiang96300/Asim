# 固定HB原生v1：前两档独立动态复核

日期：2026-11-09。结论：**FIRST_TWO_NATIVE_TEMPLATES_INDEPENDENTLY_CHECKED**。本报告只认证下述已完成范围；v1第三档没有完整result，不能计为通过；完整新CDL留出尚未执行。不代表候选全部验收或晋级，score应保持null。

审查版本：原生工作树提交1754ea63，冻结契约`experiments/fixed_hb_native/2026-11-09-v1/contract.json`。本次检查期间83份源码SHA及4份binary/config/library/DRAM依赖SHA与该契约一致。之后主Agent为资源原因另立v2（工作树bd1e520b），此处v1结果不冒充v2结果。

## 已完成的独立核查

未编译、未启动Asim，未加载巨大rw_events，也未将整份Instruction JSON载入内存。采用增量JSON解析仅取batch0完整实际指令序列，其他核查只读小型inputs/channel/references/outputs/result/receipt/config/model/dispatch制品。

| 原生v1模板 | λ1 / K16 | λ.1 / K53 |
|---|---:|---:|
| 实际Asim周期（receipt/result一致） | 18409 | 47121 |
| 独立重算96-H最大MMSE误差 | 0.6191060434% | 0.5461240318% |
| 完整96batch封存输出/独立参考FP16位比较 | 全部一致 | 全部一致 |
| 新鲜独立逐样本参考＋batch0实际指令重放 | 位一致 | 位一致 |
| 该batch完整执行指令数 | 1327 | 3029 |
| 设置完成与最后一轮实际回放内存见证 | 全部通过 | 全部通过 |

两份result均为MODELED_NUMERIC_CHECKED、numeric_status=EXECUTED、schedule_status=MODEL_RW_CHECKED_CONSERVATIVE、failed_batches为空。批次索引严格覆盖0..95；保存结果中的各batch precision_bit_equal为true且无cost_issues。

独立核查9项关键小制品的大小/SHA：asim.json、schedule.json、config.json、models.json、references.npz、outputs.npy、inputs.npz、channel.npz、dispatch.jsonl。巨大events/整份instructions的全文件hash由原评估器封存，本次没有再次读取全文件来复核，避免资源竞争；不能把本报告称为完整观察事件再审计。

从原始H/Y独立构造HᴴH+λI并求FP64 MMSE，对全部96batch重算误差，与封存task参考逐值一致，各batch误差与result差异小于1e−12。完整保存输出与独立算法output_block按FP16 uint16位比较一致。

## 实际指令及只读中间内存见证

针对每档batch0，从真实Instruction导出文件取出完整原序列，检查其opcode与typed semantic一致；Scalar源/目的都是单元素且compute_size=1；Vector工作量按rows·cols·4，分配/搬运size按FP16 rows·cols·2取64B请求数；GEMM compute_size按输出列数。实际1327/3029条与静态公式591+46K一致。

随后调用共同execute_batch执行该实际序列，期望值来自新运行的独立逐样本fixed-HB参考。只读capture放在coefficient.beta完成处与最后iteration.state写回处，不创建额外STORE、不向回放内存填参考值。

以下状态在两个时点均与独立参考或独立Scalar归约逐FP16位一致：

- α、β、lower、upper、energy_upper、P_upper；
- 完整expandedP[32,8]，由独立复点对角重复实/虚行与8列构造；
- energy_sum、energy_fourth_sum、P_square_sum、trace_sum，依据独立保存energies/P重新逐Scalar FP32运算→FP16写回计算。

所有见证从设置结束到最后一轮保持不变，没有系数或预条件器被迭代临时区覆盖。完整batch0输出也与独立参考逐FP16位一致；框架报告执行完全部指令，未出现缺语义/未初始化/边界错误。

batch0实际系数：

| λ | α | β | lower | upper |
|---|---:|---:|---:|---:|
| 1 | .2440185546875 | .52880859375 | .28662109375 | 11.4609375 |
| .1 | .20263671875 | .8232421875 | .03955078125 | 16.8125 |

这是实际Instruction数值重放的内存见证；Asim本身没有完整中间数值执行，不冒充上板读取的硬件数值。

## 源码独立性与闭包

C++真实生成固定系数，独立reference/stable_hb.py自行实现FP16运算及固定HB；native wrapper只包装其output/setup，不调用重放器。共同engine/source_views/binding/runtime没有新增HB求解分支。原生campaign绑定预算子selection SHA与K16/53/174、新H59001/59002及载荷59101/59102；精确3case/2seed/6frame/576batch覆盖和最终guard在发布成功前执行。

本次已核对契约83源码与4依赖的原版本SHA。仅延长新条目受信Asim超时和观察预算，不以修改精度参考、阈值或K消除数值差异。

## v1第三档与v2边界

主Agent告知：v1 λ.01/K174实际Asim成功141017周期，schedule通过并已写references；父评估进程在约10分钟时收到143/SIGTERM，未生成完整result；SIGTERM根因尚未确诊，不能据此断言OOM或特定超时机制。此为部分证据，不是第三档完整数值通过；本审查没有读取K174巨大JSON或假装完成其重建复核。

为降低峰值内存，另立v2仅在schedule检查完成后释放stats/trace/events等已归档大型诊断对象；C++、独立参考、阈值、K及新H590种子不变。v2需自己的冻结契约、实际运行结果与后续留出证据，不能覆盖v1失败/部分原件。新CDL质量待完成后独立续审。

本次恢复审查起点UTC2026-11-09 13:01:09；纯模型思考时间不可观测，记null。实际审查墙钟与结束时间见下方追加记录。

审查结束UTC：2026-11-09 13:07:06；可观测墙钟约357.5s；pure_model_thinking_seconds=null。

执行器补充：主Agent告知v2的nohup后台尝试被工具进程清理，未创建v2结果目录或实际运行Asim；随后改为前台PTY分阶段受监督执行，当前仅启动了该方式的v2真实Asim。本报告不声明后台持久控制器已跑通。
