# 逐操作精度模型（2026-10-30）

本轮实现自定义、版本化的modeled_fp16_fp32_v1，不宣称等同真实910B。存储/LOAD/STORE/每条指令写回FP16，RNE，渐进下溢保留；标量/Vector输入先精确提升FP32，操作FP32，写回FP16。GEMM从FP16取数，乘积FP32逐k先乘后加（非融合），按k升序每次FP32舍入，最终FP16写回。未知融合/FTZ/精度配置拒绝；溢出或NaN失败。

实块展开与同源实际指令绑定；必须由Asim precision2实际地址/size/compute_size计时，不能拿precision8周期冒充。独立同算法参考按同实块表示和顺序重新实现，不调用重放核心；任务参考仍为高精度complex solve。

输入在实块编码后统一一次RNE量化，参考使用量化H/Y/RegI/Identity；任务oracle使用原始H/Y/λ。模型actual↔algorithm硬门禁为全部FP16位模式精确一致（含signed zero），相对误差1e-10仅诊断；task相对误差目标2%，实块四象限容差2%，仅为本轮明确工程目标，不是论文通信质量验收或硬件误差界。超标保留失败，不放宽门槛。

结构FP64与FP16模型属于不同profile，不直接混排。硬件校准、完整异步事件/G4、通信统计、干净复现仍未完成；论文最终报告门禁必须保持未完成状态。

计时中存储/request/分配按P=2，Vector FP32运算compute_size按4byte/元素，二者不得混用。隐式精度转换的独立硬件成本/原生Cube累加吞吐仍未校准，周期仅作诊断，FW3完整成本闭合不标done，不发分。
