# FW4真实Core事件与稳定运行验收设计

2026-10-30。仅增加观察事件，不改算子算术或原调度。独立build -j1，仿真串行，已有原件保留。

## 声明的模型边界

Core实际can_issue_compute成功、进入执行路径时记录compute_read_begin；实际完成处理、Sram.fill之前记录read_done，之后记录write_visible。区间是保守读保护窗口，不声称NPU真实逐拍读取。DMA在真实request创建/response接收处记录owner op、fragment、request_id、DRAM/local地址、bytes、buffer。DMA源取值定义为请求发出时快照，目标按响应提交；每个fragment在自身响应处提交对应物理字节，检查器不得只取最后响应掩盖早到覆盖；LOAD整块在全部fragment响应后才完整可读。STORE源在request时快照，GM目标按fragment ack提交。该声明独立于实机校准，标签MODEL_RW_CHECKED_CONSERVATIVE。

Core::issue的真实SPAD/ACCUM flush记录buffer_reset（新batch/layer身份）。事件携带全局递增序号作为同cycle先后顺序；仅支持同频SystolicWS时钟。不能从现有pipeline CSV倒推出缺失事件。

## 绑定及检查

实际Instruction唯一ID→实际dispatch core/buffer→真实Core事件与DMA请求响应一一核对；核对opcode/基址/请求地址/bytes/fragment集合/有序时间/finish上界及pipeline区间。CSV增补request_id/op_id/fragment与compute issue/commit周期，各事件精确绑定，防止仅修改observer时间。未定义tail/padding请求显式UNSUPPORTED。拒绝未知/漏/重复/跨batch/跨buffer事件。将源视图精确展开成物理字节范围（含row/col/region stride），同一batch做RAW/WAR/WAW，初始仅Model输入DRAM可读；buffer reset防旧epoch遗留访问：跨epoch的LOAD响应/计算完成必须在reset前；STORE ack可晚于reset，但其源读取请求必须先完成，且GM输出不得与其他batch冲突。

初版只对受信source-views且一tile对应一batch的程序认证。跨tile累加、跨频时钟、未知融合、尾部多STORE、未定义读时序显式UNSUPPORTED。其他legacy/通用路径不因新增日志就自动升级。

## 稳定性验收

真实小探针：serialized正例；independent正例Cube/Vector实际重叠；missing_barrier实际WAW负例；overwrite_input保守WAR负例。typed同源程序直接调用公共数值执行器，不根据probe名称求解。归档实际stream及反例首错。

生产NS默认24core/batch96、F64与FP16模型均检查stream后才数值；重复同输入两次比较周期、逐batch输出和事件规范化摘要。删除/重复/篡改事件、错fragment/core/buffer与缺stream均拒绝且数字NOT_RUN。保留性能预筛NOT_RUN规则和失败产物。若实际NS发现调度问题则归档，不调整模型放行。

完整硬件精度、通信/CDL-B、G6不因该项通过而升级；论文最终稿继续由完整门禁控制。

资源上限：Core流256MiB/1,000,000事件，超限UNSUPPORTED；Asim每次120秒/1,000,000周期，构建-j1与仿真串行。DMA只支持零偏移、无变换、完整64byte请求；fragment局部payload偏移由同源连续views与实际DRAM请求地址映射，actual local_addr仍是SRAM entry base，不声称硬件逐元素地址/转换成本已建模。
