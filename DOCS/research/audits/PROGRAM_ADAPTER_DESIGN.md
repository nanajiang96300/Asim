# 通用Asim程序适配设计

2026-10-29，基础设施扩展；不修改src/inverse。设计先于代码，依据FRAMEWORK_PLAN与op-flow对应步骤实际执行。

## 2. 公式推导

支持实数整块LOAD/STORE、GEMM C=A@B、ADD/SUB/MUL逐元素。结构模式float64独立于硬件精度校准，无性能评分资格。仅precision=8的显式模型配置可驱动该接口；不声称Ascend原生FP64成本已校准。

DRAM/SPAD/ACCUM各region按64B请求对齐并分配不相交地址；有效元素仍为shape。GEMM m/k/n由相同region shape推导；Vector compute_size按现有Core的字节吞吐规则传元素数×8。SUB数值身份保留、计时映射ADD延迟。COPY当前不进入Asim支持范围。

数值spec作为Instruction成员由通用构建器赋值；Model向调度器返回Operation时，基础设施从实际Tile/Instruction导出操作、地址、尺寸及同一对象携带的spec。通用Operation提供region/contract header，ops从实际对象生成。保守barrier等待前一目标可用，不声称异步G4已经验证。

旧算子没有spec/header，导出原始指令并明确UNSUPPORTED；不从FormulaLogger补语义。runner只消费自己运行Asim生成的新导出，检查trace、实际对象与推导字段一致，再性能预筛，再重放。生产评估隔离与完整事件认证后置；本版仅受监督可信本地runner。

计划门禁：单元测试；独立工作区-j1构建；真实通用GEMM/逐元素正例；漏STORE/漏LOAD/错误算术/低报size/差性能/异常周期等负例；旧NS batch1运行并验证UNSUPPORTED；Agent独立只读审查。结果与源码、配置、二进制及产物SHA绑定；score=null。
