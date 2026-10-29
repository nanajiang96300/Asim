# 通用Asim程序适配设计

2026-10-29，基础设施扩展；不修改src/inverse。设计先于代码，依据FRAMEWORK_PLAN与op-flow对应步骤实际执行。

## 2. 公式推导

支持实数整块LOAD/STORE、GEMM C=A@B、ADD/SUB/MUL逐元素。结构模式float64独立于硬件精度校准，无性能评分资格。仅precision=8的显式模型配置可驱动该接口；不声称Ascend原生FP64成本已校准。

DRAM/SPAD/ACCUM各region按64B请求对齐并分配不相交地址；有效元素仍为shape。GEMM m/k/n由相同region shape推导；Vector compute_size按现有Core的字节吞吐规则传元素数×8。SUB数值身份保留、计时映射ADD延迟。COPY当前不进入Asim支持范围。

数值spec作为Instruction成员由通用构建器赋值；Model向调度器返回Operation时，基础设施从实际Tile/Instruction导出操作、地址、尺寸及同一对象携带的spec。通用Operation提供region/contract header，ops从实际对象生成。保守barrier等待前一目标可用，不声称异步G4已经验证。

旧算子没有spec/header，导出原始指令并明确UNSUPPORTED；不从FormulaLogger补语义。runner只消费自己运行Asim生成的新导出，检查trace、实际对象与推导字段一致，再性能预筛，再重放。生产评估隔离与完整事件认证后置；本版仅受监督可信本地runner。

计划门禁：单元测试；独立工作区-j1构建；真实通用GEMM/逐元素正例；漏STORE/漏LOAD/错误算术/低报size/差性能/异常周期等负例；旧NS batch1运行并验证UNSUPPORTED；Agent独立只读审查。结果与源码、配置、二进制及产物SHA绑定；score=null。

## 24核多batch扩展设计（2026-10-29）

目标：24核心、batch24/96，每batch一个独立Tile，初始core=b%24。simple scheduler首层按既有规则偏移1；实际核/buffer取自Core::issue的dispatch记录验证，不把生成时core当真实核。batch通过独立DRAM跨度隔离，本地地址由核/buffer共同区分。默认96，batch1仅为诊断兼容入口。

通用Model重复同一低层程序，ID加batch前缀保证全局唯一；数值spec仍来自实际Instruction。导出核对每batch完整操作、DRAM偏移、同步、配置与实际dispatch/完成事件，缺任一batch或跨batch别名即拒绝。结构顺序重放每batch独立，不宣称由dispatch自动证明全异步G4。

矩阵例：复瑞利Nr16/Nt8/Nrhs8，实数块A/b为16×16，所有求解GEMM对齐16³。给定A=HᴴH+λI、b=Hᴴy、X0=I/||A||2、C=2I，构造固定K12的T=A@X,R=C-T,X=X@R，最后X@b并STORE，不做额外X²。已有算子不修改，此为通用基础设施的测试程序。三路径：实际Instruction重放、独立复数同参数NS、complex128 solve。每batch误差及残差独立门禁，符号判决仅作小样本一致性诊断。

正例：batch24和96的24核运行，同输入96的单核周期对照；负例：删batch、混用batch数据、错误DRAM偏移、错误dispatch/core/buffer、错误算术、性能预筛不执行数值。独立worktree增量-j1构建，完整归档版本/配置/输入/实际程序/输出/周期/检查器hash。score仍null，真实FP16、物理异步事件、通信统计与权限隔离后置。
