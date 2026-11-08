# Q3a：实际输出的瑞利通信检查

状态：**Q3A_TRIAL_GATES_PASSED**。代码提交51d9e8a4、4aa0d0d6、360d6723；187测试通过。
固定论文入口：[paper_archive](../../../DOCS/research/paper_archive/README.md)。
当前只支持工程试验下的未编码QPSK BER/原始软符号EVM；SE=UNSUPPORTED，score=null，完整通信认证false。

## 实施与约束

直接读取通过Asim/事件/重建门禁的真实outputs.npy，配对相同H/符号/噪声的高精度MMSE。
重新核验全部制品、模型/精度、fixture输入、参考及实际输出，不按算法名调用Python标准算法替代候选。
新质量结论只来自批准计划的全部holdout；不接受校准集、挑选部分留出job或重复H。
历史数据只诊断，包括逐组状态也不签发新质量通过。

契约冻结每H的ΔBER≤0.01、Δ原始EVM≤0.01，任一超限的联合概率≤5%，至少96个独立H。
family_alpha=5%，Clopper–Pearson区间按所有组/BER/EVM/联合事件/双尾作Bonferroni分配。
当前门限用于工程验收机制，不是最终论文或业务标准。区间试验单位是H，不是同H的bits。

## 七入口历史批次：通用计量诊断

复用此前七个**显式新入口**同一runtime的10个case，默认batch96/Nr64/Nt16/RHS8/λ0.1，共960batch；
未重跑Asim，没有改动原始结果。三个Block/Merge入口有seed42/123，其余仅seed42。

| 新入口 | 每组H数 | 候选BER | 配对MMSE BER |
|---|---:|---:|---:|
| NS K16 | 96 | 0.276693% | 0.276693% |
| Cholesky | 96 | 0.276693% | 0.276693% |
| LDL | 96 | 0.280762% | 0.276693% |
| BRI K192 | 96 | 0.272624% | 0.276693% |
| Cholesky-block | 192 | 0.266520% | 0.270589% |
| LDL-block | 192 | 0.266520% | 0.270589% |
| Cholesky-merge | 192 | 0.270589% | 0.270589% |

全部未观察到试验阈值超限，但状态均DIAGNOSTIC_ONLY。单比特差异不证明某算法更优。
此外四组只有96个H，即使零超限也不足在本次同时置信要求下证明概率≤5%。
这不是算子错误，而是样本/证据角色不足。旧legacy程序失败并未被此次新入口诊断覆盖。

## 新NS完整链路与留出验收

24核/batch96/Nr64/Nt16/RHS8，λ=0.01、0.1。校准seed8101、留出8102/8103。
逐操作FP16存储/FP32算术模型，冻结网格[1,8,10,12]选K10；没有宣称整数K9已排除或全局最优。
全部6个场景共576batch真实Asim/事件/重建通过，全部与独立参考FP16位一致；
完整H/Y→xhat scope每次5862周期，最大任务误差0.14053%。不和不同输入身份的旧K16作直接加速比。

通信仅使用4个留出case，共384个H/噪声场景；每个λ有192独立H，跨λ复用相同192个H。

| λ | 每组独立H | 候选/MMSE观测BER | 候选平均原始EVM | MMSE平均原始EVM | 联合超限数/置信上界 |
|---|---:|---:|---:|---:|---:|
| 0.01 | 192 | 0 / 0 | 0.114336390 | 0.114336563 | 0 / 2.81414% |
| 0.1 | 192 | 0.274658% / 0.274658% | 0.337215006 | 0.337214143 | 0 / 2.81414% |

两组及整体均QPSK_QUALITY_CHECKED。上界是“任一差值超限的信道概率”的界，不是BER上界。
零观测BER不等于真实BER为零。33.7% EVM包含噪声/MMSE收缩，不能和0.14%的MMSE任务误差混为一谈。

## 真实入口负例

- 必需指标添加SE：UNSUPPORTED，不以其他指标替代。
- 删除一个留出case：PREREQUISITE_REJECTED，避免只保留有利样本。
- 引用此前SKIPPED_NOT_COMPETITIVE的结果：PREREQUISITE_REJECTED，通信不能覆盖前置门禁。

187测试中本轮新增16项；包含独立SciPy置信区间交叉核验、有限样本、联合超限、缩放/相位失真、
非有限/形状、数据篡改、重复信道、校准/留出/历史角色以及遗漏覆盖。

## 目录与复现

代码/命令：[communication/README](../../../verification/communication/README.md)。
先以ns-screen-contract-v2.json选K，再使用既有handoff/execution入口；通信契约为ns-heldout-contract.json。
历史诊断契约为historical-seven-contract.json。每次使用新输出目录并核对契约SHA-256。

Git中的*-v2目录保存最终小型通信结果和契约；summary.json为汇总，run_manifest.json绑定149个原件/旧清单。
大型trace仍在results/communication/2026-11-08-v1/ns-execution/，本轮原件合计约368MB，未复制旧5GB制品。
早期ns-screen（7001/7002三档）仅探索预算；实际计划使用ns-screen-v2（8101/8102/8103两档）。
早期historical-seven/ns-heldout结果保留，最终以historical-seven-v2/ns-heldout-v2为准；
后者增加联合超限置信门禁并确保历史逐组也为DIAGNOSTIC_ONLY，不覆盖先前原件。

后续Q3b：实际程序的均衡响应/系数探针与SE定义；其他算法的新留出集；高阶QAM/BLER及CDL-B。
尚未完成SE、完整通信晋级、硬件认证或Agent搜索收益对照。
