# 隐式FP16候选：真实Asim接入、重建性能与新CDL留出

2026-11-09（Asia/Hong_Kong）。状态：**ACTUAL_REPLAY_CHECKED_CDLB_QUALITY_FAILED**。

开发和实际指令映射完成；重建一致性通过，**CDL-B质量未全部通过，候选不可晋级，score=null**。本轮保留冻结K/阻尼/门限，没有在失败留出上调参后自验。此前192个CDL数值留出通过是原样本域证据，不是普遍稳定性保证。

## 实现与泛式

新增`src/inverse/bri_implicit/BRIImplicitModel.{h,cc}`、mode `bri_implicit_test`、示例配置与受信`bri_implicit_fp16_v1`适配器。设计先于实现提交，独立工作区`/home/nanajiang/asim-worktrees/bri-implicit-fp16-20261109/`串行-j1完整构建成功；二进制源版本055338a1，冻结评估器源版本5033cdc5，契约提交f942e180。

输入H/Y/λ→完整xhat；隐式Hᴴ(y−Hx)−λx、点Jacobi、单侧复状态、β×0.9和FP16补偿，K13/33/76。FP16张量/系数存储、既有固定顺序FP32临时算术，无新增持久FP32张量、无仿真器混合精度扩展。初始化、P展开、标量系数、补偿/前态复制、输出恢复与DMA全计时。

真实Instruction构造helper同步派生raw尺寸/工作量与typed views；实际导出序列通过共同engine/source_views/binding执行，不按算法名调用标准求解器。上述三个核心及numeric精度模型未改；runtime只增加有界流式读取，并在完整验证compute描述后压缩其重复字段。旧256MiB/100万事件默认预算保留，新适配器单独声明1GiB/250万事件。流式不等于恒定内存。

独立审查冻结前修正Y完整尺寸、P/out首次局部写的物理分配不足，以及对角GEMM→MUL的signed-zero差异（增加实际ADD(+0)并计费）。不需新增算术语义。源码设计见[算子文档](../../../DOCS/operators/bri_implicit_fp16_v1.md)。

## 实际Asim与重建性能

24核/batch96，Nr64/Nt16/RHS8，Cube16³；以下三份真实Asim模板使用瑞利seed57001。Asim本身不计算完整中间数值；后续CDL重放显式绑定外层输入域，不重写原template_task，也不虚构每个信道的新周期。

| λ / Es/N0 | K | 模板周期 | Asim墙钟s | 制品/事件/调度检查s | 独立参考s | 指令重建与比较s |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 0dB | 13 | 16,079 | 13.65 | 20.48 | 2.85 | 5.95 |
| .1 / 10dB | 33 | 33,999 | 28.26 | 39.91 | 5.80 | 10.35 |
| .01 / 20dB | 76 | 72,527 | 62.71 | 84.01 | 12.18 | 19.75 |

288个模板batch全部与独立同算法最终FP16输出逐位一致并过MMSE门限；最大瑞利误差分别0.8086%/0.8402%/0.5371%。三档源评估总墙钟约308.35s，通道生成另15.72s，CDL重放/参考等总墙钟123.70s。不同λ/K成本不是跨算法排名。重建约0.062/0.108/0.206s每矩阵，包括比较，不是纯算术内核时间。

主要软件开销是制品/事件/调度检查，不能把全部检查时间称为数值重建。实际峰值内存见原始`/usr/bin/time -v`日志。独立审查墙钟约404s静态、178.69s动态，可能与主工作重叠；纯模型思考时间不可观测，记null。硬件时序由负责人后续上板。

## 新CDL-B留出与失败

契约先冻结再生成H56001/H56002各96个，载荷56011/56012，全部沿用冻结K，无重试选点。真正Sionna CDL-B：3.5GHz、100ns、UT4×4/BS8×8、单极化omni、静态uplink、+1.2MHz频点、每H归一||H||F²=16。只覆盖该静态矩阵域，不是任意几何/移动/OFDM认证。三个SNR配对复用同192个H，不计成576独立信道。

| Es/N0 | K | 192 H最大MMSE相对误差 | 超过2%的H数 | 同算法FP16位一致 | 结果 |
|---|---:|---:|---:|---|---|
| 0dB | 13 | 0.87434% | 0 | 全部 | 数值/相对QPSK通过 |
| 10dB | 33 | 0.93760% | 0 | 全部 | 数值/相对QPSK通过 |
| 20dB | 76 | **5432.38%** | **1** | 全部 | **NUMERIC_QUALITY_FAILED** |

576个实际batch重放完整执行，294,912个完整实块FP16输出元素与独立算法参考逐位一致。失败样本是H56002/b95；其余191个20dB样本过2%，不抵销最坏样本硬门禁。独立报告器和独立Agent重新计算FP64 MMSE与147,456个bit判决，结果相符。未编码QPSK三档候选/MMSE误判bit分别13531/13523、8230/8231、4185/4061（每档49,152bit）。平均BER接近不能掩盖一个信道的严重失败；不是绝对业务BER、FEC/BLER或总体SE认证。

## 根因：只阻尼β缩小了稳定域

精确模态误差递推为pₖ₊₁(t)=(1+βₖ−αₖt)pₖ(t)−βₖpₖ₋₁(t)。冻结后只把β乘0.9，不能继续把原Chebyshev的包谱区间当成稳定保证。

独立重算坏点（使用FP16 H/λ/P形成系统的FP64谱诊断）：谱[0.00847195,7.63881236]完整位于名义[0.0025,8]；最后α=0.4921875、β=0.87158203125。固定系数二阶模态要求0<αt<2(1+β)，对应上界7.60515873，低于该最大特征值；末步根半径1.08440018。独立计算完整76步变系数误差多项式放大192.920838，进一步支持递推不稳定。

末步根不是单独的全程证明；完整多项式也不是逐操作FP16模型的替代。它们与实际失败/逐位一致证据共同定位算法稳定性问题。提高精度不会自动修复不稳定的递推；本轮不修改Asim、K或λ。

下一任务：联动设计α/β或重新设计带阻尼递推并复核稳定域；例如研究同时缩放方向和动量，但不能未经验证称其普遍稳定。再核查可实施保守谱界及FP16舍入余量。先独立校准选K，再冻结新候选，用全新CDL留出验收；本次坏点只作为回归/诊断，不能再当独立验收样本。

## 门禁与原七显式入口回归

240项单元测试通过。真实指令离线变异：缺语义、错误符号、缺反馈、错象限均拒绝；错误符号/反馈仅约0.321%算法差异，仍被一致性门禁拒绝，不能只看2%任务门限。零/负零RHS与独立参考位一致。变异不是故障二进制Asim实验。

同实现完整重复真实运行，指令/dispatch/trace/读写事件/输出字节相同。冻结minimum_speedup=1.01再运行，周期仍16,079，真实性能拒绝，numerics NOT_RUN且无numerics_started/references/outputs。错误SHA真实CLI在Asim前拒绝。详见[控制结果](controls.json)。

原七个**显式新入口**在同新二进制、瑞利seed57001/λ.1/batch96下全部回归，非旧legacy原始入口修复。与新隐式K33的comparison_identity完全一致：

| 入口 | K/占位 | 周期 |
|---|---:|---:|
| NS | 16 | 7,950 |
| Cholesky | 1 | 28,006 |
| LDL | 1 | 18,697 |
| 显式BRI | 192 | 23,047 |
| Cholesky Block | 1 | 21,338 |
| LDL Block | 1 | 21,665 |
| Cholesky Merge | 1 | 19,028 |
| 新隐式FP16 BRI | 33 | 33,999 |

这是同任务瑞利正确性基线成本对照，K不是各算法性能最优点。新隐式方案当前更慢，且CDL全域门禁失败，无优化收益或晋级分数声明。流水总驻留时间可能跨核/流水重叠，不能相加当总周期。

[周期/重建耗时CSV](cycles_and_replay.csv)、[流水汇总](pipeline_summary.csv)、[逐核流水汇总](core_pipeline_summary.csv)。每次真实run的`instruction_pipeline_detail.csv`保留全部指令名/周期分布，见原始制品清单。

## 固定入口、复现和归档

当前`orchestrator/codex_runtime.json`与版本化`codex_runtime_implicit_fp16_v1.json`固定82份文件，旧清单原样另存`codex_runtime_legacy7_20261108.json`。默认NS日常入口与新隐式K33日常入口均以新固定包实际运行通过。日常入口仅声明其瑞利软件模型范围；新隐式为实验候选，不代表CDL晋级。此前七入口F64覆盖继承历史，本次回归为F16。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.reconstruction.daily \
  --operator bri_implicit_fp16_v1 --iterations 33 --output results/daily_reconstruction/<fresh-id>
```

冻结campaign命令（新目录；dataset/asim均只运行一次，replay预期退出1并保留质量失败）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.reconstruction.implicit_campaign \
  --contract experiments/bri_implicit/2026-11-09-v1/contract.json \
  --sha256 3f261285425c07d38241ae0c47ed7aba91759d4c46a64a581b8cb8d1f29d7fe3 \
  --output results/bri_implicit/<fresh-id> --stage asim
# 同一目录再 --stage dataset，最后 --stage replay；失败不是归档脚本错误。
```

本轮14次真实Asim（3新模板、7回归、2控制、2固定日常），其中1次性能拒绝不执行数值。主正例/日常/回归共1248个batch，加CDL576个batch，共1824个完整数值batch槽位；不是1824个独立H。

原件固定`results/bri_implicit/2026-11-09-v1/`，输入/二进制/trace/流水/日志按路径、大小与SHA归档。[机器摘要](summary.json)、[独立审计](independent_audit.json)、[静态审查](static_audit.md)、[动态审查](dynamic_audit.md)、[稳定性诊断](stability_diagnostic.json)、[制品清单](run_manifest.json)。[工具链快照](toolchain.json)记录编译器、Python/NumPy及动态库；未执行全新OS/容器重装复现。纯字节相同的完成重复制品做硬链接去重约500MB，原路径/大小/SHA均保留，独有数据未清理；共享别名必须视为不可变，见[记录](identical_artifact_dedup.json)。

固定论文检索：[paper_archive](../../../DOCS/research/paper_archive/README.md)。规范与op-flow/op-dev/audit-operator技能已更新为实际Instruction、完整成本、强制独立数值门禁及阻尼稳定性检查；旧Claude脚本/历史报告未覆盖。

归档空间口径：冻结契约的3.5GB输出预算适用于campaign三阶段运行时；额外控制、七入口回归和两次日常验收另行执行，最终全档案299文件逻辑4,870,829,031B，去重后唯一inode内容4,370,765,895B。没有把完整扩展档案冒称为3.5GB内；复现单独三阶段不需要生成这些额外回归副本。辅助tools保存原运行/审计脚本；可移植评估入口使用上文带新输出目录的模块CLI。
