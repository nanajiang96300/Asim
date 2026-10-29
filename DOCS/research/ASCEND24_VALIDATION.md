# 24核、batch24/96 的Asim与数值闭环验证

日期：2026-10-29（Asia/Hong_Kong）。本轮已完成通用框架的多核多batch求解内核结构验证；并非完整框架、实机FP16或通信统计验收。源版本与全部证据见 [run_manifest.json](../../experiments/framework/2026-10-29-ascend24/run_manifest.json)。

## 配置与计时范围

默认工作负载改为24模拟核、batch96，生产batch采用24的倍数；batch1保留为诊断。每batch一个独立Tile，96时每核4个，24时每核1个。batch决定核间工作分配，GEMM的M/K/N分别决定Cube分块；不是batch为96就自然保证每次GEMM高效。

首批信道仍为瑞利：Nr64/Nt16/Nrhs8、96个独立信道，固定种子20261029；E|H_ij|²=1/Nr、QPSK单位符号功率、10dB噪声方差λ=0.1，y=Hs+n。原始H/y/symbols及A/b/任务参考以NPZ保存。A=HᴴH+λI、b=Hᴴy，参考x=solve(A,b)。另有独立复数NS同算法路径，X0=I/||A||₂、固定K12，最终X@b。

实际程序由通用builder生成实数块操作：A为32×32、b为32×16；GEMM为32×32×32或32×32×16，均对齐16×16×16 Cube。它不是修改旧NewtonSchulzBaselineOp。数值为FP64结构模式，Asim的precision=8；原生910B FP16乘累存与硬件成本尚未校准。

计时仅覆盖**已给A、b、X0、C=2I的求逆迭代及应用内核**。H/y准备、Gram/正则化和谱范数初值生成未计时，不能把此周期称为H/y到x_hat全流程时延。这里“闭环”指生成程序→真实Asim→预筛→实际指令重建→独立算法/任务参考→逐batch诊断→归档。

## 实际绑定与覆盖

- DRAM按每batch独立跨度分配；本地空间按core与双buffer区分。Instruction实际导出带batch/计划core，Core::issue导出实际core、buffer、时间与ID列表。
- simple scheduler首层保留既有core偏移1规则；校验实际dispatch与计划映射，不把生成时core直接当执行核。
- 每batch恢复完整操作和辅助barrier，核对DRAM偏移、opcode/shape/size/compute_size及全局ID；全batch compute/barrier完成事件核对唯一性、单元和时间，DMA按core核对数量。
- 数值仅在真实Asim、绑定检查及性能预筛通过后执行。所有batch分别比较两条参考和残差，失败batch列表保留，不用均值遮盖。
- DMA数量核对没有fragment唯一ID和读完成/写可见信息，dispatch记录也不等于全部异步hazard证明；G4仍为NOT_RUN。

## 冻结版本真实门禁

| 案例 | Asim总周期 | 结果 |
|---|---:|---|
| 24核、batch24 | 2,865 | 三路径通过，每核1 Tile |
| 24核、batch96 | 6,606 | 三路径通过，每核4 Tile |
| 1核、相同batch96/程序/输入 | 110,456 | 三路径通过，作为同输入周期对照 |
| batch96性能不具竞争力探针 | 同程序正常仿真 | SKIPPED_NOT_COMPETITIVE；数值NOT_RUN |
| batch96异常加速阈值探针 | 同程序正常仿真 | QUARANTINED_IMPLAUSIBLE_SPEEDUP；数值NOT_RUN |
| 仅最后batch输入b混用首batch数据 | 正常仿真 | NUMERIC_FAILED，仅failed_batches=[95] |

在固定FP64结构模型下，96输入的24核/单核周期比为16.72×；不是实机加速比，也不是任何历史算子优化收益。各输入的最大任务误差约6.56e-16，最大同算法误差约6.74e-16；最大归一化残差约1.79e-16。96批共12,288个符号，候选与参考逐符号判决无差异，两者都观察到81个符号错误；这是固定小样本诊断，不代表统计质量门禁通过。

trace观察到24个核同时存在Cube指令，不将其当成真实PE有效利用率。batch96共核对7,488个compute/barrier完成事件、49,152个DMA完成事件计数；每核4 Tile。

另运行了**未修改的旧NS**：M=U=32、batch96、24核，4,647周期、9,792条实际指令，各核4 Tile；因缺typed header/views仍明确UNSUPPORTED，数值NOT_RUN、score=null。它与上表新通用测试程序参数、语义、精度不同，不能比周期判胜负。

首次探索中曾误调用尚未完成增量构建的旧二进制；其忽略batch参数。该探针原件保留，恢复器因execution策略不匹配拒绝，未作为有效基线。此后所有校准运行都验证实际batch、dispatch和事件，不能仅凭Finished文本采纳周期。

## 功能门禁与独立审查

45项单元测试通过，覆盖跨batch地址、缺末batch、错core/buffer、重复dispatch、ID前缀、非法batch、DMA计数、dispatch时间、NaN策略失败归档及巨大有限参考的稳定误差。还包括“候选与参考错误总数相同、但逐符号决策不同”的负例，必须拒绝decision parity。

审查发现两个失败归档边界：范数溢出与非法NaN策略元数据。已通过先缩放再算误差、显式finite门禁及JSON安全诊断修复；独立复审在本报告限定范围通过。

C++在独立工作区-j1增量构建；本地依赖缓存继续复用，不声称G6干净依赖重建。冻结runner后运行v3套件，106个原始制品和138个源码文件hash全部核对。还从归档H/y重建A/b及solve参考逐项比较，匹配保存的输入/参考。旧原件、历史报告、算子数学源码及未跟踪中期材料保持不变。

## 重跑

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s verification/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.asim.batch_gates \
  --binary /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --cwd /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build \
  --output results/framework/<new-unique-run-id>
```

output必须不存在；评估先Asim再数值。原始制品位于results/framework/ascend24-v3-frozen/，旧NS制品位于results/framework/legacy-ns-b96-c24-v1/；Git只保存清单和摘要，不提交约57MB原始批量输出。

下一步仍需完整typed views/部分初始化、硬件精度与异步事件、评估权限隔离和正式质量/评分门禁。FW0–FW2保持in_progress，所有score=null。CDL-B尚未接入，后续需独立模型/归一化审计，不静默回退瑞利。
