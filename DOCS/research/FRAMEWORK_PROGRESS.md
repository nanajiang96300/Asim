# 框架实际进度与门禁证据

2026-10-30最新：统一重建核心与独立NS完整任务结构通过，79单元测试，三seed×96batch三路径及七旧入口回归。见[本轮报告](NS_CORRECTED_VALIDATION.md)。完整硬件精度/异步/通信/评分仍未验收。

阶段历史（2026-10-29，Asia/Hong_Kong）：七个注册入口已补源码views并执行真实Asim重建，冻结批68、兼容回归69单元测试通过；算子正确性未全通过，性能晋级BLOCKED。详见 [七入口重建报告](LEGACY_RECONSTRUCTION_REPORT.md)。下文21/29/45测试与未补语义描述为早期阶段，不能当成当前覆盖上限。

## 早期通用整块原型（阶段历史）

新增通用 `semantic_program_test` mode，不修改src/inverse或legacy算子。受信通用Model从schema生成真实Instruction，将算术spec附着于对象；基础设施在Model向调度器交付Operation时导出真实队列。runner从导出对象恢复程序，核对实际opcode、地址、shape、size、compute_size、DMA请求及辅助barrier，再核对compute/barrier完成事件集合/单元/合法区间。

执行顺序：静态完整性 → 独立Asim子进程 → 真实导出/事件核对 → 冻结性能预筛 → 数值重放 → 独立参考比较。失败/不竞争/异常速度保持数值NOT_RUN、score=null。新目录拒绝覆盖，失败也生成result.json与原始证据摘要。runner只接收本次真实进程产物，不接受Agent提供的receipt作为可信证明；目前是有人监督的本地可信运行器，尚无OS权限隔离。

FP64实数整块原型支持LOAD/STORE、GEMM、ADD/SUB/MUL；Asim配置明确precision=8。region最多32、操作最多256、每空间声明区域最多64KiB；barrier额外最多16KiB，二者地址不交叠。工作量与时序字段由views推导。该模型成本不代表硬件原生FP64校准；COPY/view/stride/partial-write/归约等未覆盖项不自动支持。

## 实际执行的门禁

独立工作区 `/home/nanajiang/asim-worktrees/fw-asim-gate-20261029`，build和日志独立，-j1构建Simulator成功。首次configure网络获取spdlog停滞后停止自己的进程树，改用既有本地依赖源码的独立副本离线配置。该构建不是G6完整干净依赖复现。构建源码commit和二进制/config/cache/日志摘要记录于 [run_manifest.json](../../experiments/framework/2026-10-29-asim-gates/run_manifest.json)。

29项单元测试通过，包含辅助barrier删除/篡改、低报size、区域总量、错误memory方向、缺/坏trace与错误单元、缺输入、未知算术、非法精度、别名和性能预筛不进入数值等。独立审查首轮指出四类边界问题，全部修复，复审在本报告限定范围内通过。

| 实际子进程门禁 | Asim结果 | 数值/门禁结果 |
|---|---|---|
| 瑞利H派生Gram，实数块表示 | 137周期，成功 | 重建与独立复数Gram参考一致，STRUCTURE_CHECKED |
| 同程序不具竞争力阈值 | 137周期，成功 | SKIPPED_NOT_COMPETITIVE；数值NOT_RUN |
| 同程序异常加速阈值 | 137周期，成功 | QUARANTINED_IMPLAUSIBLE_SPEEDUP；数值NOT_RUN |
| 真实程序插入重复相加 | 成功 | NUMERIC_FAILED，相对误差约1 |
| 真实程序漏LOAD | 触发100,000周期看门狗 | SKIPPED_ASIM_FAILED；数值NOT_RUN |
| 旧NS，4×4、batch1、默认K16 | 1,708周期，102条真实指令 | UNSUPPORTED：无typed数值header/views；未进行数值重放 |
| 旧Cholesky NoBlock v2，M4/U2、batch1 | 232周期，28条真实指令 | UNSUPPORTED：无typed数值header/views；未进行数值重放 |

表中性能阈值是预先定义的门禁探针，用来验证两条拒绝分支，不是实际优化收益。所有周期来自不同task/shape/精度，不构成算子排行榜。旧算子未补语义、未修正算法；不能拿FormulaLogger或标准算法算出的值冒充真实指令重放。

本轮`GATES_PASSED`表示这7个案例得到预期结果，其中负例成功拒绝。它不表示旧算子通过、G4通过、通信质量达标或FW0–FW2已经完整验收。所有score仍为null。

## 重跑命令

从工程根目录调用；output必须是不存在的新目录，依赖当前构建工作区仍保留。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s verification/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.asim.gates \
  --binary /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --cwd /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build \
  --output results/framework/<new-run-id>
```

原始logs、trace、实际指令、输入/参考/重建快照在 `results/framework/asim-gates-v2-frozen/`；Git保存小型run_manifest、完整逐制品SHA-256及源码hash，原件不搬迁。代码路径：src/semantic/、verification/asim/、verification/shadow/。

## 早期原型的未完成项（view部分已有下述新进展）

FW0完整五契约schema与数据集策略；FW1/FW2的typed view/stride/offset、部分初始化、独立同算法参考和复数/逐步精度；FW4异步读完成/写可见事件及hazard；权限隔离；通信质量/G6/新评分器。瑞利当前只为结构fixture提供复数H，原型通过实数块GEMM验证Gram，尚非完整MMSE均衡器。CDL-B仍显式UNSUPPORTED，后续需明确模型及归一化，不能静默回退瑞利。

下一项：先把全矩阵region扩展为明确的typed views及部分初始化掩码，为真实算子补数值语义提供基础；再选择一个实际算法做独立接入，不根据旧周期表宣布冠军。

冻结runner提交7bc2dd49后，整批7个真实门禁再次串行运行，结果一致。最终清单的源码与79个原始制品hash逐项复核匹配；首次探针目录v1仍保留原样。

2026-10-29多核更新：默认24核、batch96（24的倍数），Cube16³；24/96全batch的真实Asim→预筛→三路径数值闭环已运行，45项单元测试通过。计时为给定A/b/X0/C内核，仍非FP16/G4/通信统计验收。详见 [24核验证报告](ASCEND24_VALIDATION.md)。

## 七入口结构诊断更新

新增view/部分初始化/实际DMA槽位与完整输出检查，以及逐core/unit/name事件多重集合验证。全部七入口/每入口96batch已跑，未补算法答案，136个Instruction构造点原字段保持一致。五入口为显式歧义，BRI越界，NS输入绑定/数值失败，均未晋级。原始/详细流水资料专用目录与后续任务见 [七入口重建报告](LEGACY_RECONSTRUCTION_REPORT.md)。独立同算法参考、FP16、G4和通信门禁仍未完成。

## 2026-10-30统一重建更新

[本轮报告](NS_CORRECTED_VALIDATION.md)为最新覆盖：共同H/Y/λ及计时scope、共享view/初始化/算术核心、新NS三seed×96batch三路径结构通过。79测试通过，旧七入口保留原失败。FP16/G4/通信/G6及评分尚未验收。按[接入指南](ADDING_OPERATOR.md)增加候选与独立参考，不在重建核心添加算法名分支。prepared内核与完整任务不得直接比较；旧generic SUB→raw ADD现显式UNSUPPORTED。

2026-10-30进一步更新：FW3显式FP16/FP32模型与独立参考逐位一致，FW4合成事件hazard检查已实现；真实G4/硬件校准/通信/CDL-B/干净复现仍缺证。106测试通过，详见[论文报告入口与剩余门禁](PAPER_REPORT.md)。阶段报告不冒充最终验收。

2026-10-30稳定入口更新：实际Core读写/DMA事件已完成受限保守模型验收，固定runtime日常命令可用，121测试通过。详情及支持/拒绝边界见[稳定性报告](RUNTIME_STABILITY.md)。旧报告中的缺真实Core事件是当时状态；硬件G4/通信/全部旧算子/最终评分仍未通过，不能由该模型证书升级。

2026-10-30：七入口全指令覆盖与实际阶段重建扩展完成；714,912条描述静态扫描，首错前阶段值独立对比，130单元测试，未开放旧算子PASS/评分。见LEGACY_SEMANTIC_STAGES.md。硬件级认证由后续上板，不列为本轮开发前置。

2026-11-06：共同核心不变接入Cholesky直接解；参数适配和通用evidence移除NS专属假设。5样本432batch、日常3次288batch、4项离线拒绝及基线参数负例通过，135单元测试；工具/重放/Agent工作耗时分开归档，纯思考不可观测。见RECONSTRUCTION_GENERALITY.md。

2026-11-06后续：LDL/BRI显式映射通过主批次（864正例batch及24低K质量拒绝）、672严格阶段见证及384日常batch；144测试、独立复审通过，四核心不变。见LDL_BRI_MAPPING.md。
