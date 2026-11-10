# 受监督日常重建流程：稳定性验收与使用

2026-11-09最新：隐式FP16候选的实际Asim/重建已接通，新CDL576batch全部同算法FP16位一致，但20dB独立留出1/192样本质量失败，不能晋级。只阻尼β导致稳定域缩小已独立复核；下一步联动设计α/β并重新独立校准/新留出。240测试、七显式回归、82文件新runtime、性能/流水/耗时与负例见[完整报告](../../experiments/bri_implicit/2026-11-09-v1/README.md)。以下旧日期进展是历史证据，不能覆盖最新失败。

2026-10-30（Asia/Hong_Kong）。默认修正NS、瑞利64×16、8 RHS、batch96/24核，支持FP64结构与已声明FP16存储/FP32算术模型。**该受限范围已完成日常稳态验收**；全部旧算子、硬件G4/成本校准、正式通信/CDL-B、干净环境和无人值守控制器仍未完成，score=null。

## 1. 可以直接使用的入口

从工程根目录执行：

```bash
.venv/bin/python -m verification.reconstruction.daily
```

自动在`results/daily_reconstruction/`生成香港时区时间＋UUID目录，输出JSON中给出run_dir。默认是FP16数值模型；不代表原生910B精度或吞吐。

```bash
# FP64结构检查
.venv/bin/python -m verification.reconstruction.daily --profile structure_f64_real_block

# 固定算法参数下，与同一个固定runtime产生且已完整通过的基线预筛
.venv/bin/python -m verification.reconstruction.daily \
  --baseline results/daily_reconstruction/2026-10-30-accepted-entry-v3/result.json \
  --output results/daily_reconstruction/<new-run-id>
```

已存在的输出目录直接拒绝，不覆盖。当前固定binary只验收ns_corrected_v1；不是新候选搜索入口。新增算法/更换binary、后端或验证器需要独立工作区、fresh gate campaign、发布新的runtime配置，不能在旧包上改文件。

当前入口适用于有人监督、同一服务器串行调用。不自动重试、不跨天搜索，不提供恶意C++沙箱或候选晋级评分。

## 2. 运行顺序与退出状态

```mermaid
flowchart LR
 A[固定runtime及代码依赖校验] --> B[独立cwd与TMPDIR]
 B --> C[真实Asim]
 C --> D[制品摘要与实际指令绑定]
 D --> E[Core事件及DMA逐fragment检查]
 E --> F[冻结基线性能预筛]
 F --> G[三路径数值与位比较]
 G --> H[复核runtime身份并封存]
```

任一阶段失败都保留本次新目录与result.json。运行失败/事件缺漏/不竞争/离谱加速不执行数值。numeric_status=NOT_RUN不能解释为数值正确。低K可重放一致但任务质量不足。只有STRUCTURE_CHECKED或MODELED_NUMERIC_CHECKED返回0；其他状态不产生性能分。

## 3. 实际Core事件与保守模型

新增core-rw-v4流，实际记录：Core发射通过时read_begin、真实完成处理时read_done及fill后的write_visible；buffer真实flush/reset；每个DMA request/response的真实owner/request_id/fragment/地址/bytes/core/buffer。

计算读区间保守覆盖issue至实际完成；DMA源按request时刻快照，目标每fragment response提交。检查器按source views和实际请求地址恢复对应字节范围，不只看最后响应。LOAD旧响应/计算访问不得越过buffer reset；STORE请求快照后允许GM ack异步，但输出必须与输入及其他batch互斥。

CSV前四列保持兼容，新增request_id/op_id/fragment和实际compute issue_cycle/commit_cycle。Core流与CSV逐请求/操作绑定，防止只检查数量而漏掉时间变化。实际pipeline区间与发射/完成周期分别保存，不能把CubeWait当作Cube有效计算。

证书标签为MODEL_RW_CHECKED_CONSERVATIVE，是声明的同频SystolicWS/source-view模型检查；不是NPU真实读取微时序或完整硬件G4。check_runtime还必须与静态成本绑定、数值及独立参考共同使用，单独通过不证明算法正确。

初版明确拒绝：tail/padding DMA、不完整64byte请求、DMA局部偏移/转置/共轭、跨tile累加、未知融合和非WS时钟。预算为Core流256MiB/1,000,000事件，超限UNSUPPORTED。

## 4. 稳定性实际结果

[受信清单](../../experiments/runtime_rw/2026-10-30-v4-frozen/run_manifest.json)与[逐场景结果](../../experiments/runtime_rw/2026-10-30-v4-frozen/gates.json)。

| 场景 | 实际结果 |
|---|---|
| serialized正例 | 144周期，公共数值重建与独立矩阵参考一致 |
| independent正例 | 109周期，Cube与Vector实际计算区间重叠2周期，数值一致 |
| 删除同步导致后写先完成 | WAW拒绝 |
| 计算读保护内覆盖输入 | 保守WAR拒绝 |
| NS FP16模型，24核/96batch，含重复输入 | 7,950周期，实际事件及逐位数值通过 |
| NS FP64，24核/96batch | 14,298周期，实际事件及三路径通过 |
| 同输入重复 | 周期、输出文件、Core事件流字节完全相同 |
| 真实Asim后故意截断事件文件 | 普通运行器拒绝并归档，数值NOT_RUN |
| 不竞争预筛 | 跳过数值，NOT_RUN |
| 未支持tail任务 | Asim前明确UNSUPPORTED |

两种小正例的最终输出契约不同，109与144不作为同任务优化加速比。FP64和FP16模型周期也不混排。

另有8类离线实际制品损坏负例全部拒绝：缺完成、重复响应、错误buffer/fragment、响应时间、发射时间、缺reset及tail payload。明确为制品破坏测试，不报告模拟过的错误候选性能。

最终121项单元测试通过，覆盖byte interval索引与独立逐字节oracle、快速hazard与朴素检查的随机对照、早到fragment、epoch、位精度、runtime漂移、基线封存及manifest变化竞态。

v3完整批次曾因FP64 DMA描述重复携带全请求列表，事件流614MiB超过预算而拒绝。v4改为逐fragment地址，仍保留完整绑定，FP64降至94MiB；没有放宽资源限额或修改算子来使其通过。失败批次原位保留。

## 5. 固定版本、依赖与封存

`orchestrator/codex_runtime.json`固定已验收的Simulator、libramulator、HBM/互连配置、系统库哈希、生产验证器源码及Python/NumPy版本。动态库从归档lib选择，实际ldd解析已记录。每次运行使用独立cwd/TMPDIR，不依赖build作为工作目录。

baseline只接受同runtime/op/K/binary，必须有完整、无重复的输入/参考/输出/实际指令/事件/receipt等制品；最终顶层状态/周期/比较身份与evaluation_result.json及真实asim.json一致，不能靠改result.json伪造基线。runtime JSON从第一次读取冻结SHA，封存前复核manifest、固定文件和binary receipt；发生变化记RUN_OR_ARCHIVE_FAILED，不把旧运行标成新runtime。

默认入口已在独立工作目录实际验证P2/P4、基线复用及P8；运行时依赖复制并固定，不等于全新机器的G6复现。源码和系统库变化会拒绝旧runtime；应新建验证版本，不更新旧哈希绕过门禁。

Raw：`results/runtime_rw/2026-10-30-v4-frozen/`与对应daily目录；Git只保存摘要、路径/大小/SHA-256和固定runtime元数据。旧v1/v3探索资料、旧算子数据及用户未跟踪资料均保留。

## 6. 校验与后续

```bash
python3 scripts/check_reconstruction_archive.py experiments/runtime_rw/2026-10-30-v4-frozen/run_manifest.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s verification/tests -v
```

下一步是受监督候选协调器与工作区隔离策略，并在相同TaskContract下接入其他独立候选；继续补混合精度转换成本、真实通信suite/CDL-B及干净复现。更复杂DMA/融合/异构时钟需要新事件模型与正负例后才开放。论文最终稿仍由完整门禁控制，本报告只声明上述受限范围可用。

2026-10-30范围调整：硬件级时序认证由后续上板承担，不设为软件接入前置。本轮增加其他算子的完整语义扫描和只读阶段重建，并发布新固定runtime；原v4 runtime及回归原件保留，默认配置指向新包。见[阶段结果](LEGACY_SEMANTIC_STAGES.md)。日常入口仍只白名单已通过NS；已验收基线必须来自同runtime，不可跨新旧包复用。

2026-11-06：固定runtime新增已验收Cholesky直接解，与NS复用同一核心；`daily --operator cholesky_explicit_v1 --iterations 1`可用。首次运行/同runtime基线复用及NS默认回归均实际通过。完整算法参数绑定，参考不再强制X0；新旧包基线不能混用。见RECONSTRUCTION_GENERALITY.md。

2026-11-06后续：固定日常包新增LDL（--iterations 1）和BRI（--iterations 192），四次新包日常/基线验收通过，旧包继续归档。所有K仍需质量门禁；K1已实际质量拒绝。见LDL_BRI_MAPPING.md。

2026-11-08：新固定包白名单七新入口，新增cholesky_block_explicit_v1、ldl_block_explicit_v1、cholesky_merge_explicit_v1（均--iterations 1）。三次日常及Merge基线复用通过，重复字节一致。Scalar成本探针不进入白名单；旧Merge仍拒绝。见MERGE_AND_REMAINING_MAPPING.md。


2026-11-10固定HB更新：新固定runtime为orchestrator/codex_runtime_stable_hb_fp16_v1.json；旧implicit_v1固定清单不回写。高K事件/流水诊断造成显著内存压力，门禁后释放trace/stats/events；通用流式解析仍待做。默认日常入口仍瑞利。 证据：[固定HB原生报告](../../experiments/fixed_hb_native/2026-11-09-v2/README.md)。


2026-11-10 M1完成更新：九入口封存实际Asim制品全量回归保持统计字节、事件报告、完整输出位一致；HB K174/batch96事件检查及重建峰值1.88GiB，249测试。新runtime codex_runtime_memory_v1.json（91文件），旧90文件pin保留且配合原Git版本复现；不升级旧算子CDL质量状态。下一优先任务M2新校准的更紧谱界/复用/流水；冷构建跨环境复现、评分晋级与持久协调器仍待。详见[本轮报告](../../experiments/memory/2026-11-10-v1/README.md)。
