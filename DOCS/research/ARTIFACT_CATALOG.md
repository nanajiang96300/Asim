# 原始资料、制品清单与保留规则

2026-11-09当前：负责人已授权清理无用数据、保留最新/关键证据。已完成[29份旧trace无损压缩归档](../../experiments/storage_cleanup/2026-11-09-lossless-v1/README.md)，删除经验证的冗余未压缩副本，释放约5.61GB；原大小/SHA及gzip映射、真实恢复测试完整保留。当前82文件运行器、最新BRI/关键失败样本与所有输入/输出/参考不变。下文较早“不清理/未授权”是历史范围，不覆盖本次明确授权；旧manifest原样保留，使用前按台账显式恢复。

日期：2026-10-28（Asia/Hong_Kong）；盘点源码基点 `7cf9c5088ccc030ee3bf478468660b3cdf46cee3`。本轮所有原件原位保留，没有移动、删除或复制到外部。

## 1. 已生成清单

[快照目录](../../experiments/inventory/2026-10-28-baseline/) 包含：

- `files.jsonl`：9,140条记录，path、bytes、SHA-256、tracked、类型、分类、mtime（非实验时间）。
- `archive_map.jsonl`：211项原始/历史资料的源路径、bytes、SHA-256、retain_in_place、目标null、未请求复制、未授权清理。
- `environment.json`：HEAD、分支、工具版本、资源、范围与明确排除项。
- `manifest.json`：前三份清单本身的大小与SHA-256。manifest本体由Git内容地址保存，不做自引用。

| 范围 | 文件数 | 总bytes | 当前处理 |
|---|---:|---:|---|
| results/ | 99 | 62,981,525 | 10个run包及说明，保留原位；大部分被Git忽略 |
| DOCS/zhongqi/ | 54 | 2,015,490 | 3份Markdown、PDF及中期生成目录；未跟踪PDF/projects不批量加入Git |
| projects/ | 40 | 265,556 | 既有已跟踪架构演示制品，不重写 |
| scripts/47_trae/ | 18 | 2,226,283 | 已跟踪模型/图与源文件，保留现状 |
| 以上历史/原始合计 | 211 | 67,488,854 | 清单已落地，未执行搬迁 |
| 选定build文件 | 3 | 11,621,923 | Simulator、Simulator_test、CMakeCache；只记录摘要，不认证来源 |

其他8,926项源码/文档/配置合计67,742,499 bytes；含已跟踪.claude目录的8,339项依赖文件。`extern/**`、`.git/**`、`.venv/**`、其他忽略构建依赖不递归hash；不将盘点数称为整台服务器文件数。源码清单只纳入已有跟踪文件与明确指定的原始目录，排除并行生成的新审计子目录和清单自身。

大文件示例：LDL v2 run002 trace约11.26MB，Cholesky v2 run001/002 trace各约8.38MB，多份formula约5.35MB。它们适合制品存储，不必为追踪证据把完整trace强行加入Git。

## 2. Git和原始存储分工

Git跟踪源码、配置、当前文档、状态卡、JSON/JSONL小台账、模板、校验脚本与清单。本轮新增清单约2.74MB，主要是逐文件hash元数据，不包含大trace/模型副本。历史已跟踪的PDF/PPT/模型不在本轮清理或改写。

新run使用 `results/codex/<campaign>/<candidate>/<run>/`（目录约定，尚无执行器自动创建），每次独立log/trace/input/output/build配置快照。长期副本存储位置由实际挂载/对象存储配置决定，不能以一个未配置URI声称完成备份。

关键小摘要写入 `experiments/`，不可只留在被忽略的results目录。保留真实失败和非法摘要的原字节；解析结果另写台账，不修改原件使其合法。临时文件不得成为唯一证据。

## 3. 如需迁移，先形成复制事务

当前映射的action一律retain_in_place，destination=null，copy_status=not_requested；目前没有需要立即搬迁的数据。将来更改某项时：

1. 新建迁移manifest，引用源快照SHA，指定真实目标目录/URI、文件名、bytes、SHA-256、所有者、访问方式与保留期。推荐目标组织为 `<artifact-root>/sha256/<前2位>/<完整SHA>/<basename>`；这是命名模板，不是已存在的远端。
2. 在独立目标目录复制，禁止覆盖已有不同内容；记录复制命令与时间。
3. 对目标重新计算完整SHA-256及大小，逐项比对。源/目标一致后才标 `copied_verified`，不能以文件名或大小相同代替hash。
4. 从目标独立读取并做一次代表性复现，确认台账链接可访问。
5. 原件清理是单独决定；本轮没有清理授权，不因复制完成自动删除源。

仓库不存密码、私钥、登录令牌或带凭据URL。工具只记录指定工程文件和非敏感环境，不导出完整环境变量。生成大制品前估算磁盘预算；空间不足记INFRA_ERROR，保留已有证据。

## 4. 可重复检查

```bash
# 新快照只可写入新目录，拒绝覆盖
python3 scripts/research_inventory.py create experiments/inventory/<new-id>
# 文档尚未修改时可全量核对快照
python3 scripts/research_inventory.py check experiments/inventory/<new-id>
# 当前研究文档已更新，核对历史原件未变
python3 scripts/research_inventory.py check experiments/inventory/2026-10-28-baseline --raw-only
```

缺文件、hash变化、未支持文件类型或清单自身损坏返回非零。全量核对发现本轮正常的研究文档更新也会报变化，应通过Git差异解释，不能覆盖旧快照。原始资料核对必须保持全通过。

现存10个运行包的配置、摘要解析状态、版本、制品摘要另见 [legacy_runs.json](../../experiments/history/legacy_runs.json)，便于按算法和run追溯；它与本清单是不同视图，必须交叉一致。

## 2026-10-29七入口重建更新

[七入口重建报告](LEGACY_RECONSTRUCTION_REPORT.md)记录最新实际覆盖：真实Instruction附加views、逐元素初始化/部分写、DMA batch范围、完整输出与事件覆盖检查已落地；七入口正确性未全通过，score=null。独立同算法参考、精度/G4/通信仍缺证，不能用周期正常结束替代验收。专用原件目录`results/reconstruction_campaigns/2026-10-29-legacy7-b96-v2-frozen/`，Git小清单在`experiments/reconstruction_campaigns/`；重跑命令与hash检查见报告，旧数据不移动。

## 2026-10-30统一重建更新

[本轮报告](NS_CORRECTED_VALIDATION.md)为最新覆盖：共同H/Y/λ及计时scope、共享view/初始化/算术核心、新NS三seed×96batch三路径结构通过。79测试通过，旧七入口保留原失败。FP16/G4/通信/G6及评分尚未验收。按[接入指南](ADDING_OPERATOR.md)增加候选与独立参考，不在重建核心添加算法名分支。prepared内核与完整任务不得直接比较；旧generic SUB→raw ADD现显式UNSUPPORTED。

2026-10-30进一步更新：FW3显式FP16/FP32模型与独立参考逐位一致，FW4合成事件hazard检查已实现；真实G4/硬件校准/通信/CDL-B/干净复现仍缺证。106测试通过，详见[论文报告入口与剩余门禁](PAPER_REPORT.md)。阶段报告不冒充最终验收。

2026-10-30稳定入口更新：实际Core读写/DMA事件已完成受限保守模型验收，固定runtime日常命令可用，121测试通过。详情及支持/拒绝边界见[稳定性报告](RUNTIME_STABILITY.md)。旧报告中的缺真实Core事件是当时状态；硬件G4/通信/全部旧算子/最终评分仍未通过，不能由该模型证书升级。

2026-10-30：旧七入口阶段重建原件放在`results/legacy_semantics/`，与此前reconstruction_campaigns、runtime_rw分开。冻结v2-raw/v2-stages、新runtime与NS双精度回归均有独立清单；首轮探索原位保留。小型Git证据入口为`experiments/legacy_semantics/2026-10-30-seven-b96-v1/`。详细流水CSV留原件，Git跟踪摘要及SHA-256。硬件板级认证由负责人另行执行。

2026-11-06：跨算法普适性原件独立放在results/generality/2026-11-06-cholesky-v1，包含首轮构建失败、固定runtime、五样本及日常三次、时间台账。Git摘要位于experiments/generality/2026-11-06-cholesky-v1。

2026-11-06后续：LDL/BRI原件专用results/operator_mapping/2026-11-06-ldl-bri-v1，Git清单experiments/operator_mapping/2026-11-06-ldl-bri-v1。严格stages-v2为最终阶段证据；旧stages、入口失败与重试留存。

2026-11-08：本轮原件独立results/remaining_mapping/2026-11-08-block-merge-v1，Git清单experiments/remaining_mapping/2026-11-08-block-merge-v1。旧Merge/NoBlock重跑与成本probe放forensics，新候选放campaign，daily另分目录；统一七入口周期与详细流水摘要不混用历史数据。

## 2026-11-08 清理授权与本次执行

负责人允许删除过时且不再使用的中间结果以节省空间。本次只清理可重新下载的pip下载/wheel缓存，未删除安装包、源码、运行器、研究trace或留出输入。逐文件路径/大小/SHA、清理原因和实际释放空间见[cleanup.json](../../experiments/channel_validation/2026-11-08-v1/cleanup.json)。

后续将未引用、可确定再生的临时缓存与已归档证据区分。历史失败实验仍有证明价值；不能只因被新版本替代就删除。删除研究制品前先列引用与替代证据，保留清理台账；如果历史manifest引用的原件被压缩/转存，需另立可逆映射并验hash，不改写旧manifest。

清理后复核：pip保留1份selfcheck元数据；清理前清单中672条文件路径已消失（pip报告669个缓存制品，计数单位不同）。原清理候选清单不覆盖，准确口径见[cleanup-verification.json](../../experiments/channel_validation/2026-11-08-v1/cleanup-verification.json)。

2026-11-09第二轮空间回收：[lossless-v2](../../experiments/storage_cleanup/2026-11-09-lossless-v2/README.md)另49个旧trace原件经完整解压/原SHA校验后保留gzip，释放3,467,128,578字节。两轮78个共9,075,966,566字节，最终逐gzip哈希核对通过，见combined_check.json。最新固定HB v1中断/v2重跑和旧BRI失败均不删。旧manifest不重写，原路径复现需显式restore。

2026-11-10：[原生固定HB不可变诊断去重](../../experiments/storage_cleanup/2026-11-10-native-dedup-v1/README.md)：16对v1/v2原生诊断逐SHA相同，保留全部路径并设只读，硬链接去重额外2,687,510,851字节。不是gzip退休，无需恢复，旧manifest仍直接可核验；历史制品不可原地修改。连同两轮gzip累计回收约11.76GB，当前最新数值/失败链完整保留。

2026-11-10 BRI内循环0001：原始制品results/optimization/bri/inner-0001/独立保存，101文件manifest见experiments/optimization/bri/inner-0001/run_manifest.json；固定基线、候选commit/patch、参数、周期/精度/流水、配对基线与审计纳入paper_archive/2026-11-10-bri-inner-v1/。默认基线未切换。为预留资源，新增lossless-v3/v4两次各64个旧诊断文件完整解压校验后保留gzip，分别释放2,142,233,853和926,970,294字节，恢复按对应storage_cleanup台账执行，旧manifest不改；最新基线和候选/输入/输出不删除。
