# Asim：Claude Code＋DeepSeek 使用手册

适用版本：2026-11-12 的共享核心及 Claude 客户端。先读本手册，再读 [构建详情](../BUILDING.md) 和 [新算子原理](research/CLAUDE_NEW_OPERATOR.md)。本系统是 NPU 周期模拟器、实际指令数值重建器和受监督 Agent 工作流；运行无需真实 NPU。默认24核、batch96、Cube16×16×16。WSL2建议使用Linux文件系统；本轮已实测Linux x86_64，WSL2尚未实机验收。

## 1. 包含什么、哪些已有实测

压缩包包含系统源码、六类求逆/均衡实现及变体、DeepSeek生成的affine示例、可信验证/参考、技能规范、配置、测试、固定依赖缓存及逐文件校验清单。不包含历史实验输入/输出、trace、研究结果、build、venv、Git历史或账户/密钥。小型合成测试fixture不是实验数据。

已有实测：两个新Linux检出冷构建；六类修正入口的Rayleigh Asim/重建/FP16参考；Claude Code＋DeepSeek真实新affine算子的生成、审计、编译反馈修复、九场景864batch位一致及实际错误输入负例拒绝。affine最大FP64任务误差0.0353%，门限2%。这不说明所有历史变体、CDL-B或完整性能晋级都通过。历史证据在Git研究分支，交付包主动不附结果；SOURCE_VERSION.json记录原始版本。

## 2. 解压、离线构建、第一轮检查

目标环境：Linux x86_64 / CPython3.12，建议Ubuntu24.04、GCC13、CMake3.28，系统需要git、build-essential、cmake、python3、python3-dev。工具由企业软件源/管理员安装；包内缓存不能代替这些系统工具。建议8GiB内存、至少4GiB空闲磁盘，2vCPU默认串行，首次编译约半小时，取决于机器。AI调用另需模型服务网络。

```bash
sha256sum -c asim-claude-guide-v2.tar.gz.sha256
tar -xzf asim-claude-guide-v2.tar.gz
cd asim-claude-guide-v2
python3 scripts/check_delivery.py
# 源码压缩包没有.git，创建本地导入版本；使用者先自行配置Git身份。
git init -b codex/asim-import
git add .
git commit -m "Import Asim Claude source delivery"
python3 scripts/bootstrap.py --cache offline_cache --offline --jobs 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/test_portable.py
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/portable_smoke.py \
  --operator all --output results/first-six
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/affine_acceptance.py \
  --binary build/bin/Simulator --config configs/ascend_910b_quiet.json \
  --cwd build --output results/first-affine
```

从Git克隆的版本已有.git，不要再初始化/伪造上游commit；从源码包导入的本地commit与SOURCE_VERSION中的上游commit含义不同。offline_cache已被.gitignore排除，不会加入本地导入提交。依赖缓存的固定大小/SHA不符直接拒绝。源码包无法用Git获取未随包的旧历史，研究历史需另外从官方仓库取得。

六类smoke应得到`SIX_CASES_CHECKED`，affine应得到`NINE_CASES_CHECKED`；逐场最终状态为`MODELED_NUMERIC_CHECKED`、`MODEL_RW_CHECKED_CONSERVATIVE`、failed_batches为空。264项数值测试在无可选SciPy/Torch/Sionna环境可明确跳过2项；不得把跳过当CDL已验收。默认smoke在记录SHA后删除本次大原件，要保留用`--keep-raw`并增加磁盘预算。每个output必须是新目录。

配置内两个外部文件路径相对`build`，所以保持`--cwd build`；若改工作目录，另生成含有效绝对路径的新配置，不能沿用服务器路径。包校验在初次解压时执行；改动源码后原manifest自然不再匹配，不可覆盖它来冒充未改包。

## 3. 在Claude Code中开始工作

Claude Code/DeepSeek由使用者在用户环境配置。包不安装CLI、不预置API地址/模型名/账号。先在终端检查：

```bash
claude --version
python3 scripts/agent_entry.py doctor
claude
```

doctor只查CLI版本，不会验证API或代码生成。给对话的第一条任务可直接使用：

> 读取AGENTS.md、CLAUDE.md、DOCS/CLAUDE_CODE_USER_GUIDE.md和DOCS/research/CLAUDE_NEW_OPERATOR.md。先确认pwd、hostname、git status和资源。用独立candidate分支/worktree/build完成一轮任务，编译与Asim串行。冻结公式、任务输入/输出、K、精度和计时边界；候选不能修改评估器、参考、门限或评分规则。先Asim及事件/性能检查，合格才数值重建；失败保留日志和版本，不用自报PASS替代证据。

Claude技能文件在`.claude/skills/`，重点op-flow、op-dev、audit-operator；它们是实施规范，实际执行日志与确定性脚本才是验收。其他平台须显式读取这些Markdown，不能声称已执行不可用slash命令。旧opt-round等流程是历史实现，其分数不替代当前正确性门禁。

角色分工：协调者冻结任务/预算/可信规则；编写Agent只改候选；独立审计检查数学及真实指令；可信脚本运行Asim、重建、参考比较与评分。当前需要监督，尚不是自动跨天恢复的控制器。

## 新增：通用小型Attention模型与基础单元

新NN入口共享重建器，支持独立Softmax、GELU、FC→GELU→FC和单头Attention＋MLP；使用真实指令，输出由数值重建得到，Asim本身仍是周期模拟器。未训练固定权重用于推理一致性测试，不代表任务准确率；也不是任意ONNX模型已验证。

```bash
# 四类默认形状快检，batch96；首次CPU较慢也不要改小batch混入同一周期表。
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/nn_acceptance.py --quick --output results/nn-quick
# 4类 × 3形状 × 3种子，共36组：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/nn_acceptance.py --output results/nn-suite
# 或只运行Attention默认单例，保留未压缩原件：
.venv/bin/python scripts/agent_entry.py evaluate -- --task nn --nn-kind attention \
  --operator nn_graph_v1 --iterations 1 --profile modeled_fp16_fp32_nn_v1 \
  --binary build/bin/Simulator --config configs/ascend_910b_nn_v1.json \
  --cwd build --output results/attention-single
```

quick通过为QUICK_CHECKED，完整通过为SUITE_CHECKED。任务专用1%门限＋同算法FP16逐位一致＋中间阶段检查同时满足才通过；Softmax/GELU公式/轴/精度未知会拒绝。默认Attention为L32/D32/FF64、约8416参数，含缩放Softmax、GELU、全连接和残差；模型拓扑与发指令内核分离，另一个MLP拓扑共用同样的基础单元。

默认使用新NN配置，spad/accum字段显式256KiB、每buffer128KiB；旧六类保留其历史配置，不据此形成跨任务统一排行榜。源指令/真实Core事件、每个非线性实际持续周期、最终与中间张量指纹均验收。归约与GELU是有明确数值/成本契约的宏指令，不虚构内部微指令流水。

完整suite每场在检查全量原件SHA后保存无损tar.gz再移除临时展开文件，summary中留存SHA/恢复说明；--keep-raw可保留展开原件。新输出目录不可覆盖。已有服务器NN pin只属于当时冻结检出；新机器先构建/验收，再以自己的完整suite、六类回归和binary通过scripts/freeze_nn_runtime.py生成新pin，不覆盖旧清单。nn_acceptance --runtime可执行自己的新pin。

新增基础语义需同改生成端/binding/公共数值实现/成本/负例，独立算法参考不导入重放实现。候选仍不能改可信参考或门限。当前自动claude_codegen三文件launcher仍专用于affine演练；NN新任务由协调者冻结Task/Graph，再在对话中按批准路径实现或优化，不宣称任意模型一键自动生成已完成。

## 4. 跑现有六类算子

| 家族 | 当前可信入口 | 首次smoke固定K |
|---|---|---:|
| Cholesky NoBlock | cholesky_explicit_v1 | 1 |
| LDL NoBlock | ldl_explicit_v1 | 1 |
| Cholesky Block | cholesky_block_explicit_v1 | 1 |
| LDL Block | ldl_block_explicit_v1 | 1 |
| Newton–Schulz | ns_corrected_v1 | 16 |
| Block-Richardson | bri_explicit_v1 | 192 |

例如只跑NS：`.venv/bin/python scripts/portable_smoke.py --operator ns --output results/ns-first`。这些是该Rayleigh样本的演示参数，不能直接用于CDL或其他尺寸的精度结论。HB新入口与旧BRI显式入口要区分；原始legacy/Merge实现不能因家族名相同自动升级。

要保留完整单例，可直接调用共同runner：

```bash
.venv/bin/python scripts/agent_entry.py evaluate -- \
  --operator ns_corrected_v1 --iterations 16 --seed 42 \
  --profile modeled_fp16_fp32_real_block_v1 \
  --binary build/bin/Simulator --config configs/ascend_910b_quiet.json \
  --cwd build --output results/ns-full
```

`result.json`是状态入口；`instructions.jsonl`是实际操作及typed views，`rw_events.jsonl`/`trace.csv`是实际事件；`schedule.json`、`outputs.npy`、`references.npz`、`pipeline_statistics.json`和`instruction_pipeline_detail.csv`供审计。跨核duration_sum有重叠，不能相加当总周期，未独立观测的流水不虚构统计。

## 5. 第一个新算子：从Python到实际C++

先用已集成的affine任务演练：`Z=half(half(A@B)+C)`，非MMSE、不使用信道。三文件JSON提议协议只允许设计文档和AffineGemmModel.h/.cc；已有任务/参考/main注册由协调者冻结。示例不是任意算子的自动编译器。

```bash
# 在干净主检出中操作；目录和分支名每轮换新。
git worktree add -b codex/affine-0001 ../affine-0001 HEAD
python3 scripts/claude_codegen.py \
  --task-file agent_profiles/recipes/affine_gemm_v1.txt \
  --apply-root ../affine-0001 --replace-candidate-files \
  --output results/affine-proposal-0001 --timeout 600
# SOURCE_IMPORTED只表示成功导入，不是算子通过。
git -C ../affine-0001 diff --check
# 负责人独立审计后，仅提交这三个允许文件：
git -C ../affine-0001 add DOCS/operators/AFFINE_GEMM_DEMO_V1.md src/inverse/affine_gemm_v1/AffineGemmModel.h src/inverse/affine_gemm_v1/AffineGemmModel.cc
git -C ../affine-0001 commit -m "Freeze affine candidate 0001"
cd ../affine-0001
# 使用导入基线随包的只读缓存；每个候选自己的.deps/.venv/build。
python3 scripts/bootstrap.py --cache ../asim-claude-guide-v2/offline_cache --offline --jobs 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/affine_acceptance.py \
  --binary build/bin/Simulator --config configs/ascend_910b_quiet.json \
  --cwd build --output results/affine-0001
```

上述缓存路径假设解压目录名未改；改名后显式替换，缓存也可放企业只读公共目录。不要共享可写build/日志、并行stash或覆盖基线。

生成器实际调用`claude --bare`、无文件工具，让模型返回文件JSON，导入器检查路径/重复键/范围；密钥仅在子进程环境内。receipt中的operator_verified=false、score=null是正常提议状态。编译报错时把原始错误、接口定义和允许范围反馈模型，审计修复后冻结新commit再重跑，不能改Tensor公共API/参考/门限来接受坏候选。超时、输出截断或格式不符记录失败，不无限重试。建议每轮最多2次修复；用尽预算由协调者决定下一任务。

本例成功生成及两轮修复调用墙钟合计约301秒；此前超时另计。纯模型思考时间无法可靠获取，记null；不能把等待编译或API墙钟称纯思考时间。

## 6. 从示例扩展到真正的新算法

1. 先写独立Python数值原型，冻结输入、输出究竟是逆/均衡/中间变量、公式、初值、预条件、恢复方向、迭代、分块、精度与舍入。迭代算法先新校准选最小过门限K，再用全新留出；增加K可能撞到精度平台/稳定性问题。
2. 协调者添加可信任务/独立参考/注册：MMSE入口看`orchestrator/codex_operators.json`与`verification/operators/catalog.py`；新任务形态看`verification/tasks/affine.py`的BACKEND。只给受信Python代码选callback，不能让候选JSON指定任意参考函数。
3. 为候选指定新目录/Model/mode；只有算法/契约变化才新入口，纯优化不复制新类。参考`src/inverse/affine_gemm_v1/`、`src/semantic/SourceViews.h`以及main注册。新增Tensor维度必须满足真实C++接口，不能照抄不匹配的草稿。
4. 将Python每步映射为实际Instruction，明确DMA输入角色/偏移/字节、源和目标视图、索引/切片/转置、常量、precision_policy、buffer生命期、输出恢复、分片和屏障。24倍数batch与16对齐工作量是硬约束之一，不靠多做无效Cube提升分数。
5. 确有新指令时，由独立可信变更补Asim/真实视图绑定/数值精度/调度语义和负例，再交候选使用；未知语义必须UNSUPPORTED。不能按算法名称调用标准算法代替真实指令重放，也不能另一份FormulaLogger证明实际代码。
6. 独立审计、冻结、构建、Asim→指令/事件/性能预筛→重建与双参考→适用通信→评分→归档。新算子至少覆盖多形状/种子/边界、故意错误负例及旧入口回归。

三条数值路径必须独立：实际指令重放；同初值/参数/K/精度的Python算法；高精度任务参考。前两者一致但任务精度失败，说明算法/精度本身未达标；不一致说明映射或数值语义问题。只看2%达标会漏掉不同实现/不一致行为，所以两道门禁都需要。FP16指FP16张量及声明的FP32临时计算模型，不是硬件已认证，不能暗中用FP64实现候选。

## 7. 迭代优化：保留基线，逐轮积累证据

同算法采用同Model/mode＋独立candidate分支/worktree/build。每轮只改变一个明确方向，例如复用、布局、状态槽轮换、冗余copy、流水/同步。保持公式/K/精度不变才能称纯指令优化；改初值/预条件/谱界/K等要单列参数/外循环并重新校准、全新留出。

开始先冻结基线源码commit、binary/lib、可信验证器、配置与输入种子/内容SHA、完整结果和计时范围。每轮任务写明允许路径、假设、预算、最多修复次数、退出状态及验收；候选不改可信脚本。示例给Agent：

> 从冻结NS基线建立candidate ns-inner-0001，只研究中间buffer复用。保持K16、全部公式与精度不变；只改指定NS源文件。先交设计/差异，审计后独立build -j1。相同输入与计时下先Asim和事件检查，性能不足1.01倍或超过10倍可疑上界即停止数值、不改K重试；否则实际重建与同算法FP16逐位一致、FP64 MMSE门禁通过。保存逐流水/逐核统计、失败记录和源码/工具版本，不自动晋级。

共同runner的Python接口支持受监督性能预筛；CLI日常示例默认创建正确性基线，不自动对任意新binary晋级。协调者在可信工作区运行如下调用（base_result、candidate_binary等必须来自已冻结/已审计任务，不接受Agent自报周期）：

```python
# 从可信基线工作区启动Python；候选目录只提供binary，不执行其验证脚本。
import json
from pathlib import Path
from verification.contracts.mmse import MMSETask
from verification.contracts.core import ScreenPolicy
from verification.reconstruction.runner import evaluate
from verification.asim.runner import file_hash

# 以下变量由冻结任务提供：base_result、expected_base_sha、config、
# candidate_binary、candidate_build_dir、new_output_dir。
assert file_hash(base_result) == expected_base_sha
base = json.loads(Path(base_result).read_text())
r = evaluate('ns_corrected_v1', MMSETask(profile='modeled_fp16_fp32_real_block_v1'),
             16, 42, Path(candidate_binary), config, Path(candidate_build_dir),
             Path(new_output_dir),
             policy=ScreenPolicy(base['asim']['finish_cycle'], 1.01, 10.0),
             baseline_identity=base['comparison_identity'])
print(r['status'], r['numeric_status'])
```

同一物理配置请使用同一组固定绝对配置路径/内容；现comparison_identity含配置locator，不支持把异路径直接当同身份。当前旧daily基线要求相同binary，不能用于优化候选；正式跨路径/跨candidate基线控制仍待建设。上例是审计后的手动受监督调用，不是完整晋级器。affine的demo score=1是重复实现串接，不能当优化收益。

结果处理：`SKIPPED_ASIM_FAILED`、`SKIPPED_NOT_COMPETITIVE`或异常加速拒绝时数值不启动；`UNSUPPORTED`补可信语义后建新run；`REPLAY_MISMATCH`查映射/舍入；`ALGORITHM_QUALITY_FAILED`查算法/精度；全部拒绝均score=null并保留证据。不可删失败行或改留出集反复调参。

按`family/algorithm/baseline/candidate/run`建立台账；每场记录commit、binary/evaluator/reference/scorer/config/input SHA、参数/K/精度、计时、周期、位一致/最大任务误差、通信适用性、流水CSV、耗时/峰值内存和失败原因。`scripts/optimization_ledger.py`可校验已有hash链并生成CSV，写入接口/字段见源码和`DOCS/research/OPTIMIZATION_TRACKING.md`；包不带历史账本，请新建自己的实验目录。score未完成晋级时null。baseline/冠军用新选择记录指向，不覆盖最初基线。

## 8. 迁移到OpenCode或其他平台

更换Agent不应重写Asim/重建/独立参考/门限。当前自动提议launcher只支持Claude；`agent_entry.py`的active client只认codex/claude，不能仅把JSON改成opencode就称适配完成。本轮未安装/实测OpenCode。

先做有人监督接入：在独立候选检出运行OpenCode，要求读取AGENTS和本手册、标准及技能文件；人工审核diff/允许路径、冻结版本后，从可信工作区执行相同build、runner与质量门禁。官方当前文档给出`opencode`交互入口和`opencode run "..."`非交互入口；V2读取AGENTS.md，不自动把CLAUDE.md作为fallback，所以要显式要求读Claude规范。账户/模型/endpoint按该平台的provider配置，不能直接复制Claude认证配置。见[CLI](https://opencode.ai/docs/cli/)、[Instructions](https://opencode.ai/docs/instructions/)、[Providers](https://opencode.ai/docs/providers/)（2026-11-12查阅，部署时按实际版本核对）。

```bash
# 在独立candidate目录；模型和凭据先按平台官方文档自行配置。
opencode
# 或手动受监督的非交互任务：
opencode run "读取AGENTS.md、CLAUDE.md和DOCS/CLAUDE_CODE_USER_GUIDE.md。只审查候选允许文件，先输出设计，不改参考/评估器/门限。"
```

正式适配器应新增：CLI调用/版本检查、prompt与环境输入、平台输出解析、调用墙钟/真实模型/退出码记录、超时和修复预算、严格文件提议导入、受信文件不变检查。推荐统一为平台无关`files:[{path,content}]`或审核后的patch；保留原平台响应及SHA，不能把OpenCode响应伪装Claude成功信封。不要直接用Claude专有--bare/--tools/JSON字段套其他CLI。现有`claude_codegen --response`只校验封存Claude响应，agent_executed=false，不是通用平台适配器。

| 保持不变 | 需要适配/重新验收 |
|---|---|
| 算法/输入/K/精度/计时契约、Asim和真实指令语义 | CLI参数、认证方式、会话/工具/技能发现 |
| 重建核心、独立参考、数值/通信门限、评分 | 输出提议解析、路径保护、预算、日志/模型身份 |
| 原始基线与追加台账规则 | 新平台真实生成affine、错误负例和六类回归 |

接入验收：真实生成可构建的新算子；九组affine通过；实际错误指令拒绝；原六类回归；超时、越界写入、改参考/门限、未知语义、不足性能均正确拒绝。完成这些前只能标“接入试验”，不升级为已认证平台。提示词/白名单不是任意C++的OS沙箱，企业部署可在隔离账号/容器中运行候选。

## 9. 常见问题与证据归档

- 没有.git：按第2节初始化本地导入；worktree与冻结工具需要Git，不能删除rev-parse来绕过。
- NumPy wheel不匹配：本包仅CPython3.12/Linux x86_64；其他版本需新依赖锁并复验，不临时装任意wheel。
- 依赖SHA错误：核对包校验和/缓存，不降校验；已安装.deps被改动需新目录恢复。
- 数值没有启动：先看Asim、事件、性能预筛状态，正常拒绝不意味着重建坏；不要强行跳过前置门禁。
- 调高K反而变差：先独立研究谱界、稳定性和精度平台；留出只评估，不选参。
- CDL-B：默认smoke是Rayleigh；真实CDL依赖和新输入/选K/通信门禁另行按CHANNEL_DOMAINS执行，不把Kronecker近似当CDL认证，求逆精度不要求FEC前置。
- 磁盘不足：先估算trace；只清理自己本轮可再生成缓存，先存SHA/清单；不批量删历史报告或未跟踪资料。

自己的小型结论/契约/台账进入experiments下专门run目录；大trace/输入/输出进results，并留大小/SHA/恢复方法。论文索引固定DOCS/research/paper_archive/README.md；本包未携旧论文证据，需查Git研究档案。每次结果都区分代码存在、已运行、数值通过、通信通过和完整晋级，记录失败与缺失证据。
