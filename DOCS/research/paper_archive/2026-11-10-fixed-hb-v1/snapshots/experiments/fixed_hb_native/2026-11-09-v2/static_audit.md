# 固定HB原生生成端：独立静态审查

2026-11-09。结论：**STATIC_REVIEW_PASSED**，未发现阻断进入构建/真实Asim/实际重建的源码问题。该结论不代表编译、数值、通信或性能门禁已通过。

工作区：`/home/nanajiang/asim-worktrees/bri-fixed-hb-20261109`，提交`40bcbf0e3448cf3d62fd33166005c0fb1fbdd478`；设计文档先于代码存在。审查只读，无代码修改、编译或Asim。唯一写入为本报告。

## 核查结果

| 核查项 | 结果及源码证据 |
|---|---|
| 任务/精度/尺寸 | Model完整声明H[B,128,32]、Y[B,128,16]、RegI/Identity[32,32]、xhat[B,32,16]。Operation约束P2、请求64、24核、B为24倍数且≤96、K1..174、Nr64/Nt16/RHS8；受信catalog进一步只开放B96/F16。不是新增FP32持久张量协议 |
| 能量原值 | `BRIStableHBModel.cc:97`写独立energy标量；:98 regularize写den，:99 reciprocal写pj。:102 sum_energy、:103 scaled_energy及:110 P_energy仍读原energy；没有把e覆盖为e+λ |
| Scalar工作量 | 每个SCALAR目的view均1×1，arithmetic按17..21设置compute_size=1；SQRT真实一元SCALAR_SQRT。归约16项逐项ADD，未以一条Scalar隐藏整行/矩阵工作 |
| L4/L2/trace | :103–111的缩放、平方两次、逐项加，与主模型每条FP16写回顺序相同。P归约使用pj/128；trace用pj·原e。sumE/E4/P2/trace在循环前真实清零。norm用真实SQRT与MUL，不需MIN/MAX |
| 边界及系数 | :113–132依次生成Eu/Pu、l/u、sqrt(u/l)、den及den²，α=Q((15/16)Q(4/Q(den²)))、β=Q(Q((sqrt(u)−sqrt(l))/den)²)。独立临时区分配，未有覆盖den导致β使用错误分母 |
| α/β恒定 | α仅:128–129写入，β仅:132写入；:133起迭代循环只在:143–144读取它们。所有向量临时、P、能量和常量地址均由单调64B对齐allocator独立分配，未与α/β别名 |
| 初值与有符号零 | x从实际MUL(0)生成，再ADD(+0)规范化初值；previous/carry实际MUL(1)复制。Pg前ADD(+0)与独立模型一致；未偷用带符号零不同的对角GEMM等价声明 |
| P完整分配 | :94 allocateP先写完整32×8区域，然后每个复流逐行复制pj到实/虚对应行，16流覆盖全部32行、8列。避免首次局部写只分配一行 |
| 输出完整分配/恢复 | :153先完整分配ACCUM32×16，再四象限恢复。负虚部经MUL(-1)+ADD(+0)，与独立SUB(0,im)的零语义一致；一个MOVOUT完整覆盖每batch输出，且src_from_accum/last_inst均声明 |
| 隐式梯度及补偿 | 两个真实GEMM为H·X与Hᵀ·res；Vector减法真实MUL(-1)+ADD，两次写回。λX、Pg、动量、delta、carry及previous/state复制均生成指令。新状态写回前先更新carry，未覆盖其需要的旧X |
| DMA/batch | 四输入LOAD尺寸与物理区域匹配，base_addr=0；batched地址按完整rows·cols·P偏移。Y加载M×2R、packedY只取左列；输出stride U·2R·P。固定尺寸均64B整分，无未支持尾包 |
| raw工作量/view | 同一helper从dst.rows/cols和inner派生tile_m/k/n、size与compute_size，再附实际src/dst views。GEMM输出size按FP16 bytes、compute_size按列数；Vector按既定FP32工作字节4计费。不得把这种声明的逻辑view模型等同于上板物理gather/layout成本已认证 |
| 泛式 | shared engine/source_views/binding/runtime及numeric未增加HB算法分支。工作区与主仓这7个文件/模型的字节比对一致：studies/stable_hb、reference/stable_hb、numeric、engine、source_views、binding、runtime |

## 适配器与入口

- `stable_hb_parameters.py`的id外参数和C++ algorithm header逐项一致，包括K、复块1、系数/包谱规则、列能量及和范围、单侧表示和初值。
- `stable_hb_native.py`仅把已独立实现的逐样本数值结果包装为完整output_block、复数output及setup证据；不导入共同重放器，不替代实际执行。
- `main.cc`同时设置新mode为非语言模式并实际构建注册BRIStableHBModel；旧mode分支保留。
- runner只把超时改成受信adapter.get('asim_timeout_seconds',120)，新条目300s，旧条目默认120s不变。max_cycles仍100万；catalog/source由评估器冻结，不是候选自选超时。
- 新条目观察预算2GiB/400万事件。预算属于可执行资源边界，不是动态成功证据。

## 静态资源估算

逐条计数得到每batch：295+23K个写后barrier、591+46K条总实际指令（含4LOAD/1STORE）。K174对应4297个barrier、8595条指令。sync声明512+64K=11648个64B槽，足够覆盖全部barrier；其他区域按64B对齐且位于各自bank。

对B96/K174，约824,640个compute/barrier操作，3事件/操作，加DMA请求响应与buffer reset，预期约2,526,336观察事件，小于400万上限。只是静态描述计数；实际文件字节数、峰值RAM/swap、运行墙钟、dispatch、事件覆盖及读写依赖必须由新进程实跑验证。不要因静态预算通过就直接提高K或忽略超限。

## 动态退出条件与证据边界

1. -j1构建及真实Asim成功，实际task/algorithm/profile/配置绑定、全96batch和24核放置完整。
2. K16/53/174三档实际重建输出与独立参考逐FP16位一致；建议在系数完成处及末轮读取实际内存，只读比较α/β、l/u、P展开及归约状态，不能向回放内存填参考值。
3. 原坏点仅回归；算子新CDL H59001/59002留出冻结后验收，MMSE≤2%与相对QPSK分别报告，不能只凭前置Python192-H通过。
4. 错反馈/系数/符号/象限与缺语义等负例、重复一致性及固定runtime另验。score在适用晋级门禁完整前保持null。

外部输入域和系数稳定检查目前由独立参考/验证器拒绝，不是本C++内核生成的COMP指令。设计文档已说明这一分工。若以后宣称内核自己能动态拒绝域外数据，需新增实际比较/分支语义和对应成本；本审查不把外部检查当免费内核功能。

固定系数谱证明只针对声明的存储系统理想线性递推；不替代FP16隐式残差、补偿及输入量化误差验收。硬件认证由负责人后续上板，不是本轮软件接入前置。

审查起点UTC 2026-11-09 12:38:49；结束UTC 2026-11-09 12:46:56；可观测审查墙钟约487.65s，pure_model_thinking_seconds=null。

源码SHA-256：

- `DOCS/operators/bri_stable_hb_fp16_v1.md`: `370cb7d1a3d00e5718d89e3d772c37694ede63c6915f233616b0d135652eee20`
- `src/inverse/bri_stable_hb/BRIStableHBModel.cc`: `c45ef79c9f535176b83e09718d2013e20515d3bbd0221abdd8e6ff6d426684a7`
- `src/inverse/bri_stable_hb/BRIStableHBModel.h`: `5333116460e7d57f052f1f6046c1cb6a467ce1198bc71a8721a108d9ea04d12c`
- `verification/operators/stable_hb_parameters.py`: `98ee8f77cbb448fe8c44c07726b05fcff913a049557c8502055dc5b15a262595`
- `verification/reference/stable_hb_native.py`: `783a7e8119794b672556fc0354024b06ddae3ba4589caf95f53e47b21883c993`
- `orchestrator/codex_operators.json`: `52d9746701887533ed3142186e2ada33340fcb9a5dbe80130e8e568ffcef1dc7`
- `verification/reconstruction/runner.py`: `8b1c62e730e18e264e0e6d4b77c095555f9dd68fdb110e6e4c4c47bed5091b28`
