# 通用非线性原语与小型模型验收

日期：2026-11-12（Asia/Hong_Kong）。原始证据：`/home/nanajiang/Asim/results/tiny_nn/2026-11-12-v1/`。本报告的最终交付复现状态见文末；旧失败与中间版本保留，不把代码存在作为验收。

## 实现与范围

新增公共 `src/operations/nn/VectorUnits` 和 `NNGraphModel`，复用 SOFTMAX(8)、GELU(22) 与既有 GEMM/ADD/MUL/DMA/屏障。基础单元和模型拓扑分离，同一解释器支持 Softmax、GELU、FC→GELU→FC 与单头 Attention＋MLP 两种整图拓扑，没有按模型名返回 Python 答案。输入、权重、偏置、缩放、转置和输出都来自实际指令。Asim产生周期/事件，统一重建器计算数值，独立参考不导入重建内核。

默认B96、24核、L32/D32/FF64；8,416参数、327,680 GEMM MAC/样本。完整输入/冷权重DMA至输出STORE计时，未外置初始化。全部中间写回FP16，GEMM和非线性临时采用声明的FP32模型；没有新增FP32硬件张量。Softmax逐行稳定计算；GELU显式tanh近似。新NN配置SPAD/ACCUM各256KiB、每双buffer128KiB，最大形状布局105344/83968B均未超限。

## 验证结果

最终受信源码为b6e688ce（C++为16c28f7a），二进制SHA-256 `f8578ac30c4278511dcd7efd120429f2e27ab80dd31134f1bf06fbd607773d94`。校准种子90001与冻结留出91001/2/3分开；三种形状、四类任务各3种子，共36配置/3456次batch重放。所有输出及捕获阶段与独立同语义FP16参考位一致；最大FP64相对误差0.075541%，低于事先冻结1%门限。不同任务配对复用了相同形状/种子的输入，不把3456次重放称为3456独立样本。

| (L,D,F)，均B96 | Softmax周期 | GELU周期 | MLP周期 | Attention周期 |
|---|---:|---:|---:|---:|
|16,32,64|791|191|1506|3630|
|32,32,64|1534|346|1802|4975|
|32,64,128|1586|602|3806|8298|

逐场误差、形状、状态与阶段墙钟见[cases.csv](cases.csv)；原件位置、完整SHA及源文件/配置/库身份见[evidence_manifest.json](evidence_manifest.json)。每种任务/形状保留seed91001的[逐指令CSV、流水/逐核统计及首batch阶段指纹](pipelines/)，完整96batch与所有trace在对应无损原件包。报告副本CSV仅将CRLF规范化为LF，数值不变，原字节保留在原件包。流水duration_sum是跨核累计，不能相加当总周期；active_union也不是硬件利用率。Softmax/GELU内部是声明的宏指令，未伪造EXP/tanh微指令统计。

随行数M16→32，N32的实际Softmax Vector事件176→352周期，GELU16→32周期；整图还包括DMA和屏障，总周期不要求严格翻倍。

## 失败、反例与修复

- 首轮真实Asim崩溃：Tensor尚未注册Model就调用add_input。输入先注册后连边修复；原quick-v1、GDB和日志保留，未启动数值参考。
- quick-v2的Attention实际输出/阶段位一致，却被旧误差度量误拒绝：FP16参考先在FP16中缩放制造舍入差。现先将两边已表示数值提升FP64再归一化；新增混合存储回归，不放宽门限、不改算法。
- NN构造明确初始化外部root id、输出producer使用实际Operation id；真实输入M/U元数据修正。历史基类构造风险记录，未批量改旧算子。
- 真实错误K转置候选（独立negative-k-v2分支）仍4975周期且事件通过，但全部96batch从logits开始偏离，状态REPLAY_MISMATCH。最大FP64误差约0.198%，甚至低于1%；3个最终输出偶合位一致，也因中间阶段错误被拒绝。见[真实负例](real-negative.json)。不能只靠任务误差或合理周期认定算子正确。
- 7项真实入口/前置拒绝见[负例清单](preflight-negatives.json)：重复输入、未知节点、错误输出、容量超限、未知node属性、旧容量配置及错误精度profile。轴/公式/工作量/广播/阶段缺失等另由数值测试覆盖。未声称所有JSON层级扩展属性都有allowlist。另6项封存后内存注入（错轴、未知GELU公式、少算Softmax工作量、漏GELU、漏残差、错缩放源）均被公共验证/重建拒绝，见synthetic-negative-summary.json；这些是软件注入测试，不能称为6次真实Asim故障实验。
- 原始直接unittest因无Torch/Sionna的可选CDL测试ERROR保留于unit-final.log；b6e688ce补充显式可选依赖skip后，264项中262通过、2明确跳过；便携构建/客户端8测试通过。跳过不代表CDL验证。

原六类新二进制回归见[six-regression.json](six-regression.json)：Cholesky/LDL/Cholesky-block/LDL-block/NS/BRI，K1/1/1/1/16/192、同Rayleigh seed42、B96、Nr64/Nt16/Nrhs8、lambda0.1、旧quiet配置，周期28006/18697/21338/21665/7950/23047。这是历史配置回归，不能与新NN容量表混成统一硬件排行榜，也不升级历史失败变体或CDL状态。

## 复现与冻结

```bash
python3 scripts/bootstrap.py --cache offline_cache --offline --jobs 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/test_portable.py
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/nn_acceptance.py --output results/my-nn-suite
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/portable_smoke.py --output results/my-six
.venv/bin/python scripts/freeze_nn_runtime.py --suite results/my-nn-suite/summary.json \
  --regression results/my-six/summary.json --runtime-dir results/nn_runtime/my-v1 \
  --output orchestrator/nn_runtime_my_v1.json
.venv/bin/python scripts/nn_acceptance.py --quick --runtime orchestrator/nn_runtime_my_v1.json \
  --output results/my-pinned-check
```

从Git检出使用构建文档中的固定缓存；源码包提供offline_cache。每次使用新目录。freeze要求36唯一Cartesian、同二进制六类回归、source/evaluator/library/config指纹一致；旧pin不覆盖。summary由可信执行脚本产生，候选不得修改；本轮另逐个读取36个原件result校核summary一致性，审计证据单独保存。把该一致性复核内建到freeze是后续加固项，不宣称已经自动实现。

## 论文边界与后续

可支持“周期模拟器＋真实指令重建扩展至非线性和两个小型模型拓扑”的软件证据。不是训练后任务精度、CSI恢复、任意ONNX、多Operation跨层内存、真实910B硬件时序或性能晋级认证；score=null。硬件验证由负责人后续上板，非本轮前置。纯Agent思考时间无法观测，记null；阶段执行墙钟单独列出，不将聊天/工具等待称为思考时间。

新冻结入口orchestrator/nn_runtime_v2.json，279文件，SHA-256 `1eaacb2462662d46cff119d4b77126d8cd9149fa2aa52ae9369e1d6d4257b102`。pinned quick四类通过；错误文件SHA、重复Cartesian、来源指纹漂移均拒绝且未启动Asim/数值，见publication-negatives.json。affine九组864batch与配对重复通过，demo分数1仅表示重复基线，无研究晋级。

最终交付复现：执行中，完成后以独立delivery-validation.json和源码包SHA更新；在此记录前不宣称T5完成。
