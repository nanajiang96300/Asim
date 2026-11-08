# 写算子之前的算法参数筛选

`iterations.py` 是有人监督的离线阶段，不在固定版本的日常重建运行器内。
只复用独立逐操作 FP16 存储／FP32 算术参考；不改变算子、参考实现、重建核心或运行器 pin。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.planning.iterations \
  --contract experiments/quality_planning/2026-11-08-v1/ns_corrected_v1.json \
  --output results/quality_planning/<new-run-id>
```

契约严格拒绝未知字段、重叠的校准/留出种子、不支持的精度/信道/指标，以及超出参考调用预算的网格。
`max_reference_calls` 计一次完整 batch 的参考调用，不是单矩阵，也不是墙钟预算。
`regularizations` 定义每单位符号能量的噪声方差及匹配的 λ；H 采用现有 `1/sqrt(Nr)` 归一化。
它不是独立于天线数的接收总 SNR；不可和旧脚本未经换算的 SNR 混合。

对每个 K，全部校准场景、种子与 batch 都要求相对 MMSE 输出误差不超过门限。
遍历冻结网格，不假设误差单调。最小校准可行 K 只在留出集检查一次；失败记为
`HOLDOUT_FAILED`，不在留出集上调 K。没有校准可行点为 `NO_FEASIBLE_K`。
有缺口的网格仅声明网格内最小值；完整覆盖 1..K 才能声明该校准集内最小整数 K。
非有限值、形状错误和参考数值异常计为失败，不能跳过。

`ALGORITHM_SCREENED` 仅代表当前算法数值筛选通过。输出始终包含
`operator_verified=false`、`communication_status=NOT_RUN`、`score=null`。
SE/BER、CDL-B 请求明确拒绝，不以数值误差替代它们。
独立留出集通过不证明未见信道全部达标；正式研究需要扩大分组与留出规模。
2% 是示例工程阈值，不是已经确定的通信质量要求。

输出契约、逐 batch 曲线、输入身份、参考/筛选器源码 SHA-256、Python/NumPy 版本和墙钟。
输入可由固定版本 fixture、任务、种子重生并核验；本阶段不存大型 trace。
输出目录存在时拒绝覆盖。新工具尚未接入无人值守协调器；已有受监督的受信交接入口，见下节。
之后仍必须通过既有 Asim → 重建 → 通信门禁。

当前为保证不改动已验收的独立参考，每个 K 从头仿真，复杂度随网格的 K 总和增加。
前缀检查点复用属于下一阶段：必须与当前独立参考逐位等价后才能替代，不能提前宣称已经加速。
当前只支持既有 NS 初始化和 BRI 实 B2/ω0.5/预计算递推；新预条件或精度需要新参考与参数版本。

## Q2：计划冻结与真实执行

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.planning.handoff \
  --screen results/quality_planning/<screen-id>/ns/result.json \
  --create-baseline --output results/quality_handoff/<new-id>/ns-plan.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.planning.execution \
  --plan results/quality_handoff/<new-id>/ns-plan.json \
  --plan-sha256 <handoff输出的SHA-256> --output results/quality_handoff/<new-id>/ns
```

冻结者重新运行独立筛选，核对全部曲线/选择结果，而非相信报告PASS；额外消耗一次完整筛选预算。
计划绑定筛选原件hash、独立参考/交接器源码、注册表、已验收runtime manifest及其二进制/评估器依赖，
固定算法参数、逐场景任务/种子/精度/计时边界和校准/留出覆盖。
目前支持已验收二进制内的NS/BRI参数化执行；编写新的C++候选仍需独立工作区构建及新的受信runtime发布。
执行器只接受计划及其外部SHA-256，不接受K、seed、profile、输入或硬件配置覆盖。
计划/源码/筛选证据/运行器在执行前、每次运行前后和结束时复核；改变后不能继承旧计划。
路径目前绑定本机，需要迁移时重新核验并冻结，不是跨机可移植的签名包。

每计划最多8个串行运行，失败即停止，剩余项明确NOT_RUN；不自动调参或重试。
`--create-baseline`只是显式建立诊断基线，不宣称性能提升。
后续冻结时改用逐job的`--baseline <已封存case/result.json>`（按计划的场景/种子顺序重复传入），
可设置`--minimum-speedup`、`--maximum-speedup`。基线必须有同运行器、任务/输入、算法和完整制品证据；
性能不竞争/异常快时，核心在数值阶段前退出，没有`numerics_started.json`。

新入口复用原始`reconstruction.runner.evaluate`，不修改旧daily或固定runtime。
它另行执行冻结阈值，不能把比2%更严格的契约丢掉，也不能放宽运行器2%的已验收门限。
保留`evaluation_result.json`原始核心记录，另附runtime与计划绑定，便于后续同口径基线复用。

`PLAN_RECONSTRUCTION_CHECKED`只代表计划覆盖和受限重建门禁通过；通信仍NOT_RUN、score=null。
SHA-256证明受监督批准的内容未变化，不是数字签名，也不能隔离同一用户的恶意进程。
本轮证据与负例见 [Q2记录](../../experiments/quality_handoff/2026-11-08-v2/README.md)。
