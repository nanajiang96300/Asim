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
输出目录存在时拒绝覆盖。新工具尚未接入无人值守协调器；操作者将选中的 K 和参数
冻结为候选任务，之后仍必须通过既有 Asim → 重建 → 通信门禁。

当前为保证不改动已验收的独立参考，每个 K 从头仿真，复杂度随网格的 K 总和增加。
前缀检查点复用属于下一阶段：必须与当前独立参考逐位等价后才能替代，不能提前宣称已经加速。
当前只支持既有 NS 初始化和 BRI 实 B2/ω0.5/预计算递推；新预条件或精度需要新参考与参数版本。
