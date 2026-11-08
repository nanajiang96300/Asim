# Q3a：实际重建输出的瑞利/QPSK相对质量门禁

Q3b另有[实际I/Q响应与高斯线性SE代理](RESPONSE_PROBES.md)的显式有限样本接口；本页Q3a契约和状态不变，不将该代理静默替代必需SE。

当前支持配对的未编码QPSK BER和**原始软符号EVM**，不计算SE、BLER或高阶QAM。
它位于已有Asim/事件/数值重建之后，不改变算子或固定runtime，不重跑Asim。
候选通信输出始终来自实际`outputs.npy`；独立MMSE只作配对参考，不替代候选。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.communication.runner \
  --contract experiments/communication/2026-11-08-v1/ns-heldout-contract.json \
  --contract-sha256 <冻结契约的SHA-256> --output results/communication/<new-id>
```

契约冻结逐结果的路径/hash、角色、runtime、指标、质量门限、置信要求和最小样本数。
调用者批准这些摘要，候选不能改阈值或筛掉困难样本。摘要不是签名或OS安全边界。
读取前核对Asim成功/事件/重建门禁，再核验完整制品、实际模型、精度、输入和输出形状；
重新生成H/Y/符号与量化输入，独立求解MMSE并复核原参考和实际输出精度。
失败的数值入口不得靠通信指标翻案。运行前后核对制品和版本。

## 数据角色

- `holdout`：要求原受监督计划及每case的计划绑定，核对任务、参数和留出角色；该计划全部留出job必须包含，不能挑选有利seed/噪声档位。
- `historical_diagnostic`：历史已验收数据可统一计算指标，但顶层永远为DIAGNOSTIC_ONLY，不当作新的独立质量通过。
- 不接受校准集作为通信验收数据。不同K/算法/任务/精度分别分组，不混合提升样本量。
- 同组重复H会拒绝；同一H可以用于不同噪声/算法的配对比较，但不当作同组新增独立样本。

## 指标和统计量

QPSK决策分别比较实/虚部符号，零值固定判为非负bit。BER是未编码硬判决误码率。
原始软符号EVM为`norm(xhat-symbols)/norm(symbols)`，保留MMSE软输出的收缩偏差，
不做oracle增益/相位拟合；不是3GPP发射机EVM。
候选与MMSE使用同一H、符号和噪声，按**每个独立H**汇总全部用户/RHS的BER与EVM，再比较差值。
这些指标不承诺逐用户可靠性，也不提供绝对BER目标或编码吞吐量保证。

当前工程试验契约：每信道BER相对MMSE增量≤0.01（一个百分点），EVM增量≤0.01；
要求任一指标超限的联合信道概率≤5%，family_alpha=0.05，至少96个独立H。
这是可审计的框架试验配置，尚不是负责人确定的最终论文/业务质量规范。

对每组BER、EVM及两者联合超限事件使用Clopper–Pearson精确二项式区间；将alpha分给所有组/三事件/双尾，
通过Bonferroni形成同时置信界。独立试验单位是H，不是同一H下的相关bits/RHS。
组间允许相关；每组要求来自冻结的IID瑞利fixture且没有重复H。
参考：[SciPy精确区间说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)。
运行实现只依赖NumPy/标准库；测试额外用已有SciPy独立交叉核验，未安装时该交叉核验跳过。

上界≤约定超限概率且样本足够，得到QPSK_QUALITY_CHECKED；下界已经超过门限为QUALITY_FAILED；
否则INCONCLUSIVE。**零次超限的上界仍大于零**，不能用零观测错误宣称无错误。
历史诊断也报告区间，但它不满足新留出证据角色，不能升级质量结论。

## 明确未支持的结论

输出少量xhat不足以唯一确定均衡矩阵W，逐操作FP16还有数据相关舍入误差。
因此不由MMSE误差、原始EVM、硬判决BER或H的容量曲线冒充SE。
必需指标中出现SE/BLER/CDL-B等直接UNSUPPORTED；不会删掉指标后继续PASS。
下一步Q3b需要明确实际系数/探针协议和SE定义，再扩更多算法的新留出集及信道范围。

结果始终`score=null`、`full_communication_certified=false`；这是冻结范围内的相对QPSK试验门禁，
不是整个通信研究或完整晋级完成。CLI仅对QPSK_QUALITY_CHECKED返回0；诊断/样本不足/拒绝返回非零，
应按JSON状态区分，不把所有非零退出都当成算子错误。
