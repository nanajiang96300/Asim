# Q3b：实际指令的 I/Q 响应与线性 SE 代理

入口为 `verification.communication.probe`。这是有限样本诊断，不是完整通信认证。
先有成功的实际Asim/事件/数值重建，再复用其冻结指令；本阶段不改算子、原始输入或runtime。
探针是新目录中的离线数值实验，**没有为每个探针重跑Asim，也不产生新的性能周期**。
当前支持原有固定、无数据相关控制分支的已导出指令序列；未知语义拒绝。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.communication.probe \
  --contract experiments/response_proxy/2026-11-08-v1/ns-heldout-contract.json \
  --contract-sha256 <冻结契约SHA-256> --output results/response_proxy/<new-id>
```

## 同一套重放逻辑

1. 复用Q3a的源结果验收：实际Asim成功、事件/数值门禁通过，全部制品和原输入仍匹配。
2. 对原始Instruction及dispatch执行共同`validate_actual`，保留地址、batch、布局、原语及精度语义。
3. 固定H、RegI、Identity，只给指定batch的Y提供契约定义的探针；保持原B/Nr/Nt/RHS形状。
4. `execute_batch`执行真实序列。同算法独立精度参考只提供比较目标，不提供候选响应。
5. 每次要求完整执行、无成本歧义、结构匹配且FP16位一致；原始Y还必须重现已归档输出。

没有另写按算法名计算候选W的分支，也没有修改解释器或用FormulaLogger代替指令。
每次保存逻辑Y、原始重建输出、oracle/输出hash、执行指令数及完整响应矩阵。
密集随机探针有冻结种子并保存全部值，是受控测试输入，不是为缺失语义补随机输入。

## 从实/虚探针辨识响应

将Nr个实基向量和Nr个虚基向量打包到现有RHS列，得到实响应R、虚响应J。

\[
W=(R-iJ)/2,\qquad V=(R+iJ)/2,\qquad f(y)\approx Wy+V\bar y.
\]

检查零输入、正常Y、0.5Y、2Y、−Y、密集Y、叠加、RHS隔离和列置换。
预测误差同时检查整体与每个输出流，防止高能量流掩盖差异；检查共轭分量和RHS串扰。
超过门限则RESPONSE_MODEL_REJECTED，**不输出SE数值**；不能丢掉V后继续当作复线性算子。
精度模型与原任务一致：FP16存储/逐操作FP32算术，独立参考必须位一致。
响应代数和SE计量使用FP64分析，这不表示候选用了FP64中间运算。

每H重放次数为 `2*ceil(Nr/RHS)+9`；Nr64/RHS8为25次，另有25次独立同算法参考计算。
单契约限制最多16个源case、1..4个预选batch索引及2048次重放；先核算全部预算再开始。
holdout源必须覆盖其批准计划的全部留出case；H索引在查看响应之前冻结。

## 代理的定义和边界

仅当线性预测/共轭泄漏/串扰等检查通过时计算：

\[
G=WH,\quad \gamma_i=\frac{|G_{ii}|^2}{\sum_{j\ne i}|G_{ij}|^2+\lambda\sum_j|W_{ij}|^2},
\qquad R_{lin}=\sum_i\log_2(1+\gamma_i).
\]

使用同一个原始H、单位符号功率及白噪声λ；明确计入滤波后噪声和多流干扰。
参考W_MMSE来自独立高精度solve(HᴴH+λI,Hᴴ)，不代替候选响应。
这是**高斯输入线性接收器的归一化空间总速率代理**，不是信道容量、有限QPSK可达率或编码吞吐。
因此Nt16时出现约100的代理值而QPSK只有32 bit/复信道使用的调制上限，并不矛盾；禁止裁剪后宣称饱和通过。
参考定义背景：[Sionna LMMSE与有效噪声](https://nvlabs.github.io/sionna/v2.1.0/phy/api/mimo/sionna.phy.mimo.lmmse_equalizer.html)。本工程未依赖或集成Sionna。

FP16映射一般不是严格线性；有限测试通过没有给出全域误差界，代理**不是实际FP16可达率的严格上/下界**。
1%预测门限也不表示SE误差≤1%；代理损失只是两个线性接收器公式之间的差。
探针不覆盖全部H分布，不给置信通过或性能分，不替代Q3a真实符号BER/EVM硬门禁。
SE对输出逐流缩放/相位不敏感，仍需保留原输出契约和独立数值检查。

## 状态与归档

本轮试验门限：预测/共轭泄漏/串扰各≤1%，零偏置必须0，逐样本代理损失≤0.05 bit/s/Hz。
这些是实现验收的工程配置，不是已批准的论文质量规范。
SE_PROXY_SAMPLES_CHECKED只表示冻结样本检查通过，population_quality_certified=false、
full_communication_certified=false、performance_claim=false、score=null。
旧Q3a接口中无定义的必需“SE”仍UNSUPPORTED；必须明确请求本接口定义的代理，不能静默替换。

每次新目录保留契约、测量JSON、NPZ和摘要；原始trace/输入只读。
精确线性与预算拒绝等负例返回非零，按JSON状态区分预期拒绝和实现异常。
论文证据见 [本轮报告](../../experiments/response_proxy/2026-11-08-v1/README.md)。
