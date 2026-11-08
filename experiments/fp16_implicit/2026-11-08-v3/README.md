# FP16隐式Gram：数值可行性与独立留出验收

日期：2026-11-08。状态：**FP16_PRIMITIVE_NUMERIC_HOLDOUT_CHECKED**。本轮验证的是独立Python原语模型，未改Asim、旧算子、受信参考或runtime，未运行新Asim，score=null。

## 明确结论

在现有“FP16张量/系数存储、固定k顺序FP32临时算术”模型下，找到一条无需新增FP32张量的MMSE数值路径。全部持久迭代状态、前态、预条件器和补偿量都是FP16；不是宣称每次乘加都舍入为FP16或已上板验证。

| 独立留出目标 | 每流Es/N0 | 冻结K | 192样本最大相对误差 | 门限 |
|---|---:|---:|---:|---:|
| 直接MMSE输出 | 0dB | 13 | 0.88654% | 2%通过 |
| 直接MMSE输出 | 10dB | 33 | 0.95512% | 2%通过 |
| 直接MMSE输出 | 20dB | 76 | 1.23957% | 2%通过 |
| 仅逆矩阵输出 | 20dB | 135 | 1.43391% | 2%通过 |
| 上述逆矩阵再按FP16相乘做MMSE | 20dB | 同一逆 | 14.82567% | **不通过** |

逆矩阵目标与MMSE目标分别冻结。不能把前四项通过称为“显式逆再应用路径通过”，也不能用直接求解的周期冒充求逆周期。原七算子的状态不因本轮研究自动改变。

## 候选实际计算

当前可行基线是复块1的点Jacobi预条件Richardson变体，不是负责人所指历史最快BRI。P由每个复流的FP16能量对角加λ后取倒数，实/虚行显式复制该系数。本轮对复块4/8也做了校准诊断；更大块的原生映射与性能优化仍属后续工作。

直接MMSE迭代以g=Hᴴ(y−Hx)−λx作为隐式Gram梯度，不形成完整Gram用于残差。Chebyshev形式的系数由冻结区间[λ/4,8]生成，全部Scalar步骤存FP16，动量β再乘FP16的0.9；因此它是阻尼变体，不是未修改的最优Chebyshev多项式。区间是固定试验常量，不使用候选内谱分解，不宣称包住所有信道。

更新δ=αPg+β(x−xprev)后，使用FP16补偿c：u=Q(δ−c)，xnew=Q(x+u)，cnew=Q(Q(xnew−x)−u)。新增补偿缓冲本身是FP16。残差中的每个GEMM、ADD/SUB、MUL、DIV/SQRT结果均落FP16，所有临时算术严格按已有模型执行，不调用高精度小块solve/eig。

迭代只计算复数输出的左实块列，最后显式COPY/取负构造其余象限。它是新的表示策略，不是评估器忽略象限或修补旧输出；整个最终输出均与独立实现逐FP16位核对。CONST/COPY为数学构造/布局动作，未来需映射合法装载/复制/算术并计费，不能当作已存在的实际指令trace。

## 冻结与样本

真Sionna CDL-B使用既定3.5GHz/100ns/uplink/UT4×4、BS8×8/单极化omni/半波长/静止单时刻/+1.2MHz频点，逐H归一||H||F²=16。Nr64/Nt16/RHS8，λ1/0.1/0.01对应单位每流Es/N0 0/10/20dB。每批96为24的倍数。

校准H55001共96个，其中首24用于两个有界烟测；随后12个固定配置在完整96样本上扫描K1…256。校准规则在运行前声明：选择全部三个噪声档能通过的最小复块，再取达到1%校准余量的首个整数K。接受门限保持2%，因此冻结K不是校准集刚过2%的最小值；后者仅作为曲线诊断。

[frozen_selection.json](frozen_selection.json)先提交，再生成全新H55002/55003各96个；载荷55012/55013。仅执行冻结K，留出结果未用于调K、阻尼、块大小或重试选点。所有算法/SNR配对共享H/bits/基础噪声，不把四目标重放计成768个独立H；总共288个不同H，其中192属于独立留出。

## 验证与负例

234项测试通过。批量实数GEMM与固定F16模型逐位一致，小块预条件Cholesky/回代由原语构建并作FP64结构对照。另写独立逐样本实现，不导入候选Primitives，核对四个冻结任务共768批输出、491,520个FP16元素逐位一致，再独立重算FP64目标误差。

真实CLI把冻结K13改为14，即使给新契约正确SHA，也在读入候选/数值前因计划绑定不符拒绝，没有创建输出目录；见reject-K-result.json。非法FP32输入状态、非有限量、未知操作也有单元拒绝测试。未裁剪/替换NaN或Inf。

首轮审计的逆后应用误差用左复块范数，与评估器的全实块范数约差2.8e−7；保留原审计，新v2重新逐位核验并统一全实块指标，结论不变。当前证据为[independent_audit.json](independent_audit.json)。

QPSK仅作为诊断，每个SNR49,152bit；直接输出的误判bit计数候选/MMSE分别13526/13525、8211/8207、4041/4044。不是正式通信置信验收或绝对业务BER通过，不能把少数bit差异当算法改善。

![FP16可行性](figures/FP16_feasibility.png)

[PDF](figures/FP16_feasibility.pdf)、[完整校准K曲线](calibration_curves.csv)、[留出表](holdout.csv)、[机器摘要](summary.json)。

## 周期与下一步

原语工作量只是软件计数，不是实际Asim周期/流水占用。补偿、前态、Scalar系数与布局恢复都有成本；研究版点对角P使用通用GEMM，原生实现应在位一致验证后改成相应逐行MUL，不能靠无效矩阵运算提高Cube利用率。

下一步F16-4：把冻结的点预条件隐式MMSE候选放到独立工作区/build，阅读算子标准，明确每个缓冲/常量/复制/转换/同步，再构建→真实Asim/性能筛选→实际指令重建→独立同算法参考/MMSE/QPSK。只有这之后才能宣称算子可用或比较性能，不需为本候选引入FP32张量格式。

## 复现与归档

源程序/原语/参考/契约以每份契约SHA和Git版本为准；不同校准阶段按当时代码复现，不用当前源码强行运行旧hash。校准原计算aabdd8bf，留出c0bcd940，独立审计4aaf5ba3。原始输入/输出/日志固定results/fp16_implicit/，大小和SHA见[run_manifest.json](run_manifest.json)，既有结果未覆盖。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.studies.fp16_campaign \
  --contract experiments/fp16_implicit/2026-11-08-v3/holdout-contract.json \
  --sha256 b6d39e6ad8fafcee3f96b6e723aa981d2d1ea3526025ef9df75cb19e6c302275 \
  --output results/fp16_implicit/<new-holdout-id>
```

固定论文索引：[paper_archive](../../../DOCS/research/paper_archive/README.md)。旧v1/v2仅校准烟测，不升级为留出通过；纯模型思考时间不可观测，记null，阶段耗时见summary。
