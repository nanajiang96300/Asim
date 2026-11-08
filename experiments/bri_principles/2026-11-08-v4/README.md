# 从原始公式重新验证BRI：真实CDL-B独立Python仿真

日期：2026-11-08。本实验验证数学算法，不是Asim、指令重建、周期或实机验收。算子、原受信参考和固定runtime均未修改。输入章节原文按SHA保存在[前序目录](../2026-11-08-v1/source_chapter.tex)，勘误与完整推导见[BRI_PRINCIPLES.md](../../../DOCS/research/BRI_PRINCIPLES.md)。

## 首先修正变量定义

设A=HᴴH+λI，D=blockdiag(Ajj)，P=D⁻¹，T=PA。正确的两种形式是：

\[
X_{k+1}=X_k+\omega_kP(I-AX_k),\qquad X_0=0;
\]
\[
Y_{k+1}=Y_k+\omega_k(I-TY_k),\qquad Y_0=0,\quad X_k=Y_kP.
\]

章节第64行把B定义为P，却在第69行使用I−BY并在第74行把Y当A⁻¹。按字面会逼近P⁻¹=D。标量A=2、P=0.5时，错误式收敛到2，真正的A⁻¹是0.5。本文保留原章另立勘误，不覆盖原报告。

## 为什么加K可能变差

固定ω迭代的Eₖ₊₁=(I−ωPA)Eₖ，收敛要求ρ(I−ωPA)<1；正定条件下0<ω<2/λmax(PA)。上一轮`bri_explicit_v1`固定ω=0.5、实块2，不等于章节的复块Chebyshev求逆。当λmax≥4时该配置不保证收敛。稳定方法也可能因有限精度平台/非正规矩阵的瞬态而出现误差波动，不保证每步的普通Frobenius误差单调。

Chebyshev区间必须包含整个预条件谱。旧脚本中的上界截断/下界提高可能漏掉谱端点；自然顺序的逐根权重乘积也可能放大中间舍入误差。新版本用稳定三项递推，并通过独立根权重和谱多项式交叉核查。区间来自每样本精确谱外扩1%，是诊断上限配置，有成本，不能当成免费NPU谱估计。

## 冻结实验

- 真正Sionna2.0.1 CDL-B；3.5GHz、100ns、静态uplink、UT4×4/BS8×8、单极化V/omni、半波长间距、+1.2MHz单频点；每H归一||H||F²=16。只覆盖声明的静态矩阵域。
- 新H种子48001/48002，每种96矩阵，合计192个独立随机相位实现；Nr64/Nt16/RHS8。非此前K图或重建输入；本轮是固定样本描述性研究，不把样本当独立地理布局。
- Es/N0 0/10/20dB，复噪声及正则化λ1/0.1/0.01，每流Es=1，未编码QPSK。相同H/bits/基础噪声跨算法和SNR配对。
- 复块2/4/8；K枚举全部整数1…256。比较fixed_half、stable_stationary、chebyshev_X、chebyshev_Y_recover，共108个配置。
- FP64、NumPy complex64主机FP32，以及half_storage矩阵操作后FP16存储/complex64计算。半精度模型的小块Cholesky和回代为FP32后存半精度，不是全FP16运算，也不是Ascend逐指令模型。
- 从v3沿用的half_storage**显式采用上三角Gram/块逆与共轭镜像存储**；这是新算法存储设计，不代表原算子已有该行为。迭代X/Y不强制对称，不clip/nonfinite替换。FP64和FP32不使用这一半精度专属策略。

参考为原始H/Y构建的FP64 solve(A,I)与MMSE solve(A,HᴴY)。另对量化系统Amodel求精确逆，区分迭代误差与输入/Gram量化误差。报告真正的逆矩阵误差、I−AX残差、任务输出误差和BER。当前描述性过线定义为全部192样本有限、最大逆误差与任务误差均≤2%，不是旧SE avg_rel_error≤1%或正式业务门限。部分样本溢出时`*_max`仅为有限子集最大值，failed_channels另报，不能作为全样本上界或通过。

## 旧仿真可信度

核心T=PA、Y0=0、YP恢复在`scripts/aaa/bj_inverse.py`中是正确的。单样本、相同显式包谱区间的FP64交叉检查表明，小K结果与新推导相符；但K32时根权重顺序出现明显舍入偏离。旧脚本默认adaptive下界floor与FP16路径仍不能由此认证。精确结果见[old_helper_check.json](old_helper_check.json)。

`eval_bri_cdlb_test.py`使用Kronecker近似CDL与λ=noise*Nt、含上界截断；若干旧FP16脚本还含complex128中间计算，另有finite_quantize裁剪/替换非有限值。旧DAG自一致不等于逆矩阵正确。本次没有自动改写这些历史脚本或升级它们的结论。

## 结果与图表

完整扫描与独立审计已结束，以本目录summary、curves、audit及下方图表为准。曲线纵轴明确区分逆矩阵误差、残差和MMSE任务误差；不能把它们改名为SE。

![逆矩阵误差](figures/inverse_error_max.png)

[任务输出误差图](figures/task_error_max.png)、[逆残差图](figures/residual_max.png)、[原始曲线CSV](curves.csv)、[逐配置摘要](summary.json)、[独立谱审计](independent_audit.json)。图中相对误差超过100截顶，x为首次非有限样本；CSV保留失败数和值，不用截顶后的点发分。

## 失败尝试与版本

v1在第一配置结束导出时因NumPy布尔JSON序列化失败；v2修复该工程问题后完成8配置，朴素GEMM→逐元素FP16存储未满足Hermitian检查，拒绝而非强行修补。原输入/日志/配置保留，v3明确改变half_storage的结构存储设计后重跑。v1/v2不签发完整数值结论。v3在88个配置后遇到FP32正定/Hermitian模型拒绝，保留部分结果。v4不再因独立模型配置被拒绝而丢失其余场景，逐配置保存状态和快照；拒绝配置明确记失败，未修补它的矩阵。当前算法代码9692f34c，v3契约与原文SHA见[contract.json](contract.json)。

复现必须恢复冻结版本/依赖及原文，使用新目录，不覆盖原件：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.studies.bri_campaign \
  --contract experiments/bri_principles/2026-11-08-v4/contract.json \
  --sha256 504134075fc0a53cdfbddbd2c2cf59602b1a08f14464f2b61af61cfb41dda2e3 \
  --output results/bri_principles/<new-id>
```

原始CIR、输入、逆矩阵快照、完整指标/日志留在results/bri_principles/专用目录，[run_manifest.json](run_manifest.json)记录大小/SHA。论文检索固定[paper_archive](../../../DOCS/research/paper_archive/README.md)。

下一步先冻结正确变量/恢复、复块与实块布局、谱界来源、精度和逐操作舍入边界，选独立校准/留出集；然后才把有希望的算法映射成新Asim候选。Python通过不能自动修复旧算子或换取原周期表排名。


## 本轮完成结果

已完成108个配置的状态覆盖、27,648个K点；Python算法研究耗时440.96秒（非Agent思考时间）。225项单元测试通过。
共有2个系统模型被预检拒绝，影响8个方法配置，未改矩阵或放宽检查。其余配置完成数值迭代，但不是每个配置都满足质量门限。

### FP64 Chebyshev-X的首个达标K

| 复块大小 | 0dB | 10dB | 20dB |
|---|---:|---:|---:|
| 2 | 7 | 21 | 51 |
| 4 | 6 | 13 | 28 |
| 8 | 5 | 10 | 19 |

该表枚举1…256的全部整数，但仅针对这192个固定H、精确包谱Oracle和本轮2%逆/任务门限，不是总体最少K或Asim可部署配置。

### 显式half_storage设计的首个达标K

| 复块大小 | 0dB | 10dB | 20dB |
|---|---:|---:|---:|
| 2 | 7 | 32 | 未达标 |
| 4 | 6 | 14 | 未达标 |
| 8 | 5 | 11 | 未达标 |

半精度只对Gram/块逆作显式上三角镜像，迭代仍可能因误差平台而未达标；FP64方法有效不能代替FP16/FP32模型验收。

### 独立审计与控制实验

独立谱多项式审计覆盖216份FP64快照、41,472个逆矩阵，最大相对差7.561e-13。审计用独立块solve和Hermitian谱分解直接求pK，没有调用迭代器。

相同H上的固定ω0.5名义谱比较（此处不执行旧算子）：

| λ | 预条件器 | ρ最大值 | 不稳定H数/192 |
|---|---|---:|---:|
| 1.0 | complex_block2 | 0.8443 | 0 |
| 1.0 | contiguous_real_block2 | 1.0462 | 2 |
| 0.1 | complex_block2 | 1.2230 | 20 |
| 0.1 | contiguous_real_block2 | 2.4742 | 191 |
| 0.01 | complex_block2 | 1.5082 | 148 |
| 0.01 | contiguous_real_block2 | 2.8032 | 192 |

这直接区分了实块布局/复块定义和ω配置，不把某个固定实现的失败泛化为BRI原理无效。

配对MMSE BER见mmse_ber.json；候选BER在curves.csv中。本研究没有正式通信置信或绝对BER门禁，不能用BER接近替代逆误差验收。

### 有限精度限制与后续选择

2个被拒绝的系统为20dB下复块4/8的朴素FP32 Gram模型；它们各影响4种方法，共8配置/2048个K状态条目，没有启动对应迭代。实际执行100个方法配置的迭代，拒绝条目不作为数值测量或算法失败证明。不能把全研究的108配置状态覆盖写成108数值通过。

half_storage的量化系统精确逆本身，相对原始A⁻¹的最大误差在0/10/20dB分别约0.0763%/0.5533%/4.9739%。即使完美求解量化系统，20dB也无法满足当前2%逆误差目标；因此应先研究Gram/正则化、残差与迭代状态的精度，再调K，不能靠多迭代或裁剪“修复”这一误差。

稳定三项Chebyshev需要额外前态/动量操作，精确谱诊断也有成本；这份Python结果不继承旧四事件/低周期统计。后续先比较可负担的包谱界、明确的Hermitian计算/存储和混合精度，再冻结新候选与独立留出任务。
