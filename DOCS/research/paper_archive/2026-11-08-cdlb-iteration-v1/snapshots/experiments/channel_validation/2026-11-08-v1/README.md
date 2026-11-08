# CDL-B输入域、迭代次数与精度研究 v1

日期：2026-11-08。**实验已执行不等于全部通过**：真实CDL-B输入域的21组中，6组通过、1组数值质量失败、14组重放被拒绝。整体`CDLB_QUALITY_NOT_VERIFIED`，score=null。未修改具体算子、原精度参考、固定runtime或旧实验。

## 范围与方法

负责人确认信道编码不是求逆/均衡算子精度验证的必要项。本轮保持三路径数值验证和未编码QPSK，优先真实CDL-B、K与误差关系；不将FEC/BLER缺失作为本阶段阻塞。协议：[CHANNEL_DOMAINS.md](../../../DOCS/research/CHANNEL_DOMAINS.md)。

旧`scripts/channel/cdl.py`是指数相关Kronecker近似，本轮未将它当作真实CDL-B。本轮使用Sionna2.0.1 CDL-B，3.5GHz、100ns、UT4×4/BS8×8、单极化V/omni、半波长、默认朝向、静止单时刻、基带偏移+1.2MHz。23簇路径系数/时延保存于dataset；独立傅里叶转换与Sionna最大差约9.16e-16。每H归一||H||F²=16；对应瑞利K曲线也逐H归一。不是完整OFDM、移动轨迹、多用户或任意频点认证。

batch96、24核、Nr64/Nt16/RHS8，FP16存储、显式逐指令FP32计算；Es/N0 0/10/20dB、λ1/0.1/0.01。七个历史Asim已验收的显式新入口，NS K16、固定BRI K192/omega0.5/实块2/x0=0，直接法/实块4/Merge参数不变。这些配置不是最优K或负责人所指历史最优BRI。

使用已通过真实Asim的数据无关实际程序，显式外层CDL输入域与历史template_task分别保存，H/Y及量化程序输入同时替换，逐帧重检真实指令。没有新增Asim或虚构CDL周期。CPU信道生成在任务周期外；Gram/预条件/初始化/恢复/DMA仍全部执行并计原完整任务周期。原任务header.channel=rayleigh描述原模板来源，不冒充新的CDL标签。

## 重建结果

| 入口 | 0dB | 10dB | 20dB |
|---|---|---|---|
| NS K16 | 通过，最大误差0.1919% | 实块象限拒绝 | 实块象限拒绝 |
| Cholesky | 通过，0.3866% | 实块象限拒绝 | 实块象限拒绝 |
| LDL | 通过，0.3795% | 实块象限拒绝 | 实块象限拒绝 |
| BRI K192 | 数值失败，相对误差1419.29 | FP16溢出 | FP16溢出 |
| Cholesky Block | 通过，0.3723% | 实块象限拒绝 | 实块象限拒绝 |
| LDL Block | 通过，0.3439% | 实块象限拒绝 | 实块象限拒绝 |
| Cholesky Merge | 通过，0.1979% | 实块象限拒绝 | 实块象限拒绝 |

通过组均完成独立同算法逐FP16位一致、≤2%任务误差及相对QPSK BER/SER/EVM/置信门禁。四象限拒绝发生在该batch实际输出与同算法参考位一致之后的复数布局解码；不能因此宣称后续未完成batch通过。BRI的0dB输出全部可重建且位一致，仍被数值门禁拒绝，证明一致性与质量必须分开。没有放宽门限、跳过拒绝或改算法来通过。

留出H种子44001/44002，每组192独立H，每H两帧46001/46002；每完整组98,304bit、49,152符号。曲线H种子45001，96个不同H，和实际重建留出分离。跨算法/SNR复用H/bits/基础噪声；不能累计为4032独立H。

21组×4帧覆盖全部计划场景。实际完整成功返回29帧，即2784个完整batch，其中1帧属于最终被拒绝的组，不计该组通过。原result字段`batch_attempts=8064`实际表示计划batch槽位（执行器按帧预记96），**不是8064个batch全部执行完成**；部分拒绝帧的实际已执行batch数未记录。本报告显式纠正口径，不覆盖原件。重放耗时325.79秒；K曲线276次独立模型调用耗时173.14秒；原实验215项测试通过，随后计数口径修正回归216项通过。纯模型思考时间不可观测，记null。

## 独立复核与问题定位

独立报告器复核7个完整组、28帧、688,128次bit判决（包含失败的BRI组，但不签发其质量通过），重算完整组的FP64 MMSE，用最小距离重新判决并复核bit/符号/混淆矩阵/每H EVM及CP界；被拒绝组不混入“通过”统计。见[审计](audit.json)、[逐组质量表](quality.csv)、[诊断](diagnostics.json)。诊断另对相同H/Y的独立参考查找象限偏差/非有限计算，并计算固定BRI的高精度定常迭代谱半径；不改变受信参考或完整门禁。NS在λ0.1的独立参考首个象限偏差2.0606%，λ0.01首例12.5259%，均超过2%布局门限。固定BRI的192个留出H中，λ1/0.1/0.01分别有1/191/191个名义FP64谱半径≥1，最大值1.0419/2.4395/2.7906，支持当前定常参数的收敛性问题。

## K—误差曲线

使用独立同算法FP16/FP32和FP64模型，在固定96-H画图集上枚举冻结网格：NS [1,2,4,6,8,10,12,14,16,20,24,32]，BRI [1,2,4,8,16,32,64,96,128,192,256]；三个SNR和两个信道域。纵轴是完整均衡输出相对FP64 MMSE误差，**不是逆矩阵逐元素误差**；图中最大值和中位数帮助观察尾部、收敛和精度平台。FP16该曲线采用参考模型返回的左半复输出，不替代全实块象限及实际重建门禁。

![CDL-B K与误差](figures/cdlb_K_error.png)

![瑞利 K与误差](figures/rayleigh_K_error.png)

原始值与失败：[curves.csv](curves.csv)；有限网格首个过2%点：[curve_summary.json](curve_summary.json)。超过100的误差绘图截顶为上箭头，非有限运算用x，CSV不截断。CDL-B的NS在20dB时K16最大误差9.3038%、K32仍为9.2940%，而FP64已收敛，说明此处单纯加K不能越过精度平台；BRI固定参数无任何测试K在全部96个H上达到2%。图是描述性算法试验，不是算子性能或最小整数K认证；不根据当前留出结果调K后再自验。

## 归档、清理与后续

源代码/版本：通道与执行器6cc7a2ec，冻结契约ab30bda7，报告器c3fd814a；固定runtime及Sionna/Torch/NumPy版本和源码SHA在[contract.json](contract.json)，数据/实际输入输出/日志的大小与SHA在[run_manifest.json](run_manifest.json)。原件固定`results/channel_validation/2026-11-08-v1/`。论文固定检索[paper_archive](../../../DOCS/research/paper_archive/README.md)，旧版本不覆盖。

已清理约3.02GB可重新下载的pip缓存，逐文件清理原因/大小/SHA见[cleanup.json](cleanup.json)及[复核台账](cleanup-verification.json)；未删除安装依赖、运行器或历史研究原始数据。归档依赖的旧trace/input不是“过时即无用”，仍保留。

下一开发顺序：①根据谱半径与曲线确定固定BRI的稳定参数/预条件候选；②研究NS/直接法在病态CDL下的精度/复布局稳定性，保留独立象限检查；③先独立算法校准，再冻结新候选和全新留出，经Asim→实际重建重新验收。增加K不能自动解决发散、溢出或精度平台；不将本轮固定配置的失败推广成算法永久不可用。

## 复现与版本边界

原v1计算使用6cc7a2ec执行器及契约SHA `c697b50a8c6dd10f316e7a699e6a10b0b25e5ee98cea3d5efd6549b58b274f21`；在恢复相同Python依赖和runtime原件的隔离工作区，使用新输出目录依次执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.planning.channel_study \
  --contract experiments/channel_validation/2026-11-08-v1/contract.json \
  --sha256 c697b50a8c6dd10f316e7a699e6a10b0b25e5ee98cea3d5efd6549b58b274f21 \
  --output results/channel_validation/<new-id> --stage dataset
# 同一新目录依次 --stage replay、--stage curves；质量失败时replay返回1并保留结果。
```

此后ec3d9305只修正计划槽位/完整帧/batch计数命名，新增回归测试；没有重跑或覆盖本轮数值原件。当前代码直接运行旧契约将正确拒绝源码hash变化；新实验必须重新冻结契约，不能改旧hash以伪装原版本。报告器版本及输入摘要保存在原位report/manifest.json。

清理复核修正：原cleanup.json的deleted数组实际为清理前候选清单，包含1份pip保留的selfcheck元数据。原件保留，cleanup-verification.json记录清理前清单中672条文件路径已消失、1份保留；pip自身报告669个缓存制品，两者计数单位不同，避免将候选清单当全部已删除。
