# 独立算法选K烟测：2026-11-08-v1

实现提交：`d5903d6f`。状态：算法离线筛选，不是新算子或通信验收。
固定论文检索：[paper_archive](../../../DOCS/research/paper_archive/README.md)。

## 条件与结果

瑞利、batch24（24核倍数）、Nr8/Nt4/RHS2；QPSK单位符号能量，H缩放1/sqrt(Nr)。
λ/噪声方差分别0.1、1.0；校准seed42、留出seed123。
FP16存储/FP32逐操作模型，与现有独立参考一致；任务输出相对MMSE门限2%，每个batch都必须通过。
这只是低成本功能烟测，不代表默认batch96/Nr64/Nt16任务的最佳K，也不代表正式通信门限。

| 算法 | 冻结K网格 | 最小校准可行K | 留出最差任务误差 | 脚本墙钟 |
|---|---|---:|---:|---:|
| NS，I/trace(real A) | 1..16 | 7 | 1.8011011% | 1.770s |
| BRI，实B2/ω0.5/x0=0/预计算递推 | 1..64 | 28 | 1.2067086% | 10.181s |

两者均为 `ALGORITHM_SCREENED`，`operator_verified=false`、`communication_status=NOT_RUN`、`score=null`。
校准集覆盖了从1到选定K的全部整数，故仅能声明本校准集内最小整数K。
NS的选定点接近2%门限，应在更大留出范围评估稳定性，不能直接作为生产配置。
两算法合计164次整batch参考调用、3936条逐batch误差记录；留出集只测试选定K。
墙钟为算法脚本运行时间，不是Agent纯思考时间；后者不可观测，结果为null。

## 重跑与证据

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.planning.iterations \
  --contract experiments/quality_planning/2026-11-08-v1/ns_corrected_v1.json \
  --output results/quality_planning/<new-id>/ns
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.planning.iterations \
  --contract experiments/quality_planning/2026-11-08-v1/bri_explicit_v1.json \
  --output results/quality_planning/<new-id>/bri
```

`ns/`、`bri/`保存本次原始小结果的Git副本。每份结果包含所有K/场景逐batch误差、
任务/输入身份、独立模型参数、源码SHA-256和Python/NumPy版本。
[run_manifest.json](run_manifest.json)绑定副本与原件路径/大小/hash。
原件在`results/quality_planning/2026-11-08-v1/`，未移动/覆盖旧数据。

验证：全部159单元测试通过，其中新增8项覆盖非单调、网格最小性、最差batch、留出失败不调参、
无可行K、未知指标/信道/预算拒绝与真实FP16参考一致性。
本次不构建、不运行C++ Asim、不改算子；既有固定runtime逐文件hash另行核验。

后续：参数计划与执行器身份绑定→重建结果上的正式瑞利SE/BER→默认batch96/多SNR留出验收→CDL-B。
