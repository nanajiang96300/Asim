# 六类默认验收入口

统一任务：H/Y/λ→MMSE估计；Nr64/Nt16/RHS8、batch96、λ0.1、Rayleigh seed42。FP16存储、FP32临时计算、显式舍入/次正规保留数值模型不等于硬件认证。

| 家族 | Model/算子id | 参数 |
|---|---|---|
| Cholesky NoBlock | cholesky_explicit_v1 | 右看分解、三角求解 |
| LDL NoBlock | ldl_explicit_v1 | 单位L、D及行加权求解 |
| Cholesky Block | cholesky_block_explicit_v1 | 实块4 |
| LDL Block | ldl_block_explicit_v1 | 实块4 |
| Newton–Schulz | ns_corrected_v1 | K16，X0=I/trace(real_block_A)，最终X乘右端 |
| Block-Richardson | bri_explicit_v1 | K192，实块2，ω0.5，x0=0 |

源码位于src/inverse，入口注册见orchestrator/codex_operators.json及verification/operators/catalog.py，同算法参考位于verification/reference。example/给出六个Model请求。完整C++树保留历史实现及Merge/HB变体，以满足原factory构建依赖；存在源码不代表旧版本已通过重建。

K是本部署样本的冻结参数，不代表所有信道最小K。特别是旧显式BRI在强相关CDL-B上的有限精度问题不能由本轮Rayleigh冒烟升级；可选`--operator bri-hb`使用另一固定HB候选，不计入默认六类测试结果。
