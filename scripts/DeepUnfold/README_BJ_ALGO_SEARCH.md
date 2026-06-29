# BJ 算法层高 SE 搜索（不改算子/重建链路）

脚本：`skill/scripts/skill_bj_algorithm_se_search.py`

## 目的
- 仅在 Python 算法仿真层搜索 Block-Jacobi 参数。
- 先拿到高 SE 的稳定配置，再把配置映射到算子实现与重建链路。

## 默认搜索维度
- `bj_layers`: `12,16,24,32`
- `bj_block`: `2,4,8`
- `bj_adaptive_bounds`: `true,false`

## 快速运行（u64）
```bash
python3 skill/scripts/skill_bj_algorithm_se_search.py \
  --nr 64 --nt 64 \
  --n-sc 8 --batch 4 --trials 2 \
  --snr-db 0,5,10 --pilot-len 64 \
  --bj-layers-list 8,12,16 \
  --bj-block-list 2,4 \
  --bj-adaptive-list true \
  --tag u64_algo_only_midstat
```

## 输出
- 汇总：`result_new/bj_algo_search/<tag>/bj_algo_se_search_summary.csv`
- 最优配置：`result_new/bj_algo_search/<tag>/best_config.json`
- 每个 case 的详细指标：`result_new/bj_algo_search/<tag>/<case>/metrics.json`

## 推荐基线（当前中等采样结果）
- `layers=8`
- `block=2`
- `adaptive_bounds=true`

后续将以上参数映射到 `block_jacobi_test` 算子属性，再进行算子仿真与重建对齐。
