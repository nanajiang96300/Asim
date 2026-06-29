# BJ 数值执行模式（Python Reference）

脚本：`scripts/DeepUnfold/block_jacobi_numeric_exec_mode.py`

## 作用
- 读取 `block_jacobi` 的模型 JSON 参数（`layers/block_size/adaptive_bounds/iter_weight`）。
- 执行纯数值 BJ 迭代并评估 SE。
- 导出内部张量（`a_est/b_mat/m_half_inv/omegas/y_layers/residual_norms`）用于正确性对齐。

## 快速运行
```bash
python3 scripts/DeepUnfold/block_jacobi_numeric_exec_mode.py \
  --model-json example/block_jacobi_test_fused_p2_best.json \
  --n-sc 4 --batch 2 --trials 1 \
  --snr-db 0,5,10 \
  --out-dir result_new/bj_numeric_exec/p2_best_smoke \
  --dump-max-samples 2
```

## 输出
- `se_block_jacobi_numeric_exec.csv`
- `internal_dump/manifest.csv`
- `internal_dump/*.npz`
