# Asim Benchmark Results

## 目录结构标准

```
results/
  <algorithm>/          # 算子名称 (cholesky_block, ldl_block, ...)
    run_<NNN>/          # 按时间顺序编号，3位补零
      timestamp           # ISO 8601 格式 (UTC)
      version.txt         # git commit hash 或 "unversioned-YYYYMMDD-HHMMSS"
      operator_version.md # 拷贝自 src/inverse/<algorithm>/VERSION.md
      config.json         # --config 参数的拷贝
      models_list.json    # --models_list 参数的拷贝
      trace.csv           # ONNXIM_TRACE_CSV 输出
      formula_steps.json  # ONNXIM_FORMULA_JSON 输出
      summary.json        # 机器可读的摘要 (cycles, TPS, tiles)
      log.txt             # 仿真 stderr/stdout 完整日志
```

## 硬约束
1. **每次运行必须记录 timestamp** — 精确到秒的仿真开始时间
2. **每次运行必须记录 version** — 代码版本（git hash）和算子配置
3. **每次运行必须记录修改内容** — 如果代码有改动，记录在 `changes.md`
4. **结果不可覆盖** — 每次运行创建新 `run_NNN/` 目录
