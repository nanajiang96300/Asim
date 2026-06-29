---
name: eval-patch
description: 评估算子代码补丁：黑盒打分，返回 Score/Cycle/Cube%/Error。Use when user asks to evaluate a patch, run baseline, check operator score, or verify SE.
when_to_use: 当用户需要评估补丁效果、跑基线测试、检查算子评分、或验证 SE 数值正确性时使用。
---

## 概述

对传入的算子代码补丁执行完整评估闭环：
`git stash → apply patch → cmake → Simulator → formula_steps.json + trace.csv → UOBS Scorer → Score`

支持 --baseline 模式直接评估当前代码。

## 参数

- `--operator <name>`: 算子名称（必填）
- `--patch <path>`: 补丁文件路径
- `--baseline`: 评估当前代码（不应用补丁）
- `--nr <int> --nt <int>`: 覆盖默认维度

## 执行

1. 从 `orchestrator/operator_registry.json` 读取算子 config/mode/nr/nt
2. 评估：
   - 补丁: `bash scripts/evaluate_operator.sh --patch <p> --config <c> --mode <m> --nr <nr> --nt <nt>`
   - 基线: `bash scripts/evaluate_operator.sh --baseline --config <c> --mode <m> --nr <nr> --nt <nt>`
3. 解析 JSON，展示 Score / Status / finish_cycle / cube_util / rel_error
4. 若 BUILD_FAILED 或 SIM_FAILED：展示 stderr 最后 5 行 + 修复建议
