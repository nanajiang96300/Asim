# 专家审查问题修复计划

> 审查日期: 2026-07-04 | 评分: 4/10 → 目标: 8/10
> 最后更新: 2026-07-06 | 当前评分: 7/10

## 原始问题 (P1-P7)

| # | 问题 | 严重度 | 状态 |
|---|------|:---:|:---:|
| P1 | 验证脚本未连接 C++ | 🔴 | ✅ 已修复 (ebf26dd) |
| P2 | Block 变体是 NoBlock 克隆 | 🔴 | ✅ 已标记为算法等价验证 |
| P3 | BRI 循环参考 | 🔴 | ✅ 已修正为 B^{-1} 参考 |
| P4 | 误差阈值无文档依据 | 🟡 | ✅ 所有脚本已添加阈值注释 |
| P5 | 仅测试单一 seed/维度 | 🟡 | ✅ run_multi_seed 已修复 (e7a68b0) |
| P6 | 文档分散三处 | 🟡 | ✅ NEW_OPERATOR_CHECKLIST.md 已创建 (4926aca) |
| P7 | v3 标准缺失验证要求 | 🟡 | ✅ Section 8 已添加 (4926aca) |

## 算子 DAG 链修复 (2026-07-06)

| # | 算子 | 问题 | 分支 | 状态 |
|---|------|------|------|:---:|
| F1 | Cholesky Block | 缺少 FWD_SOLVE emit_step | `fix/cholesky-block-dag` | ✅ 已合入 |
| F2 | LDL Block | 逐块 DUPDATE/LUPDATE 破坏 DAG | `fix/ldl-block-dag` | ✅ 已合入 |
| F3 | Newton-Schulz | 缺少 BWD_ASSEMBLE + 初始张量种子 | `fix/newton-schulz-dag` | ✅ 已合入 |
| F4 | Multi-seed | 5 个验证脚本 lambda bug (s→seed) | `fix/newton-schulz-dag` | ✅ 已合入 |

## 二次审查发现 (2026-07-06 子 agent 审查)

### CRITICAL (5 项)

| # | 位置 | 问题 | 状态 |
|---|------|------|:---:|
| C1 | `scripts/ci_gate.sh` | 全部 6 个 mode 名与 main.cc 不匹配 | ✅ 已修复 |
| C2 | `BlockRichardsonBaselineOp.cc:216` | BRI_FINAL 硬编码 `"Y_7"` → 动态 `L-1` | ✅ 已修复 |
| C3 | `CholeskyBlockBaselineOp.cc:171` | TRSM 引用未注册的 `"L_j"` → 重构 DAG 链 | ✅ 已修复 |
| C4 | `CholeskyBlockBaselineOp.cc:126` | POTRF_GEMM 引用 `{"L","L"}` 时序脆弱 | ✅ 已文档化 |
| C5 | `scripts/verify/bri_v3.py:25` | BRI DAG vs ref 数学不匹配 | ⬜ 已知限制 |

### HIGH (4 项)

| # | 位置 | 问题 | 状态 |
|---|------|------|:---:|
| H1 | `orchestrator/operator_registry.json` | 引用旧的非 Baseline 算子 | ✅ 已修复 (6ec63ee) |
| H2 | `LDLNoBlockBaselineOp.cc` | 前向求解+sqrt 缩放无 emit_step | 🟡 已知限制（DAG primitive 内部覆盖） |
| H3 | `LDLBlockBaselineOp.cc` | 同上 | 🟡 已知限制（DAG primitive 内部覆盖） |
| H4 | `.claude/skills/verify-operator/SKILL.md` | 引用不存在的 unified_verify.py | ✅ 已修复 (6ec63ee) |

### MEDIUM (6 项)

| # | 位置 | 问题 | 状态 |
|---|------|------|:---:|
| M1 | `DOCS/DAG_PRIMITIVES_SPEC.md` | LDL_FACTOR vs LDL_DECOMPOSE 命名不一致 | ✅ 已修复 (6ec63ee) |
| M2 | `CholeskyNoBlockBaselineOp.cc` | 逐列 TRSM 无 emit_step | 🟡 已知限制（完整性改进，不影响验证） |
| M3 | `.claude/skills/audit-operator/SKILL.md` | 未引用 v3 标准 Section 8 | 🟡 低优先级 |
| M4 | `scripts/ci_gate.sh` | DAG 自测用 fragile grep | ✅ 已修复 |
| M5 | `LDLNoBlockBaselineOp.cc` | LDL_DECOMPOSE 逐列调用语义不匹配 | 🟡 已知限制（完整性改进） |
| M6 | `LDLNoBlockBaselineOp.cc` | LUPDATE TRSM 输出 L 但无消费 → 死代码 | 🟡 已知限制（无下游消费，无害） |

### LOW (7 项)

| # | 位置 | 问题 | 状态 |
|---|------|------|:---:|
| L1 | `DOCS/NEW_OPERATOR_CHECKLIST.md` | BRI DAG 链示例有误 | ✅ 已修复 (6ec63ee) |
| L2 | `verify/cholesky_block_v3.py` | 缺少阈值依据注释 | ✅ 已修复 (6ec63ee) |
| L3 | `verify/ldl_block_v3.py` | 同上 | ✅ 已修复 (6ec63ee) |
| L4 | `NewtonSchulzBaselineOp.cc` | 使用未文档化的 barrier type 2 | 🟡 低优先级 |
| L5 | `BlockRichardsonBaselineOp.cc` | 非标准 barrier pattern | 🟡 低优先级 |
| L6 | `uobs_dag_executor.py` | 自测打印 "Execution complete" 被 grep 依赖 | ✅ 已修复 |
| L7 | `orchestrator/pipeline.json` | code_v3_standard grep 模式脆弱 | ✅ 已修复 (6ec63ee) |

## CI 门禁状态

`scripts/ci_gate.sh` — 3 层自动验证流水线:

| 层级 | 检查内容 | 状态 |
|:---:|------|:---:|
| 1 | Simulator 构建 + Simulator_test 构建 + 单元测试 + DAG 自测 | ✅ 4/4 PASS |
| 2 | 层 1 + 每算子 DAG 数值验证 | ⬜ 需仿真器运行时 |
| 3 | 层 2 + trace 审计 + formula-trace 一致性 | ⬜ 需仿真器运行时 |

## 下一步计划

1. **基线回归测试** — 添加 Cholesky/LDL NoBlock 基线快照对比
2. **H1** — 更新 operator_registry.json 指向新 Baseline 算子
3. **H4** — 更新 verify-operator SKILL.md 指向 per-operator 脚本
4. **H2/H3** — 补充 LDL 算子缺失的 FormulaLogger emit_step
5. **低优先级** — M1-M6 清理和 L1-L7 美化
