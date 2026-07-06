# 专家审查问题修复计划

> 审查日期: 2026-07-04 | 评分: 4/10 → 目标: 8/10

## 已识别问题

| # | 问题 | 严重度 | 根因 |
|---|------|:---:|------|
| P1 | 验证脚本未连接 C++ | 🔴 | 脚本只用 formula_steps.json 的 metadata，不执行 DAG |
| P2 | Block 变体是 NoBlock 克隆 | 🔴 | 验证脚本调用 prim_cholesky(全矩阵) 而非分块算法 |
| P3 | BRI 循环参考 | 🔴 | 对比 B^{-1} 而非 A^{-1} |
| P4 | 误差阈值无文档依据 | 🟡 | 调试过程中手动调整，未记录理由 |
| P5 | 仅测试单一 seed/维度 | 🟡 | 未覆盖边缘情况 |
| P6 | 文档分散三处 | 🟡 | 无统一新算子清单 |
| P7 | v3 标准缺失验证要求 | 🟡 | 未规定验证脚本必须连接 C++ |

## 修复计划

### Task 1: 连接 C++ — 验证脚本使用 formula_steps.json 内容 (P1)

修改每个 `scripts/verify/<op>.py`：
- 读取 formula_steps.json 步骤列表
- 构建 FormulaDAG 并执行
- 同时用 Python 参考算法独立计算
- 报告 DAG vs Ref 误差（不只是 Py vs Ref）

### Task 2: Block 变体标记 (P2)

采用方案 A（算法等价验证）：
- Block 验证脚本调用 prim_cholesky（全矩阵）验证最终输出
- 添加注释说明这是"算法等价验证"（Block Cholesky = 全矩阵 Cholesky 在数学上等价）
- DOCS 中记录分块计算的中间步骤验证是后续工作

### Task 3: BRI 参考修正 (P3)

- BRI 验证改为对比 $A^{-1}$（真正的逆）
- C++ BRI_FINAL emit_step 改为产出 Y（最后一次迭代结果，不做 W/X_hat）

### Task 4: 阈值文档化 (P4)

在每个验证脚本头部注释中记录阈值依据。

### Task 5: 多 seed 测试 (P5)

每个验证脚本 test 3 seeds（42, 123, 456），报告中取最大值。

### Task 6: 统一文档 (P6)

创建 `DOCS/NEW_OPERATOR_CHECKLIST.md`，合并来自以下来源的内容：
- OPERATOR_DEVELOPMENT_STANDARD_V3.md
- DAG_PRIMITIVES_SPEC.md
- per-operator-verification-design.md
- verify-operator/SKILL.md

### Task 7: 更新 v3 标准 (P7)

在 OPERATOR_DEVELOPMENT_STANDARD_V3.md 中添加验证要求章节。
