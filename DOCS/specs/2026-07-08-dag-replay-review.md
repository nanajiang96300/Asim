# DAG 重放流程架构评估报告

> 审查日期: 2026-07-08 | 子 agent 全面架构审查

## 总体评价

DAG 重放方案是适合本项目的正确架构选择。3 层原语分级（Core/Algorithm/Operator）结构良好，C++ → JSON → Python 全链路清晰。发现的问题均为**实现不完整**而非架构缺陷，均可在现有架构内修复。

## 新发现问题 (12 项)

### CRITICAL (3 项)

| # | 位置 | 问题 | 建议修复 |
|---|------|------|------|
| C1 | `newton_schulz_v3.py:28` | NS 验证对比 A^{-2} vs A^{-1}（代数不匹配）。DAG 输出 X_{K-1}@X_{K-1} ≈ A^{-2}，但 Python 参考是 X ≈ A^{-1} | Python 参考添加最终 GEMM: `A_ref = fp16(prim_gemm(X, X))` |
| C2 | `bri_v3.py:22` | BRI 验证三重不匹配：硬件算 X_hat=Y@H@Yin，DAG 记录 Y@Y，参考用 B^{-1}。三者是不同数学量 | 从验证套件中移除 BRI，或扩展 DAG 匹配实际硬件链路 |
| C3 | `reference_inverse_registry.py:138` | `dag.build(data["steps"])` API 错误 — FormulaDAG 无 build() 方法。静默返回 None，回退到 per-algorithm 函数 | 改为 `dag = FormulaDAG(data["steps"])` |

### HIGH (2 项)

| # | 位置 | 问题 | 建议修复 |
|---|------|------|------|
| H1 | `uobs_dag_executor.py:312` | 初始张量为所有 batch 注册相同值（实际硬件每个 batch 有不同 H 矩阵） | 支持 per-batch 初始张量或仅接受单 batch DAG |
| H2 | `newton_schulz_v3.py:37` | `A_dag = A_ref` fallback 将 DAG 失败静默转为 PASS（error=0.0） | 删除 fallback，让 None 传播为错误 |

### MEDIUM (4 项)

| # | 位置 | 问题 |
|---|------|------|
| M1 | `uobs_dag_executor.py:370` | TRSM 2-input pass-through 是死代码（无算子触发） |
| M2 | `uobs_dag_executor.py:236` | DIAG_INV/MATRIX_INV_2x2/SQRT_SCALE 注册但从未被 emit |
| M3 | `CholeskyBlockBaselineOp.cc:126` | POTRF_GEMM shapes 标注 {B,B} 但实际矩阵是 {U,U} |
| M4 | `uobs_dag_executor.py:436` | 自测仅覆盖 6/12 原语 |

### LOW (3 项)

| # | 位置 | 问题 |
|---|------|------|
| L1 | `_base.py:15` | load_dag() 在旧格式 JSON 上崩溃（list 而非 dict） |
| L2 | `CholeskyNoBlockBaselineOp.cc:145` | 逐列发射 U 次冗余 CHOLESKY 步骤 |
| L3 | `reference_inverse_registry.py:215` | `_execute_dag_inverse` 死代码 + 额外 bug |

## 方案对比结论

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|:---:|
| **DAG 重放**（当前） | 算子无关、无需仿真器、分离关注点 | 迭代方法存在近似间隙 | ✅ 适合本项目 |
| C++ 数值追踪 | 消除抽象间隙 | 侵入性太强、需重写全部指令处理 | ❌ |
| 黄金 trace 对比 | 简单、覆盖所有指令 | 脆弱、仅能做回归测试 | 辅助使用 |

## 评分: 7/10

架构设计正确，但 BRI/NS 验证存在代数不匹配，需修复后方可达到 9/10。

> 审查完整报告由子 agent 生成，详见对话记录。
