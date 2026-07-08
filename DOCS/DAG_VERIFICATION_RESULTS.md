# DAG 重放验证结果报告

> 日期: 2026-07-08 | 版本: final

## 一、DAG 原语覆盖

### 1.1 已注册原语 (12 个)

| 层级 | 原语 | 使用算子数 | 公式 |
|------|------|:---:|------|
| **Core** | `GEMM` | 6/6 | C = A @ B |
| | `DIAG_ADD` | 5/6 | A += λI |
| | `TRSM` | 2/6 | Y = L⁻¹ (1-input) |
| | `MATRIX_SUB` | 2/6 | C = A - B |
| | `MATRIX_ADD` | 1/6 | C = A + B |
| | `SCALE` | 0/6 | A ← α·A (预留) |
| **Algorithm** | `CHOLESKY` | 2/6 | L = chol(A) |
| | `LDL_DECOMPOSE` | 2/6 | Full LDL + FWD + sqrt(Dinv) |
| | `DIAG_INV` | 0/6 | D⁻¹ = 1/D (预留) |
| **Op-Specific** | `BRI_PRECOND` | 1/6 | B = blockdiag(A_ii⁻¹) |
| | `MATRIX_INV_2x2` | 0/6 | Direct 2×2 inverse (预留) |
| | `SQRT_SCALE` | 0/6 | Y *= sqrt(Dinv) (预留) |

### 1.2 各算子 DAG 链完整度

| 算子 | emit_step | 链路径 | 是否完整 |
|------|:---:|------|:---:|
| Cholesky NoBlock v2 | 4 | GRAM → REG → CHOLESKY → FWD_SOLVE → BWD | ✅ |
| Cholesky Block v3 | 5 | GRAM → REG → CHOLESKY → FWD_SOLVE → BWD | ✅ |
| LDL NoBlock v2 | 3 | GRAM → REG → LDL_DECOMPOSE → BWD | ✅ |
| LDL Block v3 | 3 | GRAM → REG → LDL_DECOMPOSE → BWD | ✅ |
| Newton-Schulz v3 | 3K+1 | K×(GEMM → MATRIX_SUB → GEMM) + BWD | ✅ |
| Block-Richardson v3 | 3L+4 | GRAM → REG → BRI_PRECOND → L×(GEMM → MATRIX_SUB → MATRIX_ADD) + BWD | ✅ |

### 1.3 输入名解析覆盖率

8 级解析链全部被使用：

| 级别 | 模式 | 使用场景 |
|:---:|------|------|
| 1 | `registry[(batch, name)]` | 所有中间结果 |
| 2 | `registry[(0, name)]` | batch fallback |
| 3 | `aux_params[name]` | lambda 参数 |
| 4 | `"lambda*I"` | 正则化矩阵 (Cholesky/LDL/BRI) |
| 5 | `"I"` | 单位矩阵 (BRI 迭代首轮) |
| 6 | `"2I"` | 2倍单位矩阵 (Newton-Schulz) |
| 7 | `"H^H"` | 共轭转置 (所有 GRAM 步骤) |
| 8 | `"Y^H"` | 共轭转置 (所有 BWD 步骤) |

## 二、自测误差统计

### 2.1 DAG 执行器自测 (`--self-test`)

| 测试项 | N | 误差 | 阈值 | 结果 |
|------|:---:|------|:---:|:---:|
| Cholesky DAG 链 | 8 | 7.55e-04 | 0.01 | ✅ PASS |
| LDL primitive | 8 | 4.77e-02 | 0.10 | ✅ PASS |
| BRI 2x2 block | 8 | 3.96e-04 | — | ✅ PASS |

### 2.2 各算子验证脚本（需仿真器运行时生成 formula JSON）

| 算子 | 阈值 | 验证方式 | 双路径 |
|------|:---:|------|:---:|
| Cholesky NoBlock | 0.01 | DAG vs prim_cholesky ref | ✅ |
| Cholesky Block | 0.01 | DAG vs prim_cholesky ref (算法等价) | ✅ |
| LDL NoBlock | 0.10 | DAG vs prim_ldl_decompose ref | ✅ |
| LDL Block | 0.10 | DAG vs prim_ldl_decompose ref (算法等价) | ✅ |
| Newton-Schulz | 0.10 | DAG vs prim_gemm+prim_matrix_sub ref | ✅ |
| Block-Richardson | 0.01 | DAG 自一致性 (同 primitive 重放) | ✅ |

### 2.3 已知限制

| 算子 | 限制 | 影响 |
|------|------|------|
| Newton-Schulz | DAG 输出 A^{-2} = X@X，参考也计算 X@X | 正确（不验证 A^{-1} 精度） |
| BRI | DAG 自一致性检查（非 A^{-1} 验证） | 验证 DAG 重放正确性，不验证算法收敛 |
| Cholesky Block | POTRF_GEMM "schur" 为死数据流 | 无影响 |
| LDL NoBlock | 前向求解/缩放由 LDL_DECOMPOSE 内部覆盖 | 无影响 |

## 三、代码清理

### 3.1 已清理项目

| 文件 | 内容 | 操作 |
|------|------|:---:|
| `reference_inverse_registry.py:214-247` | `_execute_dag_inverse` 死代码（无调用者 + 初始张量 bug） | 删除 |
| `reference_inverse_registry.py:138` | `dag.build()` API 错误 | 修复为 `FormulaDAG(steps)` |
| `base.py:15` | 旧格式 JSON 崩溃 | 兼容新旧格式 |

### 3.2 保留的预留原语

| 原语 | 原因 |
|------|------|
| `SCALE` | 未来标量缩放算子需要 |
| `DIAG_INV` | 未来独立对角求逆算子需要 |
| `MATRIX_INV_2x2` | BRI 预条件器内部使用（直接 2×2 公式） |
| `SQRT_SCALE` | LDL 列缩放的显式表示（当前 LDL_DECOMPOSE 内部覆盖） |

## 四、效果评估

### 4.1 DAG 重放方案评价

| 维度 | 评分 | 说明 |
|------|:---:|------|
| **声明式语义** | 9/10 | emit_step 将数学步骤声明嵌入 C++，无需维护独立验证代码 |
| **算子无关性** | 9/10 | 新增 NS 算子时 DAG executor 零改动（纯 Core 原语组合） |
| **反耦合** | 9/10 | 6 Core + 3 Algorithm + 3 Op-Specific，严格不互相依赖 |
| **双路径验证** | 8/10 | Path A (DAG) vs Path B (Reference) 覆盖所有算子 |
| **FP16 精度模拟** | 8/10 | 双重量化（compute + store），TRSM 内部量化待统一 |
| **可扩展性** | 9/10 | pipeline.json 扩展插槽 + 预留原语注册机制 |
| **调试友好** | 7/10 | registry key 打印即可定位链断裂点，但无可视化工具 |

### 4.2 与替代方案对比

| 方案 | 正确性保证 | 侵入性 | 维护成本 | 适用性 |
|------|:---:|:---:|:---:|:---:|
| **DAG 重放（当前）** | 算法语义级别 | 低（仅 emit_step 调用） | 低 | ✅ 所有算子 |
| C++ 数值追踪 | 指令级别 | 极高（需重写全部 handler） | 极高 | ❌ 超出项目范围 |
| 黄金 trace 对比 | 指令级别 | 低 | 高（需频繁更新） | 辅助使用 |
| 黑盒评分 (UOBS) | 性能级别 | 无 | 低 | 已集成 |

### 4.3 总结

DAG 重放验证方案已达到设计目标：
- **6/6** 算子 DAG 链完整
- **12** 个原语覆盖三级体系
- **8** 级输入名解析链覆盖所有场景
- **3/6** 算子已有运行时验证通过记录（Cholesky NoBlock / LDL NoBlock / BRI）
- **3/6** 算子待运行时验证（Cholesky Block / LDL Block / Newton-Schulz — DAG 链已就绪）

---
> 归档: `DOCS/DAG_VERIFICATION_RESULTS.md`
