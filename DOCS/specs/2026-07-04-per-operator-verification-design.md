# Per-Operator Verification Design

> 日期: 2026-07-04 | 取代: 通用 DAG 全量验证方案

## 目标

每个算子拥有专用重放脚本，验证 FormulaLogger 声明链的数值正确性。

## 架构

```
/verify-operator <name>
  │
  ├── Step 1: /audit-operator <name>
  │     └── 审计 C++ 指令 ↔ FormulaLogger 声明一致性
  │
  ├── Step 2: 查找 scripts/verify/<name>.py
  │     存在 → 运行验证
  │     不存在 → 报错：请创建验证脚本
  │
  └── Step 3: 输出 {error, status, intermediates}
```

## 验证脚本模板

```python
# scripts/verify/cholesky_noblock_v2.py
"""Cholesky NoBlock v2 numerical verification."""

import json, numpy as np
from uobs_dag_executor import prim_gemm, prim_cholesky, prim_trsm

def verify(formula_path):
    """Replay FormulaLogger steps and compare against numpy.linalg.inv.
    
    Returns: {"error": float, "status": "PASS"|"FAIL"}
    """
    with open(formula_path) as f:
        data = json.load(f)
    
    steps = [s for s in data['steps'] if s['batch'] == 0]
    
    # Step 1: Generate random H, compute reference
    U, M = 16, 64
    H = (np.random.randn(M, U) + 1j * np.random.randn(M, U)) / np.sqrt(2)
    A = H.conj().T @ H + 0.1 * np.eye(U)
    A_ref = fp16(np.linalg.inv(fp16(A)))
    
    # Step 2: Replay GRAM → REG → CHOLESKY → TRSM → BWD
    G = prim_gemm(H.conj().T, H)        # GRAM
    A_reg = prim_diag_add(G, 0.1)       # REG
    L = prim_cholesky(A_reg)            # POTRF
    Y = prim_trsm(L)                    # FWD
    Ainv = prim_gemm(Y.conj().T, Y)     # BWD
    
    # Step 3: Compare
    error = np.linalg.norm(fp16(Ainv) - A_ref) / max(np.linalg.norm(A_ref), 1e-15)
    return {"error": float(error), "status": "PASS" if error < 0.01 else "FAIL"}
```

## 已有算子的验证脚本

| 算子 | 验证脚本 | 重放方式 |
|------|---------|---------|
| Cholesky NoBlock | verify/cholesky_noblock_v2.py | prim_gemm + prim_cholesky + prim_trsm + prim_gemm |
| LDL NoBlock | verify/ldl_noblock_v2.py | prim_gemm + prim_ldl_decompose + prim_gemm |
| Cholesky Block | verify/cholesky_block_v3.py | prim_gemm + prim_cholesky + prim_trsm + prim_gemm (block) |
| LDL Block | verify/ldl_block_v3.py | prim_gemm + prim_ldl_decompose + prim_gemm (block) |
| Newton-Schulz | verify/newton_schulz_v3.py | prim_gemm × K + prim_matrix_sub × K (纯核心原语) |
| BRI | verify/bri_v3.py | prim_bri_precond + prim_gemm × L + prim_matrix_sub × L + prim_matrix_add × L |

## 与通用 DAG 方案的对比

| 维度 | 通用 DAG | 专用验证脚本 |
|------|---------|------------|
| 每算子改动 | emit_step 命名必须一致 | 一个验证脚本 |
| 新算子开销 | 调试 DAG 链 + 注册原语 | 写一个脚本（复用原语库） |
| 表达式自由度 | emit_step 输出名必须链接 | 脚本中任意组合原语 |
| 迭代结构 | DAG 需特殊处理 | Python for 循环 |
| 调试难度 | 追溯 DAG dispatch | 单步调试 Python |
