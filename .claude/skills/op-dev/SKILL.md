---
name: op-dev
description: 按 v2.0 标准创建新 NPU 算子：生成 .h/.cc、注册 mode、创建 config、注册 scorer，自动接入 UOBS 黑盒打分和 /opt-round 优化。Use when user wants to create a new NPU operator, add an algorithm, or implement a new inverse method.
when_to_use: 当用户要求新建算子、添加新算法、实现新的求逆方法、或需要按标准流程创建算子文件时使用。参数: 算子名称和简要描述。
---

## 概述

按照 `DOCS/OPERATOR_DEVELOPMENT_STANDARD.md` v2.0 标准，引导完成新算子的 5 步开发流程。

## 5 步流程

### Step 1: 创建算子类 (.h + .cc)

文件位置: `src/operations/<Name>.h`, `src/operations/<Name>.cc`

必须包含:
- 继承 `Operation`，实现构造函数、`initialize_tiles`、`initialize_instructions`
- `#include "../FormulaLogger.h"`
- 在 `initialize_instructions()` 开头调用:
  ```cpp
  FormulaLogger::instance().set_algorithm("算子名", block_size, layers, matrix_dim);
  ```
- 每个数学步骤后调用 `FormulaLogger::instance().emit_step()`

### Step 2: 注册 mode + 创建 config

- `src/main.cc`: 添加 `else if (mode == "xxx_test")` 分支
- `example/xxx_test.json`: 创建测试配置，必填 matrix_m/nr, matrix_k/nt, batch_size, attributes

### Step 3: 注册到编排系统

- `orchestrator/operator_registry.json`: 添加条目
  ```json
  "name": {"config":"example/xxx_test.json","mode":"xxx_test",
    "source":"src/operations/X.cc","header":"src/operations/X.h",
    "nr":64,"nt":16,"dimensions":[...],"attributes":[...],"description":"..."}
  ```
- `scripts/reference_inverse_registry.py`: 用 `@register("算子名")` 注册参考逆函数

### Step 4: 编译验证

```bash
cmake --build build_asim --target Simulator -j$(nproc)
```

### Step 5: 黑盒验证

```bash
/eval-patch --operator <name> --baseline
# 检查 formula_steps.json 的 _metadata 字段
```

## UOBS 原语表

| op_type | 数学语义 |
|---------|---------|
| GEMM | C = A @ B |
| DIAG_ADD | A ← A + λI |
| CHOLESKY | L = chol(A) |
| TRSM | X = L⁻¹B |
| DIAG_INV | 对角/块对角求逆 |
| MATRIX_INV_2x2 | 2×2 直接求逆 |
| MATRIX_SUB | C = A - B |
| MATRIX_ADD | C = A + B |
| SCALE | A ← αA |

## 参考

- 标准文档: `DOCS/OPERATOR_DEVELOPMENT_STANDARD.md`
- 参考实现: `src/operations/CholeskyInvOp.cc`, `src/operations/BlockRichardsonOp.cc`
- 现有注册表: `scripts/reference_inverse_registry.py`
