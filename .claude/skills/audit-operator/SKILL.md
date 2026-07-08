# Audit Operator — 算子正确性审查

## 用途

写完一个 NPU 算子后，自动审查 C++ 指令序列是否与数学公式 / Python 参考严格对应。

## 调用方式

```
/audit-operator <operator_name> [--python-ref <path>] [--design-doc <path>]
```

- `<operator_name>`: 算子名称，如 `cholesky_noblock_v2`、`ldl_noblock_v2`、`block_richardson`
- `--python-ref`: Python 参考实现路径（默认 `scripts/algo/<name>.py`）
- `--design-doc`: 设计文档路径（默认 `DOCS/specs/*<name>*design.md`）

## 审查流程

### Step 1: 定位代码

从 `orchestrator/operator_registry.json` 查找算子对应的 C++ 文件路径，或直接在 `src/inverse/<algorithm>/` 下搜索。

### Step 2: 加载公式基准

优先级：
1. Python 参考实现（`scripts/algo/`）— 如果存在，作为公式的权威表达
2. 设计文档中的数学推导（`DOCS/specs/`）
3. 论文 `DOCS/Ascend_new.tex` 中的算法章节

### Step 3: 逐项审查

#### 3.1 指令类型检查
每条 SCALAR 指令的 opcode 是否与公式中的数学操作匹配：
- 乘法 → `SCALAR_MUL`
- 除法 → `SCALAR_DIV`
- 减法 → `SCALAR_SUB`
- 开方 → `SCALAR_SQRT`
- 加法 → `SCALAR_ADD`

#### 3.2 运算数检查
每条指令的 `src_addrs` 是否与公式中的操作数对应：
- 二元操作（MUL/DIV/SUB/ADD）必须有 2 个源操作数
- 一元操作（SQRT）必须有 1 个源操作数
- 源操作数的 SPAD 基地址必须在之前被写入或 MOVIN 加载

#### 3.3 循环结构检查
- 外循环（j/c 列循环）边界是否正确（0..U-1）
- 内循环（k 累加循环）边界是否正确（0..j-1 或 c..i-1）
- `SCALAR_SUB` 是否存在以执行 Schur complement 减法

#### 3.4 恒等元合成检查
当公式需要常数 1.0（如 `1/L[c,c]`、`1/D[j]`），检查是否通过 `DIV(Reg/Reg)` 合成 unity 而非直接用 `addr_Reg` 当作 1。

#### 3.5 公式日志覆盖检查
`FormulaLogger::emit_step()` 是否覆盖了所有关键数学阶段：
- GRAM（GEMM）
- REG（DIAG_ADD）
- 每列分解（CHOLESKY/TRSM/DIAG_INV）
- 后向装配（GEMM）

#### 3.6 结构一致性检查（Cholesky vs LDL）
如果存在成对算子（如 Cholesky-NoBlock + LDL-NoBlock）：
- SPAD 布局是否对应（同名区域是否在相同偏移）
- Barrier 位置是否一致
- Phase 划分是否相同

#### 3.7 DAG 链完整性检查（v3 标准 Section 8）
依据 `DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md` Section 8 验证要求：
- emit_step 输出名 ≡ 下游输入名（DAG 链连通性）
- 链是否形成完整路径：初始输入 → ... → "Ainv"
- 验证脚本 `scripts/verify/<op_name>.py` 是否存在
- 验证脚本是否正确使用 `load_dag()` + `compute_error()` + `run_multi_seed()`

### Step 4: 输出审查报告

格式：

```
## 审查报告: <operator_name>

### 基准来源
- Python 参考: <path> (存在/不存在)
- 设计文档: <path> (存在/不存在)

### 审查结果

| ID | 行 | 严重度 | 描述 |
|----|-----|--------|------|
| XX | NNN | HIGH/MEDIUM/LOW | 具体问题描述 |

### 严重度定义
- CRITICAL: 导致数值完全错误的缺失步骤
- HIGH: 操作数、opcode 或循环边界错误
- MEDIUM: 结构不一致，但不影响正确性
- LOW: 注释、命名建议

### 结论
- 当前状态: PASS / FAIL
- 已修复: N 项
- 剩余: M 项
```

## 示例

```
/audit-operator cholesky_noblock_v2
/audit-operator ldl_noblock_v2 --python-ref scripts/algo/ldl_noblock.py
```

## 与其他 Skill 的关系

```
op-dev (创建算子) → audit-operator (审查) → eval-patch (评估) → opt-round (优化)
```
