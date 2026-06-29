# LDL-NoBlock (ldl_noblock)

## 算法概述
逐列 LDL 分解求逆，B=1 退化形式。有两个独立实现。

## 版本

### ldl_basic (LDLDecompNoBlockOp)
- **文件**: `LDLDecompNoBlockOp.h/.cc` + `LDLNoBlockModel.h/.cc`
- **实现方式**: `LDLDecompOp` 的薄包装，强制 `block_size=1, pack_blocks=1`
- **特点**: 代码量小（29 行），但依赖 `LDLDecompOp` 父类

### ldl_aligned (LDLDecompNoBlockAlignedOp)
- **文件**: `LDLDecompNoBlockAlignedOp.h/.cc` + `LDLDecompNoBlockAlignedModel.h/.cc`
- **实现方式**: 独立实现，right-looking LDL，显式匹配 Cholesky-NoBlock 结构逐元素对齐
- **特点**: 代码量大（265 行），专门为公平对比 Cholesky-NoBlock 设计

## 区别
| 特性 | ldl_basic | ldl_aligned |
|------|-----------|-------------|
| 实现方式 | 包装 LDLDecompOp | 独立实现 |
| 指令结构 | 继承自 block 版本 | 逐元素 right-looking |
| 对比目标 | 通用 LDL | 精确对比 Cholesky-NoBlock |
| Mode 名 | `ldl_noblock_test` | `ldl_noblock_aligned_test` |

## 版本历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06 | 从 operations/ 迁移到 inverse/ldl_noblock/，两个变体合并到一个目录 |
