# 显式LDL直接求解映射

2026-11-06。保留所有旧实现。共用任务H/Y→xhat，24核、默认batch96、Nr64/Nt16/RHS8，瑞利。

## 1. 文件

src/inverse/ldl_explicit/LDLExplicitModel.{h,cc}；verification/reference/ldl.py；受信catalog参数工厂。重放engine/source_views/binding/runtime不增加算法分支。

## 2. 公式推导

A=R(H)ᵀR(H)+λI=LDLᵀ，L单位下三角。右看式Dj=Ajj，Ltail,j=Atail,j/Dj，Atail←Atail−(Ltail,j Dj)Ltail,jᵀ。解Lz=HᵀY，再逐行w_j=z_j/Dj，最后Lᵀx=w；逆矩阵为L⁻ᵀD⁻¹L⁻¹但任务无需显式生成逆。D权重作用于行，不使用旧实现的列缩放。

## 3. SPAD与初始化

共同H/Y/RegI/Identity；A/L为2Nt方阵，L=I显式MUL(one)初始化；D为1×2Nt；rhs/forward/backwork为2Nt×2Nrhs；weighted列与T/negativeT暂存；ACCUM整块输出。D每个元素在使用前真实写入。one=Reg00/Reg00，zero=I00-I00，neg1=zero-one。

## 4. 指令/FormulaLogger映射

Dj写入用单元素SCALAR_MUL；整列DIV/MUL为真实Vector，scalar_rhs显式，compute_size按全部视图元素×算术字节。秩一更新GEMM，减法MUL(-1)+ADD；前/后解有真实秩一更新，不遗漏累加。每步emit_step并附SourceViews，每步保守barrier。DMA base0，完整STORE。

## 5. 验证

独立高精度LDL参考和独立逐指令FP16/FP32模型；后者不导入共同engine/precision。同任务真值独立solve。默认与第二尺寸、两个seed，先Asim/检查再数值；错误行权重、单位对角和未知指令离线负例明确拒绝。编译-j1，独立工作区/build串行独占；日志和耗时在results/operator_mapping/2026-11-06-ldl-bri-v1。结果：默认FP64 24647周期、FP16模型18697周期；两seed模型最大任务误差0.2543%以内，逐位一致；第二尺寸F64 4689周期。阶段实际L/D见证及日常入口通过，144单元测试，详见DOCS/research/LDL_BRI_MAPPING.md。
