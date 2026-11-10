# BRI inner-0001：状态缓冲区轮换

2026-11-10，设计先于代码。family=bri，algorithm=bri_fixed_hb_fp16_v1；baseline=bri-hb-base-v1；candidate=bri-hb-inner-0001。同Model/mode/opid bri_stable_hb_fp16_v1，独立worktree codex/bri-inner-0001，主工作区基线不改。

假设：HB更新后原先previous←x、x←value两个32×8 FP16 COPY（各MUL(1)+barrier）可由生成指令时轮换三个物理地址角色替代，减少每迭代4条实际指令，不增加免费运算。设原物理(A,B,C)对应(previous,x,value)，下一轮使用(B,C,A)。所有迭代结束后输出读取当前逻辑x，K mod3全覆盖；K16/53/174恰对应1/2/0。不改变已完成的carry计算、初始化或任何数值公式。新value区域是原previous所在的完整已分配区域，原值在moment以后已无读依赖；每个new写后原barrier保留，删除的仅两次copy及各自barrier。真实观察事件必须确认无读写/epoch风险，不作硬件证明。

公式仍为原冻结fixed HB：g=Hᵀ(y−Hx)−λx，δ=αPg+β(x−previous)，补偿更新/谱界/αβ/正零/输出恢复不变。MUL(1)对已有限FP16状态不改位（含保留次正规/有符号零），但仍以重建和正负零控制验证；不使用随机输入填补或同名标准算法代替程序。

修改范围仅src/inverse/bri_stable_hb/BRIStableHBModel.cc末轮两个copy替换为生成期地址轮换；Model/main/catalog/reference/evaluator/Asim精度不改。regions保持原分配，typed view随逻辑地址生成。预期每batch指令591+42K（基线591+46K）；所有输入DMA/初值/预条件/谱界/补偿/输出恢复继续计时。

冻结K16/53/174、λ1/.1/.01、Nr64/Nt16/RHS8/batch96/24核；正交顺序：构建→独立审计→冻结版本→实际Asim→事件/性能预筛（minimum_speedup1.001，maximum4）→同算法位比较与FP64MMSE→全新静态CDL-B H61001/61002、载荷61101/61102各96（原H590只回归，不选K）→未编码QPSK原门限→流水/台账/归档。候选需在同输入/硬件/计时identity下优于各档基线18409/47121/141017。失败不换K或门限，不自动重试。

预算：一个候选×三档，编译-j1/Asim串行；候选原始制品≤3.5GiB，运行前可用空间目标≥4.5GiB，最低512MiB；总墙钟≤5400s，冻结契约/输入/工具链/评估器来源。基线现有封存Asim与质量证据保留，评估器M1兼容重验另列，不将旧结果改写成新运行。
