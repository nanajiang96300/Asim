# BRI inner-0001 独立静态审计

日期：2026-11-10（Asia/Hong_Kong）。候选工作树 `/home/nanajiang/asim-worktrees/bri-inner-0001`，提交 `88acdaed5e2f565f76a1a9f91f789ff07a440c67`。设计 `DOCS/operators/bri_inner_0001.md` 先于代码。审查者为独立子Agent；仅只读检查源码及少量无输入样本的FP16原语/符号映射，不编译、不运行Asim、不加载巨大JSON，不修改算子/评估器/参考。

结论：**STATIC_STATE_ROTATION_REVIEW_PASSED_WITH_DYNAMIC_GATES_REQUIRED**。未发现阻塞问题。静态结论不证明实际周期、真实Core事件或完整任务/通信精度通过，score仍应null。

## 1. 修改范围

`git diff HEAD^ HEAD`只包含 `src/inverse/bri_stable_hb/BRIStableHBModel.cc:150` 附近两条copy移除与三行注释/轮换；与主线基点d26206b9相比，新增设计/版本管理文档和空间计划，唯一C++变更仍为该处。Model/main/catalog/reference/evaluator/精度模型/任务/K均不变。

原步骤：每轮完成carry后 `previous←x`、`x←value`，copy助手是MUL(1)并各带一个barrier。候选改为生成期逻辑地址角色轮换，不执行新数学算法。由于原张量数值已经驻留于三个独立物理区，重新选择后续指令的真实src/dst是合法的静态存储重用，不是省略初始化或用元数据替代数值计算。

## 2. 三缓冲及依赖

源码第24—26行分配x、previous、value各U×R（本域32×8）独立、64B对齐的完整SPAD区；固定region声明未缩小或重叠。第89—92行x先通过实际MUL/ADD(+0)归零，previous实际copy初始化；value在第一轮第147行的完整矩阵ADD首次全写，因此没有读未初始化value。

设本轮物理区(A,B,C)逻辑角色为(previous,x,value)，carry写完以后两次swap得到(B,C,A)。对逻辑状态来说，下一轮previous正确指向旧x，x正确指向本轮value，下一轮value可以覆盖原previous。原previous在第142—143行差分/动量后已不再读取；旧x仍在第148行difference_new中读取，该读取与第149行carry计算均发生在轮换之前。所有三个物理区大小相同；轮换只改addr_type值，不会改header region范围。

每个保留的arithmetic仍由助手生成实际Instruction、typed view和原barrier，特别是new/difference_new/carry链全部保留。SourceViews.h按值创建json view，所以已发出的前序指令不会因随后C++地址变量swap被追溯修改。后续迭代源地址和最终输出地址根据轮换后的逻辑变量生成。

静态数据依赖没有暴露出新增的读写冲突；但PIPE_BARRIER/scoreboard具体发射时序及buffer epoch仍须通过该候选的真实Core观察事件检查，不以源码barrier存在替代动态证据。删除的barrier和其他barrier序号变化都是程序变化，不能拿基线事件直接签发候选。

## 3. K mod3及输出

从初始逻辑(previous,x,value)对照原region标签，输出x的物理槽如下：

| K mod3 | 逻辑previous物理槽 | 输出x物理槽 | 下轮value物理槽 |
|---|---|---|---|
| 1 | 初始X | 初始new | 初始previous |
| 2 | 初始new | 初始previous | 初始X |
| 0 | 初始previous | 初始X | 初始new |

独立用纯符号轮换检查K1…174；K16、53、174分别覆盖余数1、2、0。源码第155—158行所有输出re/im/re_copy/negative_im均使用最终`x`变量，未硬编码初始X区域，布局恢复不会因K mod3改变而错读初始槽。实际动态见证应读取输出指令的真实src或明确的角色映射，不能只按header的region名字“X”判断最终逻辑x。

## 4. FP16位等价边界

已独立穷举全部65,536个binary16位型并排除非有限值：63,488个有限位型执行FP32 multiply(1)后写回FP16，全部uint16位相同，包含2,046个正/负次正规位型及+0/-0。这与本工程F16 policy（FP16存储、FP32临时、RNE、subnormal preserve、无fusion）一致。

因此被删除的两个状态copy在声明的有限状态域中是逐位恒等，轮换不改变递推的舍入点；没有改为更高精度持久状态。此结论不涵盖NaN/Inf、FTZ硬件模式或任意其他数值政策；非有限输入/中间值仍须按既有门禁拒绝。仍需真实程序正负零RHS及全部batch位比较，以发现实现/索引错误。

## 5. 指令数与性能口径

每轮copy助手各为MUL+barrier，因此静态预期删除4K条指令/batch。保留21个arithmetic调用/轮，每个包含一个实际compute与barrier，即42K条，setup/output未改：候选591+42K；基线591+46K。

| K | 基线/batch | 候选预期/batch | 96batch预期总数 |
|---|---:|---:|---:|
| 16 | 1327 | 1263 | 121248 |
| 53 | 3029 | 2817 | 270432 |
| 174 | 8595 | 7899 | 758304 |

这些仅为源码结构推导，不是实际导出的指令计数或周期实测。不能按指令减少比例线性估计周期。实际模型可能因存储角色/scoreboard行为改变而影响流水，必须执行同task/input/config/measurement identity的Asim和性能预筛。

## 6. 后续动态验收必须覆盖

1. 构建/冻结candidate commit、binary/library/config/evaluator/参考与契约，基线原件保留；同opid必须用candidate_id+commit区分，不能混写记录。
2. 三档真实Asim完成，实际导出指令数和输出地址映射符合预期；独立检查实际Core事件/完整DMA/24核96batch、无缺事件或读写/epoch冲突。
3. Asim/性能预筛通过后才执行数值。完整96batch逐位比较独立同算法参考/基线，再与FP64 MMSE比较；K16/53/174覆盖三角色槽。全新CDL/QPSK按冻结种子与门限执行，不调K。
4. 正/负零RHS重建；缺语义/错误符号/错误最终输出槽等负例按实际候选指令构造。
5. 候选已删除`iteration*.previous`与`iteration*.state`指令，旧控制脚本或见证若按这些suffix定位会失效。必须改用该候选真实存在的new/carry与输出指令定位，缺目标时明确拒绝，不能静默跳过。

## 可执行的低成本审查片段

```python
import numpy as np
raw=np.arange(65536,dtype=np.uint16)
half=raw.view(np.float16)
mask=np.isfinite(half)
actual=np.multiply(half[mask].astype(np.float32),np.float32(1),dtype=np.float32).astype(np.float16)
assert np.array_equal(actual.view(np.uint16),raw[mask])
for K in range(1,175):
    previous,x,value='previous','x','value'
    for _ in range(K):
        previous,x,value=x,value,previous
    assert x=={0:'x',1:'value',2:'previous'}[K%3]
```

纯Agent思考时间不可观测，为null；本审计未记录可信的完整墙钟起点，因此不虚构审查耗时。静态审查通过可继续动态门禁，不等于软件候选已验收或晋级。
