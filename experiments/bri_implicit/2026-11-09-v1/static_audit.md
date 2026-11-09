# 隐式FP16候选独立静态审查

日期：2026-11-09（Asia/Hong_Kong）。审查Agent：implicit_operator_audit。仅此报告文件写入；源码/文档未修改，未运行构建或大数值实验。

## 范围与结论

审查实际C++生成端、冻结独立算法参考、受信adapter、流式观察事件校验、受监督campaign及op-flow/op-dev skill。初轮发现与逐轮复核均保留在下表。动态Asim/重建/MMSE/QPSK尚不能由静态审查签发。

当前静态状态：**STATIC_REVIEW_PASSED**。主要阻断均已修；76d6afd2已将最终replay guard移至成功result发布之前，复核通过。该结论只覆盖审阅代码与声明任务，不代替动态Asim/重建/通信验收。

## 发现与修正

| 发现 | 证据与状态 |
|---|---|
| HIGH：Y区域M×R而实际加载M×2R | 已改完整Y分配，复核通过 |
| HIGH：expandedP/输出第一次局部写只分配局部大小 | 已加入allocateP/allocateOutput完整实际Vector初始化，复核通过 |
| 对角GEMM改MUL存在signed-zero位差异 | 微检查g=-0、p=2：MUL为0x8000、dense累加为0x0000；已加计费ADD(+0)规范化梯度，新增零/下溢回归 |
| 旧观察事件预算不适合K76/batch96 | 采用有界iter_stream，原256MiB/100万默认保持；新受信适配器声明1GiB/250万。每个compute描述/semantic先核验再仅保留core/cycle/order；DMA/reset完整保留 |
| HIGH：zip缺覆盖可能all([])通过 | 已加入精确3case/2seed coverage及最终3group/6frame/实际576batch检查，空/缺/逆序测试 |
| HIGH：模板结果不完整绑定冻结任务 | 已加入OP、完整task/algorithm、seed57001、binary、实际receipt/model/config/路径及必需制品核验 |
| 源码/依赖允许遗漏 | 已用source_map精确闭包覆盖verification（含独立参考study）、candidate/main/SourceViews/registry；依赖精确binary/config/lib/DRAM，冻结Python/NumPy |
| 成功封存后才guard | dataset/asim末尾先guard再封存；76d6afd2已将replay最终guard移至成功result发布之前，复核通过 |
| 文档与代码状态不一致 | ADDING_OPERATOR顶部已有当前摘要；设计已改expandedP，一轮仅单条MUL；C++报错已纠正implicit/FP16 |

## 原理与生成端检查

当前C++限定FP16存储、Nr64/Nt16/RHS8、batch为24倍数且≤384、K1..76。点能量、预条件实虚复制、系数递推、动量0.9、隐式残差、补偿更新、前态及最终四象限恢复，未发现其他静态运算错误。三档λ1/.1/.01中Q(Qλ/4)与Q(λ/4)微检查位一致。

全部操作可由现有GEMM/ADD/MUL/单元素SCALAR_SUB/DIV与scalar_rhs表达，无需给共同重建核心增加BRI分支。CONST/COPY/负号已由实际算术表达并计费；FormulaLogger只作索引，不代替真实Instruction语义。

## 事件校验与实验边界

流式版本在丢弃重复compute描述之前完整校验FIELDS、core/buffer、dispatch周期和read_begin semantic。后续trace/order/epoch/RAW/WAR/WAW仍使用原门禁，新增描述篡改回归覆盖压缩前拒绝。流式并非恒定内存，原始instructions、DMA与精简事件仍占空间。

campaign计划先运行三份真实batch96瑞利模板（λ/K为1/13、.1/33、.01/76），通过真实Asim/事件/数值门禁再用全新预冻结CDL H56001/56002各96与帧56011/56012重放。每组三SNR共享192个H，不累加成576独立H。输出保留template_task/source_finish_cycle与CDL域说明，不制造CDL新Asim周期。新计时将独立参考与instruction replay/comparison分开；validate_actual绑定开销仍在两项之外，应由总墙钟解释。

op-flow/op-dev现以实际指令重建与独立参考为强制验收，保留旧UOBS流程但不把其DAG自一致当数值通过。规范不要求不同算法同布局/同指令数，而要求同任务/原语/精度/计时边界。

## 时间与限制

审查轮次墙钟：第一轮146秒、第二轮79秒、第三轮80秒；第四轮02:35:36至02:37:15 UTC，约99秒。四轮累计约404秒；轮间等待不计入。以上包含读取、工具调用和判断，不是纯模型思考时间；pure_model_thinking_seconds=null。未执行本campaign，测试计数来自主Agent而非本审查独立跑测。

## 审阅文件SHA-256

下面记录报告生成时文件内容；之后修改需重新复核对应差异。

| 文件 | SHA-256 |
|---|---|
| `/home/nanajiang/Asim/verification/reconstruction/implicit_campaign.py` | `d710af21f36fe131ae10f0a2530511635441ed7a0be0e7a699fb03914014dafb` |
| `/home/nanajiang/Asim/verification/schedule/runtime.py` | `1c60a92cc2e88be25710033c4ed47e9d91e7396fcb2f102a7db82ff20cad1be6` |
| `/home/nanajiang/Asim/verification/reconstruction/runner.py` | `2928fc4874b3694230e40632f2c95bff49b100ce4c1cea1c1041fb50f592e61f` |
| `/home/nanajiang/Asim/verification/asim/runner.py` | `ed5380147e8576860fe51bc98b87feec86134651f2fd291f5eb0d54586b9686b` |
| `/home/nanajiang/Asim/verification/operators/catalog.py` | `04b975fb0358c9e3986e32ed6ca2e928d120346b33ec912656302bd16d6f00a2` |
| `/home/nanajiang/Asim/verification/operators/implicit_parameters.py` | `9eccbe72e46ba902fbf615e7e2c742747a65efd094572e7352f9dc48fef4d120` |
| `/home/nanajiang/Asim/verification/reference/bri_implicit.py` | `1a1c0b94cc13281faa7fcd68f123cb098524c11c1edce8b210fb8d15f396d045` |
| `/home/nanajiang/Asim/verification/communication/qpsk_link.py` | `1d53806472eef335eeb8e6d3ff9cb520956c77977d97dd34b4934c065bba73e8` |
| `/home/nanajiang/Asim/orchestrator/codex_operators.json` | `8f8bd291773180aba8b08e6f1cba8198c46ea27d76cffc265cad1cb226b760a5` |
| `/home/nanajiang/Asim/.claude/skills/op-flow/SKILL.md` | `102bd9af008bc51b5bd24767ac2c02f8c3ebccefc129b6b4ba3c6701cbf1e6f6` |
| `/home/nanajiang/Asim/.claude/skills/op-dev/SKILL.md` | `4abdf91212d5b260c68078bfbefe0e00a3b912f322926309e30968ba87abba55` |
| `/home/nanajiang/Asim/verification/tests/test_implicit_mapping.py` | `130acb272b2a82004ad9c129b99b4a71b29bca987d51e1141c0f4b3ba165acd1` |
| `/home/nanajiang/asim-worktrees/bri-implicit-fp16-20261109/src/inverse/bri_implicit/BRIImplicitModel.cc` | `271a3f6a3392586952b014e5e3c6df2ca6e4649e69cee89c9d424e55aa6c9bb6` |
