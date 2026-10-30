# 新算子接入共同重建框架

2026-10-30。统一数值核心已实现；未知指令、缺语义或成本不能自动通过。当前只有结构float64与逻辑complex128诊断，不支持硬件FP16或CDL-B。

## 1. 冻结共同任务

`verification/contracts/mmse.py:MMSETask`定义H/Y/λ、batch/Nr/Nt/Nrhs、信道、表示与计时边界；默认96/64/16/8，24核，Cube16³。

- `mmse_h_y_to_xhat_v1`：外部仅提供H/Y及I/λI共同常量；从输入DMA至xhat写回，Gram、HᴴY、算法初始化/预条件、迭代和恢复全计时。
- `prepared_solver_kernel_v1`：给定A/b/X0/C，仅作旧内核诊断；prepared fixture仍用旧专用工具，本轮共同full-scope runner拒绝套用。

实块编码/解码是共享表示边界，解码检查全部四象限；不等于真实复数硬件成本认证。比较必须具有同任务、同输入hash、同硬件配置与同计时范围的`comparison_identity`。优化预筛须引用同identity基线；不能拿旧NS、其他维度或其他seed周期直接套阈值。

## 2. 在独立工作区生成候选

保留旧实现，新增Model/mode；每个Instruction构造处写入`source-views-v1`。程序header用`source-views-program-v1`，记录真实Model inputs/outputs/regions/capacities、profile及完整task_contract与algorithm参数。

指令的src/dst、row/col/shape、transpose/conjugate、opcode、广播须具体；MUL/DIV矩阵与标量组合必须声明`scalar_rhs`，按完整物理矩阵计Vector工作量。单个SCALAR只能操作一个元素；归约逐条展开或新增受信原语及成本测试，不能将批量SCALAR计为一次。辅助barrier也必须出现在实际trace。

目前公共原语：LOAD/STORE/COPY（CPU结构路径）、GEMM、ADD/SUB/MUL/DIV/SQRT，部分初始化/切片/转置/共轭。支持数学原语不等于所有Asim路径均可生成：Vector没有独立SUB，所以新候选显式MUL(-1)+ADD；旧generic schema的SUB→raw ADD映射现在UNSUPPORTED，不能沿用旧结构PASS当成真实减法验证。COPY/FILL/其他归约的Asim成本未实现时拒绝。

## 3. 只注册生成端与独立参考

受信配置`orchestrator/codex_operators.json`记录mode、source、scope、profile、独立参考模块/函数。目录由评估器维护，候选不能改评估器、配置、参考或门槛。

当前最小插件协议：Model接收共同尺寸、task_contract与iterations；参考函数`simulate_ns(H,Y,lambda,iterations)`返回output与X0等证据。此协议适用于当前迭代候选；新分解算法需在`operators/catalog.py`扩充生成参数/参考回调的适配契约，**不得在reconstruction核心添加按算法名求解的分支**。不声称仅一行JSON就能接入任意参数族。

数值路径：旧legacy source-views与新候选→`reconstruction/source_views.py`绑定实际Instruction→`engine.py:ArrayMemory/execute_views`；通用structure-program→shadow适配器→同一核心。旧`legacy/replay.py`只保留兼容导入/历史输入fixture。独立同算法参考不导入重放核心；任务真值独立solve。

## 4. 固定门禁

`reconstruction/runner.py:evaluate`：验证契约→构造共同输入→新目录真实Asim→实际任务/成本/dispatch/compute+Scalar+DMA事件检查→冻结性能预筛→独立算法与任务参考→实际逐batch重放→三路径比较→详细流水/原件归档。

初次正确性基线显式`baseline_creation=true`；性能竞争力与异常收益两个分支保持数值NOT_RUN。已有基线时调用者必须提供baseline_identity，跨任务/输入/硬件/范围直接拒绝。当前是有人监督的受信Python调用接口，非候选自报周期，未实现OS权限隔离。

actual↔algorithm失败为REPLAY_MISMATCH；一致但任务误差超阈值为ALGORITHM_QUALITY_FAILED；全部逐batch满足才STRUCTURE_CHECKED。后者仍不代表FP16/G4/通信/G6或评分验收，score=null。

新增原语时更新共同engine、实际Instruction投影、成本派生与负例；新增算法时只写候选与独立算法参考。不允许根据算法名调用numpy逆来代替真实重放，不依赖FormulaLogger的另一份声明。

## 5. 命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s verification/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.reconstruction.runner \
  --binary /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --cwd /home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build \
  --operator ns_corrected_v1 --seed 42 --iterations 16 \
  --output results/unified_reconstruction/<new-run-id>
```

CLI显式选择precision8结构profile并写入receipt；不会把precision2模拟结果冒充FP16数值。output必须不存在；构建/仿真都串行，-j1；原件按完整清单校验后才能标记归档，不覆盖旧run。
