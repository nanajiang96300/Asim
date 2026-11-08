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

新增精度模型`modeled_fp16_fp32_real_block_v1`已接通：FP16存储、FP32逐指令/逐k算术，独立参考必须输出完整output_block，按全部FP16位比较；P2分配/搬运与P4 Vector工作量分别校验。硬件校准与转换成本仍未验收，不能凭该模型晋级。见[论文阶段报告](PAPER_REPORT.md)。

当前生产入口要求core-rw-v4及CSV请求身份/issue/commit字段；缺或旧schema时拒绝数值，不静默退回旧重放。固定日常runtime只白名单已验收NS；新增算子先跑独立campaign，再发布新runtime包。见[稳定入口](RUNTIME_STABILITY.md)。

2026-10-30：旧七入口增加完整静态覆盖及只读阶段观测；见[阶段报告](LEGACY_SEMANTIC_STAGES.md)。`execute_batch(captures=...)`仅用于诊断，不创建指令、不继续越过歧义。新算法仍需独立生成端/算法参考，不将阶段目标当回放实现。硬件级认证由上板承担。

2026-11-06：Cholesky直接求解已接入同一核心。catalog通过受信`parameter_module/parameter_function`生成完整algorithm参数；参考可返回通用`evidence`，不再强制X0。NS兼容；直接法CLI当前需显式`--iterations 1`，冻结algorithm不包含迭代数。完整普适性范围与耗时口径见[本轮报告](RECONSTRUCTION_GENERALITY.md)。

2026-11-06：新增LDL与B2实块BRI，仍不修改四个重放/绑定/事件核心。迭代上限通过受信adapter的max_iterations声明（NS32、BRI256）；直接法1占位。阶段观察只读，严格shape/finite与完整清单/覆盖绑定；见LDL_BRI_MAPPING.md。

2026-11-08：三个剩余映射通过，共用四核心保持不变。融合归约必须真实生成GEMM/明确原语，禁止SCALAR compute_size>1伪装向量；旧Merge因此继续拒绝。七新入口/151测试/统一周期证据见MERGE_AND_REMAINING_MAPPING.md。
