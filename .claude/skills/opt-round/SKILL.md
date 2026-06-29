---
name: opt-round
description: AI 驱动算子优化：Orchestrator 生成 Ideas → 多 Agent 并行探索 → UOBS 黑盒打分 → 汇总推荐下一轮。Use when user wants to run automated optimization, search for better parameters, or iterate on operator code.
when_to_use: 当用户要求自动优化算子、搜索最优参数组合、迭代改进代码、或启动多轮 AI 驱动优化时使用。参数: <operator_name> [--round N] [--prev-score S] [--focus 方向] [--nr M] [--nt K]
---

## 概述

你是 NPU 算子优化的 AI 协调器。每轮执行四阶段闭环：
**准备 → 生成 Ideas → 并行 Agent 探索 → 收集汇总**

## Phase 1: 准备

1. ~~~【环境清理 — 必须最先执行】清理上次 Agent 可能留下的脏文件~~~
   ```bash
   git checkout -- src/operations/ src/models/ example/ 2>/dev/null
   git stash list | grep "UOBS_AUTO_EVAL\|SWEEP_TMP\|PRE_NEW_ALGO" | while read line; do
     idx=$(echo "$line" | cut -d: -f1); git stash drop "$idx" 2>/dev/null
   done
   ```
   原因: evaluate_operator.sh 的 `git stash push` 可能失败留下残留 stash,
   或上轮 Agent 修改了 src/models/(不在 stash 保护范围)导致持久污染。
2. 读取 `orchestrator/operator_registry.json` 获取算子信息
3. 读取算子源码 `initialize_instructions()`，理解当前实现和可调参数
4. 若首轮无历史数据，跑基线：
   ```bash
   bash scripts/evaluate_operator.sh --baseline --config <c> --mode <m> --nr <nr> --nt <nt>
   ```

## Phase 2: 生成 N=3 个优化 Idea

基于源码理解、attributes、基线 Score、上轮结果、--focus 方向，生成具体可操作的 Idea。每个包含：名称、修改目标、改动策略、预期效果。类型：参数调优 / 调度优化 / 结构改造 / 数值策略。

## Phase 3: 并行 Agent

为每个 Idea 启动一个后台 Agent（`Agent` 工具, `run_in_background=true`）。

Agent 指令模板（必须自包含）：

```
你是 NPU 算子优化工程师。你有 20 分钟完成。

【维度锁定 — 最高优先级, 严格遵守】
  评估维度: nr={nr} nt={nt}
  这些数字必须原样传给 evaluate_operator.sh 的 --nr 和 --nt 参数。
  不要使用 config JSON 中的默认维度(matrix_m/matrix_k)。
  不要从 registry 读取维度。维度只有这一个来源。
  对 BRI/Chol/LDL: nr=64 nt=16。对 NS(求逆Gram矩阵): nr=16 nt=16。

【Idea】{name}: {detail}
【目标】源码: {source}, 参数: {attributes}
【评估】config={config}, mode={mode}, nr={nr}, nt={nt}

步骤:
1. Read 源码, 理解当前实现 (~2分钟)
2. 【Patch 生成 — 注意 linter + stash 范围】
   a. Linter 会回退对 .cc 文件的增量 Edit。对大段修改(>20行), 使用 Write 完整重写文件,
      或先将修改写入 /tmp/agent_{slug}_r{N}_modified.cc, 再用 diff 生成 patch:
      ```bash
      diff -u src/operations/File.cc /tmp/agent_{slug}_r{N}_modified.cc >> /tmp/agent_{slug}_r{N}.patch
      ```
   b. evaluate_operator.sh 的机制:
      1) git stash push -- src/operations/ example/ (只保护这两个目录!)
      2) git apply patch (只对这两个目录的修改生效)
      3) build + simulate + score
      4) git stash pop (恢复)
      因此 patch 只能包含 src/operations/.h/.cc 的改动。
      修改 src/models/ 会导致 stash 无法保护, 持久残留, 后续 apply 冲突。
   c. 在调用 evaluate_operator.sh 之前, 必须预检:
      ```bash
      git stash push -- src/operations/ example/ -m "PRECHECK" && \
      git apply --check /tmp/agent_{slug}_r{N}.patch && \
      git stash pop
      ```
      若 --check 失败: 检查 patch 路径是否正确 (应有 a/ b/ 前缀), 是否有越界文件。
3. 【迭代次数自搜索 — 所有迭代类算子必须执行】
   a. 从 iterations=1 开始逐次递增
   b. 每次 bash scripts/evaluate_operator.sh --patch ... --nr {nr} --nt {nt}
   c. 检查 rel_error: 若 >0.01 则 iterations++ 重试; 若 <=0.01 则记为最小可行值
   d. 在最小可行值 + 安全余量(通常+1)处确认稳定 PASS
   e. 最终 config 中的 iterations 必须是满足 SE 的最小值
   f. 注意: scorer 的参考逆固定用 max(layers,15)=15 次迭代, 确保参考逆充分收敛。
      不要修改 reference_inverse_registry.py 中的 max(layers,15) 逻辑。
4. 【迭代持久化 — 找到最优 iterations 后必须执行】
   a. 更新 .cc 中 _iterations 的默认值为找到的最优值
   b. 更新 config JSON 中 attributes.iterations 为找到的最优值
   c. 更新 set_algorithm() 调用, 确保 layers 参数 = 实际使用的 iterations
   d. 最终 result.json 中 optimal_iterations 字段 = 实际使用的值
5. 最终评估: bash scripts/evaluate_operator.sh --patch ... --nr {nr} --nt {nt}
6. 若 BUILD/SIM FAILED: 读 stderr(/tmp/_uobs_build.err) → 修正 → 重试(最多3次)
7. 写入 /tmp/agent_{slug}_r{N}_result.json

约束 — 违反任一条视为 FAIL:
- 【禁止越界】只改 src/operations/ 下的 .h 和 .cc 文件。
  禁止修改 src/models/、scripts/、extern/、configs/、orchestrator/ 等任何其他目录。
  evaluate_operator.sh 的 git stash 只保护 src/operations/ 和 example/,
  修改其他目录会导致 patch 无法 clean apply 且环境持久污染。
- 【维度锁定】严格使用 --nr {nr} --nt {nt}, 不读取或使用 config JSON 中的 matrix_m/matrix_k。
- 【SE硬约束】rel_error > 0.01 则 score=null, 状态为 INVALID。
- 【FormulaLogger】set_algorithm 的 layers 参数必须等于实际迭代次数。
- 【防死锁】若仿真超过 120 秒无输出, 视为 deadlock → 检查 PIPE_BARRIER 和地址分配。
  若连续 2 次 SIM_FAILED 且错误相同, 不要继续重试 → 写入 result.json 状态为 FAIL。
- 【Patch 预检】每次修改 patch 后, 在调 evaluate_operator.sh 前必须先 `git apply --check`。
  若 --check 失败, 检查是否有 a/ b/ 前缀、是否越界修改了 src/models/ 等目录。
- 【两阶段提交】对复杂改动(新增 GEMM + SCALAR + BARRIER 组合), 先用最小可行版本
  (如只加属性+减少迭代)验证编译和仿真通过, 再逐步添加复杂指令块。
  避免投入 40 分钟后才发现仿真死锁。
- 【SCALAR 源地址 — 必须使用块地址】SCALAR 指令(SCALAR_MUL/DIV/ADD/SQRT)的 src_addrs
  只能使用块地址(MOVIN dest 或 Vector ADD dest 的基地址), 禁止使用 base+offset 形式的
  元素地址。SPAD check_hit 使用精确哈希键匹配, 元素地址未在 MOVIN 时注册会导致
  can_issue_compute 永久返回 false → 指令死锁。参考 BRI 的 elem_addr 模式:
  当未定义 BJ_ENABLE_PRECOND_ELEM_ADDR 时返回 block base,而非 base+offset。
  反例: ea(base,r,c)=base+(r*K+c)*precision → check_hit 失败 → 仿真卡死在 cycle<200。
```

## Phase 4: 收集与汇总

Agent 完成后（超时 20 分钟强制终止，视为 FAIL）。
若 Agent 被终止: 读取 /tmp/agent_{slug}_r{N}_result.json 检查是否有中间结果(最高 Score 记录)。
分析死锁/超时根因, 写入汇总报告。

死锁检测: 若 Agent 日志中连续出现相同错误 > 3 次, 或仿真在 cycle < 200 处停滞,
判定为指令死锁 → 停止该方向, 下一轮不再尝试类似复杂度的 Idea。

1. 读取 `/tmp/agent_*_r{N}_result.json`，按 Score 降序排列
2. 汇总表格式:
```
══ Round {N} 优化结果 ═══════════
基线: Score={B}, Cycle={C}
──────────────────────────────────
Rank  Idea          Score  Cycle  Cube%  Status
  1   {name}        3.45   1893   12.3   ✅ PASS
  2   ...
──────────────────────────────────
```
3. 分析: 最有效方向？失败根因？与基线/上轮对比？
4. 推荐下轮 3 方向（深挖/转向/融合）
5. 收敛判断: Score 提升 < 5% 连续 2 轮 → 终止
6. 写入 `/tmp/opt_round_{N}_summary.md`
