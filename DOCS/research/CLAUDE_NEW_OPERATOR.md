# Claude Code + DeepSeek 新算子受监督闭环

统一共享核心、独立客户端配置；模型编写C++，可信框架构建和验收。选用真实仿射矩阵乘加作为最小新算子：`Z=half(half(A@B)+C)`。它不是MMSE任务，不套用信道/BER认证。实际运行证据见experiments/claude_operator/2026-11-11-v1/README.md。

## 使用

先按BUILDING.md部署Linux环境并在用户环境配置Claude Code + DeepSeek；不复制服务器用户配置或凭据。`claude --version`、`python scripts/agent_entry.py doctor`仅说明入口可用，不证明代码或数值正确。

```bash
# 在已配置的共享核心检出内执行；候选目录必须新建、干净、独立。
git worktree add -b codex/my-affine-candidate ../my-affine-candidate HEAD
python3 scripts/claude_codegen.py --task-file agent_profiles/recipes/affine_gemm_v1.txt \
  --apply-root ../my-affine-candidate --replace-candidate-files --output results/my-affine-proposal
# 检查receipt与git diff，只允许设计文档及该算子.h/.cc；冻结commit后独立审计。
# 构建/评估在该候选目录中串行执行：
cd ../my-affine-candidate
python3 scripts/bootstrap.py --jobs 1
# 默认配置的外部文件路径相对build；改用其他cwd时须显式提供有效配置。
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/affine_acceptance.py \
  --binary build/bin/Simulator --config configs/ascend_910b_quiet.json --cwd build --output results/affine-run
```

该任务模板、类/运行mode和可信affine任务集成已由负责人冻结。新算法不能只改名称套此任务：负责人须新增独立Python算法、任务输入输出/精度/计时契约和注册，保留共同Asim/事件/重放核心。候选Agent不允许修改这些可信文件。现helper是本例三文件提议协议，不是任意任务或持久协调器。

构建错误由协调器把原始编译错误和接口证据反馈给模型，模型生成修复提议；审核范围/冻结新commit后重新构建。不得改Tensor公共API、门限或参考来接纳错误候选。真实示例保留首次编译失败和DeepSeek修复，未以人工编写C++冒充模型生成。

## 原理与约束

Python原型明确FP16存储、逐k FP32乘加、GEMM结束写回FP16、再ADD写回FP16。Agent把矩阵映射到真实LOAD/GEMM/ADD/STORE并附原操作的索引/地址/视图；模型只提出三个文件的JSON，确定性导入器拒绝额外路径、重复键和参考/评估器修改。采用`claude --bare --tools ''`避免个人plugin/hook和文件工具扩大作用域。API密钥只在子进程内存，提议receipt不保存环境。它不是OS级安全沙箱；模型进程运行身份和本机安全由部署方管理。

Asim真实周期仿真先行；真实Core读写/trace与指令绑定通过才执行数值。重放从实际Instruction及typed views读取DMA输入/中间缓冲、切片和运算，FormulaLogger只诊断。共同引擎与所有原求逆算子共享，不新增一份仿射专用重放器。未知语义UNSUPPORTED，不能跳过。

三条数值路径独立：实际指令重放；冻结Python同算法FP16参考（逐位一致）；原始double输入的FP64任务参考（2%相对Frobenius误差门限）。测试3种含非方形的16对齐形状、3seed、batch96，每seed含零输入/零偏置/符号零边界。不同形状不合成同一性能排行榜。

评分由可信脚本在完整数值/事件验收后计算；本例同输入同实现重复基线的cycles比值只演示评分串接，research_score=null、promotion=false，不宣称优化收益。Asim过而数值错的真实负例必须拒绝评分。受监督审计、提交和人工反馈尚未被持久自动控制器取代。

## 本机兼容经验

本轮CLI2.1.272的bare模式文件工具能力受限，因此使用无工具JSON提议，避免把模型声称写文件当真写入。初轮个人hook/输出预算导致两次600秒超时，修正协议后成功。MAX_THINKING_TOKENS=0/alwaysThinkingEnabled=false只是请求，第三方模型是否内部思考不能据此证明；仅记录调用墙钟，纯思考时间为null。

官方参考：[DeepSeek Claude Code集成](https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code/)、[Anthropic兼容API](https://api-docs.deepseek.com/guides/anthropic_api/)、[Claude设置](https://code.claude.com/docs/en/settings-reference)。用户现有API配置优先，不把本机模型名视为永久默认。
