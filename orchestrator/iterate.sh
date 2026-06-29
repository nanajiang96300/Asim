#!/bin/bash
# ============================================================================
# iterate.sh — 多轮 AI 驱动算子优化自动化脚本
# 用法: bash orchestrator/iterate.sh <operator_name> [max_rounds] [--no-auto]
# 示例: bash orchestrator/iterate.sh block_richardson 5
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REGISTRY="$SCRIPT_DIR/operator_registry.json"
RESULT_DIR="$ROOT/result_new/orchestrator_runs"
LOG_DIR="$RESULT_DIR/logs"

# ── 参数解析 ──────────────────────────────────────────────
OPERATOR="${1:-}"
MAX_ROUNDS="${2:-5}"
NO_AUTO="${3:-}"

if [ -z "$OPERATOR" ]; then
  echo "用法: bash orchestrator/iterate.sh <operator_name> [max_rounds]"
  echo ""
  echo "可用算子:"
  python3 -c "
import json
with open('$REGISTRY') as f:
    reg = json.load(f)
for name, info in reg['operators'].items():
    print(f'  {name:20s} — {info[\"description\"][:60]}')
"
  exit 1
fi

# ── 验证算子存在 ──────────────────────────────────────────
OP_INFO=$(python3 -c "
import json
with open('$REGISTRY') as f:
    reg = json.load(f)
if '$OPERATOR' not in reg['operators']:
    exit(1)
print(json.dumps(reg['operators']['$OPERATOR']))
") || { echo "错误: 未知算子 '$OPERATOR'。请检查 operator_registry.json"; exit 1; }

OP_CONFIG=$(echo "$OP_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['config'])")
OP_MODE=$(echo "$OP_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['mode'])")
OP_SRC=$(echo "$OP_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['source'])")
OP_NR=$(echo "$OP_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['nr'])")
OP_NT=$(echo "$OP_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['nt'])")

# ── 准备 ──────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$RESULT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="$RESULT_DIR/${OPERATOR}_${TIMESTAMP}"
mkdir -p "$SESSION_DIR"

# ── 收敛参数 ──────────────────────────────────────────────
PREV_BEST_SCORE=0
STAGNANT_ROUNDS=0
CONVERGENCE_THRESHOLD=0.05   # 提升 < 5% 视为停滞
MAX_STAGNANT=2               # 连续 N 轮停滞则终止

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AI 驱动算子优化 — 多轮迭代自动化                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  算子:     $OPERATOR"
echo "║  源文件:   $OP_SRC"
echo "║  配置:     $OP_CONFIG"
echo "║  模式:     $OP_MODE"
echo "║  维度:     nr=$OP_NR, nt=$OP_NT"
echo "║  最大轮次: $MAX_ROUNDS"
echo "║  收敛阈值: ${CONVERGENCE_THRESHOLD} (Score 提升比例)"
echo "║  输出目录: $SESSION_DIR"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 多轮迭代 ──────────────────────────────────────────────
for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Round $round / $MAX_ROUNDS  (上轮最优 Score = $PREV_BEST_SCORE)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  ROUND_DIR="$SESSION_DIR/round_${round}"
  mkdir -p "$ROUND_DIR"

  # 调用 Claude Code 执行 /opt-round
  if [ "$NO_AUTO" = "--no-auto" ]; then
    echo ""
    echo "  [手动模式] 请在 Claude Code 中执行:"
    echo "    /opt-round $OPERATOR --round $round --prev-score $PREV_BEST_SCORE"
    echo ""
    read -p "  按 Enter 继续下一轮..."
  elif command -v claude &>/dev/null; then
    claude --print --permission-mode acceptEdits \
      "/opt-round $OPERATOR --round $round --prev-score $PREV_BEST_SCORE" \
      2>&1 | tee "$LOG_DIR/round_${round}_$(date +%H%M%S).log" || true
  else
    echo ""
    echo "  [半自动模式] 未找到 claude CLI。请在 Claude Code 中执行:"
    echo "    /opt-round $OPERATOR --round $round --prev-score $PREV_BEST_SCORE"
    echo ""
    echo "  执行完成后，Agent 结果将写入 /tmp/agent_*_r${round}_result.json"
    echo "  本脚本将自动检测并继续。"
    echo ""
    # Wait for results
    for i in $(seq 1 120); do
      if ls /tmp/agent_*_r${round}_result.json 2>/dev/null | grep -q .; then
        echo "  检测到 Agent 结果，继续..."
        break
      fi
      sleep 5
    done
  fi

  # 从本轮结果中提取最优 Score
  # Agent 将结果写入 /tmp/agent_*_r{N}_result.json (由 /opt-round skill 指定)
  CURRENT_BEST=0
  AGENT_OUTPUTS=(/tmp/agent_*_r${round}_result.json)

  for result_file in "${AGENT_OUTPUTS[@]}"; do
    if [ -f "$result_file" ]; then
      SCORE=$(python3 -c "
import json
with open('$result_file') as f:
    d = json.load(f)
print(d.get('score', 0))
" 2>/dev/null || echo "0")
      if [ "$(echo "$SCORE > $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
        CURRENT_BEST=$SCORE
      fi
      # 持久化：将 Agent 结果复制到本轮目录
      cp "$result_file" "$ROUND_DIR/" 2>/dev/null || true
    fi
  done

  # 备份：也检查 ROUND_DIR 中的结果（Agent 可能直接写入）
  if [ "$CURRENT_BEST" = "0" ]; then
    for result_file in "$ROUND_DIR"/agent_*_r${round}_result.json "$ROUND_DIR"/*_result.json; do
      if [ -f "$result_file" ]; then
        SCORE=$(python3 -c "
import json
with open('$result_file') as f:
    d = json.load(f)
print(d.get('score', 0))
" 2>/dev/null || echo "0")
        if [ "$(echo "$SCORE > $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
          CURRENT_BEST=$SCORE
        fi
      fi
    done
  fi

  echo ""
  echo "  >>> Round $round 最优 Score: $CURRENT_BEST"

  # ── 收敛判断 ────────────────────────────────────────────
  if [ "$PREV_BEST_SCORE" != "0" ]; then
    IMPROVEMENT=$(python3 -c "
prev = $PREV_BEST_SCORE
curr = $CURRENT_BEST
if prev > 0 and curr > 0:
    imp = (curr - prev) / prev
    print(round(imp, 4))
else:
    print(99.0)
")
    echo "  >>> 提升幅度: ${IMPROVEMENT}"

    if [ "$(echo "$IMPROVEMENT < $CONVERGENCE_THRESHOLD" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
      STAGNANT_ROUNDS=$((STAGNANT_ROUNDS + 1))
      echo "  >>> 本轮提升 < ${CONVERGENCE_THRESHOLD} (停滞 $STAGNANT_ROUNDS/$MAX_STAGNANT)"
      if [ "$STAGNANT_ROUNDS" -ge "$MAX_STAGNANT" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  ✅ 收敛: 连续 $MAX_STAGNANT 轮提升 < ${CONVERGENCE_THRESHOLD}                        ║"
        echo "║  最终最优 Score: $CURRENT_BEST                               ║"
        echo "║  总轮次: $round                                              ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        break
      fi
    else
      STAGNANT_ROUNDS=0
    fi
  fi

  PREV_BEST_SCORE=$CURRENT_BEST
  echo ""
done

# ── 汇总 ──────────────────────────────────────────────────
if [ "$STAGNANT_ROUNDS" -lt "$MAX_STAGNANT" ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  达到最大轮次 ($MAX_ROUNDS)，优化结束                          ║"
  echo "║  最终最优 Score: $PREV_BEST_SCORE                             ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
fi

echo ""
echo "  完整记录: $SESSION_DIR"
echo "  日志:     $LOG_DIR"
