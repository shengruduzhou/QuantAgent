# v8 训练 tmux 监控速查

## 当前运行

| 项目 | 值 |
|---|---|
| Session | `qa_v8_sweep` |
| Universe | top-500 by liquidity (`runtime/reports/v8/pipeline/universe_top500.txt`) |
| Date range | 2019-01-02 → 2024-12-31 |
| Horizons | short_5d → mid_5d_30d → long_30d_120d (顺序训练) |
| GA | population=48, generations=20, top-K=30 |
| Output root | `runtime/reports/v8/pipeline/v8_sweep_<timestamp>/` |
| Log | `runtime/logs/v8_pipeline/v8_sweep_<timestamp>.log` |

## tmux 操作

```bash
# 切换到运行画面（3 panes：driver / log tail / status）
tmux attach -t qa_v8_sweep

# 离开 tmux（不停止训练）
Ctrl-b 然后按 d

# 列出 panes
tmux list-panes -t qa_v8_sweep

# 仅看主 driver pane
tmux attach -t qa_v8_sweep \; select-pane -t 0

# 强制停止整个训练
tmux kill-session -t qa_v8_sweep
```

## 命令行监控（不进 tmux）

```bash
# 找当前日志
LOG=$(ls -t runtime/logs/v8_pipeline/v8_sweep_*.log | head -1) && echo $LOG

# 跟踪日志，过滤掉警告噪音
tail -F $LOG | grep -vE "FutureWarning|daily = merged"

# 看每个 horizon 的 metrics（持续刷新）
watch -n 30 'for h in short_5d mid_5d_30d long_30d_120d; do
  echo ">> $h"; cat runtime/reports/v8/pipeline/v8_sweep_*/$h/backtest/metrics.json 2>/dev/null
done'

# 看 GA fold OOS 结果
cat runtime/reports/v8/pipeline/v8_sweep_*/short_5d/ga/walk_forward_backtest.json | jq '.[].best_loss'
```

## 看产物（训练结束后）

```bash
RUN=$(ls -td runtime/reports/v8/pipeline/v8_sweep_* | head -1)

# 三个 horizon 的 metrics
for h in short_5d mid_5d_30d long_30d_120d; do
  echo "=== $h ==="
  cat $RUN/$h/backtest/metrics.json
done

# ensemble 输出
cat $RUN/ensemble_summary.json
ls -la $RUN/

# 各 horizon 的 GA factor weights
for h in short_5d mid_5d_30d long_30d_120d; do
  echo "=== $h factor_weights ==="
  cat $RUN/$h/ga/factor_weights.json
done

# 每个 horizon 的成交记录 + 风险事件
ls $RUN/*/backtest/
```

## 常见状况

| 现象 | 原因 | 处理 |
|---|---|---|
| pane 0 显示 `READY_TO_CLOSE` 然后 60s 后消失 | sweep driver 正常结束 | 看 `[end]` 时间和 `ensemble_summary.json` |
| 日志只有 FutureWarning 不动 | GA 还在 fold 内迭代 | 等；每个 fold 大概 10-30s |
| `[fatal] horizon X failed` | CLI 出错 | 看上面 Traceback；driver 会继续下一个 horizon |
| 三个 horizon 都 skip 在 ensemble | 三个都失败了 | 用 `tmux attach` 进 pane 0 看完整错误 |

## 重新跑

```bash
# 用更激进的 GA 设置
GA_POP=96 GA_GEN=40 bash scripts/launch_v8_pipeline_sweep_tmux.sh

# 用更长时间窗口
START_DATE=2015-01-02 END_DATE=2024-12-31 bash scripts/launch_v8_pipeline_sweep_tmux.sh

# 用更大 universe
UNIVERSE_FILE=自己的列表.txt bash scripts/launch_v8_pipeline_sweep_tmux.sh
```
