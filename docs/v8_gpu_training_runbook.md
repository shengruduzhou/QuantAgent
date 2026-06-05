# v8 GPU 训练 Runbook

## 当前运行

| 项 | 值 |
|---|---|
| tmux session | `qa_v8_deep` (3 panes: driver / log / nvidia-smi) |
| Universe | top-500 by liquidity (`runtime/reports/v8/pipeline/universe_top500.txt`) |
| Train window | 2018-01-02 → 2023-06-30 |
| OOS window | 2023 中 + embargo 30 bdays → 2024-12-31 |
| Horizons | short_5d → mid_5d_30d → long_30d_120d (顺序训练 + ensemble) |
| Model | FT-Transformer (4 blocks × 8 heads × d_token=128) |
| Epochs | 40 / horizon (early stop patience 10) |
| Batch | 8192 (AMP mixed-precision) |
| GPU | NVIDIA RTX 3090 24GB |
| Top-K | 30 names equal-weight |
| Log | `runtime/logs/v8_deep/v8_deep_<ts>.log` |
| Output | `runtime/reports/v8/deep/v8_deep_<ts>/` |

## 数据源验证（已完成）

| Source | Status | Coverage |
|---|---|---|
| qlib_local | ✅ | 1999-2020 (3875 sym, 4943d) |
| silver_panel | ✅ | 1999-2026 (3872 sym, 15.1M rows) |
| akshare | ⚠️ 3/5 probes | 日 K + 龙虎榜 + 资金流 OK; min60 + 财务 EM 端点已变 |
| tushare | ⚠️ no token | 包装好，缺 `TUSHARE_TOKEN` 环境变量 |
| baostock | ⚠️ missing | `pip install baostock` 即可 |
| **alpha181_gold** | ✅ | **7.1GB / 246 fields / 2018-01 → 2026-05** — 训练主源 |

完整报告：`runtime/reports/v7/v8/datasets_verification.{md,json}`

## tmux 操作

```bash
# 进入观察 3 panes
tmux attach -t qa_v8_deep

# pane 切换
Ctrl-b 然后 数字键

# 离开但保留训练
Ctrl-b 然后 d

# 杀掉训练
tmux kill-session -t qa_v8_deep
```

## 不进 tmux 查 GPU 状态

```bash
# 一行实时 GPU 占用
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader

# 5 秒刷新
watch -n 5 nvidia-smi

# 训练 log 过滤掉警告
LOG=$(ls -t runtime/logs/v8_deep/v8_deep_*.log | head -1)
tail -F $LOG | grep -vE "FutureWarning|daily = merged|pkg_resources"

# 看 epoch loss
tail -F $LOG | grep -E "epoch|loss|metrics"
```

## 结果查看（训练结束后）

```bash
RUN=$(ls -td runtime/reports/v8/deep/v8_deep_* | head -1)

# 三个 horizon 的 OOS metrics
for h in short_5d mid_5d_30d long_30d_120d; do
  echo "=== $h ==="
  cat $RUN/$h/backtest/metrics.json
  echo
done

# ensemble 输出
cat $RUN/ensemble_summary.json

# 看 daily decision report
cat $RUN/short_5d/daily_decision_report.md

# FT-Transformer 训练详细
ls $RUN/short_5d/ft/
cat $RUN/short_5d/ft/metrics.json | head -20
```

## 重启 / 调参

```bash
# 更长训练 + 更大模型
MAX_EPOCHS=80 D_TOKEN=256 N_BLOCKS=6 bash scripts/launch_v8_deep_sweep_tmux.sh

# 不同 universe
UNIVERSE_FILE=自己的列表.txt bash scripts/launch_v8_deep_sweep_tmux.sh

# 不同时间窗
TRAIN_START=2015-01-02 TRAIN_END=2022-12-31 TEST_END=2024-12-31 bash scripts/launch_v8_deep_sweep_tmux.sh

# 用 CPU（不推荐，慢）
PYTHON_BIN=$(pwd)/AI_quant_venv/bin/python ./scripts/launch_v8_deep_sweep_tmux.sh
# 然后在 train-v8-deep 调用里把 --require-gpu 改成 --no-require-gpu
```

## 性能基准（已测）

| Scope | Epochs | 用时 | Sharpe (OOS) | 备注 |
|---|---|---|---|---|
| smoke: 50 sym × 4 yr | 5 | 8s GPU train + 4s inference + 4s backtest = 19s 总 | -0.18 | 仅 verify GPU 通路 |
| 当前: 500 sym × 5.5 yr | 40 | 预估 8-12 min train / horizon × 3 = 30-40 min 总 | 待出 | 真正 production |

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `[fatal] --require-gpu set but CUDA not available` | torch 没有 CUDA build / GPU 占用 | `nvidia-smi` 看占用；`AI_quant_venv/bin/python -c "import torch; print(torch.cuda.is_available())"` |
| pane 0 卡在 "loading dataset" | 7.1GB parquet 加载慢 | 等 30-60 秒 |
| `OOM` / `CUDA out of memory` | batch_size 太大 | `BATCH_SIZE=4096 bash scripts/launch_v8_deep_sweep_tmux.sh` |
| OOS Sharpe 为负 | 模型欠拟合 / overfit / regime mismatch | 加 epochs；扩 universe；扩 train 窗口；用更长 horizon |
| `pane 0 显示 READY_TO_CLOSE` | 训练正常结束 | 看 ensemble_summary.json |
