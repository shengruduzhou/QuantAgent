# V7 Real-Data Training Pipeline / 真实数据训练流程

V7 的 real-data path 从 Qlib CN market data、AkShare/TuShare PIT financial data、valuation snapshots 和 Alpha101-style factors 构建 gold training dataset，再通过 walk-forward out-of-sample training 产出 metrics、predictions、target weights、paper/backtest report。全流程默认不启用 live trading，不使用 synthetic fallback。

## Storage

默认 Windows root 是 `E:\AI量化\`：

```text
E:\AI量化\data\v7\raw\
E:\AI量化\data\v7\silver\
E:\AI量化\data\v7\gold\training_dataset\
E:\AI量化\data\v7\manifests\
E:\AI量化\models\v7_alpha\
E:\AI量化\predictions\
E:\AI量化\target_weights\
E:\AI量化\reports\v7\
E:\AI量化\logs\
```

`QUANTAGENT_HOME` 覆盖全局 root，`QUANTAGENT_DATA_ROOT` 只覆盖 data tier。

## Qlib Setup

```powershell
quantagent storage-info-v7 --ensure

quantagent setup-qlib-v7 `
  --region cn `
  --interval 1d `
  --run `
  --allow-community-fallback
```

`setup-qlib-v7` dry-run 时只打印官方 Qlib command；`--run` 使用 `qlib.tests.data.GetData`。失败时 fail-loud，并给出 official release tarball、community mirror 或 `scripts/dump_bin.py` 的人工 fallback 指令。

`download-qlib-v7` 仅保留为 deprecated alias，文档和新脚本应使用 `setup-qlib-v7`。

## OOS Training Rule

`run-full-real-training-v7` 不允许 sample-in prediction/backtest。流程是：

1. build gold dataset。
2. 用 configured split interface 产生 train / validation folds。
3. 每个 fold 只在 train rows 上 fit。
4. 只对 validation rows 写 `walk_forward_predictions.csv`，`sample_role=validation`。
5. full pipeline 只读取 validation-only predictions 构建 target weights。
6. backtest 只跑 out-of-sample target weights。

如果 predictions 不是 validation-only，full pipeline 会直接 raise。

## Commands

```powershell
quantagent build-market-panel-v7 `
  --provider-uri E:\AI量化\data\raw\qlib\cn_data `
  --symbols 600519.SH,000858.SZ `
  --start-date 2020-01-01 `
  --end-date 2026-05-15

quantagent build-akshare-v7 `
  --symbols 600519.SH,000858.SZ `
  --start-date 2020-01-01 `
  --end-date 2026-05-15 `
  --allow-network

quantagent build-valuation-v7 --as-of-dates 2026-05-15 --allow-network

quantagent build-labels-v7 `
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet

quantagent materialize-factors-v7 `
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet `
  --backend polars

quantagent build-training-dataset-v7 `
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\AI量化\data\v7\labels.parquet `
  --fundamentals-root E:\AI量化\data\v7\raw\akshare\fundamentals

quantagent train-alpha-v7 `
  --dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet `
  --model ft_transformer `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent run-full-real-training-v7 `
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\AI量化\data\v7\labels.parquet `
  --sector-map E:\AI量化\data\v7\silver\sector\sector_map.csv `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5
```

## Optimization

`optimize-alpha-v7` uses the same split interface as the trainer and optimizes validation metrics, not sample-in metrics. Supported objectives:

- `rank_ic_mean`
- `rank_ic_stability`
- `turnover_adjusted_net_return`
- `max_drawdown`
- `sharpe_like`
- `information_ratio_like`
- `hit_rate`

The report is written to `E:\AI量化\reports\v7\optimization\optimization_report.json` by default.
