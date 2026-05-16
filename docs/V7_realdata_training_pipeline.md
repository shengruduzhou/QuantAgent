# V7 Real-Data Training Pipeline / 真实数据训练流程

V7 的 real-data path 从 Qlib CN market data、AkShare/TuShare PIT financial data、valuation snapshots 和 Alpha101-style factors 构建 gold training dataset，再通过 walk-forward out-of-sample training 产出 metrics、predictions、target weights、paper/backtest report。全流程默认不启用 live trading，不使用 synthetic fallback。

## Storage

默认 Windows root 是 `E:\Project\QuantAgent\runtime\`：

```text
E:\Project\QuantAgent\runtime\data\v7\raw\
E:\Project\QuantAgent\runtime\data\v7\silver\
E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\
E:\Project\QuantAgent\runtime\data\v7\manifests\
E:\Project\QuantAgent\runtime\models\v7_alpha\
E:\Project\QuantAgent\runtime\predictions\
E:\Project\QuantAgent\runtime\target_weights\
E:\Project\QuantAgent\runtime\reports\v7\
E:\Project\QuantAgent\runtime\logs\
```

`QUANTAGENT_HOME` 覆盖全局 root，`QUANTAGENT_DATA_ROOT` 只覆盖 data tier。

## Qlib Setup

```powershell
cd E:\Project\QuantAgent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[data,training,research,optimization]"
pip install pyqlib akshare polars lightgbm xgboost torch
$env:QUANTAGENT_HOME = "E:\Project\QuantAgent\runtime"

quantagent storage-info-v7 --ensure

quantagent setup-qlib-v7 `
  --region cn `
  --interval 1d `
  --target-dir E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
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
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
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
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet

quantagent materialize-factors-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --backend polars

quantagent build-training-dataset-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\Project\QuantAgent\runtime\data\v7\labels.parquet `
  --fundamentals-root E:\Project\QuantAgent\runtime\data\v7\raw\akshare\fundamentals

quantagent train-alpha-v7 `
  --dataset E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\training_dataset.parquet `
  --model ft_transformer `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent run-full-real-training-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\Project\QuantAgent\runtime\data\v7\labels.parquet `
  --sector-map E:\Project\QuantAgent\runtime\data\v7\silver\sector\sector_map.csv `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent run-paper-backtest-v7 `
  --target-weights E:\Project\QuantAgent\runtime\target_weights\target_weights.parquet `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --initial-cash 1000000 `
  --output-dir E:\Project\QuantAgent\runtime\reports\v7\paper_report

quantagent generate-paper-report-v7 `
  --target-weights E:\Project\QuantAgent\runtime\target_weights\target_weights.parquet `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --initial-cash 1000000 `
  --output-dir E:\Project\QuantAgent\runtime\reports\v7\paper_report
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

The report is written to `E:\Project\QuantAgent\runtime\reports\v7\optimization\optimization_report.json` by default.

## Paper Report Outputs

`run-paper-backtest-v7` / `generate-paper-report-v7` writes:

- `selected_stocks.csv`
- `target_weights.parquet` from the upstream optimizer
- `trades.csv`
- `failed_orders.csv`
- `holdings.csv`
- `pnl.csv`
- `paper_report.json`
- `paper_report.md`
- `paper_report.html`

Summary fields include `initial_cash`, `final_nav`, `realized_money_earned_lost`, `gross_return`, `net_return_after_estimated_costs`, `total_estimated_fees`, `total_estimated_slippage`, `max_drawdown`, `trade_count`, and `failed_order_count`.
