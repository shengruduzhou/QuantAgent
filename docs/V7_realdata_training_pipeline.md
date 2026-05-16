# V7 Real-Data Training Pipeline / 真实数据训练流程

V7 real-data path 从 Qlib CN market data、AkShare/TuShare PIT financial data、valuation snapshots 和 factor DSL 构建 gold training dataset，再通过 purged / embargo walk-forward OOS training 产出 metrics、predictions、target weights、paper/backtest report。默认不启用 live trading，不使用 synthetic fallback。

## Storage

默认 Windows root：

```text
E:\Project\QuantAgent\runtime\
  data\raw\qlib\cn_data\
  data\v7\raw\akshare\fundamentals\
  data\v7\silver\market_panel\
  data\v7\silver\fundamentals\
  data\v7\silver\valuation\
  data\v7\silver\factors\
  data\v7\gold\training_dataset\
  data\v7\manifests\
  models\v7_alpha\
  predictions\
  target_weights\
  reports\v7\
  logs\
```

`QUANTAGENT_HOME` 覆盖全局 root；`QUANTAGENT_DATA_ROOT` 只覆盖 data tier。

## Qlib Setup

```powershell
cd E:\Project\QuantAgent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
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

quantagent check-qlib-v7 `
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --region cn `
  --symbols SH600519,SZ000001 `
  --start-date 2018-01-01 `
  --end-date 2020-09-25
```

> Qlib CN instruments use uppercase exchange prefix (`SH600519`,
> `SZ000001`). The official free Qlib CN release covers
> `2000-01-04 .. 2020-09-25`; pass an explicit range inside that window,
> or prepare a custom dump (`scripts/dump_bin.py` against your own PIT
> CSV/Parquet OHLCV) for more recent dates.

`setup-qlib-v7` dry-run 只打印官方 Qlib command；`--run` 使用 `qlib.tests.data.GetData`。失败时 fail-loud，并给出 community mirror、自备 CSV/Parquet `scripts/dump_bin.py` 和 `check-qlib-v7` 验证路径。`download-qlib-v7` 仅保留为 deprecated alias。

## AkShare Setup

```powershell
quantagent build-akshare-v7 `
  --symbols 600519.SH,000001.SZ `
  --start-date 2020-01-01 `
  --end-date 2024-12-31 `
  --allow-network

quantagent smoke-akshare-v7 `
  --symbols 600519.SH,000001.SZ `
  --start-date 2024-01-01 `
  --end-date 2024-12-31 `
  --as-of-date 2024-12-31 `
  --allow-network

quantagent build-valuation-v7 `
  --as-of-dates 2024-12-31 `
  --symbols 600519.SH,000001.SZ `
  --allow-network
```

AkShare provider 会写 income、balance_sheet、cashflow、financial_indicator、dividend。每个 manifest 包含 `source`、`function_name`、`params`、`row_count`、`schema_hash`、`fetched_at`、`warnings`、`failed_symbols`。网络、限流、空表、字段缺失、接口变更不会生成假数据。

## OOS Training Rule

`run-full-real-training-v7` 不允许 sample-in prediction/backtest。流程是：

1. build gold dataset。
2. 使用 configured split interface 生成 train / validation folds。
3. 每个 fold 只在 train rows fit。
4. 只对 validation rows 写 `walk_forward_predictions.csv`，`sample_role=validation`。
5. full pipeline 只读取 validation-only predictions 构建 target weights。
6. backtest 只跑 OOS target weights。

如果 predictions 不是 validation-only，full pipeline 会直接 raise。

## Command Chain

```powershell
quantagent build-market-panel-v7 `
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --symbols SH600519,SH600036,SZ000001 `
  --start-date 2018-01-01 `
  --end-date 2020-09-25

quantagent build-labels-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet

quantagent materialize-factors-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --backend polars

quantagent build-training-dataset-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\Project\QuantAgent\runtime\data\v7\labels.parquet `
  --fundamentals-root E:\Project\QuantAgent\runtime\data\v7\raw\akshare\fundamentals `
  --valuation E:\Project\QuantAgent\runtime\data\v7\silver\valuation\valuation.parquet

quantagent train-alpha-v7 `
  --dataset E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\training_dataset.parquet `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent train-alpha-v7 `
  --dataset E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\training_dataset.parquet `
  --model ft_transformer `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent optimize-alpha-v7 `
  --dataset E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\training_dataset.parquet `
  --search-space configs/example_search_space.json `
  --sampler grid `
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
```

## Paper Report Outputs

`run-paper-backtest-v7` / `generate-paper-report-v7` writes:

- `selected_stocks.csv`
- `target_weights.parquet`
- `trades.csv`
- `failed_orders.csv`
- `holdings.csv`
- `pnl.csv`
- `paper_report.json`
- `paper_report.md`
- `paper_report.html`

Summary fields include `initial_cash`, `final_nav`, `realized_money_earned_lost`, `gross_return`, `net_return_after_estimated_costs`, `total_estimated_fees`, `total_estimated_slippage`, `max_drawdown`, `trade_count`, and `failed_order_count`。

## Limits

该系统只输出研究、OOS metrics、target weights、paper/backtest PnL 和风险证据，不承诺真实盈利。任何 live execution path 都必须另行审查，并且默认关闭。
