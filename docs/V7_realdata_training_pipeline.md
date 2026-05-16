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

`QUANTAGENT_HOME` 覆盖全局 root，`QUANTAGENT_DATA_ROOT` 只覆盖 data tier。

## Data Sources

- Qlib：适合官方 CN free dump 覆盖期内的 OHLCV、technical features、labels、training slices、backtest base。
- AkShare market：适合 2020-09-25 之后的近年 A 股 OHLCV；`available_at` 使用下一 business day。
- AkShare/TuShare fundamentals：income、balance_sheet、cashflow、financial_indicator、dividend、valuation fields，所有 PIT 财务字段必须有 `report_period`、`ann_date`、`available_at`。
- EvidenceRecord：policy、announcement、news 原文必须保留 `source`、`published_at`、`available_at`、`raw_hash`、`confidence` 并进入 EvidenceStore。

## Qlib Setup

```powershell
cd E:\Project\QuantAgent
.\.venv\Scripts\Activate.ps1
$env:QUANTAGENT_HOME = "E:\Project\QuantAgent\runtime"

.\.venv\Scripts\quantagent.exe setup-qlib-v7 `
  --region cn `
  --interval 1d `
  --target-dir E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --run `
  --allow-community-fallback

.\.venv\Scripts\quantagent.exe check-qlib-v7 `
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --region cn `
  --symbols SH600519,SZ000001 `
  --start-date 2018-01-01 `
  --end-date 2020-09-25
```

Qlib CN instruments 使用 uppercase exchange prefix：`SH600519`、`SZ000001`。官方 free dump 覆盖 `2000-01-04 .. 2020-09-25`；更近日期要么自备 PIT CSV/Parquet 后用 Qlib `scripts/dump_bin.py`，要么使用 AkShare market panel。

## AkShare Setup

```powershell
.\.venv\Scripts\quantagent.exe build-akshare-market-panel-v7 `
  --symbols 600519.SH,600036.SH,000001.SZ `
  --start-date 2021-01-01 `
  --end-date 2026-05-15 `
  --allow-network

.\.venv\Scripts\quantagent.exe build-akshare-v7 `
  --symbols 600519.SH,600036.SH,000001.SZ `
  --start-date 2015-01-01 `
  --end-date 2026-05-15 `
  --allow-network

.\.venv\Scripts\quantagent.exe build-valuation-v7 `
  --as-of-dates 2026-05-15 `
  --symbols 600519.SH,600036.SH,000001.SZ `
  --allow-network
```

AkShare provider 写 income、balance_sheet、cashflow、financial_indicator、dividend。每个 manifest 包含 `source`、`function_name`、`params`、`row_count`、`schema_hash`、`fetched_at`、`warnings`、`failed_symbols`。网络、限流、空表、字段缺失、接口变更不会生成假数据。

## OOS Training Rule

`run-full-real-training-v7` 不允许 sample-in prediction/backtest。流程是：

1. build gold dataset。
2. 使用 configured split interface 生成 train / validation folds。
3. 每个 fold 只在 train rows fit。
4. 只对 validation rows 写 `walk_forward_predictions.csv`，`sample_role=validation`。
5. full pipeline 只读取 validation-only predictions 构建 target weights。
6. backtest 和 paper report 只跑 OOS target weights。

如果 predictions 不是 validation-only，full pipeline 会直接 raise。

## Command Chain

```powershell
.\.venv\Scripts\quantagent.exe build-labels-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet

.\.venv\Scripts\quantagent.exe materialize-factors-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --backend polars

.\.venv\Scripts\quantagent.exe build-training-dataset-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\Project\QuantAgent\runtime\data\v7\labels.parquet `
  --fundamentals-root E:\Project\QuantAgent\runtime\data\v7\silver\fundamentals `
  --valuation E:\Project\QuantAgent\runtime\data\v7\silver\valuation\valuation.parquet

.\.venv\Scripts\quantagent.exe train-alpha-v7 `
  --dataset E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\training_dataset.parquet `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

.\.venv\Scripts\quantagent.exe optimize-alpha-v7 `
  --dataset E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\training_dataset.parquet `
  --search-space configs/example_search_space.json `
  --sampler grid `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

.\.venv\Scripts\quantagent.exe run-full-real-training-v7 `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --labels E:\Project\QuantAgent\runtime\data\v7\labels.parquet `
  --model ridge `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5 `
  --optimizer-backend auto `
  --objective max_expected_alpha `
  --initial-cash 1000000 `
  --paper-report-output-dir E:\Project\QuantAgent\runtime\reports\v7\paper_report
```

## Paper Report Outputs

`run-full-real-training-v7`、`run-paper-backtest-v7`、`generate-paper-report-v7` 写：

- `selected_stocks.csv`
- `target_weights.parquet`
- `trades.csv`
- `failed_orders.csv`
- `holdings.csv`
- `pnl.csv`
- `paper_report.json`
- `paper_report.md`
- `paper_report.html`

Summary fields include `initial_cash`、`final_nav`、`realized_money_earned_lost`、`gross_return`、`net_return_after_estimated_costs`、`total_estimated_fees`、`total_estimated_slippage`、`max_drawdown`、`trade_count`、`failed_order_count`。

`selected_stocks.csv` 直接回答“买了什么股、什么时候第一次买、交易几次、估算盈利多少”。`trades.csv` 回答每次什么时候买/卖。`pnl.csv` 回答 daily NAV、daily PnL、drawdown。

## LLM Provider

LLM provider 支持 `disabled`、`openai`、`openai-compatible`、`gemini`。Gemini/Gemma 通过 Google AI Studio API key 调用：

```powershell
$env:GOOGLE_API_KEY = "<local-only-never-commit>"
$env:QUANTAGENT_LLM_PROVIDER = "gemini"
$env:QUANTAGENT_LLM_ENABLED = "true"
$env:QUANTAGENT_LLM_ALLOW_NETWORK = "true"
$env:QUANTAGENT_LLM_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
$env:QUANTAGENT_LLM_MODEL = "<google-ai-studio-model-id>"
$env:QUANTAGENT_LLM_API_KEY_ENV = "GOOGLE_API_KEY"
```

LLM 不能生成 orders；Optimizer 也不能生成 orders，只能输出 `target_weights`。Order intent 仍然只能由 `OrderManager` 在 dry-run / VirtualBroker path 中生成。

## Limits

系统只输出研究证据、OOS metrics、target weights、paper/backtest PnL 和风险证据，不承诺真实盈利。任何 live execution path 都必须另行审查，并且默认关闭。
