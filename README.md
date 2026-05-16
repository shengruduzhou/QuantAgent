# QuantAgent V7 / A-share PIT Research System

QuantAgent V7 是面向 A 股现实交易约束的 Point-in-Time quant research system。它不是 toy LLM trading demo，不提供 financial advice，不承诺收益，也不默认连接真实券商。工程目标是用真实数据生成可审计的 out-of-sample metrics、risk reports、target weights、paper-trading/backtest outputs，供后续人工判断。

## Safety Boundary

- `QMTGateway` 默认 `dry_run=True`、`live_trading_enabled=False`。
- Agent / LLM 只能输出 evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer / portfolio construction 只能输出 `target_weights`。
- 只有 `OrderManager` 可以把 `target_weights` 转成 order intents。
- QMT submit 前必须经过 risk gate、kill switch、execution constraint simulation、reconciliation、audit replay。
- Production / real-data path 禁止 synthetic fallback；mock data 只允许 tests 和 smoke examples。

## Storage Layout

所有大数据、模型、predictions、target weights、reports、logs、cache 默认写到仓库外：

```text
E:\AI量化\
  data\v7\raw\{qlib,akshare,tushare,disclosures}\
  data\v7\silver\{market_panel,fundamentals,valuation,disclosures,factors}\
  data\v7\gold\training_dataset\
  data\v7\manifests\
  models\v7_alpha\
  predictions\
  target_weights\
  reports\v7\
  logs\
  cache\
```

`quantagent.config.paths.quant_paths()` 是单一来源。`QUANTAGENT_HOME` 覆盖全局 root，`QUANTAGENT_DATA_ROOT` 只覆盖 data tier。

```powershell
$env:QUANTAGENT_HOME = "E:\AI量化"
quantagent storage-info-v7 --ensure
```

## Real-Data Command Chain

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[data,training,research,optimization]"

quantagent storage-info-v7 --ensure

quantagent setup-qlib-v7 --region cn --interval 1d --run --allow-community-fallback

quantagent check-qlib-v7 `
  --provider-uri E:\AI量化\data\raw\qlib\cn_data `
  --symbols 600519.SH,000858.SZ `
  --start-date 2020-01-01 `
  --end-date 2026-05-15

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
  --model lightgbm `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent train-alpha-v7 `
  --dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet `
  --model ft_transformer `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent optimize-alpha-v7 `
  --dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet `
  --search-space configs/example_search_space.json `
  --sampler grid `
  --objective rank_ic_mean `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent predict-alpha-v7 `
  --model-dir E:\AI量化\models\v7_alpha `
  --feature-dataset E:\AI量化\data\v7\gold\training_dataset\training_dataset.parquet

quantagent build-target-weights-v7 `
  --predictions E:\AI量化\predictions\predictions.parquet `
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet

quantagent walk-forward-backtest-v7 `
  --target-weights E:\AI量化\target_weights\target_weights.parquet `
  --market-panel E:\AI量化\data\v7\silver\market_panel\market_panel.parquet

quantagent v7-live-readiness-report `
  --metrics E:\AI量化\models\v7_alpha\metrics.json `
  --paper-report E:\AI量化\reports\v7\paper_report.json
```

`run-full-real-training-v7` 现在只使用训练阶段产生的 validation-only out-of-sample predictions 来构建 target weights 和 backtest；不会再对同一 gold dataset 做 in-sample prediction/backtest。

## Training and Models

- `train-alpha-v7 --model ridge|elastic_net|lightgbm|xgboost|ft_transformer`。
- LightGBM / XGBoost 缺依赖时默认 fail-loud；只有显式 `--allow-model-downgrade` 才能回退 ridge。
- `ft_transformer` 使用 PyTorch checkpoint `ft_transformer.pt`，缺 PyTorch 时 fail-loud。
- `train-deep-alpha-v7` 保留原 deep alpha trainer；默认输出到 `E:\AI量化\models\v7_alpha\deep`。
- `predict-alpha-v7` 支持 classical、deep alpha、FT-Transformer artifact layouts。
- split interface: `--split-mode expanding|rolling|purged|chronological`，并支持 `--valid-size-days`、`--min-train-days`、`--rolling-train-days`、`--embargo-days`、`--purge-days`。`purge_days` 未显式传入时默认不小于最大 label horizon。

## Feature and PIT Contract

- Qlib 负责 market panel、technical features、labels、backtest base。
- AkShare / TuShare 负责 financial statements、financial indicators、valuation fields、disclosure dates。
- Financial rows 必须包含 `report_period`、`ann_date`、`available_at`。
- Training feature schema 禁止 label leakage、raw forward label leakage、same-day close-derived leakage。
- `available_at <= trade_date` 是 gold training dataset 的硬约束。
- `quantagent.factors.expr` 支持 pandas 和 Polars backend：`build_factor_frame(frame, backend="polars")`。

## Validation

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall -q src
git diff --check
quantagent --help
```

`git diff --check` 在 Windows checkout 可能报告 CRLF warning；真实 whitespace error 必须修复。
