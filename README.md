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
E:\Project\QuantAgent\runtime\
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
$env:QUANTAGENT_HOME = "E:\Project\QuantAgent\runtime"
quantagent storage-info-v7 --ensure
```

## Real-Data Command Chain

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

quantagent check-qlib-v7 `
  --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data `
  --symbols 600519.SH,000858.SZ `
  --start-date 2020-01-01 `
  --end-date 2026-05-15

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
  --model lightgbm `
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
  --objective rank_ic_mean `
  --split-mode rolling `
  --purge-days 126 `
  --embargo-days 5

quantagent predict-alpha-v7 `
  --model-dir E:\Project\QuantAgent\runtime\models\v7_alpha `
  --feature-dataset E:\Project\QuantAgent\runtime\data\v7\gold\training_dataset\training_dataset.parquet

quantagent build-target-weights-v7 `
  --predictions E:\Project\QuantAgent\runtime\predictions\predictions.parquet `
  --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet `
  --optimizer-backend auto `
  --objective max_expected_alpha

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

quantagent v7-live-readiness-report `
  --metrics E:\Project\QuantAgent\runtime\models\v7_alpha\metrics.json `
  --paper-report E:\Project\QuantAgent\runtime\reports\v7\paper_report\paper_report.json
```

`run-full-real-training-v7` 现在只使用训练阶段产生的 validation-only out-of-sample predictions 来构建 target weights 和 backtest；不会再对同一 gold dataset 做 in-sample prediction/backtest。

## Training and Models

- `train-alpha-v7 --model ridge|elastic_net|lightgbm|xgboost|ft_transformer`。
- LightGBM / XGBoost 缺依赖时默认 fail-loud；只有显式 `--allow-model-downgrade` 才能回退 ridge。
- `ft_transformer` 使用 PyTorch checkpoint `ft_transformer.pt`，缺 PyTorch 时 fail-loud。
- `train-deep-alpha-v7` 保留原 deep alpha trainer；默认输出到 `E:\Project\QuantAgent\runtime\models\v7_alpha\deep`。
- `predict-alpha-v7` 支持 classical、deep alpha、FT-Transformer artifact layouts。
- split interface: `--split-mode expanding|rolling|purged|chronological`，并支持 `--valid-size-days`、`--min-train-days`、`--rolling-train-days`、`--embargo-days`、`--purge-days`。`purge_days` 未显式传入时默认不小于最大 label horizon。

## LLM API Setup

LLM is optional and disabled unless both config and environment allow it. Keep real keys out of git:

```powershell
$env:OPENAI_API_KEY = "<set-in-local-shell-only>"
$env:QUANTAGENT_HOME = "E:\Project\QuantAgent\runtime"
```

Config supports:

```yaml
llm_skills:
  provider: openai              # openai | openai-compatible | disabled
  endpoint: https://api.openai.com/v1/responses
  model: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
  allow_network: false
```

LLM outputs remain limited to explanation、归因、财报摘要、新闻可信度、主题链路分析和风险解释；orders 只能来自 deterministic/cvxpy target-weight builder 后的 `OrderManager` dry-run path。

## Feature and PIT Contract

- Qlib 负责 market panel、technical features、labels、backtest base。
- AkShare / TuShare 负责 financial statements、financial indicators、valuation fields、disclosure dates。
- Financial rows 必须包含 `report_period`、`ann_date`、`available_at`。
- Training feature schema 禁止 label leakage、raw forward label leakage、same-day close-derived leakage。
- `available_at <= trade_date` 是 gold training dataset 的硬约束。
- `quantagent.factors.expr` 支持 pandas 和 Polars backend：`build_factor_frame(frame, backend="polars")`。
- `materialize-factors-v7` 同步写 `<output>.manifest.json`，记录 factor expression、lookback、required columns、backend 和 Polars fallback。
- `run-paper-backtest-v7` 输出 `selected_stocks.csv`、`trades.csv`、`failed_orders.csv`、`holdings.csv`、`pnl.csv`、`paper_report.json`、`paper_report.md`、`paper_report.html`。

## Validation

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall -q src
git diff --check
quantagent --help
```

`git diff --check` 在 Windows checkout 可能报告 CRLF warning；真实 whitespace error 必须修复。
