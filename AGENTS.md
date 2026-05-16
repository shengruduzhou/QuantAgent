# QuantAgent V7 Codex 指令 / Codex Guide

## 项目目的 / Project Purpose

QuantAgent V7 是面向 A 股散户现实约束的 PIT quant research system。仓库不是 toy LLM trading demo，不提供 financial advice，不承诺收益。研究输出必须经过测试、回测、风控、paper trading 和 live-readiness reporting，才允许讨论任何 live execution path。

V7 覆盖：

- Point-in-Time financial provider (TuShare / AkShare) + local Parquet/CSV cache。
- Daily Evidence Ingestion Layer (`data/ingestion/*` + `EvidenceStore`)；每条 `EvidenceRecord` 必须带 source、published_at、available_at、raw_hash 和 confidence。
- Qlib CN market panel、technical features、label generation、training slices、backtest base。
- Dynamic theme discovery、industry chain graph、stock pool hard gate。
- Fundamental due diligence、Financial Fraud Risk、News Credibility、Intrinsic Valuation。
- Multi-horizon Alpha: 1 / 5 / 20 / 60 / 120 / 126 days，默认 Ridge，ElasticNet optional，Deep Alpha disabled unless real training code is wired。
- Purged walk-forward CV、model artifacts、metrics、acceptance gates。
- A-share execution simulation：T+1、limit-up/down、suspension、ST、lot size、volume cap、slippage、cost、partial fills、failed order audit。
- QMT execution-preparation、Risk Gate、Kill Switch、Reconciliation、Audit Replay。

## A 股安全约束 / A-share Safety Constraints

- No live trading by default：默认禁止实盘交易。
- `QMTGateway` 必须默认 `dry_run=True`，`live_trading_enabled=False`。
- Agents never emit orders：LLM / Agent 只能输出 evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer never emits orders：Portfolio Construction 只能输出 `target_weights`。
- Only `OrderManager` converts target weights into order intents。
- QMT submit 前必须通过 risk gate、kill switch、execution constraint simulation、reconciliation、audit replay。
- Production mode must not use synthetic fallback；mock data 只允许在 tests 和 smoke examples。
- 不允许新增任何 guaranteed profitability 或收益保证表述。

## Code Style / 代码规范

- Python code、comments、docstrings、function names、class names、variable names、test names、config keys 使用 English。
- Markdown 必须 Chinese-English mixed，以中文说明为主，保留关键 English terms。
- 新增财务字段必须同时定义 `report_period`、`ann_date`、`available_at`，否则不能进入 PIT cache。
- Optional dependencies must degrade gracefully；real-data commands must report actionable install/setup errors。
- 优先做 wrappers、adapters、integration seams，不删除仍被引用的 SOTA components。
- 删除 unused code 或 obsolete `.md` 前，必须证明未被 imports、CLI、tests、README、AGENTS 或 docs 引用。
- 所有 silver / gold artifact 必须伴随一份 `data/v7/manifests/<dataset>.json`（`quantagent.data.manifest.DataManifest`）。

## Real-Data Commands / 真实数据命令

```powershell
quantagent download-qlib-v7 --target-dir ~/.qlib/qlib_data/cn_data --region cn
quantagent check-qlib-v7 --provider-uri ~/.qlib/qlib_data/cn_data --symbols 600519.SH
quantagent build-market-panel-v7 --provider-uri ~/.qlib/qlib_data/cn_data --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --allow-network
quantagent build-valuation-v7 --as-of-dates 2026-05-15 --allow-network
quantagent build-labels-v7 --market-panel data/v7/silver/market_panel/market_panel.parquet --output data/v7/labels.parquet
quantagent build-training-dataset-v7 --market-panel data/v7/silver/market_panel/market_panel.parquet --labels data/v7/labels.parquet --fundamentals-root data/v7/silver/fundamentals --output data/v7/gold/training_dataset/training_dataset.parquet
quantagent train-alpha-v7 --dataset data/v7/gold/training_dataset/training_dataset.parquet --output-dir artifacts/v7_alpha
quantagent train-deep-alpha-v7 --dataset data/v7/gold/training_dataset/training_dataset.parquet --output-dir artifacts/v7_alpha/deep --horizons 1,5,20,60,120,126
quantagent run-real-training-v7 --market-panel data/v7/silver/market_panel/market_panel.parquet --labels data/v7/labels.parquet --fundamentals-root data/v7/silver/fundamentals
quantagent evaluate-alpha-v7 --metrics artifacts/v7_alpha/metrics.json --paper-report reports/v7/paper_trade_report.json
quantagent walk-forward-backtest-v7 --target-weights reports/v7/target_weights.csv --market-panel data/v7/silver/market_panel/market_panel.parquet
quantagent paper-trade-v7 --target-weights reports/v7/target_weights.csv --market-panel data/v7/silver/market_panel/market_panel.parquet
quantagent v7-live-readiness-report --metrics artifacts/v7_alpha/metrics.json --paper-report reports/v7/paper_trade_report.json
```

Qlib CN official command:

```powershell
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## Acceptance Gates / 验收门槛

Model 不能标记 production-ready，除非：

- zero PIT violations。
- 每个 horizon 有足够 training rows、symbol coverage、date coverage。
- out-of-sample RankIC stability 为正。
- turnover-adjusted net return after cost 通过阈值。
- max drawdown 低于配置阈值。
- no single factor dominates unrealistically。
- 至少一个 adverse market regime 验证通过。
- paper trading report exists before live readiness can pass。
- 结果不是只在 mock data 上成立。

## Testing Commands / 测试命令

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
```

## Provider 责任 / Provider Responsibilities

- Qlib：行情、technical factors、label generation、training slices、backtest base。
- TuShare / AkShare：财务报表、财务指标、估值字段、公告披露日期。
- TradingView public pages：sentiment / attention context，不作为基本面或行情真值。
- Policy、announcement、news 原文必须保留 `source / published_at / available_at / raw_hash / confidence` 并进入 `EvidenceStore`。
