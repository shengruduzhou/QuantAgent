# QuantAgent V7 Codex 指令 / Codex Guide

## 项目目的 / Project Purpose

QuantAgent V7 是面向 A 股散户现实约束的 PIT quant research system。仓库不是 toy LLM trading demo，不提供 financial advice，不承诺收益。研究输出必须经过测试、回测、风控、paper trading 和 live-readiness reporting，才允许讨论任何 live execution path。

V7 覆盖：

- Point-in-Time financial provider (TuShare / AkShare) + local Parquet/CSV cache。
- Daily Evidence Ingestion Layer (`data/ingestion/*` + `EvidenceStore`)；每条 `EvidenceRecord` 必须带 source、published_at、available_at、raw_hash 和 confidence。
- Qlib CN market panel、technical features、label generation、training slices、backtest base。
- Dynamic theme discovery、industry chain graph、stock pool hard gate。
- Fundamental due diligence、Financial Fraud Risk、News Credibility、Intrinsic Valuation。
- Multi-horizon Alpha: 1 / 5 / 20 / 60 / 120 / 126 days。**生产模型 = FT-Transformer sleeves**（`cli/v8_deep.py train-v8-deep` + `configs/production_blend.json`，单命令物化 `scripts/materialize_production_composite.py`）。Ridge/ElasticNet 为 v7 classical 基线；`models/v7_deep_alpha.py` 等启发式 scorer 非生产（见文件头 STATUS WARNING）。
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
- **评测窗口纪律（2026-07 起）**：`configs/quarantined_windows.json` 定义禁评窗（被烧 holdout 2025-09-01→2026-05-18；冻结新鲜窗 2026-05-19+，正式首读 ≥120 交易日）。可信评测唯一入口 = `scripts/baseline_protocol.py` variant C（守卫 fail-closed）；任何数字引用须带 trust class（见 `BASELINE_TRUST_CLASSIFICATION.md`）。改进验收规则见 `ACCEPTANCE_RULES.md`。

## Code Style / 代码规范

- Python code、comments、docstrings、function names、class names、variable names、test names、config keys 使用 English。
- Markdown 必须 Chinese-English mixed，以中文说明为主，保留关键 English terms。
- 新增财务字段必须同时定义 `report_period`、`ann_date`、`available_at`，否则不能进入 PIT cache。
- Optional dependencies must degrade gracefully；real-data commands must report actionable install/setup errors。
- 优先做 wrappers、adapters、integration seams，不删除仍被引用的 SOTA components。
- 删除 unused code 或 obsolete `.md` 前，必须证明未被 imports、CLI、tests、README、AGENTS 或 docs 引用。
- 所有 silver / gold artifact 必须伴随一份 `<lake_root>/manifests/<dataset>.json`（`quantagent.data.manifest.DataManifest`）。
- 大数据/模型/报告默认写入 `E:\Project\QuantAgent\runtime\`（Windows）或 `~/AI_quant`（POSIX），通过 `QUANTAGENT_HOME` 环境变量覆盖。`quantagent.config.paths.quant_paths` 是单一来源。

## Real-Data Commands / 真实数据命令

```powershell
quantagent storage-info-v7 --ensure
quantagent setup-qlib-v7 --region cn                      # 仅打印官方下载命令
quantagent setup-qlib-v7 --region cn --run --allow-community-fallback   # 若 pyqlib 已装则直接下载
quantagent download-qlib-v7 --target-dir E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data --region cn
quantagent check-qlib-v7 --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data --symbols 600519.SH
quantagent build-market-panel-v7 --provider-uri E:\Project\QuantAgent\runtime\data\raw\qlib\cn_data --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --allow-network
quantagent build-valuation-v7 --as-of-dates 2026-05-15 --allow-network
quantagent build-labels-v7 --market-panel E:\Project\QuantAgent\runtime\data\v7\silver\market_panel\market_panel.parquet
quantagent build-training-dataset-v7 --market-panel ... --labels ... --fundamentals-root ...
quantagent train-alpha-v7 --dataset ...
quantagent train-alpha-v7 --dataset ... --model lightgbm           # real LightGBM, fail-loud if missing
quantagent train-alpha-v7 --dataset ... --model xgboost --allow-model-downgrade   # ridge fallback only with the flag
quantagent train-deep-alpha-v7 --dataset ... --horizons 1,5,20,60,120,126
quantagent optimize-alpha-v7 --dataset ... --search-space search.json --sampler grid
quantagent predict-alpha-v7 --model-dir ... --feature-dataset ...
quantagent build-target-weights-v7 --predictions ... --market-panel ... --sector-map ...
quantagent run-real-training-v7 --market-panel ... --labels ... --fundamentals-root ...
quantagent run-full-real-training-v7 --market-panel ... --labels ... --sector-map ...   # dataset → train → predict → target_weights → backtest
quantagent evaluate-alpha-v7 --metrics ... --paper-report ...
quantagent walk-forward-backtest-v7 --target-weights ... --market-panel ...
quantagent walk-forward-backtest-v7 --predictions ... --market-panel ... --sector-map ...   # optimiser runs first
quantagent paper-trade-v7 --target-weights ... --market-panel ...
quantagent v7-live-readiness-report --metrics ... --paper-report ...
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
- 至少一个 adverse market regime 验证通过（`evaluate_adverse_regime` 真实计算 bottom-quartile-day rank-IC；旧的 `adverse_regime_passed=True` 硬编码已删除）。
- paper trading report exists before live readiness can pass。
- 结果不是只在 mock data 上成立。

## Data 防错铁律 / Data Anti-Footgun

- `AkShareSectorProvider` 必须用 per-board membership endpoint 或 local mapping；**绝不** 把所有 industry 当成 cross-join 应用到每个 symbol。
- 财务报表合并必须走 `pit_wide_merge_statements`：按 statement type 加列前缀（income_revenue / balance_total_assets / cashflow_operating_cash_flow / indicator_*），按 PIT 四键 outer-merge，重复 `(symbol, report_period, available_at)` 必须 raise。
- 真实数据 manifest（`DataManifest`）必须包含 provider、source paths、generated_at、row_count、date_range、symbols、schema report、PIT violations、duplicate rate、warnings 和 content hash。

## New Modules / 新增模块

- `quantagent.config.paths` — 统一 `E:\Project\QuantAgent\runtime\` 存储布局，环境变量 `QUANTAGENT_HOME` / `QUANTAGENT_DATA_ROOT` 覆盖。
- `quantagent.training.splitters` — expanding / rolling / purged / chronological 走式 walk-forward 切分。
- `quantagent.training.optimize` — alpha 超参 grid / random search，默认写 `E:\Project\QuantAgent\runtime\reports\v7\optimization\`。
- `quantagent.factors.expr` — Alpha101-style 符号化因子 DSL，`Rank(TsMean(Returns(Close, 1), 5))`，零 lookahead 测试覆盖。
- `quantagent.models.ft_transformer` — FT-Transformer 表格架构（PyTorch 可选）。
- `quantagent.training.ft_transformer_trainer` — FT-Transformer trainer（AMP / checkpoint resume / 时序 validation 切分）。
- `quantagent.cli.v7_storage` — `storage-info-v7` / `setup-qlib-v7`。
- `quantagent.cli.v7_optimize` — `optimize-alpha-v7`。

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
