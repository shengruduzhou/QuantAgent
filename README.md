# QuantAgent V7 / A-share PIT Research System

QuantAgent V7 是面向 A 股散户现实约束的 Point-in-Time quant research system。它不提供 financial advice，不承诺收益，不默认连接真实券商；目标是在真实数据、成本、slippage、T+1、涨跌停、停牌、ST、流动性、换手、回撤和 kill-switch 约束下，研究 out-of-sample net risk-adjusted return。

## 安全边界 / Safety Boundary

- 默认 `live_trading_enabled=false`、`dry_run=true`、`virtual_broker_only=true`。
- Agent 只能输出 structured evidence、views、constraints、confidence、risk flags、audit logs。
- Optimizer / Portfolio Construction 只能输出 `target_weights`。
- 只有 `OrderManager` 可以把 `target_weights` 转成 order intents。
- 任何 QMT submit path 前必须通过 Risk Gate、Kill Switch、execution constraint simulation、reconciliation 和 audit replay。
- Production mode 不允许 synthetic fallback；mock data 只用于 tests 和 smoke examples。

## 真实数据流程 / Real-Data Flow

```text
download / prepare Qlib CN data
-> build-market-panel-v7
-> build-akshare-v7 or build-fundamentals-v7
-> build-labels-v7
-> build V7 training dataset
-> train-alpha-v7
-> walk-forward-backtest-v7
-> paper-trade-v7
-> v7-live-readiness-report
```

Qlib 负责 market OHLCV、technical features、labels 和 backtest base。TuShare / AkShare 负责 financial statements、financial indicators、valuation fields 和 disclosure dates，并必须保存 `report_period / ann_date / available_at`。TradingView public pages 只作为 sentiment / attention context，不作为行情或基本面真值。

## 安装 / Install

核心依赖保持轻量：

```powershell
pip install -e .
```

真实数据依赖使用 optional extra：

```powershell
pip install -e ".[data]"
pip install -e ".[research]"
```

`data` 包含 AkShare、TuShare、pyarrow；`research` 保留 pyqlib/vectorbt。缺少 optional dependency 时，import 不应破坏离线研究流程；真实数据 CLI 会给出可操作错误。

## CLI / 命令

```powershell
quantagent download-qlib-v7 --target-dir ~/.qlib/qlib_data/cn_data --region cn
quantagent build-market-panel-v7 --provider-uri ~/.qlib/qlib_data/cn_data --symbols 600519.SH --start-date 2020-01-01 --end-date 2026-05-15
quantagent build-akshare-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --allow-network
quantagent build-labels-v7 --market-panel data/v7/market_panel.parquet --output data/v7/labels.parquet
quantagent train-alpha-v7 --dataset data/v7/training_dataset.parquet --output-dir artifacts/v7_alpha
quantagent walk-forward-backtest-v7 --target-weights reports/v7/target_weights.csv --market-panel data/v7/market_panel.parquet
quantagent paper-trade-v7 --target-weights reports/v7/target_weights.csv --market-panel data/v7/market_panel.parquet
quantagent v7-live-readiness-report --metrics artifacts/v7_alpha/metrics.json --paper-report reports/v7/paper_trade_report.json
```

Qlib 官方 CN download command 是：

```powershell
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## 关键配置 / Configs

- `configs/v7.default.yaml`：默认 strict_local research config。
- `configs/v7.mock.yaml`：tests 和 smoke examples 使用，不能作为 production-ready 依据。
- `configs/v7.realdata.yaml`：真实数据 data quality gates 和 execution constraints。
- `configs/v7.train.yaml`：Ridge / ElasticNet training 和 acceptance gates。
- `configs/v7.paper.yaml`：VirtualBroker paper trading 和 live-readiness prerequisites。

## 验证 / Validation

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
```

`git diff --check` 在 Windows checkout 中可能报告 CRLF warning；真正 whitespace error 必须修复。

## 文档 / Docs

- [V7 系统架构与 Agent 接口](docs/V7_系统架构与Agent接口.md)
- [V7 PIT 数据与财务特征](docs/V7_PIT数据与财务特征.md)
- [V7 算法风控回测与验收](docs/V7_算法风控回测与验收.md)
- [V7 证据摄取与交易规则](docs/V7_证据摄取与交易规则.md)
