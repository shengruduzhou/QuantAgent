# QuantAgent V7：A 股 PIT 研究与纸面交易系统 / A-share PIT Research and Paper Trading System

QuantAgent V7 面向 A 股散户现实约束，构建 Point-in-Time 数据、证据摄取、主题研究、基本面尽调、多周期 Alpha、组合构建、风控、回测和 dry-run 执行准备的闭环。系统不提供 financial advice，不默认连接真实券商，也不保证盈利；目标是在交易成本、滑点、T+1、涨跌停、停牌、ST、流动性、换手、回撤和 kill-switch 约束下研究 out-of-sample positive expected value。

## 安全边界 / Safety Boundary

- `live_trading_enabled=false`、`dry_run=true`、`virtual_broker_only=true` 是默认边界。
- Agent 只能输出 evidence、scores、constraints、risk flags、audit logs 和 rationale，不能输出 orders。
- Optimizer / Portfolio Construction 只能输出 `target_weights`，不能输出 broker orders。
- `OrderManager` 是唯一允许把 `target_weights` 转成 order intents 的组件。
- 任何 QMT submit path 之前必须经过 Risk Gate、Kill Switch、execution constraint simulation、reconciliation 和 audit replay。
- Stock Pool Hard Gate 打空时不会回退到 full universe；pipeline 返回 `stock_pool_gate_failed`，`target_weights` 为空，audit reason 为 `empty_after_stock_pool_hard_gate`。

## 主流程 / V7 Daily Flow

```text
V7DataStage
-> V7EvidenceStage
-> V7ThemeStage
-> V7FundamentalStage
-> V7AlphaStage
-> V7PortfolioStage
-> V7RiskStage
-> V7ReportStage
```

公开入口保持兼容：

```python
from quantagent.services.v7_pipeline_service import run_daily_v7_research

result = run_daily_v7_research("configs/v7.mock.yaml", as_of_date="2026-05-14")
```

## 数据职责 / Provider Responsibilities

- Qlib 只负责 market data、technical features、label slicing 和 backtest base，不承载财务事实。
- TuShare / AkShare 负责 financial statements、financial indicators、valuation fields 和 disclosure dates，并必须写入 `report_period / ann_date / available_at`。
- TradingView public pages 只作为 sentiment / attention context，不作为基本面或行情真值。
- Policy、disclosure、news 原文必须保留 `source / published_at / available_at / raw_hash / confidence` 并进入 `EvidenceStore`。

## 证据与爬虫 / Evidence and Crawler

`src/quantagent/data/crawler/` 提供 production-style public crawler facade：

- global token-bucket rate limit，默认不超过 5 requests/second；
- per-domain token-bucket rate limit，默认不超过 1 request/second；
- robots.txt 与 domain allowlist；
- canonical URL normalization；
- ETag / Last-Modified support；
- proxy provider interface，仅用于可靠性和受控出口；
- CAPTCHA / blocked page detection，检测后标记 blocked，不绕过验证码、不重试风暴。

`NewsIngestor` 不再只靠 keyword tagging。它按 exact keyword seed、company/security dictionary、industry/theme ontology、optional embedding reranker、cross-source validation 分层处理。Rumor 会降低 confidence；single low-reliability source 不能生成 core-pool trade signal。

## 数据质量 / Data Quality

`V7DataHub` 返回 `data_mode.quality_report`，覆盖 row count、missing columns、source reliability、duplicate rate 和 PIT violation count。`EvidenceStore.quality_report()` 对 evidence partitions 做同样报告。

## CLI / 命令

```powershell
quantagent validate-v7 --config configs/v7.default.yaml
quantagent run-daily-v7 --config configs/v7.mock.yaml --date 2026-05-14 --output-dir reports/v7
quantagent build-fundamentals-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --provider akshare --allow-network
quantagent check-qlib-v7 --provider-uri D:\qlib_data\cn_data --symbols 600519.SH --start-date 2026-05-01 --end-date 2026-05-15
```

## 验证 / Validation

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
```

`git diff --check` 在 Windows checkout 中可能显示 CRLF warning；实际 whitespace error 仍需修复。

## 文档 / Docs

- [V7 系统架构与 Agent 接口](docs/V7_系统架构与Agent接口.md)
- [V7 证据摄取与交易规则](docs/V7_证据摄取与交易规则.md)
- [V7 PIT 数据与财务特征](docs/V7_PIT数据与财务特征.md)
- [V7 hardening migration](docs/V7_hardening_migration.md)
