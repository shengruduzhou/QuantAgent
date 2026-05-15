# V7 系统架构与 Agent 接口 / System Architecture and Agent Contracts

## 目标 / Goal

QuantAgent V7 是 A 股 PIT research pipeline，不是 LLM trading demo。它把 real-data providers、EvidenceStore、Theme Discovery、Industry Chain Reasoner、Stock Pool Hard Gate、Fundamental Due Diligence、Financial Fraud Risk、News Credibility、Multi-Horizon Alpha、Portfolio Construction、Risk Gate、Backtest Attribution 和 Audit 放在同一条可回放链路里。

核心边界：Agent 不产生 orders；Portfolio Construction 只产生 `target_weights`；`OrderManager` 才能产生 dry-run order intents。

## Stage Pipeline / 阶段化 Pipeline

`run_daily_v7_research(config, as_of_date)` 仍是公开 API，但内部拆成 stage-level services：

| Stage | 中文职责 | English responsibility |
| --- | --- | --- |
| `V7DataStage` | 加载 PIT bundle 与 LLM dry-run seam | Load `V7DataHub` bundle and runtime clients |
| `V7EvidenceStage` | 政策、公告、新闻证据标准化 | Normalize policy, disclosure and news evidence |
| `V7ThemeStage` | 主题发现与证据驱动产业链 | Discover themes and reason industry chain |
| `V7FundamentalStage` | 财务、估值、fraud、宏观行业分析 | Build fundamentals, valuation, fraud and economics views |
| `V7AlphaStage` | universe、factor applicability、stock pool gate、alpha | Build universe, enforce gates and generate alpha |
| `V7PortfolioStage` | sleeve 与 `target_weights` | Construct portfolio target weights only |
| `V7RiskStage` | execution constraints、risk gate、backtest、audit | Apply risk, execution simulation, backtest and audit |
| `V7ReportStage` | JSON-safe report | Assemble stable report payload |

## Agent Contract / Agent 契约

`src/quantagent/v7/agent_contracts.py` 现在同时提供 static specs 和 runtime validators：

- `validate_agent_specs()` 检查所有 `AgentSpec` 都不能输出 `OrderIntent`。
- `validate_agent_output(agent_name, payload)` 递归检查 runtime payload。
- `assert_agent_output_valid(agent_name, payload)` 在违规时抛出 `AgentContractViolation`。

Runtime rules：

- 禁止 `order / orders / order_intent / order_intents / broker_order` 等字段。
- Evidence-like payload 必须包含 `source / available_at / raw_hash / confidence`。
- Portfolio、hedge、execution、risk、backtest 等 downstream decision 必须带 `audit_trail / audit_log / audit / decision_id / rationale` 之一。

## Stock Pool Gate / 股票池硬门槛

`stock_pool_gate` 是 alpha 前置 hard gate。允许进入 alpha 的成员只能来自：

- `core_beneficiary_pool`
- `strong_correlation_pool`
- `optional_satellite_pool` 且 `source_confidence >= threshold`

`watchlist / exclusion / false_association / no_factor_coverage` 会被剔除。若 gate 后为空，pipeline 不再回退到 full universe，而是返回：

```text
stock_pool_gate.gate_failed = true
stock_pool_gate.audit_reason = empty_after_stock_pool_hard_gate
multi_horizon_alpha = {}
portfolio_plan.target_weights = {}
risk_report.risk_passed = false
audit_log.final_decision_reason = empty_after_stock_pool_hard_gate
```

## Data Layer / 数据层

`V7DataHub` 是统一入口：

- `strict_local` 缺 required tables 会抛 `V7DataQualityError`。
- `enforce_pit_fundamentals=true` 会丢弃 `available_at > as_of_date` 的财报行。
- `data_mode.quality_report` 输出 row count、missing columns、source reliability、duplicate rate、PIT violation count。

Provider 分工固定：

- Qlib：行情、技术因子、label、训练切片、回测底座。
- TuShare / AkShare：财务报表、指标、估值字段、公告披露日期。
- TradingView public pages：sentiment / attention context。

## Tests / 测试入口

关键测试包括：

- `tests/test_v7_theme_research_pipeline.py`
- `tests/test_v7_architecture_contracts.py`
- `tests/test_v7_evidence_store_and_gate.py`
- `tests/test_v7_crawler.py`
- `tests/test_v7_pit_financial.py`
- `tests/test_v7_execution_and_cli.py`

这些测试覆盖 order boundary、runtime contract、hard gate fail-closed、crawler safety、Qlib/AkShare schema 和 EvidenceStore PIT quality。
