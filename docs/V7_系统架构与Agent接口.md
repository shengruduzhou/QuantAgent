# QuantAgent V7 系统架构与 Agent 接口 / System Architecture and Agent Contracts

## 目标 / Goal

QuantAgent V7 是面向 A 股散户现实约束的 Point-in-Time 量化研究系统。把政策红头文件、产业新闻、宏观环境、行业景气、产业链、主题股票池、基本面尽调、Financial Fraud Risk、News Credibility、多周期 Alpha、Factor Applicability、Risk Gate、A-share execution simulation、Backtest Attribution 和 Audit 全部组合进同一个 PIT 闭环。

**核心边界**：Agent 不允许生成真实交易订单；Portfolio Construction 只能输出 `target_weights`，`OrderManager` 才能在 Risk Gate 通过后生成 dry-run order intents，再经 `VirtualBroker` 模拟成交并写入审计日志。

## 分层架构 / Layered Architecture

V7 使用六层结构，所有层都必须保存 `as_of_date`、data version、evidence hash 和 confidence：

1. **数据采集层 / Data Ingestion Layer**：政策、公告、财报、行业、新闻、市场、执行约束数据。统一通过 `data/ingestion/*` 下 `PolicyIngestor / DisclosureIngestor / NewsIngestor / FinancialIngestor / OrderContractIngestor / RegulatoryPenaltyIngestor` 进入系统；每个 ingestor 共用 `EVIDENCE_COLUMNS` schema 与 `SourceCredibilityRegistry`。
2. **结构化证据层 / Evidence OS**：`DailyEvidenceJob` 把所有 ingestor 输出合成单张 PIT-safe 表并写入 `EvidenceStore`（按 `available_at` 分区），`EvidenceRecord` 字段含 `source_authority` / `source_reliability` / `published_at` / `available_at` / `horizon_days` / `decay_half_life` / `cross_validation_count` / `raw_hash`。
3. **Agent 层 / Agent Layer**：所有 Agent 只输出 evidence、score、constraint、risk flag、view 或 report，不输出 broker order。
4. **研究图层 / Research Graph**：Theme Discovery、Industry Chain Reasoner（证据驱动，无静态模板）、Thematic Universe、Stock Pool Hard Gate、Factor Applicability（真 sector slice）、Multi-Horizon Alpha（Ridge / ElasticNet / V7 Deep）组成 DAG。
5. **组合与风险层 / Portfolio and Risk Layer**：Portfolio Construction 输出 sleeve allocation 与 `target_weights`，Hedge Decision + `RetailHedgeFeasibilityChecker` 负责降权、现金缓冲、ETF 反向对冲与不可执行 action 剥除；Risk Gate 与 Kill Switch 守门。
6. **执行模拟、回测与审计层 / Execution Simulation, Backtest and Audit**：`EventDrivenThemeBacktester` 提供单日权重重放，`FullPipelineBacktester` 按日期滚动调用 `daily_step` 回调完成全链路 PIT 回测；`VirtualBroker` / `audit_replay` 完成 T+1、涨跌停、停牌、ST、流动性约束下的模拟成交与 audit log。

## 数据层 / Data Layer

V7 数据 entrypoint 是 [`V7DataHub`](../src/quantagent/data/v7_datahub.py)：

```text
V7DataHub
  -> LocalV7ResearchProvider (data/v7/*.csv)
  -> FinancialStatementCache (data/v7/fundamentals/*.parquet)
  -> Online providers (政策 / 公告 / 新闻 / TradingView / Qlib)
```

- `strict_local` 模式要求 `policies + base_universe + market_state + market_panel + fundamentals` 全部存在，缺一即抛 `V7DataQualityError`。
- `enforce_pit_fundamentals=true` 会丢弃所有 `available_at > as_of_date` 的财报行。
- `use_financial_cache=true` 时 `V7DataHub` 会从 `FinancialStatementCache` 读取 PIT 财报并通过 `build_financial_features` 投影为 V7 schema。
- 财务数据从 TuShare Pro / AkShare 拉取后写入本地 Parquet 缓存（见 `docs/V7_PIT数据与财务特征.md`），永远不会来自 Qlib。

## Agent 层 / Agent Layer

V7 实际生效的 Agent 模块按职责分布在以下目录：

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| `daily_evidence_job` | `src/quantagent/data/ingestion/daily_evidence_job.py` | 把所有 ingestor 输出汇总为单张 PIT 表 |
| `evidence_store` | `src/quantagent/data/ingestion/evidence_store.py` | 按 `available_at` 分区落盘 + `read_visible` |
| `policy_ingestor / news_ingestor / disclosure_ingestor` | `src/quantagent/data/ingestion/*` | Active discovery + local cache，支持 RSS / sitemap |
| `policy_parser` | `src/quantagent/themes/policy_parser.py` | 红头文件结构化解析 |
| `policy_schema_extractor` | `src/quantagent/themes/policy_schema_extractor.py` | 可选 LLM-based schema 抽取 |
| `theme_extractor` | `src/quantagent/themes/theme_extractor.py` | Theme Discovery 与生命周期 |
| `industry_chain_reasoner` | `src/quantagent/themes/industry_chain_reasoner.py` | 证据驱动产业链图谱（无静态模板） |
| `company_exposure_mapper` | `src/quantagent/themes/company_exposure_mapper.py` | 通过 ChainNode 动态 score 判断 direct / bottleneck，不再硬编码 node id |
| `stock_pool_selector` | `src/quantagent/themes/stock_pool_selector.py` | 主题股票池分级（observability） |
| `stock_pool_gate` | `src/quantagent/themes/stock_pool_gate.py` | 硬门槛：alpha 模型前过滤 watchlist / exclusion / false / 无因子覆盖 |
| `theme_universe_builder` | `src/quantagent/themes/theme_universe_builder.py` | 主题股票池构建 |
| `news_credibility_agent` | `src/quantagent/credibility/news_credibility_agent.py` | 新闻可信度 |
| `financial_statement_agent` | `src/quantagent/fundamental/financial_statement_agent.py` | 基本面打分 |
| `fraud_risk_agent` | `src/quantagent/fundamental/fraud_risk_agent.py` | Financial Fraud Risk 打分 |
| `intrinsic_valuation` | `src/quantagent/fundamental/intrinsic_valuation.py` | DCF / DDM / 相对估值 |
| `order_contract_agent` | `src/quantagent/fundamental/order_contract_agent.py` | 订单 / 合同 evidence |
| `economic_analyzer` | `src/quantagent/fundamental/economic_analyzer.py` | 行业景气与宏观 |
| `factor_applicability_agent` | `src/quantagent/factors/factor_applicability_agent.py` | 因子适用性 walk-forward 验证（`member.sector` slice） |
| `long_horizon_factors` | `src/quantagent/factors/long_horizon_factors.py` | 中长周期因子 |
| `v7_classical_alpha` | `src/quantagent/models/v7_classical_alpha.py` | Ridge / ElasticNet 多周期 alpha（default） |
| `v7_deep_alpha` | `src/quantagent/models/v7_deep_alpha.py` | 多塔深度 alpha（optional） |
| `retail_hft_risk` | `src/quantagent/risk/retail_hft_risk.py` | 散户 / HFT 不对称风险 |
| `retail_hedge_feasibility` | `src/quantagent/portfolio/retail_hedge_feasibility.py` | 剥掉账户不可执行的 hedge action |
| `full_pipeline_backtester` | `src/quantagent/backtest/full_pipeline_backtester.py` | 按日 PIT 滚动重放整个 V7 pipeline |
| `llm_orchestrator` | `src/quantagent/agents/llm_orchestrator.py` | LLM skill orchestration |

所有 Agent 输出都进入 `EvidenceRecord` 或 V7 typed record，禁止直接生成订单。

## 核心 Schema / Core Schemas

`src/quantagent/v7/schemas.py` 定义了下列 frozen dataclass，覆盖 Agent / Research / Portfolio / Risk 全链：

```text
EvidenceRecord
ThemeProfile
ChainNode
ChainEdge
ThematicUniverseMember
StockPoolSelectionReport
FundamentalScore
FundamentalDueDiligenceReport
FraudRiskScore
NewsCredibilityScore
FactorApplicability
MultiHorizonAlpha
MarketRegimeSnapshot
TechnicalTimingPlan
PortfolioPlan
HedgeDecision
ExecutionConstraintReport
RiskGateReport
BacktestAttributionReport
AuditLogRecord
```

## Daily DAG / 日级 DAG

`src/quantagent/v7/dag.py` 定义了每日研究流程的有向图。`validate_dag` 用于在 CLI 启动时检测拓扑环。

## 安全契约 / Safety Contract

- 任何调用 `Risk Gate` 之前的环节都不能修改持仓。
- `OrderManager.target_weights_to_order_intents` 是唯一允许生成 `OrderIntent` 的接口。
- `VirtualBroker` 必须默认 `dry_run=True`。
- Live trading 路径要显式 `live_trading_enabled=true` + `dry_run=false`，且必须通过 `RiskGate.check` 与 `KillSwitch.check`。

## 测试入口 / Tests

| 测试 | 用途 |
| --- | --- |
| `tests/test_v7_theme_research_pipeline.py` | Theme → Chain → Universe → Daily Service smoke |
| `tests/test_v7_architecture_contracts.py` | DAG、order boundary、PIT、target_weights 约束 |
| `tests/test_v7_real_data_ready_components.py` | 真实数据 ready 时各组件可独立运行 |
| `tests/test_v7_execution_and_cli.py` | CLI 入口与执行约束 |
| `tests/test_v7_5_new_modules.py` | V7.5 新模块（fraud / valuation / economic / retail HFT） |
| `tests/test_v7_pit_financial.py` | PIT 财务 provider / cache / feature build |
| `tests/test_v7_walk_forward_sleeve.py` | walk-forward sleeve allocator |
| `tests/test_v7_ingestion_layer.py` | Source registry + ingestor 输出格式 |
| `tests/test_v7_evidence_store_and_gate.py` | EvidenceStore PIT 落盘 + stock_pool_gate 硬门槛 |
| `tests/test_v7_classical_alpha.py` | Ridge / ElasticNet baseline 训练 + 预测 |
| `tests/test_v7_retail_hedge_feasibility.py` | RetailHedgeFeasibilityChecker 拒绝不可执行 action |
| `tests/test_v7_factor_applicability_sector.py` | `member.sector` slice 回归（不再被 chain_node 污染） |
| `tests/test_v7_full_pipeline_backtester.py` | Full-pipeline PIT 回测 T+1 / cap / audit |
| `tests/test_v7_docs_exist.py` | 文档与配置存在性 |
