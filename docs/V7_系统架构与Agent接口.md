# QuantAgent V7 系统架构与 Agent 接口 / System Architecture and Agent Contracts

## 目标 / Goal

V7 的目标是把 V6 的 production scaffold 升级为面向 A 股散户现实约束的研究与组合系统：政策红头文件、产业新闻、宏观环境、行业景气、产业链、主题股票池、基本面尽调、Financial Fraud Risk、News Credibility、多周期 Alpha、Factor Applicability、Risk Gate、A-share execution simulation、Backtest Attribution 和 Audit 全部进入同一个 Point-in-Time 闭环。

参考 TradingAgents 的 multi-agent role separation、bull / bear debate、risk management 和 structured output 思路，但 QuantAgent V7 不照搬其 Trader 直接生成交易提案的路径。V7 中 Agent 不允许生成真实交易订单；Portfolio Construction 只能输出 `target_weights`，`OrderManager` 才能在 Risk Gate 通过后生成 dry-run order intents。

## 分层架构 / Layered Architecture

V7 使用六层结构，所有层都必须保存 `as_of_date`、data version、evidence hash 和 confidence：

1. 数据采集层 / Data Ingestion Layer：政策、公告、财报、行业、新闻、市场、执行约束数据。现有 `src/quantagent/data/providers/` 可扩展 provider adapter，`mock_provider.py` 继续作为 deterministic fixture。
2. 结构化证据层 / Evidence OS：所有原始信息转为 `EvidenceRecord`，并记录 source authority、source reliability、cross validation、decay、horizon、risk flags 和 Point-in-Time validity。
3. Agent 层 / Agent Layer：20 个 Agent 只输出 evidence、score、constraint、risk flag、view 或 report，不输出 broker order。
4. 研究图层 / Research Graph：Theme Discovery、Industry Chain Graph、Thematic Universe、Factor Applicability 和 Multi-Horizon Alpha 组成有向 DAG。
5. 组合与风险层 / Portfolio and Risk Layer：Portfolio Construction 输出 sleeve allocation 与 `target_weights`，Hedge Decision 与 Risk Gate 负责降权、禁入、现金缓冲和 kill switch。
6. 执行模拟、回测与审计层 / Execution Simulation, Backtest and Audit：`EventDrivenBacktester`、`VirtualBroker`、`AuditReplay` 复用 V6 基础，新增 V7 attribution schema。

## 现有模块可扩展 / Existing Extension Points

以下 V6 模块可以直接扩展为 V7 implementation seam：

- 数据与 PIT：`src/quantagent/data/providers/`、`src/quantagent/data/feature_store.py`、`src/quantagent/data/point_in_time.py`、`src/quantagent/data/event_store.py`。
- Agent evidence：`src/quantagent/agents/policy_agent.py`、`news_agent.py`、`sentiment_agent.py`、`sector_rotation_agent.py`、`financial_statement_agent.py`、`agent_router.py`。
- 基本面与造假风险：`src/quantagent/fundamental/scores.py`、`quality.py`、`valuation.py`、`forensic_accounting.py`。
- 因子治理：`src/quantagent/factors/lifecycle.py`、`governance.py`、`evaluation.py`。
- 模型与多周期输出：`src/quantagent/models/v6_model_system.py`、`src/quantagent/models/v6_outputs.py`、`src/quantagent/models/multitower.py`。
- 组合、风控、执行：`src/quantagent/portfolio/allocator.py`、`src/quantagent/risk/risk_gate.py`、`src/quantagent/risk/kill_switch.py`、`src/quantagent/backtest/engine.py`、`src/quantagent/execution/order_manager.py`、`src/quantagent/execution/virtual_broker.py`、`src/quantagent/execution/reconciliation.py`。

## 新增 V7 模块 / New V7 Modules

当前已新增 `src/quantagent/v7/` 作为 schema 和 contract 层。后续实现应按下列文件结构落地，先做 shared internals，再做 V6 compatibility wrapper：

```text
src/quantagent/v7/
  schemas.py                 # V7 typed records: EvidenceRecord, ThemeProfile, PortfolioPlan, RiskGateReport
  agent_contracts.py         # 20 Agent specs and order-boundary contract
  dag.py                     # Daily DAG task graph
  scoring.py                 # Lifecycle, news, fraud, universe and execution scoring helpers

src/quantagent/policy/
  v7_policy_parser.py        # policy title, authority, subsidy, pilot city, target year parser
  authority.py               # central/ministry/local/media authority scoring

src/quantagent/theme/
  discovery.py               # Theme Discovery algorithm
  lifecycle.py               # theme lifecycle and invalidation state machine
  industry_chain_graph.py    # chain nodes and relation graph
  universe_builder.py        # dynamic thematic stock pool

src/quantagent/credibility/
  news.py                    # News Credibility scoring
  source_registry.py         # source reliability registry

src/quantagent/research/
  fundamental_due_diligence.py
  fraud_risk.py
  stock_research_card.py

src/quantagent/models/
  v7_multi_horizon.py        # 1D/5D/20D/60D/120D/126D alpha
  v7_alpha_outputs.py

src/quantagent/risk/
  hedge_decision.py
  v7_risk_acceptance.py

src/quantagent/reports/
  v7_daily_report.py
  v7_audit_report.py
```

## EvidenceRecord Schema / 证据结构

`EvidenceRecord` 是 V7 的最小可信事实单元。字段已经落在 `src/quantagent/v7/schemas.py`，数据库字段应保持同名：

```text
evidence_id
source
source_type
source_authority_level
timestamp
published_at
effective_start_date
effective_end_date
symbol
sector
industry
theme
sub_theme
chain_node
event_type
direction
magnitude
confidence
evidence_quality
source_reliability
cross_validation_count
decay_half_life
horizon_days
rationale
raw_reference
hash
point_in_time_valid
risk_flags
```

约束规则：

- `published_at <= as_of_date`，否则 `point_in_time_valid=false`。
- `effective_start_date` 和 `effective_end_date` 用于政策、订单、补贴、试点、财报有效期。
- `hash` 必须由结构化字段稳定生成，不能由 LLM 自填。
- `raw_reference` 保存 URL、公告编号、交易所文件 ID、政策文号或 vendor row id。
- 任何 `rumor_risk`、`fraud_risk`、`data_missing`、`future_leakage` 都必须进入 `risk_flags`。

## 数据库表结构 / Database Schema

建议先用 DuckDB / SQLite / Parquet hybrid，后续再迁移到 PostgreSQL。所有表必须有 `as_of_date` 或 `published_at`，避免 future leakage。

```sql
CREATE TABLE evidence_records (
  evidence_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_authority_level DOUBLE NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  published_at TIMESTAMP NOT NULL,
  effective_start_date DATE,
  effective_end_date DATE,
  symbol TEXT,
  sector TEXT,
  industry TEXT,
  theme TEXT,
  sub_theme TEXT,
  chain_node TEXT,
  event_type TEXT NOT NULL,
  direction DOUBLE NOT NULL,
  magnitude DOUBLE NOT NULL,
  confidence DOUBLE NOT NULL,
  evidence_quality DOUBLE NOT NULL,
  source_reliability DOUBLE NOT NULL,
  cross_validation_count INTEGER NOT NULL,
  decay_half_life DOUBLE NOT NULL,
  horizon_days INTEGER NOT NULL,
  rationale TEXT NOT NULL,
  raw_reference JSON,
  hash TEXT NOT NULL,
  point_in_time_valid BOOLEAN NOT NULL,
  risk_flags JSON
);

CREATE TABLE theme_profiles (
  as_of_date DATE NOT NULL,
  theme_name TEXT NOT NULL,
  theme_category TEXT NOT NULL,
  theme_strength DOUBLE NOT NULL,
  policy_strength DOUBLE NOT NULL,
  market_strength DOUBLE NOT NULL,
  industry_fundamental_strength DOUBLE NOT NULL,
  capital_flow_strength DOUBLE NOT NULL,
  news_sentiment_strength DOUBLE NOT NULL,
  lifecycle_stage TEXT NOT NULL,
  expected_horizon_days INTEGER NOT NULL,
  theme_confidence DOUBLE NOT NULL,
  bubble_risk DOUBLE NOT NULL,
  crowding_score DOUBLE NOT NULL,
  expiry_date DATE NOT NULL,
  key_evidence JSON,
  opposing_evidence JSON,
  required_follow_up_data JSON,
  PRIMARY KEY (as_of_date, theme_name)
);

CREATE TABLE industry_chain_nodes (
  as_of_date DATE NOT NULL,
  theme_name TEXT NOT NULL,
  node_id TEXT NOT NULL,
  node_name TEXT NOT NULL,
  dependency_strength DOUBLE,
  bottleneck_score DOUBLE,
  domestic_substitution_score DOUBLE,
  supply_shortage_score DOUBLE,
  price_elasticity DOUBLE,
  profit_elasticity DOUBLE,
  demand_visibility DOUBLE,
  policy_support_score DOUBLE,
  technology_barrier DOUBLE,
  competition_intensity DOUBLE,
  listed_company_count INTEGER,
  evidence_ids JSON,
  PRIMARY KEY (as_of_date, theme_name, node_id)
);

CREATE TABLE thematic_universe_members (
  as_of_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  company_name TEXT NOT NULL,
  theme TEXT NOT NULL,
  sub_theme TEXT,
  chain_node TEXT,
  exposure_type TEXT NOT NULL,
  exposure_score DOUBLE NOT NULL,
  revenue_exposure_estimate DOUBLE,
  profit_exposure_estimate DOUBLE,
  evidence_count INTEGER NOT NULL,
  source_confidence DOUBLE NOT NULL,
  fundamental_score DOUBLE NOT NULL,
  valuation_score DOUBLE NOT NULL,
  quality_score DOUBLE NOT NULL,
  fraud_risk_score DOUBLE NOT NULL,
  liquidity_score DOUBLE NOT NULL,
  market_attention_score DOUBLE NOT NULL,
  theme_lifecycle_stage TEXT NOT NULL,
  entry_date DATE NOT NULL,
  expiry_date DATE NOT NULL,
  last_validated_at TIMESTAMP NOT NULL,
  watchlist_status TEXT NOT NULL,
  removal_reason TEXT,
  PRIMARY KEY (as_of_date, symbol, theme)
);
```

补充表：

- `fundamental_scores`：商业模式、收入暴露、利润弹性、订单、产能、治理、估值和 margin of safety。
- `fraud_risk_scores`：Beneish、Piotroski、Altman、accruals、cashflow、receivables、inventory、related party、regulatory penalty、audit opinion。
- `news_credibility_scores`：source reliability、primary source、official flag、cross validation、rumor risk、contradiction flags。
- `multi_horizon_alpha`：`alpha_1d`、`alpha_5d`、`alpha_20d`、`alpha_60d`、`alpha_120d`、`alpha_126d`、prediction interval 和 contribution。
- `factor_applicability`：factor universe、sector、theme、regime、horizon、decay、RankIC、ICIR、capacity、crowding、lifecycle。
- `portfolio_plans`：sleeve weights、target weights、cash、hedge、limits 和 position reason。
- `risk_gate_reports`：rejected、reduced、blocked、required cash buffer、kill switch 和 rationale。
- `audit_logs`：decision id、input data versions、model version、feature version、evidence hashes、risk gate result 和 final decision reason。

## Agent 接口 / Agent Interface

所有 Agent 实现必须遵守以下 contract：

```python
class V7Agent:
    name: str
    point_in_time_required: bool = True
    can_emit_orders: bool = False

    def run(self, inputs: Mapping[str, Any], as_of_date: str) -> AgentOutputEnvelope:
        ...
```

输出约束：

- 允许输出 `EvidenceRecord`、`ThemeProfile`、`ChainNode`、`ThematicUniverseMember`、`FundamentalScore`、`FraudRiskScore`、`NewsCredibilityScore`、`MultiHorizonAlpha`、`FactorApplicability`、`TechnicalTimingPlan`、`PortfolioPlan`、`HedgeDecision`、`ExecutionConstraintReport`、`RiskGateReport`、`BacktestAttributionReport`、`AuditLogRecord`。
- 禁止输出 `OrderIntent`、broker order、QMT submit payload。
- 每个输出必须有 confidence、rationale 或 evidence hash。
- Agent 可被 disabled、degraded 或 replaced；禁用时输出 `RiskFlag(data_missing)`，不能静默缺失。

## Agent 职责表 / Agent Responsibility Matrix

| Agent | 输入 / Inputs | 输出 / Outputs | 现有模块 / Existing | 新增模块 / New |
|---|---|---|---|---|
| Policy Agent | PolicyDocument, PolicyTaxonomy | EvidenceRecord, ThemePolicyScore | `agents/policy_agent.py` | `policy/v7_policy_parser.py` |
| Theme Discovery Agent | EvidenceRecord, MarketBreadth, SectorFlow | ThemeProfile, RiskFlag | new | `theme/discovery.py` |
| Industry Chain Graph Agent | ThemeProfile, IndustryChainSeed | ChainNode, ChainEdge | new | `theme/industry_chain_graph.py` |
| Thematic Universe Builder | BaseUniverse, ChainGraph, FundamentalScore | ThematicUniverseMember | `data/universe.py` | `theme/universe_builder.py` |
| Fundamental Due Diligence Agent | FinancialStatement, Announcement | FundamentalScore | `fundamental/scores.py` | `research/fundamental_due_diligence.py` |
| Valuation Agent | ValuationPanel, IndustryPercentile | ValuationScore | `fundamental/valuation.py` | wrapper |
| Financial Fraud Risk Agent | FinancialStatement, RegulatoryDisclosure | FraudRiskScore, RiskFlag | `fundamental/forensic_accounting.py` | `research/fraud_risk.py` |
| News Credibility Agent | NewsItem, Announcement, PolicyDocument | NewsCredibilityScore | `agents/news_agent.py` | `credibility/news.py` |
| Sentiment Agent | CredibleNews, SocialPanel | SentimentScore | `agents/sentiment_agent.py` | wrapper |
| Market Regime Agent | MarketPanel, MacroPanel, IndexPanel | MarketRegimeSnapshot | `quant_math/regime.py` | wrapper |
| Sector Rotation Agent | SectorPanel, MarketRegimeSnapshot | SectorRotationScore | `factors/sector_rotation.py` | wrapper |
| Factor Applicability Agent | FactorLifecycleReport, Universe | FactorApplicability | `factors/lifecycle.py` | wrapper |
| Multi-Horizon Alpha Agent | FeatureStore, Evidence, FactorApplicability | MultiHorizonAlpha | `models/v6_model_system.py` | `models/v7_multi_horizon.py` |
| Technical Timing Agent | OHLCV, Alpha, ThemeProfile | TechnicalTimingPlan | `quant_math/technical_indicators.py` | wrapper |
| Portfolio Construction Agent | Alpha, Timing, Risk, Universe | PortfolioPlan target_weights | `portfolio/allocator.py` | wrapper |
| Hedge Decision Agent | PortfolioPlan, Regime, Risk | HedgeDecision | new | `risk/hedge_decision.py` |
| A-Share Execution Agent | PortfolioPlan, MarketState, PositionState | ExecutionConstraintReport | `backtest/engine.py` | wrapper |
| Risk Gate Agent | PortfolioPlan, ExecutionReport, FraudRisk | RiskGateReport | `risk/risk_gate.py` | wrapper |
| Backtest & Attribution Agent | PortfolioPlan, MarketPanel, Evidence | BacktestAttributionReport | `backtest/engine.py` | attribution extension |
| Audit Agent | Evidence, RiskGate, Attribution | AuditLogRecord | `execution/audit.py` | `reports/v7_audit_report.py` |

## TradingAgents 参考边界 / Reference Boundary

可借鉴：

- Analyst / Researcher / Risk Manager 的分工。
- structured output schema 的 provider-independent 形式。
- checkpoint resume 和 decision log。
- bull / bear / neutral 风险辩论。

不可照搬：

- Trader 直接决定 Buy / Sell 的路径。
- 单 ticker prompt-driven decision 作为最终交易决策。
- 忽略 A 股 T+1、涨跌停、停牌、ST、最小交易单位和散户流动性约束。

QuantAgent V7 的最终输出是可审计 research conclusion、thematic universe、`target_weights`、entry / exit condition、risk condition 和 audit log，不是未经验证的真实交易指令。
