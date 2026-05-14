# QuantAgent V7：A 股 PIT 多 Agent 研究与风控系统 / A-share PIT Multi-Agent Research OS

QuantAgent V7 是面向 A 股散户现实约束的 PIT (point-in-time) 量化研究、基本面尽调、多周期 Alpha、组合风控、回测与模拟执行系统。它**不是投资建议系统**，也**不是真实券商自动下单系统**；默认只输出 research、backtest、paper / virtual trading、audit replay 和风险提示。

## 设计目标 / Goal

V7 主链：

```text
Data Providers (TuShare / AkShare / Local PIT cache)
  -> Daily Evidence Ingestion Layer (single seam for outside-world data)
       SourceCredibilityRegistry (OFFICIAL_PRIMARY → SOCIAL_MEDIA)
       PolicyIngestor      (active sitemap / RSS discovery; PDF parsing seam)
       DisclosureIngestor  (上交所 / 深交所 / 北交所 / 巨潮资讯)
       NewsIngestor        (regulated > tier1 > tier2 > self media)
       FinancialIngestor   (TuShare / AkShare PIT cache)
       OrderContractIngestor / RegulatoryPenaltyIngestor
  -> EvidenceStore (partitioned parquet/csv keyed by available_at)
  -> News Cross-Validator (跨源确认 / 反证 / 同源转载 / 盘后判断)
  -> Theme Discovery + Industry Chain Reasoner
       (证据驱动，禁用静态模板回退)
  -> Evidence-driven Company Exposure Mapper
       (ChainNode 动态 role/score 替代硬编码 node-id)
  -> Thematic Universe (核心 / 强相关 / 卫星 / 观察 / 排除)
  -> Fundamental + Fraud + News Credibility Agents
  -> Multi-Horizon Alpha
       Ridge / ElasticNet 经典基线 (default, walk-forward friendly)
       V7DeepAlpha multi-tower (optional, off by default)
  -> Factor Applicability Hard Gate (walk-forward, 真 sector slice)
  -> Stock Pool Hard Gate (core+strong 允许；卫星需高 confidence；false 一律屏蔽)
  -> Adaptive Long-Horizon Factor Weights
  -> Market Regime + Sector Rotation
  -> Long-Short Allocator + Portfolio Construction
  -> Lifecycle Trading Rules (per-stage caps and exits)
  -> Hedge Decision Engine
       + Tool-based Hedge (ETF + cash + beta reduction)
       + RetailHedgeFeasibilityChecker (剥掉不可执行的 hedge action)
  -> Risk Gate + Kill Switch
  -> A-share Execution Simulation (T+1 / 涨跌停 / 停牌 / ST / 流动性)
  -> OrderManager (唯一允许产生 order intent 的节点)
  -> Full-Pipeline PIT Backtester (available_at 滚动重放)
  -> VirtualBroker / Audit Replay
```

相对早期版本的核心升级 / Key upgrades vs earlier V7:

- **Evidence Ingestion Layer 成形**：所有外部数据（政策 / 公告 / 新闻 / 财报 / 订单 / 处罚）都通过 `data/ingestion` 下统一 ingestor + `EvidenceStore` 落盘，输出 `EVIDENCE_COLUMNS` schema (含 `source_authority`、`source_reliability`、`available_at`、`horizon_days`、`decay_half_life`、`raw_hash`、`point_in_time_valid`)。
- **Active discovery 抓取**：政策 / 公告 / 新闻 ingestor 支持从 `SourceProfile.discovery_urls / rss_urls / sitemap_urls` 主动发现新文章，而不是只读静态 URL 列表。
- **Stock Pool 硬门槛**：`stock_pool_gate` 在 alpha 模型之前过滤掉 watchlist / exclusion / false-association / 无因子覆盖 的标的，alpha / 组合 / 风控 / 回测都基于这个过滤后的 universe。
- **Factor Applicability sector 修正**：`factor_applicability_agent` 用 `member.sector`（行业分类），不再把 `chain_node` 错当 sector，sector slice 真正可用。
- **公司映射不再硬编码 node id**：`company_exposure_mapper` 通过 `ChainNode.bottleneck_score / domestic_substitution_score / dependency_strength / demand_visibility` 动态判断 `DIRECT_EXPOSURE` vs `CRITICAL_BOTTLENECK`，不再依赖 `_DIRECT_PRODUCT_NODE_IDS` / `_BOTTLENECK_NODE_IDS` 写死表。
- **经典 ML alpha baseline**：新增 Ridge / ElasticNet 多周期模型 (`v7_classical_alpha.py`)，可在 walk-forward 训练后输出 `MultiHorizonAlpha`；deep 模型从默认转为可选。
- **散户对冲可执行性**：`RetailHedgeFeasibilityChecker` 把模型推荐的 hedge action 剥成账户实际可执行的 cash + inverse ETF + 减仓 子集，且记录 audit 备注。
- **全链路 PIT 回测**：`backtest.full_pipeline_backtester` 按日期滚动调用 `daily_step` 回调，PIT slice (`build_pit_evidence_slice`) 保证不读未来数据；T+1 fill、turnover cost、单票上限内建。

## 安全边界 / Safety Boundary

- 默认不连接真实券商，默认使用 `VirtualBroker`。
- `LLM / agent` 只能输出 `EvidenceRecord`、score、view、constraint、risk flag、audit log，不能输出 order。
- 模型只能输出 alpha、confidence、prediction interval、risk penalty 和 factor contribution。
- Optimizer / Portfolio Construction 只能输出 `target_weights`。
- 只有 `OrderManager` 可以把 `target_weights` 转成 order intents。
- `Risk Gate`、`Kill Switch`、execution constraint simulation 和 reconciliation 必须在任何 QMT submit path 前完成。
- 真实券商接入必须显式配置 `live_trading_enabled=true` 且 `dry_run=false`，并保持默认关闭。

## 文档 / Docs

- V7 架构与 Agent 接口：[`docs/V7_系统架构与Agent接口.md`](docs/V7_系统架构与Agent接口.md)
- V7 算法、风控、回测与验收：[`docs/V7_算法风控回测与验收.md`](docs/V7_算法风控回测与验收.md)
- V7 PIT 数据与财务特征：[`docs/V7_PIT数据与财务特征.md`](docs/V7_PIT数据与财务特征.md)
- V7 证据摄取与交易规则：[`docs/V7_证据摄取与交易规则.md`](docs/V7_证据摄取与交易规则.md)

## 快速验证 / Quick Validation

```powershell
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:\Users\shanh\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
```

## V7 CLI / 命令入口

```powershell
# 验证 DAG 与安全边界
quantagent validate-v7 --config configs/v7.default.yaml

# 一日完整研究流程（默认 strict_local，缺数据会显式失败）
quantagent run-daily-v7 --config configs/v7.default.yaml --date 2026-05-15 --output-dir reports/v7

# Mock 数据 smoke test
quantagent run-daily-v7 --config configs/v7.mock.yaml --date 2026-05-15 --output-dir reports/v7

# 拉取 PIT 财务数据到本地 Parquet 缓存
quantagent build-fundamentals-v7 --symbols 600519.SH,000858.SZ --start-date 2020-01-01 --end-date 2026-05-15 --provider tushare --allow-network

# 在 sleeve 日收益面板上跑 walk-forward 学习
quantagent walk-forward-v7 --sleeve-returns reports/v7/sleeve_returns.csv --splits 4 --embargo-days 5
```

## 数据模式 / Data Modes

`configs/v7.default.yaml` 默认是 `strict_local`：

- 缺少 `policies / base_universe / market_state / market_panel / fundamentals` 任一表时，`V7DataHub` 会直接抛 `V7DataQualityError`，不允许 synthetic fallback。
- `enforce_pit_fundamentals=true` 会进一步丢弃所有 `available_at > as_of_date` 的财报行。
- `stock_pool_gate.enabled=true`、`alpha_model.enabled=true`（Ridge）、`retail_hedge_feasibility.enabled=true` 是默认开关。

`configs/v7.mock.yaml` 是 mock 模式：

- 允许内置 deterministic synthetic 输入。
- 仅用于 smoke test（Theme Discovery、Industry Chain Reasoner、Universe、Multi-Horizon Alpha、A-share Execution Constraints、Risk Gate、Audit Log）。
- 永远不会生成真实交易订单。

## Evidence Ingestion / 证据摄取

`configs/v7.default.yaml` 的 `ingestion` section 控制 daily evidence job：

```yaml
ingestion:
  enabled: true
  cache_root: data/v7/evidence
  store_root: data/v7/evidence/store
  write_to_store: true
  enabled_sources: [policy, disclosure, news, financial, order_contract, regulatory_penalty]
  policy:
    urls: []
    allow_network: false
    active_discovery: true     # 通过 SourceProfile.discovery_urls 主动发现
    max_articles_per_source: 25
  disclosure:
    allow_network: false
    active_discovery: true
  news:
    urls: []
    allow_network: false
    active_discovery: true
  source_registry: []           # 用户自定义 source 覆盖默认 registry
```

权威排序固定为：`OFFICIAL_PRIMARY > OFFICIAL_SECONDARY > EXCHANGE_DISCLOSURE > REGULATORY_PENALTY > REGULATED_MEDIA > TIER1_FINANCIAL_MEDIA > TIER2_FINANCIAL_MEDIA > INDUSTRY_MEDIA > SELF_MEDIA > SOCIAL_MEDIA`。

每条 evidence 都带：

```
source / source_type / source_authority / source_reliability
is_primary_source / is_official
published_at / available_at / ingested_at
symbol / company_name / affected_symbols
theme_candidates / chain_node_candidates
event_type / confidence / cross_validation_count / contradiction_count
horizon_days / decay_half_life / rumor_risk_flag
raw_hash / point_in_time_valid
```

`EvidenceStore` 按 `available_at` 分区落盘，`read_visible(as_of_date)` 只返回当日及之前可见的行 — 这是全链路 PIT 回测的真值源。

## PIT 财务数据闭环 / PIT Financial Data Loop

```text
TuShare / AkShare
  -> TuShareFinancialProvider / AkShareFinancialProvider
  -> FinancialStatementCache (data/v7/fundamentals/*.parquet)
  -> apply_point_in_time_filter (available_at <= as_of_date)
  -> build_financial_features (quality / growth / valuation / fraud-ready columns)
  -> derive_v7_financial_columns (投影到 V7 EvidenceRecord schema)
  -> V7DataHub.fundamentals
  -> score_fraud_risk + score_financial_statements + intrinsic valuation
  -> long_horizon_factors + (Ridge / ElasticNet / V7 Deep Alpha)
```

财务数据从不来自 Qlib：Qlib 仅负责行情、技术因子和回测底座；财务事实由 TuShare / AkShare provider 拉取，写入 PIT 缓存后由 V7DataHub 注入。

## Full-pipeline PIT 回测 / Full-pipeline PIT Backtest

`quantagent.backtest.full_pipeline_backtester` 提供一个轻量编排：传入日期列表和 `daily_step` 回调（通常包装 `run_daily_v7_research`），它会按日滚动调用、按 `execution_lag_days` 应用 T+1 fill、累积 NAV、turnover cost 和 PIT audit。`build_pit_evidence_slice(frame, as_of_date)` 提供严格的 `available_at <= as_of_date` 切片，避免任何未来函数。
