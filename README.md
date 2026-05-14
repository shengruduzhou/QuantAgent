# QuantAgent V7：A 股 PIT 多 Agent 研究与风控系统 / A-share PIT Multi-Agent Research OS

QuantAgent V7 是面向 A 股散户现实约束的 PIT (point-in-time) 量化研究、基本面尽调、多周期 Alpha、组合风控、回测与模拟执行系统。它**不是投资建议系统**，也**不是真实券商自动下单系统**；默认只输出 research、backtest、paper / virtual trading、audit replay 和风险提示。

## 设计目标 / Goal

V7 主链：

```text
Data Providers (TuShare / AkShare / Local PIT cache)
  -> Daily Evidence Ingestion Layer
       PolicyIngestor / DisclosureIngestor / NewsIngestor /
       FinancialIngestor / OrderContractIngestor / RegulatoryPenaltyIngestor
  -> Source Credibility Registry (OFFICIAL_PRIMARY → SOCIAL_MEDIA)
  -> Point-in-Time Evidence OS (available_at 严格过滤)
  -> News Cross-Validator (跨源确认 / 反证 / 同源转载 / 盘后判断)
  -> Policy & Theme Agents (政策红头文件解析、主题发现)
  -> Industry Chain Reasoner (证据驱动产业链图谱，禁用模板回退)
  -> Evidence-driven Company Exposure Mapper (drops hard-coded alias)
  -> Thematic Universe (核心 / 强相关 / 卫星 / 观察 / 排除)
  -> Fundamental + Fraud + News Credibility Agents
  -> Multi-Horizon Alpha (1 / 5 / 20 / 60 / 120 / 126 天)
  -> Factor Applicability Hard Gate (walk-forward 验证)
  -> Adaptive Long-Horizon Factor Weights
       (per theme / sector / lifecycle / regime / horizon)
  -> Market Regime + Sector Rotation
  -> Long-Short Allocator -> Portfolio (binding sleeve_weights override)
  -> Walk-Forward Sleeve Allocator (long / medium / short / hedge / cash)
  -> Lifecycle Trading Rules (per-stage caps and exits)
  -> Portfolio Construction
  -> Hedge Decision Engine + Tool-based Hedge (ETF + cash + beta reduction)
  -> Risk Gate + Kill Switch
  -> A-share Execution Simulation (T+1 / 涨跌停 / 停牌 / ST / 流动性)
  -> OrderManager (唯一允许产生 order intent 的节点)
  -> VirtualBroker / Audit Replay
```

相对早期版本的核心升级：

- **Point-in-Time 财报闭环**：`TuShareFinancialProvider`、`AkShareFinancialProvider`、`FinancialStatementCache` 把利润表 / 资产负债表 / 现金流量表 / 财务指标 / 公告披露日期写入本地 Parquet，每条记录强制 `available_at`，回测/研究永不读到未来披露。
- **PIT 数据强约束**：`V7DataHub` 默认要求 `policies + base_universe + market_state + market_panel + fundamentals`；`enforce_pit_fundamentals` 会丢弃所有 `available_at > as_of_date` 的财报行。
- **产业链证据强约束**：`IndustryChainReasonerConfig.strict_no_template_fallback=True` 时，没有足够证据的主题不会再回退到静态 AI_COMPUTE_TEMPLATE。
- **因子适用性硬门槛**：`factor_applicability.hard_gate=True` 时，未通过 walk-forward `production / validation` 检验的因子不会进入 V7 Deep Alpha 模型。
- **学习型 Sleeve 分配**：`walk_forward_sleeve_allocator` 用过去的 sleeve 日收益做 walk-forward grid search，输出 long / medium / short / hedge / cash 的权重区间，而不是固定先验。

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

`configs/v7.mock.yaml` 是 mock 模式：

- 允许内置 deterministic synthetic 输入。
- 仅用于 smoke test（Theme Discovery、Industry Chain Graph、Universe、Multi-Horizon Alpha、A-share Execution Constraints、Risk Gate、Audit Log）。
- 永远不会生成真实交易订单。

## 数据与远程抽取 / Data And Remote Extraction

V7 支持显式 `online` 数据模式接入 policy、news、disclosure、TradingView public pages、Qlib、AkShare、TuShare provider。

- 网络调用默认关闭，必须配置 `allow_network=true` 且 `allow_synthetic_fallback=false` 才会尝试真实抓取。
- TuShare 财务数据需要 `TUSHARE_TOKEN` 环境变量；AkShare 是兜底通道。
- TradingView public pages 只作为 sentiment context，不作为官方行情或基本面真值。
- `policy_extraction` 是 OpenAI-compatible remote schema extraction seam，默认 `enabled=false`；启用后它只把红头文件、公告和新闻抽取成 `EvidenceRecord`、theme、sub-theme、chain nodes、confidence 和 risk flags，不允许输出 order 或 trade advice。

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
  -> long_horizon_factors + Deep Alpha
```

财务数据从不来自 Qlib：Qlib 仅负责行情、技术因子和回测底座；财务事实由 TuShare / AkShare provider 拉取，写入 PIT 缓存后由 V7DataHub 注入。
