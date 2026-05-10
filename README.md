# QuantAgent V4 → V5 / 大 A AI Quant OS

QuantAgent 是一个面向大 A 市场的 AI Quant OS 最小可落地闭环。它不是 LLM trading demo，而是把 research、point-in-time feature store、multi-tower model、factor governance、portfolio optimizer、A-share backtest 和 QMT dry-run execution-preparation 连接起来。

> **V5 设计已落地**：见 [docs/v5_design.md](docs/v5_design.md)。V5 在 V4 骨架上做了「结构性减法 + 模型 / 因子 / agent 智能化」：MoE 多塔融合、可学习因子门控、Conformal 校准、Agent 在线可信度、Regime-aware 优化器、HRP 兜底。V4 路径完全兼容并继续可用。

## 核心边界 / Core Boundary

系统默认不启用 live trading。LLM Agents 只输出 structured evidence 和 AgentView；Optimizer 只输出 target weights；OrderManager 才能把 target weights 转成 order intents；QMTGateway 默认 dry-run。

```text
FeatureStore -> V4MultiTowerModel -> AgentView -> blend_alpha_and_views
-> solve_v4_portfolio -> EventDrivenBacktester -> OrderManager -> QMTGateway(dry-run)
```

## V4 功能 / V4 Capabilities

- 数据层 / Data Layer：`FeatureStore`、`PITJoiner`、`EventStore`、`UniverseBuilder` 支持 synthetic offline flow。
- A 股规则 / A-share Rules：`AshareRuleEngine` 支持主板、创业板、科创板、北交所、ETF、可转债 placeholder、futures hedge placeholder。
- 因子治理 / Factor Governance：`FactorDAG`、`FactorLifecycleReport`、group metrics 支持 active/degraded/retired/watch。
- 模型层 / Model Layer：`V4MultiTowerModel` 包含 sequence tower、snapshot tower、structured event tower 和 quantile heads。
- Agent 层 / Agent Layer：`EvidenceRecord`、`AgentView`、`AgentRouter`、BL posterior adapter，全程不产生 orders。
- 组合层 / Portfolio Layer：`blend_alpha_and_views` 与 `solve_v4_portfolio` 输出 target weights、cost、turnover、rejected symbols。
- 回测层 / Backtest Layer：V4 backtester 支持 T+1、limit rules、suspension、partial fill、reject reasons、cost attribution。
- 执行准备 / Execution Preparation：`QMTGateway` 默认 dry-run，支持 idempotency、audit logs、query stubs、kill switch。

## 快速验证 / Quick Check

```powershell
python -m pytest tests/
python -m pytest tests/test_ashare_rules_v4.py tests/test_v4_services.py
```

如果本地只暴露 Python launcher，可以使用：

```powershell
py -3.12 -m pytest tests/
```

## CLI 示例 / CLI Examples

```powershell
quantagent build-features-v4
quantagent infer-v4
quantagent build-portfolio-v4
quantagent backtest-v4
quantagent paper-trade-v4 --dry-run
```

所有命令默认使用 synthetic fixtures 或本地小文件，不需要 internet、broker 或真实账户。

## 安全声明 / Safety Notice

本文档和代码不构成投资建议 / financial advice。V4 默认 `live_trading_enabled=false` 且 `dry_run=true`。任何真实 QMT submit path 必须显式关闭 dry-run，并通过 RiskGate、KillSwitch、reconciliation 和审计。

## 文档入口 / Docs

- [V5 设计 / V5 Design](docs/v5_design.md) ← **当前演进目标**
- [V4 架构 / Architecture](docs/v4_architecture.md)
- [数据与 Feature Store / Data](docs/v4_data_and_feature_store.md)
- [模型训练 / Model Training](docs/v4_model_training.md)
- [Agent Views / Structured Evidence](docs/v4_agent_views.md)
- [组合与回测 / Portfolio Backtest](docs/v4_portfolio_backtest.md)
- [QMT 执行准备 / Execution](docs/v4_execution_qmt.md)
- [Roadmap / 路线图](docs/v4_roadmap.md)
