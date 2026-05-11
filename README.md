# QuantAgent V6：AI Quant OS

QuantAgent V6 是面向 A 股研究、验证、虚拟交易和审计回放的 AI Quant OS。它不是投资建议系统，也不是真实券商下单系统；当前阶段的 live trading 仅指 virtual live trading、historical live replay 和 paper trading。

## 目标 / Goal

V6 主链是 `ExternalDataProviders -> PIT FeatureStore -> FactorPipeline -> V6 Model -> Agent Evidence Committee -> BL posterior -> Regime-aware Optimizer -> RiskGate -> OrderManager -> VirtualBroker -> Reconciliation -> Audit Replay -> Validation Report`。真实行情、真实新闻、宏观、财务和资金流可以通过 provider adapter 用于训练、验证和历史回放，但核心测试默认使用 mock / fixture，不依赖外部网络。

## 安全边界 / Safety Boundary

- 默认不连接真实券商，默认使用 `VirtualBroker`。
- `LLM / agent` 只能输出 `EvidenceRecord`、risk warning、reasoning summary 和 `AgentView`，不能输出 order。
- 模型只能输出 alpha、confidence、risk_score、uncertainty 和 factor_gate。
- Optimizer 只能输出 `target_weights`。
- 只有 `OrderManager` 可以把 `target_weights` 转成 order intents。
- 真实券商接入必须未来单独启用，并且默认关闭。

## 快速运行 / Quickstart

```powershell
python -m pytest tests/ -q
quantagent validate-v6 --config configs/v6.default.yaml
quantagent replay-v6 --config configs/v6.default.yaml --scenario 2020_covid_volatility_replay
quantagent paper-trade-v6 --config configs/v6.default.yaml --dry-run
```

## 配置 / Config

主配置是 `configs/v6.default.yaml`，provider 配置在 `configs/data_providers.v6.yaml`，历史回放场景在 `configs/replay_scenarios.v6.yaml`，风险限制在 `configs/risk_limits.v6.yaml`，模型配置在 `configs/model.v6.yaml`。API key 只能来自环境变量或 `.env`，不得写入代码。

## 文档 / Docs

V6 设计和验收文档位于 `docs/V6_*.md`。这些文档描述数据、模型、Agent、组合优化、虚拟交易、生产审计、运维演练和验收标准。
