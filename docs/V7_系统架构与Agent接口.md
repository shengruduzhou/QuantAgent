# V7 系统架构与 Agent 接口 / Architecture and Agent Contracts

## 核心边界 / Core Boundary

V7 的核心接口保持清晰：

- Agent 输出 evidence、views、constraints、confidence、risk flags、audit logs。
- Portfolio Construction 输出 `target_weights`。
- `OrderManager` 是唯一把 `target_weights` 转换为 order intents 的组件。
- `VirtualBroker` 是默认 paper / replay path。
- QMT live submit 默认关闭。

## Daily Research API / 每日研究入口

公开 API 保持兼容：

```python
from quantagent.services.v7_pipeline_service import run_daily_v7_research

result = run_daily_v7_research("configs/v7.mock.yaml", as_of_date="2026-05-14")
```

Mock config 只用于 tests 和 smoke examples。Production / strict_local 不允许 synthetic fallback。

Theme Discovery、Financial Fraud Risk、News Credibility 和 Fundamental Due Diligence 都必须输出可审计结构，不允许只保存 natural language。

## Real-Data Architecture / 真实数据架构

新增 real-data modules：

| Module | Responsibility |
| --- | --- |
| `data/bootstrap/qlib_bootstrap.py` | Document / wrap Qlib CN preparation and export market panel |
| `data/bootstrap/akshare_bootstrap.py` | Batch AkShare financial ingestion into PIT cache |
| `data/v7_dataset_builder.py` | Build trainable feature matrix from market, fundamentals, evidence, theme and risk features |
| `data/v7_label_builder.py` | Build 1/5/20/60/120/126 day forward labels |
| `data/v7_quality_gates.py` | Enforce data and model acceptance gates |
| `training/v7_experiment.py` | Purged walk-forward Ridge / ElasticNet training and artifact writing |
| `backtest/ashare_execution_simulator.py` | A-share execution simulation through OrderManager and VirtualBroker |

## Provider Contracts / Provider 分工

- Qlib：market OHLCV、technical features、labels、training slices、backtest base。
- TuShare / AkShare：financial statements、financial indicators、valuation fields、disclosure dates。
- TradingView public pages：sentiment / attention context only。
- Policy / announcement / news：must preserve `source / published_at / available_at / raw_hash / confidence`。

## CLI Surface / 命令接口

```powershell
quantagent download-qlib-v7
quantagent build-market-panel-v7
quantagent build-akshare-v7
quantagent build-labels-v7
quantagent train-alpha-v7
quantagent walk-forward-backtest-v7
quantagent paper-trade-v7
quantagent v7-live-readiness-report
```

这些命令不会开启 live trading。真实数据命令在缺少 dataset、token、network permission 或 optional dependency 时必须报告 actionable error。
