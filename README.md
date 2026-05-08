# QuantAgent

QuantAgent 是一个面向生产级 AI 量化系统的工程骨架。当前阶段只专注于**训练模型和验证 Alpha**，QMT / MiniQMT 会在后续作为独立交易网关接入，不进入模型训练代码。

核心拆分：

```text
训练与研究层：数据、因子、标签、短线 Alpha、长线评分、风险模型、文本模型
策略决策层：信号融合、风控门、目标仓位
执行网关层：QMT / IBKR / Alpaca 等 broker adapter，后续再接
```

重要原则：

```text
模型不直接下单
策略不直接调用 QMT
QMT 只做行情、查询、下单、撤单、回报
当前第一阶段只训练模型，不接实盘账户
```

## 当前已有内容

```text
configs/strategy.default.yaml          默认策略参数
configs/training/short_alpha.yaml      短线 Alpha 训练配置
docs/production_ai_quant_blueprint.md  中文生产级技术方案
docs/ai_quant_os_v2.md                 Agent 与 target weight 架构边界
docs/math_optimization_layer.md        数学优化层说明
src/quantagent/data/                   特征、标签、数据集构建
src/quantagent/models/                 Alpha Transformer 模型骨架
src/quantagent/training/               loss、walk-forward、训练入口
src/quantagent/quant_math/             IC、Regime、风险、成本、优化器
src/quantagent/fundamental/            DCF、Reverse DCF、质量和财务风险
src/quantagent/agents/                 Agent 结构化输出和仲裁
src/quantagent/strategy/               信号融合、风控门、仓位 sizing
tests/                                 核心决策层测试
```

## 训练短线 Alpha 的流程

准备两个文件：

```text
data/raw/prices.parquet      个股日频 OHLCV
data/raw/benchmark.parquet   基准指数日频 OHLCV，例如 000300.SH
```

最低字段：

```text
trade_date,symbol,open,high,low,close,volume,amount
```

构建特征和标签：

```powershell
quantagent-build-daily-features `
  --prices data/raw/prices.parquet `
  --benchmark data/raw/benchmark.parquet `
  --benchmark-symbol 000300.SH `
  --output data/processed/daily_features.parquet
```

训练短线模型：

```powershell
quantagent-train-short-alpha --config configs/training/short_alpha.yaml
```

输出：

```text
models/checkpoints/short_alpha/best.pt
```

## 不训练基线

Phase 1 可以先跑不训练版本，用于建立可解释、可回测的 baseline：

```text
OHLCV
  -> RSI / MACD / Bollinger / ATR / ADX / Donchian / VWAP
  -> mean-reversion / momentum-breakout rule signal
  -> DCF / Reverse DCF / quality / fraud risk
  -> AgentSignal evidence arbitration
  -> TargetWeight
  -> optimizer
```

关键文件：

```text
src/quantagent/quant_math/technical_indicators.py
src/quantagent/strategy/rule_signals.py
src/quantagent/strategy/weight_adapter.py
src/quantagent/fundamental/valuation.py
src/quantagent/fundamental/quality.py
src/quantagent/agents/arbitration.py
```

## 快速验证

```powershell
python -m pytest
python -m quantagent.cli demo-decision --ticker 300750.SZ
```

当前本机环境还没有可用的 `python` / `py` 命令，所以我暂时无法在这台机器上执行测试。安装 Python 3.11+ 后即可验证。

## 技术路线

```text
Phase 1: 日频短线 Alpha Transformer / TCN
Phase 2: walk-forward 回测和 Rank IC / ICIR 评估
Phase 3: 概率化 Alpha 融合、Regime 调整、组合优化
Phase 4: 长线 Fundamental Ranking Model
Phase 5: 新闻、财报、政策文本模型
Phase 6: paper trading
Phase 7: QMT Gateway 小资金实盘
```

详细方案见 [docs/production_ai_quant_blueprint.md](docs/production_ai_quant_blueprint.md)。
AI Quant OS v2 见 [docs/ai_quant_os_v2.md](docs/ai_quant_os_v2.md)。
数学层说明见 [docs/math_optimization_layer.md](docs/math_optimization_layer.md)。
