# AI Quant OS v2

This project now treats LLM and multi-agent systems as evidence processors, not order generators.

## Core Contract

Every research component ultimately emits one of these structured objects:

```text
AgentSignal    -> evidence-backed signal, confidence, risk penalty
AlphaPrediction -> probabilistic alpha, volatility, downside risk, confidence
TargetWeight   -> target portfolio weight, not an order
```

Forbidden:

```text
LLM -> market order
Agent vote -> final trade
News sentiment -> direct alpha without validation
Backtest-free strategy -> production execution
```

Allowed:

```text
LLM -> structured event
Agent debate -> evidence and risk tags
Model -> probabilistic alpha
Optimizer -> target weights
RiskGate -> approval or blocking
QMT -> execution only, later
```

## Repository Roles

TradingAgents is mapped to the short-horizon event desk:

```text
Technical Signal Agent
Event Shock Agent
Statistical Regime Agent
Microstructure Agent
Short Risk Agent
Trader Agent that emits target weights only
```

ai-hedge-fund is mapped to the long-horizon fundamental desk:

```text
DCF / Reverse DCF
Quality and moat
Growth and TAM
Policy alignment
Ownership flow
Tail risk and fraud risk
Long thesis memory
```

The original personality-style agents should become testable factor models.

## Phase 1: No-Training System

The first runnable version should use:

```text
OHLCV technical indicators
Rule signals
Fundamental DCF / reverse DCF
Quality and fraud risk scores
Regime filter
Transaction cost model
Continuous optimizer
Risk report
```

This gets the system to a backtestable, explainable baseline before GPU training.

## Phase 2+

```text
Phase 2: LightGBM / CatBoost ranker, purged CV, dynamic agent weights
Phase 3: PatchTST / iTransformer / TSMixer, event embeddings, anomaly models
Phase 4: constrained RL with target-weight action space
Phase 5: paper trading and QMT execution gateway
```

## Weight-Centric Flow

```text
features
  -> short_rule_signal / long_horizon_score / agent_signal
  -> raw weights
  -> optimizer target weights
  -> A-share rounding and risk gate
  -> future QMT Gateway
```

QMT is intentionally absent from this layer.
