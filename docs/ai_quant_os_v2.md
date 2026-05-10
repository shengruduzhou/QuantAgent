# AI Quant OS v2 / V2 设计回顾

V2 的核心价值是把 LLM 和 multi-agent 系统定位为 evidence processors，而不是 order generators。V4 保留这个边界，并把 `AgentSignal` 升级到 `EvidenceRecord` 和 `AgentView`。

## 核心合约 / Core Contract

研究组件输出结构化对象：`AgentSignal`、`AlphaPrediction`、`TargetWeight`。V4 进一步要求 Agent 输出 `EvidenceRecord`，再由 `AgentRouter` 转成 Black-Litterman views、portfolio constraints、risk warnings 或 no-trade flags。

禁止路径 / Forbidden path：

```text
LLM -> market order
Agent vote -> broker submit
Optimizer -> broker order
```

允许路径 / Allowed path：

```text
LLM -> structured evidence
Model -> probabilistic alpha
Optimizer -> target weights
OrderManager -> dry-run order intents
QMTGateway -> dry-run audit by default
```

## 与 V4 的关系 / V4 Relation

V2 是边界定义，V4 是可运行闭环。V4 增加 point-in-time feature store、A-share rule engine、multi-tower model、factor lifecycle、event-driven backtest 和 QMT dry-run gateway。

## 测试 / Tests

```powershell
python -m pytest tests/test_agent_views_v4.py tests/test_v4_services.py
```
