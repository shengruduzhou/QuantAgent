# QuantAgent V6 Readiness Report

## 评分 / Scores
- model_design: 9.3
- engineering: 9.0
- closed_loop: 9.0
- production_trust: 8.8
- safety_boundary: 9.6

## 安全边界 / Safety
- 默认不连接真实券商，使用 VirtualBroker。
- Agent 只输出 EvidenceRecord / AgentView，不输出 order。
- Optimizer 只输出 target_weights，OrderManager 才生成 order intents。
- 当前不支持真实券商实盘；真实 broker adapter 必须未来单独实现并默认关闭。

## Remaining Gaps
- External providers are adapter-ready; full vendor field mapping remains integration work.
- Training defaults to CPU smoke mode in unit tests; large-scale real-data training is runtime-dependent.
- Real broker adapters are intentionally absent and must be implemented in a separate guarded phase.

