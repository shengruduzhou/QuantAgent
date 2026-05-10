# V4 Agent Views / 结构化 Agent 观点

V4 Agent 层不输出 orders。Policy、flow、commodity、fundamental evidence 都转换为 `EvidenceRecord`，再由 `AgentRouter` 转成 `AgentView`、constraints、risk warnings 或 no-trade flags。

## Schema / 数据结构

`EvidenceRecord` 包含 source、timestamp、symbol、sector、event_type、horizon_days、direction、magnitude、confidence、decay_half_life、rationale、raw_reference。

`AgentView` 包含 view_id、symbols、sparse exposure、q、omega、confidence、constraints、expires_at、evidence。

## BL 适配器 / Black-Litterman Adapter

`agent_views_to_bl_views` 将 AgentView 转成 P、q、Omega。`posterior_alpha_from_agent_views` 输出 posterior alpha，不产生订单。

## 审计 / Audit

`write_audit_jsonl` 以 deterministic JSONL 记录 evidence 和 views，方便回放。

## 测试 / Tests

```powershell
python -m pytest tests/test_agent_views_v4.py
```
