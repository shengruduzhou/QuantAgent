# V6 Agent 证据系统 / Agent Evidence OS

## 目标 / Goal
Agent 只生成 structured evidence，不能生成 order、target_weight 或 broker instruction。

## 架构 / Architecture
`AgentCommittee` 调度 sentiment、news、policy、flow、commodity、sector 和 financial_statement evidence；`AgentRouter` 接入 `AgentReliability`，把 EvidenceRecord 转成 BL-style AgentView。

## 数据流 / Data Flow
新闻和财务等原始记录被 schema 化为 EvidenceRecord，Router 按 reliability 调整 q 和 omega，posterior alpha 由 `bayesian_arbitration.py` 生成。

## 关键模块 / Key Modules
`agents/agent_committee.py`、`agents/agent_router.py`、`agents/agent_reliability.py`、`agents/news_agent.py`、`agents/bayesian_arbitration.py`。

## CLI 使用方式 / CLI
```powershell
quantagent build-portfolio-v6 --config configs/v6.default.yaml --date 2026-03-31
```

## 配置方式 / Config
`configs/v6.default.yaml` 的 `agents` 节控制开关和 reliability halflife / cold_start / min_score / max_score。

## 安全边界 / Safety
LLM 可做摘要和事件抽取，但输出必须 schema validation 并降级为 EvidenceRecord；LLM 不可用时 fallback 到 lexicon / rules / mock evidence。

## 测试方式 / Testing
`tests/test_v6_agent_reliability_router.py` 验证 reliability 会影响 q 和 omega。

## 验收标准 / Acceptance
Agent evidence 写入 audit 所需字段：source、timestamp、symbol、event_type、direction、magnitude、confidence、reliability、rationale、raw_reference hash。

