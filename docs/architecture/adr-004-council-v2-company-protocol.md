# ADR-004 — Council v2：11 岗公司共同决策协议

- 状态 / Status: Accepted
- 日期 / Date: 2026-08-27
- 取代范围 / Supersedes in part: [ADR-003 的七岗 Council v1](adr-003-atlas-fusion-and-council.md)

## 背景 / Context

Council v1 固定七个研究角色；Strategy preflight 又形成一套八角色 ID。两套契约与
产品要求的 11 岗公司流程不一致，旧 override 还只按 run + role 生效：候选、产物或
阈值变化后，历史人工改判可能被错误套用到新的裁决。

## 决策 / Decision

Council v2 以 `services.quant_api.services.council.COUNCIL_ROLES` 为唯一 registry，
固定四阶段与 11 岗：

1. Data admission：`data_acquisition / data_quality / microstructure`；
2. Research validation：`factor_integrity / model_validation / fusion_search`；
3. Portfolio delivery：`portfolio_risk / execution_realism`；
4. Independent decision：`challenger / compliance / governance`。

Strategy preflight 复用同一 registry。CIO / governance 必须消费前十岗的结构化
verdict；任一前置 `blocked` 或 `unknown` 时，主席不得返回 `pass`。

v2 promotion thresholds 固定为 PBO ≤ 0.25、DSR ≥ 0.95、SPA p-value ≤ 0.05。
协议版本、角色、顺序、veto scope 与阈值共同生成 `policyFingerprint`，由 roster、
review、UI 和 override 一起携带。

缺失 required evidence 必须为 `unknown`。研究可以继续，但 `eligibleForHumanGate=false`；
`liveEligible` 在本层永远为 false。对照组或单因子胜出是有效的负面研究结论，但不是
可晋级策略，因此必须阻止 promotion，不能包装成可晋级 warning。

每条 v2 override 冻结：

- `protocolVersion` 与 `policyFingerprint`；
- `subjectContentHash` 与 `candidateId`；
- `findingHash` 与 `originalVerdict`；
- author、reason、new verdict 与 recordedAt。

只有全部 scope 精确匹配时 override 才参与 effective decision。v1 无版本记录、或
候选 / 产物 / finding / policy 已变化的记录，只以 historical / stale 状态展示。
`unknown → pass` 不被接受，因为理由文本不能制造缺失证据。

## 后果 / Consequences

- 角色数量、UI、Strategy preflight 与正式审查不再漂移；
- 历史审计记录完整保留，但不会无声改变新协议的结论；
- 阈值或证据变化会要求重新人工复核，这是有意的安全成本；
- ADR-003 继续记录 L2 Fusion 与 Council v1 的历史决策，不回写其 Accepted 事实。
