# V7 Hardening Migration / V7 强化迁移说明

## 目的 / Purpose

本次 hardening 把 V7 从 mock-friendly research pipeline 推进到更严格的 real-data、PIT-safe、risk-controlled paper-trading scaffold。公开 API 保持兼容，但几个 unsafe fallback 被移除。

## 行为变化 / Behavior Changes

- `run_daily_v7_research(config, as_of_date)` 内部拆成 `V7DataStage` 到 `V7ReportStage`。
- Stock Pool Hard Gate 为空时 fail closed，不再回退到 `universe_members`。
- Gate fail 时 `multi_horizon_alpha={}`，`portfolio_plan.target_weights={}`，`risk_report.risk_passed=false`。
- Audit reason 固定为 `empty_after_stock_pool_hard_gate`。
- News tagging 从 keyword-only 升级为 keyword、security dictionary、ontology、optional reranker、cross-validation 分层。
- Public crawler 迁到 `src/quantagent/data/crawler/`，默认网络关闭，严格限速，不绕过 CAPTCHA。
- Qlib/AkShare 增加 health check 和 schema report。
- Agent contract 增加 runtime validators，禁止 order payload，强制 evidence traceability 和 downstream audit trail。

## 迁移建议 / Migration Notes

- 依赖旧 gate 回退行为的测试或脚本需要改为检查 `stock_pool_gate_failed`。
- 真实数据模式下请补齐 `available_at`，不要用 `report_period` 替代。
- 如果使用 Qlib，请先准备本地 `provider_uri`，再运行 `quantagent check-qlib-v7`。
- 如果使用 AkShare，请把 empty response 和 schema warning 当作数据质量问题处理，不要静默使用 mock fallback。

## 验证 / Validation

```powershell
C:.\AppData\Local\Programs\Python\Launcher\py.exe -m pytest tests/
C:.\AppData\Local\Programs\Python\Launcher\py.exe -m compileall src
git diff --check
git grep "old full-universe gate fallback marker"
git grep "agents_can_emit_orders"
```
