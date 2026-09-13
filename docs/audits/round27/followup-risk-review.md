# Round 27 风控复核 / Risk follow-up

## 已确认的独立发现

独立只读 agent risk_recheck 对 ab273de529759e16ea5c9c77ae4a3a9ce550b47b 的裁决为 request changes，复现两个 P1：

- 历史有效 NAV / daily volume 掩盖当前缺失测量，第二笔订单仍到达 broker。
- 时间戳 2026-08-17T22:30:00-04:00 与 2026-08-18T02:30:00Z 属于同一 Shanghai session，但 canonical literal date 与恢复日期不同，重启可漏计订单。

复现只使用 synthetic broker doubles，没有实盘调用。该报告不是修改后 approve。

## 本次修改

- 当前订单缺少启用约束所需的有限正测量时，明确记 unmeasured，即使历史完整也不放行。
- canonical trade date、idempotency date、risk session 共用 Shanghai session helper。
- 增加 NAV / volume / both 缺失 × restart / no restart 六项回归，以及跨时区重启回归。

## 验证与限制

- 2026-09-12 本地 tests/execution/test_order_manager_pretrade_risk.py：19 passed。
- 原提交 ab273de 的 GitHub test job 103370036753：3591 passed，52 skipped，706 warnings；这不是本次新提交的验收。
- 原提交 audit job 103370034847 因缺 testing_expert、quant_expert_tester、ai_quant_expert_auditor 的 exact-head 审核失败，未绕过。
- 本地 frontend build 成功；启动 QA fixture 服务后浏览器调用未返回，环境再次离线。浏览器验收仍 NOT_RUN。
- 本次两文件修改由已记录 patch 和固定远程 base 恢复保存，需新提交 CI 与独立复核。
- 历史旧格式错日 ledger 的迁移、账户级并发风险序列化和 open-order operational recovery 不在此修复中；不声明 live readiness。
- PR 保持 Draft，不合并。
