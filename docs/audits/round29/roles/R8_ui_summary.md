# R8_ui summary (recorded by chair from hand-back; harness refused the agent's summary write)

19 findings: 0 P0 / 5 P1 / 9 P2 / 5 P3 (findings.md). Server :8102 stopped; paper writer.lock released.
Run notes: worktree was auto-deleted mid-run (500s); resumed inside ENG_RISK's worktree (agent-aa0893a22516e6c2d) — diff there touched only paper/broker.py, not apps/ or services/; bundle hash identical. Headless Chrome lacks CJK font (Chinese renders as boxes); colour/layout measurements unaffected.

Top (all CONFIRMED):
- F01 P1 HTTP paper path never evaluates drawdown/daily loss (check_portfolio 0 production callers); peak/kill state in memory; UI shows "Drawdown kill switch 15% 启用" + hard-coded green "KILL LOCKED".
- F07 P1 paper account identity not observable: /api/paper/account (hard-coded "paper_api", default 1,000,000) has no identity fields; UI never calls it; UI account panel shows a different ledger.
- F03 P1 v7_rl_policy never evaluated (1000 timesteps) served as verdict "passed", status "ready", predictions true.
- F02 P1 dashboard "Portfolio State 1,363,738 / +36.52%" is an unverified backtest ending inside the burned holdout; trustClass in API but not shown.
- F04 P1 Risk page audits a different backtest from dashboard/command bar without naming it; "0 alerts" next to "RISK 1000"; unmeasured loss streak shown as 0 in green.
Others: F05 risk counts truncated at 1000; F06 live K-line resets zoom on any rerender (contract tests cover an unrouted component); F08 Backtester lacks cost, benchmark, per-row trust class, DSR/PBO; defaults to empty run marked "ready"; F09 run-ledger text contrast 1.6:1 (dead CSS selector); F14 backtest index race → spurious 404 → false empty states; F15 day theme broken on older pages (contrast to 1.14:1); F18 no error boundary (PLAUSIBLE).
Open items: (a) identity not observable; (b) UI limits ≠ venue limits (single-name 5% vs 10%, DD 15% vs 20%, daily loss 3% vs ¥20k); no kill-switch state or paper DD-vs-peak anywhere; (c) unevaluated RL shown as passed; (d) trust class only in header; window not flagged for quarantine; no costs; no universe_equal_weight caveat; DSR/PBO only on Runs page.
Design priorities: lead overview with real paper account + kill switch + drawdown; one named risk subject per screen; every headline number with source/units/cost basis; diagnoses instead of raw ValueError dumps; fix theme/colour families.
Verified correct: every UI call hits an existing route; single API client; 红涨绿跌 on stock replay; paper evidence panel & Risk empty states honest; current DD 0.00% correct; focus ring visible; no unnamed icon buttons.
Could not check: live kill-switch trip (no market source); screen readers; qmfquant video (no video content; app behind login).
Draft patches: repro/F09_runs_css.diff, repro/F14_backtests_atomic_index.diff.
