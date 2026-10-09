# R2_risk summary (recorded by chair from hand-back; harness refused the agent's summary write)

Counts: P0 ×4 · P1 ×6 · P2 ×4 · P3 ×1. Details/verbatim repro: findings.md (F01–F14).

## Ranked
| # | Sev | Title | Conf |
|---|---|---|---|
| F01 | P0 | Paper venue crashes on 2nd name (`check_order` values the book with only the order's price → UnpriceablePosition); ledger left FILLED+phantom SUBMITTED; introduced by DEF-039 fix 1d1fc0a; all tests single-symbol | CONFIRMED |
| F02 | P0 | `check_portfolio` (DD 20%, daily loss 20k, gross ≤1, portfolio kill latch) zero production calls; BUY FILLED at −24.41% DD | CONFIRMED |
| F08 | P0 | v7 executable backtest reports DD vs rolling-252 peak (synthetic −22.21% vs true −32.98% passes 25% gate; real data −60.4% vs −92.7%); arithmetic annualisation; missing benchmark = 0%; overwrites metrics["max_drawdown"] feeding gates/Optuna | CONFIRMED |
| F03 | P0 | Venue passes only reference_price+session_volume ⇒ industry limit & stale-quote never fire; approvals default True | CONFIRMED |
| F04 | P1 | Kill switch & turnover in-memory; restart clears; no API to trigger/view | CONFIRMED |
| F05 | P1 | max_orders_per_symbol_per_day only in reconcile(); 8 orders vs limit 5 | CONFIRMED |
| F07 | P1 | /api/paper/orders has no market source → refused after RISK_APPROVED+SUBMITTED ⇒ phantom working order | CONFIRMED |
| F09 | P1 | No sector map ⇒ sector cap skipped but reported 0.3 (100% single-sector book); paper run-loop lacks --sector-map | CONFIRMED |
| F10 | P1 | Vol target/DD ladder/Kelly/beta only in training backtest, orphan helpers, never-constructed RiskGate; production targets always 100% gross | CONFIRMED (grep) |
| F06 | P2 | NaN session volume on HTTP venue ⇒ uncapped fill, zero impact | CONFIRMED |
| F11 | P2 | Real-data risk profile | CONFIRMED |
| F12 | P2 | Parent→child/TCA (PR #133) zero production call sites; late restart releases overdue slices at once | CONFIRMED/PLAUSIBLE |
| F13 | P2 | Free-text author can flip council portfolio_risk blocked→pass via POST /api/council/overrides | PLAUSIBLE |
| F14 | P3 | Square-root impact not vol-scaled (~4× too high typical); several declared flags never read | CONFIRMED |

Repro: test_r2_continuous_multiname.py (1F), test_r2_adversarial_venue.py (8F/7P), test_r2_http_no_market_phantom.py (1F), test_r2_executable_backtest_drawdown.py (3F), test_r2_target_weights_sector_cap.py (1F/1P). Failing tests assert correct behaviour.

## Controls vs reports
Hard-blocked on order paths: qty>0, finite limit price, lot/max qty, limit-up buy/limit-down sell/suspended/ST buy, band, T+1, cash, single-name ≤10% buy, ≤200k CNY/order, fat-finger 10%, idempotency, single-writer lock.
NOT controlled: vol targeting; drawdown/daily loss (F02); VaR/ES; beta (RiskGate never constructed); sector at venue (F03) and at targets only with map (F09); order rate (live-only); participation metered not refused. Only measured in backtests (DD under-measured F08).
Critique verdicts: (a) PARTIAL (deterministic static limits, not LLM, no vol use); (b) TRUE on production path; (c) TRUE order path / PARTIAL targets; (d) FALSE (sqrt impact, caps exist; flaws F06/F14); (e) LLM bypass FALSE, mixing PARTIAL (post-trade never runs; council override F13); (f) PARTIAL (per-order hard blocks; no portfolio breaker; kill switch not persistent, no trigger).

## Open items
(a) Parent-child/TCA: not wired; Perold IS math OK; planner invariants OK; issues: late-restart burst, no session check, child orders without limit price, iceberg not randomised.
(b) Account identity: continuous account identity IS observable (/api/paper/execution-evidence → accountIdentity; PaperExecutionEvidencePanel.tsx:281-284). Missing: /api/paper/account lacks portfolioId/accountInstanceId/identity SHA (1M initial cash hard-coded in container.py); UI never calls /paper/account|orders|policy; neither exposes risk state. Spec: riskState{limits, riskEngineAttached, killSwitch{active,scope,reason,triggeredAt}, peakEquity, drawdownFromAllTimePeak, sessionTurnover, unpriceableSymbols} + accountIdentity on /paper/account + account-and-risk card in TPlusOneExecutionWorkspace.

## Real data (≤2025-08-29, two runs identical SHA-256)
Top-50 book net: vol 31.3%, VaR95/99 3.19%/5.86%, ES97.5 5.91%, MaxDD −92.7%, 2,121 sessions underwater. EW universe: vol 24.1%, MaxDD −43.7%, 819 underwater. Most-traded-300 proxy: vol 26.0%, MaxDD −72.9%. Beta 1.108 vs EW (corr .853), 0.939 vs MT300. Rolling-60 vol 16–60%. HS VaR95 Kupiec p=.41 but Christoffersen p=.0003 (P(exc|exc)=14%); EWMA-normal VaR99 48 vs 20.8 expected (p=3e-7). Max sector weight median 12%, p95 22%, max 46% (current-snapshot map). Liquidity >10% of 20d ADV: 0% at NAV 10M, 60.5% at 100M, 100% at 1B. Cost drag 9.8%/yr (measurement vehicle, not alpha claim).

## Recommended minimal design
1 deterministic risk middleware before RISK_APPROVED (full marks, PIT industry, quote timestamps; unmeasured ⇒ reject; risk state persisted as ledger events). 2 DD ladder vs all-time peak 5/10/15/20% → 100/75/50/25% then reduce-only, in daily_loop, same function as backtest. 3 vol targeting gross=clip(σ*/σ̂,0.2,1). 4 quarter-Kelly cap only. 5 ES97.5 limit; beta via gross or index futures (转融券 suspended since 2024-07-11); construct RiskGate. 6 order-rate rules on paper path (CSRC HFT 300/s or 20,000/day). 7 report all-time-peak DD + duration + geometric CAGR.

## Verified correct
Venue blocks limit-up/ST/suspended/NaN-inf price/50% single name (7 pass); continuous loop refuses bad market values; no override path; upstream "approved" cannot skip checks; Shanghai session date in OMS; DEF-033/034/040, round-27 restart, LIVE_DISABLED not regressed. 0/15 equity-curve artifacts affected by rolling peak.
Not checked: end-to-end 81% sector book (F01 crashes first); QMT; PIT sector map & certified CSI300 (absent); SSE rules page unfetchable (gmw.cn reprint + CSRC used); F13 not executed.
