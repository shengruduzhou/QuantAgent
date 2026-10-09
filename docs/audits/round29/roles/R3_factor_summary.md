# R3_factor summary (recorded by chair from hand-back; harness refused the agent's summary write)

Counts: P0 0 · P1 5 · P2 7 · P3 2 (13 CONFIRMED, F10 PLAUSIBLE). Details + verbatim repro in findings.md.

Top: F06 E1 pre-registered real-data result (≤2025-08-29, 158 distinct price-volume features, 389 OOS dates, determinism run identical): pooled rankIC IC-weighted 0.0765 < best single 0.0822 < ridge 0.0962 < LightGBM 0.1088; R1 TRUE (+0.0126, t_HAC 3.51, 5/6 folds, decile spread 172 vs 138 bp/5d); gain = interactions (depth-1 ≈ ridge; full trees +0.0099, t 2.62). Holds after removing sealed limit-up entries (+0.0127, t 3.50). Gross, no costs/capacity, micro-caps included.
F01 sealed limit-up never excluded (mask_limit_up has no producer; 144,840 rows, mean fwd5 +4.0% vs +0.15%; inflates decile spread 13–17%; entry_feasible constant True).
F03 nonlinear comparison CLI has no quarantine guard, folds anchored at panel end (all in quarantined windows); never run on real data.
F02 round-23/24 factor screens used 2,558 dates incl. 168 burned + 42 frozen-fresh sessions.
F05 LLM expression parsed with eval (1,175 reachable classes); Delay(close,-1) accepted = lead.
Others: F04 pruning sign-blind (65 vs sign-aware 107 survive cost; 1-day churn vs 5-day spread); F07 Fuyao available_at same-day; F08/F09 lifecycle ledger + validity gate not wired, no Fama-MacBeth, multiple-testing is a caller-set flag; F11 certified labels.parquet overwritten with same-close label (99.5% rows differ from dataset.parquet); F14 RD-loop reuses one validation window and feeds IC back to LLM (no embargo, no trial count; 53/96 accepted); F10 model comparison day-to-day turnover vs 5-day rebalance (PLAUSIBLE); F12/F13 absent sentiment/policy filled 0.0; stale availability text.

Answers: Not stuck in RSI/MACD — features are 15 base PV + 83 Alpha101 + 64 GTJA191 with real OOS signal; BUT 100% of model features are daily price/volume: fundamentals 0 rows, no earnings-surprise/analyst-consensus source or code, sentiment 51 news rows from one day, no alternative data. Ranking is per-date full cross-section (no batch-local rank). Missing → median rank (documented). No industry/size neutralisation anywhere on the certified path. LLM only proposes factor expressions, never trades.
Not checked: costs/capacity for E1; ST exclusion (panel not ST-PIT complete); full RD-Agent paper; UI.
