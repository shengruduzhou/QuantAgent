# R1_backtest summary (recorded by chair from hand-back; harness refused the agent's summary write)

Counts: P0 1 · P1 7 · P2 6 · P3 1 (F01–F15 in findings.md). Read 12/12 assigned URLs + 11 searches.

Required real-data re-backtest: production_composite.parquet is not built on the certified panel (manifest says likely_overfit) and the canonical evaluator crashes on it (F01) ⇒ ran a preregistered transparent book instead: signal 0.5·rank(−vol_20d)+0.5·rank(−ret_20d), top-50 EW, 8 bps slippage, 1M CNY, run_strict_backtest_v8 on certified gold (hash 10b63ba024dd7428), 2017-01-04..2025-08-29, two runs NAV-hash identical; gross engine returns vs independent vectorised: daily-return corr 0.97/0.99 (no look-ahead signs).

| Series | Ann | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| Daily book net | −25.57% | 16.2% | −1.74 | 92.3% |
| Weekly book net | −9.51% | 16.9% | −0.51 | 68.2% |
| Weekly gross (approx) | −1.06% | — | 0.02 | 45.3% |
| Universe EW | +6.17% | 23.3% | 0.37 | 44.2% |
| CSI300/500/1000 price | +3.49%/+1.26%/−1.62% | | | |
Turnover one-way 66×/yr daily, 28×/yr weekly; min commission makes daily effective 7.6 bps vs 3; statutory pre-2023 stamp ⇒ −27.5% / −10.5%. Trust class research_certified_panel_factor_backtest_st_incomplete. No v8.9 headline reproduced or supported.

Ranked: F01 P1 canonical evaluator crashes on own v7 panel (missing suspended rows fatal since #119; amount NaN 2017–2020) ⇒ no post-fix canonical number for any artifact. F03 P0 flat stamp duty (−1.0..−1.9 pts/yr understated cost). F10/F10b P1 real backtest above. F12 P1 UI lists 42 backtests, all pre-fix, none with canonical timing stamp; 15 from _VOID_pre_round20_backtests (one +126%/yr) (repro/scan_ui_backtests.py). F04 P1 sort-order dependent rotation (3,549 insufficient_cash rejects in real run). F09 P1 undated limits (4,711 ChiNext limit-up closes missed pre-2020). F05 P1 gold mask_is_st UNKNOWN 100% incl. SZ (388 SZSE intervals ignored). F02 P1 v7 panel qfq levels + raw volume + 3 vendors stitched.
P2/P3: F14 total_cost omits slippage (423k shown vs 749k paid); F07 CPCV/triple-barrier/uniqueness 0 call sites, no MinTRL, baseline_protocol no DSR/PBO/SPA; F08 per-run trial count; F11 impact lacks vol term, close fills may take 10% full-day volume; F13 three fill clocks & three cost tables; F06 gold no suspended rows; F15 quarantine fallback list omits fresh window, UI export fills benchmark gaps with 0.

Verified correct: strict timing T close → next-session close, same-session fills fail closed; certified labels match (hand-checked 600519); certified gold back-adjusted, ÷factor reproduces raw exactly; delisted names present; golden buy leg exact; post-2023 stamp correct; DSR formula & PBO structure correct; SPA/PBO/DSR wired fail-closed in research gates; legacy RL env refuses untradable reward unless acknowledged; no regressions.
Could not check: cross-engine parity runs (code reading only); PIT of fundamentals & index membership; SSE/BSE ST (no dated register); delisted holdings frozen at last close (flatters slightly).
Repro scripts point at removed worktree; rerun with PYTHONPATH=/home/shanhefu/QuantAgent/src (same commit 22b3f6c).
