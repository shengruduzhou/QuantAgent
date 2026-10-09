# R5_test — Round 29 (recorded by chair from hand-back; harness refused the agent's file writes)

Evidence: repro/, data_evidence/, logs/, screens/, raw_ak118/, raw_ak119/, xcheck/, reading_log.md. Re-ran against clean main @22b3f6c at main_22b3f6c/ after worktree deletion. Counts: 5 P1, 8 P2, 3 P3.

## AkShare verdict — why the user's akshare data "还是不太对"
1. P1 v7 silver market_panel.parquet (the panel actually used by baseline_protocol, paper/daily_loop, forward inference) is 99.6% akshare:sina from 2020; prices qfq as-of ~2026-05-18, volume/amount raw; all rows stamped point_in_time_valid=True. 2021-01..2025-08: 2,731,327/4,103,672 closes differ from certified U0 raw by >0.5%; VWAP outside [low,high] on 1,948,734 rows; re-downloading qfq today changes 2024 prices (600519 −2.4%, 000001 −5.6%); 1,901 post-seam ex-dividend days stored raw on top of qfq history.
2. P1 all 15,971 akshare:east_money rows have volume exactly 0.01× U0 (lots not shares).
3. P1 "qfq" differs by source: EastMoney/Tencent qfq subtractive (raw−qfq constant), Sina multiplicative (matches gold hfq_factor to 5e-7); Tencent qfq daily returns off up to 2.83% with no ex-date in window. EastMoney intermittently RemoteDisconnected (3/15 then 0/15; upstream #7330) ⇒ failover mixes sources per symbol.
4. P1 Tencent units: STAR volume inflated 100× by `_normalize_akshare_daily` and U0 TencentSource (ratio 100.000032 on 58/58 days; upstream PR #7328: STAR volume native shares; gold 688806.SH shows 1.0e11 CNY/day turnover). Pinned akshare 1.18.84 (= 1.19.1) skips ×100 for any `sz000` symbol ⇒ 000001.SZ volume 0.01× labelled shares.
5. P2 version drift: venv 1.18.60 vs pyproject pin 1.18.84; different Tencent shapes.
Minimal fix: rebuild akshare rows with adjust="" from Sina only; per-board volume units (688/689 = shares) + amount/(volume×close)≈1 check; install pinned version and fail on mismatch; keep Tencent out of failover until sz000 guarded.
Verified correct: Sina raw matches U0 exactly on 5 boards incl. BSE 920000; EastMoney ×100 normalisation right; gold hfq_factor agrees with Sina hfq.

## Full-stack runtime
Frontend: typecheck pass; vitest 124 passed / 1 skipped (live-API); vite build OK. API: 99 GET routes swept, one 5xx, none >3 s. Browser: 27 routes + dawn/day themes; charts readable (host lacks CJK fonts).
Findings (failing tests repro/test_r5_api_honesty.py, repro/test_r5_tencent_volume_units.py: 6 fail, 1 control pass, twice):
- P2 /market-playbooks raises React #310 (hook after early return) ⇒ root empty, whole app blank.
- P2 /selection/runs/{bad}/…/decision-chain 500 (KeyError) not 404.
- P2 /risk/overview?backtestId=<unknown> silently returns another run's risk with status ready.
- P2 consecutiveLossDays 0 with no equity curve; top bar "RISK CLEAR" while loading.
- P2 "1000 active violations" = first page of historical order_skipped events, not a count.
- P2 dashboard headline/backtests from burned/frozen holdout unlabelled.
- P2 /factors/{f}/ic returns ready with all null.
- P3 drawdown axis label overlap; API writes into shared runtime at startup; ~33 server routes never called by UI.

## Cross-check of R1/R2/R3/R11 repros (clean main, twice)
11 test files: 32 tests fail as claimed, 33 controls pass, no import/setup errors. R1 5 (limit dating, stamp, rotation); R2 14 (multi-name fixture verified to price both names); R3 4; R11 2 (+7 PR #156 tests fail on main as expected). 9 read-only probe scripts ran clean (evidence printers). Broken: R1 diag_bp_missing_bar_main.py (depends on R1 cwd). Not run: e1_run, usefulness_run, certified backtest. Refuted: none.

Top 5: F01 v7 akshare panel qfq+raw volume+forged PIT flag; F02 EastMoney lots; F03 STAR ×100; F04 sz000 ÷100 at pinned version; F05 playbooks page crash.
