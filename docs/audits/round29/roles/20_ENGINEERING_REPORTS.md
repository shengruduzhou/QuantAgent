# Round 29 engineering reports (recorded by chair from hand-backs)

## ENG_RISK — agent/round29-risk-middleware (merged)
6a73863 F06/F07 venue refusals terminate canonical order; snapshots reject unmeasured values · 088b198 F01/F03 whole-book marks, industry, quote age · 8d357fc F02/F04 portfolio limits at venue + persisted risk state · 0193fa7 F05 per-symbol order count on every submission · 82f1952 F03 sector-map plumbing · d72e618 F09 sector cap published unenforced without map · 39ff2b1 /api/paper/account accountIdentity + riskState.
Full suite 3745 passed / 47 skipped / 0 failed (scratch QUANTAGENT_HOME). Chair follow-ups: 44e2fff daily-loss breaker 5% of opening equity (was 20k CNY ⇒ ~1 in 10 days); 2dc4125 canonical sector_level_1 map accepted (repo's own map crashed both loader & optimiser); 0e513f0 HTTP service settles T+1 on session roll (pre-existing).
Deferred: F10 ExposurePolicy; reconcile() keeps legacy silent per-symbol skip for forensic replay.

## ENG_DATA — agent/round29-data-foundation (merged)
abf6eb4 R1-F05 per-exchange ST · a2bd5fd R3-F01 tri-state limit masks from raw prices, labels refuse without them · 321dec5/eb775f4 R1-F06/F01/F02 certified raw execution panel + baseline_protocol --panel verification · ceb3fb8 R3-F11 label convention stamp, certified gold immutable (root cause: workstation "重建 Labels" → build-labels-v7 overwrote certified labels) · 25a024b/43dc6b0/424b6a5/26c11aa perf, phase split, USD B-share limits UNKNOWN, delisting write-off.
Extra: raw panel had no corporate actions ⇒ dividends booked as losses; panel now carries ca_cash_per_share/ca_share_ratio and the strict simulator credits holdings. U0 rights_ratio is actually 转增. R1-F10 raw-panel numbers lacked CA credits.
Targeted 900 passed / 17 skipped; full suite timed out at 63% under load (0 failures to then).
Data: runtime/data/gold/full_universe_r29/ dataset 10,582,839 rows / 5,790 symbols hash aad5261cfe4bbf99 (gold certificate, no failed/unknown checks, peak RSS 25.9 GB); labels sha f22dbbee2b52ee2b convention delay1_close_t1_to_close_t1_plus_h_v1; execution_panel 11,081,730 rows hash 2c9dd58b75528be7 (+149,746 MISSING_UNEXPLAINED, 7,373 SUSPENDED, 261 DELISTED; 35,923 CA steps matched dividend records, 1,102 credited at value).
Mask change vs certified 10b63ba024dd7428: limit_up_at_t1 drops 158,511 (key absent before); SZ ST TRUE 191,525 (dropped st_at_t); SH/BJ ST UNKNOWN; seasoning TRUE 345,677→166,522; UNKNOWN limit-up t+1 kept 23,619.
FIRST POST-TIMING-FIX CANONICAL BACKTEST (variant C on r29 execution panel; weekly top-50 0.5·rank(−vol20)+0.5·rank(−ret20); signals to 2025-08-20; exec 2017-01-04..2025-08-21; 8 bps): ann −11.31%, total −64.49%, Sharpe −0.569, MaxDD 73.36%, Calmar −0.154, PSR 0.046, MinTRL undefined (Sharpe≤0), bootstrap 95% CI [−1.275, 0.116], not deflated; benchmark EW hfq TR +6.90% ⇒ excess −18.21%; costs 543,725 CNY (slippage 245,092); 9 delisting write-offs 70,567 CNY; run1==run2 nav sha256 22917ec3…0096. Trust: research_certified_r29_panel_st_incomplete.
Deferred: SSE/BSE dated ST register absent; default --panel still legacy (reported unverified); rename rights_ratio; 13,460 in-life gaps without prior close excluded; certified dir labels still the overwritten file.

## ENG_AKSHARE — agent/round29-akshare-truth (merged)
0d24450 per-response volume unit (x1/x100 must put VWAP in [low,high] on ≥95% rows, else UNIT_AMBIGUOUS), refuse vendor qfq/hfq, gate Tencent failover · 3c18e1f refresh never splices sources or vendor-adjusted prices · 7690cc0/60e7015 scripts/audit_market_panel_units.py · af8d4a5 fail loudly on akshare version ≠ pin (override QUANTAGENT_AKSHARE_VERSION_OVERRIDE) · 8570fc6 acquisition/probe raw only · 77fbe97 docs unit truth.
Closed: STAR ×100, sz000 ÷100 (still in 1.18.84, caught live), EastMoney lots per response, qfq refused, version drift. R5 repro 4/4 pass. akshare-related 713 passed / 16 skipped on 1.18.60 and 1.18.84.
Audit v7 panel 2021-01..2025-08: 4,126,420 rows all stamped PIT-valid ⇒ REFUTED; 3,325,346 non-raw, 3,263,319 qfq-as-of-fetch; sina VWAP-in-range 63.4%, close match U0 33.5%; east_money volume 0.01× U0 on 100%.
Live smoke (5 symbols, 2024-01, raw, 1.18.84) vs U0: close/volume/amount ratio 1.000 for 600519.SH, 000001.SZ, 300750.SZ, 688981.SH, 920000.BJ.
VENV: akshare now 1.18.84 (the pin), nothing else changed.
Deferred: v7 silver panel + 13 U0 Tencent-served symbols not rebuilt; TickFlow/minute adapters still ×100; version check only on daily-bar provider.

## R9 — agent/round29-deadcode (merged)
e8fdcfd 7 zero-ref modules (−532) · ac58ed6 execution/risk_kill_switch shim · 77e762d clean_room dataset/risk · 3e275e1 2 crashing RL scripts on burned holdout · 00423bf unreachable v8 tail · 7b602b9 CPCV/triple-barrier/uniqueness + guard test · 65bf186 F01 acceptance producer forged 0.0 (empty summary passed drawdown gate) · d4ce8e8 F02 --env-config dropped fields. 27 files +233/−1471. Tip on exact archive: 3794 passed / 47 skipped / 0 failed.
Open: F03 v7/agent_contracts.py lists 6 nonexistent extension files; F04 model API shows burned-holdout PPO eval numbers unstamped; F05 P1 target-weights builder can select a name with no bar that day (gaps read as tradable). F06 refuted (pandas downcast warnings harmless).
Branch adjudication: authoritative-acceptance-calendar VALUABLE-UNMERGED (port next round); 9 others SUPERSEDED (#131, #80/#83, 9fd56d6, #133, #130+3e275e1, continuous_execution #83/#126/#128/#129/#131 ×2, cli/nonlinear+nonlinear_promotion, 9fd56d6).

## ENG_DATA follow-up (R10-F01/F05) — merged b6b8e74
2ae07cd zero-volume bars = no-trade (mask_no_trade; carried hfq close; build_labels drops no_trade_at_t 25,518 + entry_zero_volume 25,489; execution panel NO_TRADE_ZERO_VOLUME 25,047; verify refuses TRADED zero-volume; certificate no_zero_volume_rows). Real checks: 000836.SZ 2016-02-25 hfq 67.54→67.57 (not ×3); 000038.SZ 2017-07-06 60.68→60.69 (not +60%).
3c9ef35 holding-period dividend tax FIFO lots (20% ≤1m, 10% ≤1y, 0% after; 财税〔2015〕101号); 送股 par-value tax not modelled.
Rebuild r29 v2: dataset 10,561,245 rows hash 5e358c198532f2ac (certificate granted, 0 failed/unknown), labels sha 30bede0bd4071eb7; execution panel 11,081,259 rows hash 682740a03aec19bb; is_suspended 158,407→183,099.
CANONICAL v2 (weekly top-50 low-vol+reversal, variant C, signals ≤2025-08-19, exec 2017-01-04..2025-08-21, 8 bps; trust research_certified_panel_st_incomplete): ann −11.73%, total −65.93%, Sharpe −0.592, MaxDD 74.25%, Calmar −0.158, PSR 0.039, MinTRL undefined, CI [−1.300, 0.097]; benchmark EW hfq TR +6.89% ⇒ excess −18.62%; costs 541,465 (slippage 243,168); dividends gross 58,062, tax 11,255; 343 CA credits; 9 write-offs 68,883. run1==run2 nav sha256 e0103695…54c79dc.
