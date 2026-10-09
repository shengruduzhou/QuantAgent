# R7_expert (recorded by chair from hand-back; harness refused report files). Target c042afd.
Strategies via scripts/baseline_protocol.py (variant C, top-50, 8 bps, r29 execution panel; weekly signals 2018-01-05..2025-08-20; twice, byte-identical):
(a) value/quality tilt (R4 (d), v7 fundamentals): ann +4.33%, vol 17.7%, Sharpe 0.33, PSR 0.82, MinTRL 6,135 sessions, CI [−0.41, 1.12], MaxDD 33.2%, turnover 34.8%/wk, cost 422,014 CNY (slippage 210,519), excess vs EW all-A (+8.44%) −4.11 pp.
(b) low-vol (−vol_60d): +2.52%, 13.6%, 0.25, 0.76, 10,383, [−0.48, 1.05], 36.4%, 14.4%/wk, 163,131 (76,537), −5.92 pp.
Caveats: EW all-A daily rebalanced, cost-free, includes ST/untradeable; (a) on survivorship-biased 3,638-name fundamentals universe. Neither citable as edge.
Breaker sim (paper 20% DD, 5% daily loss; EOD closes): both latch the same reduce-only switch, never reset without human, peak never resets. (a) latches 2018-02-06 (−5.46% day) → reduce-only exposure 12% after 20 sessions → final −7.2% vs backtest +38%. (b) latches 2018-06-25 (DD −20.3%); 1,043/1,850 sessions below 80% of peak ⇒ re-latches after clearing. Daily-loss fired 8× (a), 4× (b). Verdict: tools can stop, cannot manage drawdown (no de-gross ladder / vol target / re-entry).
F1 P1 baseline_protocol.py:153 maps SH/BJ ST UNKNOWN (3.74M rows) → is_st False; unknown limit-up → True. SH/BJ = 69.2% (b) / 65.9% (a) of buy value. out.json lacks trust_class; 'research_certified_r29_panel_st_incomplete' exists nowhere in code. repro/st_exposure.py
F2 P1 backtests vs paper rules differ: strict simulator applies none of paper RiskLimits ⇒ reported paths unreachable under shipped limits. repro/breaker_sim.py
F3 P2 evaluator out.json drops vol, turnover, fees, impact; Backtester shows turnover 暂无. repro/run_capture.py
F4 P2 Evidence Center headlines non-citable run (ReportsPage.tsx:120: 36.52%, Sharpe 1.45, MaxDD 6.09%, burned holdout, pre-fix clock) with no caveat; flows into JSON/HTML exports.
F5 P2 Factor Lab "NO ARTIFACT" for all 275 factors though factor_ic_{alpha101,gtja191}.json exist; fusion page 0 results; default panel old gold; no nonlinear evidence anywhere in UI.
F6 P2 default launcher doesn't set QUANTAGENT_PAPER_SECTOR_MAP ⇒ HTTP paper account refuses every buy (industry_unmeasured) — fail-safe but unusable.
F7 P2 PLAUSIBLE index_hedge.py flat 50 bps/yr basis+roll; multi-% IC/IM discounts 2018–2024 unmodelled (no futures data).
F8 P3 duplicate run names; lagging 60d "bear" label (+5.3%/yr bench in bear); metrics.json end_date = --end not last NAV date.
UI: (i) paper risk state YES; (ii) citable backtests YES — none of 29; default selection a burned-holdout run; (iii) window+costs YES (costs mostly 未记录); (iv) factor/nonlinear evidence NO.
Gaps vs references: High — DD ladder / vol target / stop applied identically in backtest and paper (akquant risk layer in both); vol_target_scalar only in stage-8 script with fillna(1.0); pre-register (Kaminski-Lo). Medium — per-strategy budgets (owner_strategy_id); pluggable analyzer; futures instrument/roll model. Low — options hedging, multi-timeframe feeds. Industry 0.8 warn / 0.7 stop tiers vs single 20% latch.
Reading 10/10 pages + 8 searches; Kaminski-Lo PDF 405.
