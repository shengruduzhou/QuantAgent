# R10_postchange findings (audit of c042afd vs 22b3f6c)

### R10-F01 [P1] Vendor zero-volume flat bars are TRADED/not-suspended; CA credits land on stale prices
- Location: src/quantagent/data/ashare/execution_panel.py (build_execution_panel: gap_classification/suspension from masks only); gold_bridge build_masks/build_labels; ashare_execution_simulator_impl._apply_corporate_actions
- Claim: 25,313 TRADED rows (195 symbols, <=2025-08-29) have volume=amount=0 and flat OHLC; 25,108 carry suspension_status FALSE (measured "not suspended"). 54 factor steps land on such rows (19 bonus/transfer): raw close is not re-based, so close x new factor jumps (000836.SZ 2016-02-25 10转20: hfq x3.0; 000038.SZ 2017-07-06: +60%) and the simulator credits bonus shares at the stale pre-ex close -> NAV of the position inflated by share_ratio until trading resumes. Gold dataset keeps 20,851 rows whose t+1 entry bar has zero volume (entry infeasible) and 20,757 rows with zero volume at t; low-vol factors rank these first (vol_20d=0): 239/21,000 top-50 picks on 72/420 dates of the ENG_DATA canonical signal.
- Reproduction: repro/zero_volume_bars.py, repro/ca_step_sanity.py (data_evidence/*.json)
- Impact: label domain + benchmark hfq series contaminated; NAV marks inflated during stale-CA windows; "certified" suspension mask claims FALSE where unmeasured.
- Fix: classify volume==0 & flat OHLC vendor bars as SUSPENDED (or UNKNOWN->fail-closed) before masks/labels; re-base carried close across CA on those rows exactly as gap rows; add a certificate check "zero-volume bars marked not suspended == 0".
- Confidence: CONFIRMED (real data)

### R10-F02 [P1] _hold_untradeable_positions runs after the long-only projection: sector cap breached while reported enforced
- Location: src/quantagent/portfolio/v7_target_weights.py:699 (build_v7_target_weights) / _hold_untradeable_positions
- Claim: frozen held name is re-inserted after _project_and_validate_long_only_target; other names in its sector are only scaled for gross, not for the sector cap. Repro: sector C = 0.36 > cap 0.30 (held limit-down C name 0.20); diagnostics still sector_cap_enforced=True.
- Reproduction: repro/test_r10_hold_untradeable.py::test_sector_cap_survives_holding_a_limit_down_name -> "sector gross after hold: {'A': 0.24, 'B': 0.24, 'C': 0.36, 'D': 0.16} held: 0.2" FAIL
- Also: a predicted-but-rowless held name is frozen on every date forever (0.1 on 4/4 dates) (P2).
- Fix: pass frozen names as reserved holdings INTO _project_and_validate_long_only_target (it already reserves current holdings), then validate; release no_market_row holds after N sessions or when the name has no further prediction.
- Confidence: CONFIRMED

### R10-F03 [P1] clean_label_mask uses the global calendar; gold labels exit on the symbol's own (1+h)-th next traded bar
- Location: src/quantagent/backtest/quarantine.py:141 (clean_label_mask); callers research/model_comparison.py:585, scripts/prune_and_cost_price_factors.py:82, scripts/evaluate_full_universe_price_factors.py:79
- Claim: rows 2025-06-01..08-27 that pass the mask but whose own label exit is >= 2025-09-01 (burned holdout): h1 13 rows/9 symbols, h5 41/15, h20 164/33; max own exit 2025-09-16. Calibration: labels.entry_close_t1 equals next TRADED bar hfq close on 100% of 324,185 rows (so the per-symbol rule is the real one).
- Reproduction: repro/quarantine_underdrop2.py (run1==run2 byte-identical; data_evidence/quarantine_underdrop2_run*.json)
- Fix: store exit_date_{h}d in labels at build time and mask on it; or compute exits per symbol from the traded-bar calendar (max(global, own)).
- Confidence: CONFIRMED

### R10-F04 [P2] Round-trip trade stats on raw panels ignore corporate actions
- Location: src/quantagent/backtest/strict_v8.py (_realized_round_trip_pnl)
- Claim: FIFO matching on fills only; a 10转10 round trip with NAV -0.06% reports gross_loss 250,400 CNY, win_rate 0. Dividends never enter trade P&L; bonus shares are an unmatched sell.
- Reproduction: repro/test_r10_roundtrip_stats_ca.py FAIL
- Fix: feed corporate_action_audit into lots (scale qty/price on ex-date, add cash to the lot) or drop trade stats on CA panels.
- Confidence: CONFIRMED

### R10-F05 [P2] Dividend credits gross of dividend tax; not disclosed in metrics/trust stamp
- Location: ashare_execution_simulator_impl._apply_corporate_actions (docstring only)
- Claim: A-share 差别化 dividend tax (<=1m 20%, 1m-1y 10%) withheld at sale; weekly book (69% turnover/rebalance) holds <1m. Canonical run credited 57,316 CNY -> ~11.5k CNY overstated (~0.2%/yr). Cash-in-lieu for fractional bonus shares is also not A-share practice (P3).
- Fix: holding-period tax on sale or at minimum stamp "dividends_gross_of_tax" in metrics.json.
- Confidence: PLAUSIBLE (magnitude estimated from canonical run audit totals)

### R10-F06 [P2] Certified-panel path of baseline_protocol publishes no trust class although SH/BJ ST is UNKNOWN on 4.88M rows
- Location: scripts/baseline_protocol.py (_load_verified_panel meta has no trust_class; out["trust_class"] only from legacy/quarantine)
- Claim: ENG_DATA report says trust "research_certified_r29_panel_st_incomplete" but that string exists nowhere in code; result.json of canonical run has no trust_class. Variant C cannot exclude SH/BJ ST names (is_st FALSE for UNKNOWN). Limit masks DO handle ST-unknown fail-closed (two-world check, verified).
- Fix: stamp trust_class from manifest st_coverage_exchanges != all.
- Confidence: CONFIRMED (grep + result.json)

### R10-F07 [P2] LLM factor DSL has no column allowlist: label-side columns in the r29 dataset are leads the parser accepts
- Location: factor_synthesis.parse_expression / llm_factor_proposer._parse_factors; expression_safety.expression_leakage_reasons has zero callers
- Claim: r29 dataset.parquet carries entry_close_t1, forward_return_*, entry_feasible. Column(name='entry_close_t1') parses, leak scanner returns (), RD loop accepted it. (Column('forward_return_5d') itself was not accepted.)
- Reproduction: repro/test_r10_label_column_lead.py FAIL ("accepted: True")
- Fix: allowlist leaves to _MARKET_COLUMNS + declared feature columns; call expression_leakage_reasons in _parse_factors.
- Confidence: CONFIRMED (synthetic panel); real exposure requires --market-panel pointing at dataset.parquet

### R10-F08 [P2] Latched kill switch has no operator clear path
- Location: paper/broker.py clear_kill_switch (zero callers in services/ and cli/)
- Claim: switches now persist (correct), but no API/CLI clears them; a 5% daily-loss or 20% drawdown latch makes the HTTP/continuous paper account reduce-only forever without code-level intervention.
- Fix: authenticated CLI `paper clear-kill-switch --scope --key --confirm` journaling KILL_SWITCH_CLEARED.
- Confidence: CONFIRMED (grep)

### R10-F09 [P3] CI workflow still lists the deleted CandlestickChart.test.tsx
- Location: .github/workflows/ci.yml:168
- Claim: vitest filter silently matches nothing (step passes 6 files); K-line coverage moved to vnext/market/MarketCandlestickChart.test.tsx not in that step.
- Confidence: CONFIRMED (ran the step: 6 passed, exit 0)

### R10-F10 [P3] Out-of-box HTTP paper venue refuses every BUY (no QUANTAGENT_PAPER_SECTOR_MAP default); run_quant_ui.sh does not set it. UI shows REFUSING UNMEASURED (honest). Document or default to the repo sector map.

### R10-F11 [P1] Canonical evaluator drops amount-unmeasured symbols wholesale: 57 of 65 are delisted (median lifetime hfq -92.7%)
- Location: scripts/baseline_protocol.py evaluate() certified-panel branch (`unmeasured = set(panel.loc[~amount_measured,'symbol'])`)
- Claim: excluded set = 65 symbols (52 sina_truncated, 13 tencent); 57/65 (87.7%) have a DELISTED row vs 4.5% universe-wide; 85.7% have negative lifetime hfq return, median -92.7%. The r29 gold dataset keeps 56,029 rows of 63 of them (54,090 eligible_for_training=True, amount NaN on all), so any model trained on it predicts them and the evaluator silently removes them -> survivorship-shaped optimistic bias; disclosed only as a count ("excluded_symbols_amount_unmeasured": 60) with no trust downgrade. Canonical ENG_DATA run unaffected (excluded_prediction_rows 0).
- Reproduction: data_evidence/amount_unmeasured_symbols.txt
- Fix: report excluded prediction rows' hfq outcome (with write-off) as a bias bound and downgrade trust_class when >0; or let the simulator trade on measured volume with amount marked estimated (explicitly stamped).
- Confidence: CONFIRMED

### R10-F12 [P3] Benchmark drops delisted names at their last price while the strategy writes them off at 0 (conservative for strategy; asymmetric).
### R10-F13 [P3] Sells-first assumes sell proceeds fund buys in the same close auction; a real broker's available-cash check at order entry (14:57-15:00) does not include unfilled sell proceeds. Model caveat for signal_t_close_next_session_close_v1.
