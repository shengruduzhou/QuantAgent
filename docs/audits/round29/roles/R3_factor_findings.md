# R3_factor findings — Round 29 (independent phase)

Worktree: /home/shanhefu/QuantAgent/.claude/worktrees/agent-ae14755d64f57520b @ 22b3f6c
Started 2026-09-30.

## PRE-REGISTRATION (written BEFORE any experiment result was seen)

Experiment E1 — linear vs nonlinear factor fusion, certified full-universe panel.

Data: runtime/data/gold/full_universe/{dataset,factors_alpha101,factors_gtja191,labels}.parquet, folds.json.
- Features: 15 base features (dataset.parquet manifest feature_columns) + 83 non-placeholder Alpha101
  (18 all-NaN placeholders dropped) + 64 GTJA191, minus exact duplicates found at load time
  (keep first of any bit-identical pair). Same feature set for every arm.
- Label: forward_return_5d (delay-1: entry close(t+1), exit close(t+6)).
- Domain: entry_feasible == True and label finite (the repository's own "tradable domain").
- Date subsample (stated): every 5th trading date (non-overlapping 5-day holding periods),
  all symbols on those dates. Offset 0 from the first panel date.
- Hard window: only rows whose label_end_5d <= 2025-08-29 are used anywhere (train or test).
- Folds: folds.json folds 0..5 (train_start..train_end expanding, 20-day embargo); fold 5 test is
  truncated at label_end_5d <= 2025-08-29. Additional purge: drop train rows whose label_end_5d >= test_start.

Arms (all fitted on the training fold only; per-date cross-sectional transforms are stateless):
- (a) IC-weighted linear composite: features cross-sectionally ranked to [-0.5,0.5] per date,
      weight_k = mean daily rank IC of feature k on training dates; missing rank -> 0 (cross-sectional median).
- (b) Ridge on the same ranked inputs, target = per-date rank of forward_return_5d in [-0.5,0.5];
      alpha fixed = 1e-3 * n_train_rows (mild shrinkage), fit_intercept=True.
- (c) LightGBM on raw (untransformed) feature values, NaN native, same ranked target.
- (d) LightGBM on the ranked inputs of (a)/(b) (NaN kept as NaN), same ranked target.
  LightGBM fixed params: objective=regression(L2), learning_rate=0.05, num_leaves=63,
  min_data_in_leaf=2000, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10,
  max 1000 rounds, early stopping 50 on an inner validation = last 15% of training dates
  (with a 2-sampled-date gap), then that best_iteration is used. seed=7. No tuning on test.

Metrics per fold and pooled over all OOS test dates:
 daily rank IC (Spearman pred vs forward_return_5d) mean, std, ICIR=mean/std,
 t_HAC = Newey-West t of the mean IC (Bartlett, lag = floor(4*(T/100)^(2/9))),
 top-minus-bottom decile mean 5-day return spread (equal weight), and its t_HAC.

Pre-registered decision rule (fixed now):
 R1. "Nonlinear beats linear" iff best-of(c,d) vs best-of(a,b) [best chosen by POOLED mean IC]
     has (i) pooled paired daily-IC difference with t_HAC >= 2.0, AND (ii) positive fold-mean IC
     difference in >= 4 of 6 folds, AND (iii) higher pooled top-minus-bottom decile spread.
     If any fails -> "no evidence that nonlinear fusion beats linear on these features".
 R2. A composite "has OOS signal" iff pooled t_HAC >= 3.0 (Harvey-Liu-Zhu hurdle) AND
     mean IC > 0 in >= 5 of 6 folds.
 R3. "Raw-input LightGBM is harmed by non-stationary raw scales" iff (d) beats (c) on pooled
     mean IC with paired t_HAC >= 2.0.
 Results are reported regardless of which way they go; no arm/param is changed after seeing test output.
 (A single run is not a production result: no cost model, no capacity, no multiple-testing over arms
  beyond the 4 declared here.)

Addendum to pre-registration (still before any result): 
- (e) DESCRIPTIVE ONLY, not in the decision rule: LightGBM on ranked inputs with num_leaves=2 / max_depth=1
  (additive, no interactions), max 2000 rounds, otherwise same params. (d) vs (e) separates
  "interaction" gain from "nonlinear marginal shape" gain.
- LightGBM early stopping: the early-stopped model (trained on first 85% of train dates) predicts at
  best_iteration; no refit on the full training window.
- Spearman IC for the (a) weights is computed as Pearson on per-date ranks with pairwise-NaN exclusion
  (approximation for features with missing values; exact for predictions, which have no NaN).
- Descriptive (not in the rule): top-decile churn between consecutive sampled dates, and best-single-feature
  (sign-adjusted by training IC) baseline.


---

### R3_factor-F01 [P1] Certified panel's "tradable domain" never excludes sealed limit-up at the t+1 entry: the limit-up branch is dead code (no producer of `mask_limit_up`)
- Location: src/quantagent/data/ashare/gold_bridge.py:477-480 (`build_labels`: `if "mask_limit_up" in result.columns:`); scripts/build_u0_full_universe_gold.py (never builds that column); consumers: scripts/evaluate_full_universe_price_factors.py, scripts/prune_and_cost_price_factors.py, any trainer filtering on `entry_feasible`.
- Claim: `build_labels` docstring promises "Entry is infeasible when ... sealed at limit-up at t+1 -- you cannot buy a locked limit-up, and pretending otherwise is where the old phantom alpha came from." The check is guarded by `if "mask_limit_up" in result.columns`, and `grep -rn mask_limit_up src scripts` finds NO producer — only tests/data/test_gold_bridge.py:210 injects the column by hand. So on the certified 10.9M-row panel the limit-up exclusion never ran (manifest `rows_dropped` has no `limit_up_at_t1` key). Also: because the builder DROPS infeasible rows, `entry_feasible` is constant True on dataset.parquet (10,917,401/10,917,401), so every "restricted to entry_feasible" screen is a no-op filter.
- Reproduction: `python repro/f01_limitup_entry.py data_evidence/f01_limitup_entry_run1.json` (read-only). Verbatim:
  `"entry_feasible_rows": 10917401` (= rows_total), `"entry_feasible_but_t1_sealed_limit_up": 144840`, `"share_of_entry_feasible": 0.01327`, `"rows_le_2025_08_29_flagged": 129954`, `"mean_fwd5_flagged": 0.0401` vs `"mean_fwd5_other": 0.0015` (medians -0.0018 vs -0.0018), `"manifest_rows_dropped_has_limit_up_key": false`.
  Sealed proxy = adjusted close-to-close return >= board band - 0.5pp AND close == high on the t+1 bar (10%/20% STAR/20% ChiNext from 2020-08-24/30% BSE; ST 5% not modelled => undercount).
- Economic impact: 1.33% of the "tradable" rows are unbuyable and carry a mean 5-day forward return 27x the rest (fat right tail: limit-up streak continuation). Any model/IC/decile spread measured on this domain can harvest phantom return from names that cannot be bought — exactly the failure the docstring names. Affects factor_ic_*.json, factor_pruning_report.json and any fusion/model comparison on the certified panel (quantified for E1 below).
- Proposed fix: build `mask_limit_up` (tri-state, from raw prices + board/ST band + exchange rounding) in `gold_bridge.build_masks`, and make `build_labels` FAIL (not skip) when the column is absent: `if "mask_limit_up" not in result: raise`/record UNKNOWN in the certificate. Test: a builder-level test running `build_masks`->`build_labels` on a panel with a sealed limit-up bar (no hand-injected column) must drop the row; fails today.
- Confidence: CONFIRMED (dead branch by grep; row count measured on the real panel).
- References: https://akquant.akfamily.xyz/guide/cross_section_checklist/ (tradability filters before cross-sectional ranking) — see reading_log.


### R3_factor-F02 [P1] Round-23/24 factor IC screen and pruning/post-cost report read labels from BOTH quarantined windows, including the FROZEN fresh holdout
- Location: scripts/evaluate_full_universe_price_factors.py:55-69 (`--start` only, no end, no quarantine import); scripts/prune_and_cost_price_factors.py:70-78 (no date filter at all). Outputs: runtime/data/gold/full_universe/factor_ic_{alpha101,gtja191}.json, factor_pruning_report.json.
- Claim: configs/quarantined_windows.json forbids reading returns in 2025-09-01..2026-05-18 (burned) and 2026-05-19.. (frozen fresh holdout) for evaluation/tuning. Neither script imports `quantagent.backtest.quarantine`; both consume every labelled date of dataset.parquet. The shipped reports show `"periods": 2558` / `"n_dates": 2553..2558`.
- Reproduction (reads only date counts, no returns):
  dates with >=50 labelled rows = `2558 first 2016-01-04 last 2026-07-16`; `in burned 2025-09-01..2026-05-18: 168`; `in frozen >=2026-05-19: 42`; `<=2025-08-29: 2348`. 2558 == the report's `periods` for gtja150/gtja158 => all 210 quarantined dates were used. runtime/state/holdout_access_log.jsonl has 3 lines, 0 from these scripts (no forensic override was logged).
- Impact: the factor ranking ("GTJA stronger", top_by_net_spread, the 90-factor `independent_set` ordering) was selected with 42 sessions of frozen-fresh-holdout labels. Any model later built from that selection and scored on the fresh window is contaminated by construction; the chair's FRESH first read (~2026-11) must treat these factor choices as holdout-informed.
- Proposed fix: both scripts call `quarantine.clamp_panel_window`/`check_window` before loading labels (fail closed, forensic override logged), and the two JSONs are regenerated on <=2025-08-29. Test: run each script's main on a 3-date synthetic panel with one date inside a quarantined window -> must raise QuarantineViolation (fails today).
- Confidence: CONFIRMED.

### R3_factor-F03 [P1] The nonlinear-fusion comparison (`qa audit-nonlinear-factors` / `run_model_comparison`) has no quarantine guard and anchors every fold at the panel end: on the certified panel all 6 folds fall inside quarantined windows and both "holdout" folds inside the FROZEN fresh holdout; it has never been run on real data
- Location: src/quantagent/cli/nonlinear.py:20-107; src/quantagent/research/model_comparison.py:581-593 (`anchor="end"`, no `anchor_end`, no quarantine); scripts/audit_nonlinearity_and_style_alpha.py:144 (`--start 2021-01-01`, no end; DEFAULT_DATASET `training_dataset_alpha181_exec_v89_plus7clean_fund.parquet` does not exist under runtime/data/v7/gold/training_dataset/).
- Claim: with CLI defaults (6 folds x 40 days, 2 holdout folds) the whole OOS evidence is the LAST 240 sessions of whatever panel is passed. For the certified panel (ends 2026-07-23): selection folds 2025-07-28..2026-03-26 (burned window), holdout folds 2026-03-27..2026-07-23 (frozen fresh holdout). No check refuses this. Also 240 of ~2,560 sessions = the verdict on the user's #1 capability would rest on ~9% of history in one regime. No runtime/reports/nonlinearity* output exists => the governed nonlinear path has never produced a real-data verdict; certified factor files are separate parquet files and no command joins them into the single panel this CLI requires.
- Reproduction (plumbing only, no labels read): `python repro/f03_nonlinear_cli_fold_plan.py` -> data_evidence/f03_fold_plan.json, verbatim: fold 0-3 `"role": "selection"`, valid 2025-07-28..2026-03-26, `"quarantine_hits": ["2025-09-01..2026-05-18"]`; fold 4 holdout `"valid": "2026-03-27..2026-05-27"` hits both windows; fold 5 holdout `"valid": "2026-05-28..2026-07-23"`, `"quarantine_hits": ["2026-05-19..2027-12-31"]`.
- Impact: first real run of the flagship nonlinear audit would burn the fresh holdout and produce a regime-local verdict.
- Proposed fix: `run_model_comparison` takes `anchor_end` = last date with label end <= last non-quarantined date (via quarantine.clamp_panel_window) and raises QuarantineViolation otherwise; CLI exposes `--end` defaulting to the clamp. Test: fold windows from a panel spanning 2025-2026 must all end <= 2025-08-29 (fails today).
- Confidence: CONFIRMED (fold plan computed with the repository's own splitter/config).


### R3_factor-F04 [P1] Round-24 post-cost pruning report is sign-blind and horizon-inconsistent: cost makes negatively-signed factors rank HIGHER, 42 cost-surviving factors are counted as failures, and a 1-day churn cost is netted against a 5-day spread
- Location: scripts/prune_and_cost_price_factors.py:140-153 (`"net_spread": gross - cost`, `spread["abs_net"] = spread["net_spread"].abs()`, `"net_positive_count": int((spread["net_spread"] > 0).sum())`), and the churn loop :115-133 (consecutive-SESSION churn vs `forward_return_5d`).
- Claim: (1) For a factor with negative gross spread (used short-top/long-bottom), net should be |gross| - cost. The code computes gross - cost, so abs_net = |gross| + cost: the higher the turnover cost, the higher the factor ranks in `top_by_net_spread` AND in the greedy `independent_set` order (which iterates in abs_net order). Verbatim from the shipped report: gtja158 `gross_spread -0.007589, cost_per_period 0.00147, net_spread -0.009059, abs_net 0.009059` (> |gross|). (2) `net_positive_count` = 65 counts only positively-signed factors. (3) Churn is measured between consecutive sessions (median 0.3417) but the spread is a 5-day forward return; the honest per-rebalance cost for a 5-day spread is the 5-day churn.
- Reproduction: `python repro/f04_pruning_recompute.py cache/e1_off0.parquet data_evidence/f04_pruning_recompute_run1.json` (every 5th date, <=2025-08-22 so labels end <=2025-08-29; 5-day churn). Verbatim: `"factors": 146`, `"report_convention_net_positive_count(gross-cost>0)": 65`, `"sign_aware_net_positive_count(|gross|-cost>0)": 107`, `"negative_gross_factors": 49`, `"negative_gross_but_sign_aware_net_positive": 42`, `"median_churn_5d": 0.7234` (vs report daily 0.3417). Example: alpha020 gross -0.00658, churn_5d 0.817 -> report-style net -0.00854 (ranks as if strong), sign-aware net +0.00462.
- Impact: the report's headline ("65 of 146 survive costs") and its ordering are wrong in the direction the commit warns against ("would have retired 30 factors that pay for their own trading"): 42 more factors pay for their own trading; cost is understated ~2x (daily vs 5-day churn) for every factor; greedy independence pruning keeps high-churn negative factors over low-churn ones.
- Proposed fix: `sign = np.sign(gross); net = sign*gross - cost` (or orient each factor by training-period IC sign), measure churn at the label horizon (every h-th date) and report per-rebalance cost; sort by sign-aware net. Test: two synthetic factors with gross -0.01 and churn 0.1 vs 0.9 -> the low-churn one must rank first and both count as net-positive (fails today).
- Confidence: CONFIRMED (arithmetic from the shipped JSON + recomputation on the real panel, pre-quarantine).


### R3_factor-F05 [P1] LLM factor loop: LLM text is executed with Python `eval` (empty builtins is not a sandbox) and the "zero look-ahead" DSL accepts negative lags, which no guard catches
- Location: src/quantagent/factors/factor_synthesis.py:1986-1992 (`parse_expression` = `eval(expr_repr, {"__builtins__": {}}, _PARSE_NAMESPACE)`), called on raw LLM output at src/quantagent/factors/llm_factor_proposer.py:258-261 (`_parse_factors`); src/quantagent/factors/expr.py:140-154 (`Delay.periods` unchecked; module docstring claims "zero look-ahead"); src/quantagent/factors/expression_safety.py:5-7 (regex only knows `Ref|Shift(x,-k)`, `Lead|Future|LookAhead(`).
- Claim: (a) `parse_expression("Delay(expr=Column(name='close'), periods=-1)")` builds a LEAD; evaluated, row t equals close(t+1). `expression_leakage_reasons(repr(...))` returns `()`. synthesize_factors_rd_agent has no truncation/PIT recomputation test, so the only defence is the DSL, which is wrong. (b) the parser is `eval`: `parse_expression("().__class__.__base__.__subclasses__()")` returns a list of 1175 live classes, i.e. arbitrary object-graph traversal (the standard route to os.system) on LLM-controlled text. The module docstring promises the LLM "never writes free-form Python".
- Reproduction: `PYTHONPATH=<worktree>/src pytest -q repro/test_f05_llm_expression_guard.py` -> `3 failed`. Verbatim: `AssertionError: parse_expression accepted a lead: Delay(close,-1)[t] == close[t+1]` (`array([11., 12., 13., 14., 15.])`); `expression_leakage_reasons("Delay(expr=Column(name='close'), periods=-1)") returned no reason`; `parse_expression evaluated arbitrary Python and returned list with 1175 reachable classes`.
- Evidence it has not fired yet: runtime/state/factor_loop_memory_v89.jsonl (96 records) contains 0 `periods=-`/`window=-` expressions. So no accepted factor is known to be contaminated; the control is simply absent.
- Impact: an LLM (or prompt-injected text in its memory/RAG directive) can emit a look-ahead factor whose IC would be enormous and pass the IC gate; and can execute code in the research process that has credential env vars loaded.
- Proposed fix: replace eval with an `ast.parse(mode="eval")` walker that only allows Call(Name in DSL whitelist), keyword args, Constant numbers/strings; validate in DSL constructors `periods >= 1`, `window >= 1`; add a truncation test in the RD loop (factor value at t unchanged when rows > t are deleted). The three tests above pass after the fix.
- Confidence: CONFIRMED.
- References: https://github.com/microsoft/RD-Agent (RD-Agent runs generated code in a Docker sandbox; this port replaced the sandbox with a DSL but parses it with eval); https://akquant.akfamily.xyz/advanced/llm/.


## E1 addendum 2 (declared AFTER run-1 results were seen, BEFORE the sensitivity run) 
Run-1 results exist (below). Declared now, before computing it: a SENSITIVITY re-evaluation of the SAME
run-2 predictions (no refit, no param change) on the domain minus rows whose t+1 entry bar is sealed
limit-up (F01 proxy). Same metrics, same R1/R2 rules. It is a robustness check of a known domain defect,
not a new arm; if R1 flips under it, that is reported as the headline caveat. Run-2 must reproduce run-1
daily metrics bit-for-bit (determinism check).


### R3_factor-F06 [P2] E1 REAL-DATA RESULT: on the certified panel, the production-style additive IC-weighted fusion is the WORST arm (below its own best single factor); ridge beats it by +0.020 IC and LightGBM beats ridge by +0.0126 IC (t_HAC 3.51, 5/6 folds) — pre-registered rule R1 = TRUE; the gain is from interactions, not marginal nonlinearity
- Location (what is being judged): src/quantagent/fusion/schemes.py:186-205 (`ic_weighted`/`ic_ir_weighted`/`inverse_volatility` = per-factor weights from marginal IC, collinearity-blind); src/quantagent/research/model_comparison.py (has ridge + GBM arms but has never been run on real data, F03).
- Setup: exactly as pre-registered above (158 features after dropping 18 all-NaN placeholders and 4 bit-identical duplicates — NEW: `gtja158 == intraday_range` is a 4th identity, (H-L)/C); every 5th trading date, 466 dates, 1,927,611 rows; label delay-1 forward_return_5d with true exit date <= 2025-08-29; folds.json train/embargo windows, fold 5 test truncated at 2025-08-29 (22 dates). Script repro/e1_run.py; outputs data_evidence/e1_run1_{summary.csv,daily.csv,decision.json,log.txt}. Wall 666 s.
- Pooled OOS (389 test dates, gross, no costs) — verbatim from e1_run1_summary.csv:
  | arm | mean rankIC | ICIR | t_HAC | top-bottom decile 5d spread | top decile 5d ret | top-decile churn/5d |
  |---|---|---|---|---|---|---|
  | a IC-weighted linear | 0.0765 | 0.615 | 12.74 | 87.4 bp | 32.8 bp | 0.684 |
  | s best single (train-IC pick) | 0.0822 | 0.640 | 11.47 | 97.6 bp | 53.9 bp | 0.461 |
  | b ridge (ranked) | 0.0962 | 0.875 | 15.62 | 137.9 bp | 71.7 bp | 0.666 |
  | c LightGBM raw | 0.1088 | 0.975 | 18.71 | 171.7 bp | 85.7 bp | 0.669 |
  | d LightGBM ranked | 0.1052 | 0.992 | 19.15 | 164.7 bp | 81.5 bp | 0.709 |
  | e LightGBM stumps (additive, descriptive) | 0.0953 | 0.857 | 15.44 | 124.8 bp | 71.3 bp | 0.674 |
  Per-fold mean IC (0..5): a .077/.073/.060/.080/.080/.117; b .101/.087/.088/.106/.083/.151; c .119/.103/.102/.099/.108/.156; d .111/.102/.093/.099/.108/.149.
- Pre-registered verdicts (e1_run1_decision.json): R1 best_linear=b_ridge, best_nonlinear=c_lgb_raw, pooled IC diff +0.01255, diff t_HAC 3.51, folds_diff_pos 5/6, spread 171.7 vs 137.9 bp -> `R1_nonlinear_beats_linear: true`. R2: all six arms t_HAC >= 3 and IC > 0 in 6/6 folds -> all "have OOS signal". R3 (ranked beats raw): d - c = -0.0036, t -1.61 -> false (LightGBM is not hurt by raw hfq scales on this feature set).
- Descriptive (post-hoc, not rule-bearing): b - a = +0.0198 IC, t_HAC 3.94, 6/6 folds; s - a = +0.0057, t 1.13 (158-factor IC-weighted blend does not beat one factor); e - b = -0.0010, t -0.28 (additive nonlinear = linear); d - e = +0.0099, t 2.62 (the tree gain is INTERACTIONS, consistent with Gu-Kelly-Xiu 2020). No decay across folds (folds 0-2 vs 3-5 IC: c .108 vs .110), i.e. no McLean-Pontiff decay visible to 2025-08 in this universe.
- Economic / decision impact: (1) the fusion schemes the repo actually exposes (all additive IC-derived weight vectors, DEF-037) are the weakest way to combine these factors — IC-weighting 158 collinear factors double-counts correlated clusters; a single ridge already adds ~26% IC. (2) There IS measurable, fold-stable interaction value on the certified price-volume panel, so the nonlinear path is worth finishing (F03 blocks running it honestly). (3) NOT a production result: gross of costs/impact/capacity, full universe incl. micro-caps, domain contaminated by F01 limit-up rows (sensitivity below), ST not excluded (st_pit_complete=false), single sample offset.
- Proposed follow-up: add ridge/orthogonalised-IC scheme to fusion/schemes.py as a fitted control; run model_comparison on a quarantine-clamped certified panel (after F03) with this E1 as the pre-registered reference.
- Confidence: CONFIRMED (run 1; determinism run 2 in progress — see E1 determinism note below).
- References: Gu, Kelly, Xiu 2020 RFS https://dachxiu.chicagobooth.edu/download/ML.pdf ; Harvey-Liu-Zhu 2016 t>3 https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314 ; qlib per-date CSRankNorm https://qlib.org.cn/en/latest/

### R3_factor-F07 [P2] Fuyao financial statements are stamped available_at = disclosure DATE, so an after-close report is joined onto the same session's row (violates the user's rule 1 and the repo's own trading_calendar rule)
- Location: src/quantagent/data/providers/fuyao_provider.py:573-574 (`frame["available_at"] = frame["ann_date"]`, time-of-day truncated by `_ms_series_to_shanghai_date` :494-495); contrast src/quantagent/data/trading_calendar.py:3-10 ("only actionable from the next trading session") and akshare_financial_provider.py:137,358 (`available_lag_days = 1`).
- Claim: a statement disclosed 2024-04-25 20:00 Shanghai gets available_at 2024-04-25; `merge_pit_features` (backward as-of on available_at <= trade_date) attaches it to the 2024-04-25 row, whose v7 label opens at close(2024-04-25) and whose decision is taken at that close.
- Reproduction: `PYTHONPATH=<worktree>/src pytest -q repro/test_f07_fuyao_after_close_available_at.py` -> `AssertionError: after-close disclosure (available_at=2024-04-25 00:00:00) joined onto the same session's row: net_profit=[123.0]` / `1 failed`.
- Impact: latent today (runtime has no Fuyao financial data; the v7 training manifest reports fundamentals_rows=0), but Fuyao is the pinned fundamentals source in CLAUDE.md; the first fundamentals build through it would leak every evening disclosure one session early (earnings-announcement returns are exactly the largest ones).
- Proposed fix: resolve Fuyao `available_at` via TradingCalendar with lag >= 1 session (or keep the timestamp and use next session if time >= 15:00), same contract as akshare; add the test above.
- Confidence: CONFIRMED.


### R3_factor-F08 [P2] Factor lifecycle state machine is a dormant control: zero production call sites, no ledger file, ACTIVE is unreachable by design, and nothing the model trains on is gated by it
- Location: src/quantagent/factors/lifecycle_state.py (FactorLifecycleLedger, decide_lifecycle_transition; default path runtime/state/factor_lifecycle.jsonl); src/quantagent/factors/lifecycle.py (build_factor_lifecycle_report).
- Claim: `grep -rn "FactorLifecycleLedger|decide_lifecycle_transition|replay_lifecycle" src scripts` finds only the package `__init__` re-exports and a docstring in factors/governance.py; `build_factor_lifecycle_report` is called only by scripts/run_factor_research_cycle.py, which pulls BaoStock (unreachable from this host: outbound 80/443 only). `ls runtime/state/factor_lifecycle.jsonl` -> No such file. The module itself states shadow->active "is intentionally not issued here". No training/dataset builder references lifecycle state (grep in training/, data/dataset_builder, cli/v7_train.py: none). So candidate->shadow->active->degraded->retired is not fed by monitoring data and does not decide any feature set; the 146 certified factors have no lifecycle state at all.
- Impact: the user's "factor lifecycle management" requirement is unmet in practice; a factor that decays (McLean-Pontiff ~58% post-publication) would stay in every model until someone notices by hand.
- Proposed fix: a scheduled job that runs evaluate_factor_candidate on a quarantine-clamped rolling window of the certified factor files and appends observe() transitions; the training feature list is then read from ledger state (active|shadow) instead of "all columns". Test: training feature resolver returns only ledger-active factors (fails today: no resolver).
- Confidence: CONFIRMED (grep + missing file).

### R3_factor-F09 [P2] The factor validity gate exists but has never been applied to the certified factors; no Fama-MacBeth anywhere; multiple testing is a caller-supplied boolean; single-factor hurdle is t>=2.0, not HLZ t>=3
- Location: src/quantagent/factors/governance_metrics.py:31-49 (FactorGateConfig: min_newey_west_rank_t_stat=2.0, max_library_abs_correlation=0.85, decay/capacity present), :53-71 (FactorPromotionContext.multiple_testing_passed: bool supplied by caller), only caller scripts/run_factor_research_cycle.py:361 (BaoStock). `grep -rlin "macbeth" src scripts` -> nothing.
- Claim: the round-23 IC screen and round-24 pruning (the only measurements ever made on the 146 certified factors) bypass this gate entirely (they are standalone scripts, F02/F04). Positive-only test `mean_ic < min_mean_rank_ic` rejects negatively-oriented factors unless a direction is registered upstream. With 146 candidates screened, t>=2 per factor admits ~7 false positives at 5% even before overlap; HLZ/BHY imply t~3.
- Impact: there is no enforced, real-data validity verdict for any factor that could reach a model; "which factors are valid" currently rests on F02/F04 reports that are quarantine-contaminated and sign-blind.
- Proposed fix: one CLI that runs evaluate_factor_candidate over the certified factor files (quarantine-clamped), computes BHY-adjusted p-values across the whole tested family, and writes the lifecycle ledger (F08). Add a Fama-MacBeth (with NW lag >= h-1) slope test for multi-factor marginal contribution.
- Confidence: CONFIRMED (grep/call-site evidence).

### R3_factor-F10 [P2] model_comparison economic leg charges cost on day-to-day top-K turnover while modelling an H-day rebalance (horizon-inconsistent cost, same class as F04)
- Location: src/quantagent/research/model_comparison.py:376-419 (`_topk_daily_returns`: `cost_per_day = cost_bps/1e4/horizon`; `target_turnover = len(names - previous)/len(names)` with `previous` = previous SESSION's target).
- Claim: the docstring models one rebalance per H days with cost "spread over the same H days", but turnover is measured between consecutive sessions, not between rebalances H apart. For a ranking that drifts, H-day turnover is ~2x the 1-day turnover (E1: top-decile churn per 5 days 0.67-0.71 vs round-24's per-session decile churn median 0.34), so net returns of high-churn arms are overstated; for a ranking that oscillates the bias reverses.
- Impact: the economic criterion `min_net_return_delta` in the nonlinear verdict is biased (usually toward the higher-churn arm, typically GBM). Not run on real data yet (F03), so no shipped number is corrupted.
- Proposed fix: measure turnover vs the target H sessions earlier (or simulate H staggered sub-books). Test: synthetic monotonically drifting ranking where 1-day turnover=0.2 and 5-day=1.0 must be charged 1.0*cost per 5 days.
- Confidence: PLAUSIBLE (code reasoning; not reproduced with a test).

### R3_factor-F11 [P2] The certified directory's labels.parquet was overwritten after certification with a DIFFERENT (same-close, non-delay-1) label convention; readiness only checks the file exists
- Location: runtime/data/gold/full_universe/labels.parquet (mtime Aug 1, after manifest Jul 29); writer schema = v7_label_builder (open..adjust_factor, label_end_*, forward_excess_*, forward_rank_*, horizons 1/5/20/60/120) whereas scripts/build_u0_full_universe_gold.py:401 writes symbol, trade_date, entry_close_t1, forward_return_{1,5,20}d; src/quantagent/safety/readiness_tiers.py:195 checks `.exists()` only.
- Reproduction (read-only): on the E1 sample (1,927,661 rows) `labels.forward_return_5d` vs `dataset.forward_return_5d`: 1,918,850 rows differ (>1e-9), max |diff| 2.0118. For 600519.SH 2016-01-04: labels = -0.048376 = close(t+5)/close(t)-1, label_end_5d 2016-01-11 (t+5); dataset = -0.075322 = close(t+6)/close(t+1)-1. I.e. labels.parquet[t] == dataset[t-1] (shifted one session, no delay).
- Impact: any consumer (or auditor) that takes "the certified labels" from labels.parquet gets a zero-delay label and the wrong label_end for purging (I nearly did: E1 had to recompute exit dates). manifest label_hash no longer describes the file next to it.
- Proposed fix: restore the builder's labels.parquet (or move the v7 file elsewhere) and make readiness verify label_hash, not existence. Test: hash(labels.parquet) == manifest.label_hash.
- Confidence: CONFIRMED.

### R3_factor-F12 [P3] Sentiment/news family is effectively absent (51 news rows, all available_at 2026-06-04) yet core_policy fabricates core_sentiment_score/core_policy_score = 0.0 for every row and feeds weighted overlays
- Location: src/quantagent/factors/core_policy.py:144-153 (missing CORE column -> 0.0; NaN -> 0.0), :186-190 (`... if sentiment_col else 0.0`, `fillna(0.0)`); consumers src/quantagent/ensemble/regime_conditional_overlay.py:39-40 (weights 0.18 policy / 0.09 sentiment), scripts/overlay_regime_split.py.
- Reproduction: `runtime/data/v7/raw/news/news_canonical.parquet` -> 51 rows, available_at min=max=2026-06-04.
- Impact: overlays labelled policy/sentiment-aware carry a constant for all of 2016-2025; harmless to ranking but misleading as "alternative data in use". The summary flag evidence_sentiment_available exists but is not enforced by consumers.
- Proposed fix: drop absent evidence columns (or NaN + explicit unavailable flag) instead of 0.0; consumers refuse weights on unavailable columns.
- Confidence: CONFIRMED (data count; wiring by grep).

### R3_factor-F13 [P3] Stale PIT policy strings: v7_feature_groups declares "close-derived; available_at = next trading day" while the builder now stamps available_at = trade_date (post DEF-026)
- Location: src/quantagent/data/v7_feature_groups.py:262,272,282,325,336 vs src/quantagent/data/v7_dataset_builder.py:158.
- Impact: documentation/UI that renders these strings misstates the PIT contract. Fix: update strings. Confidence: CONFIRMED (code read).


## E1 determinism + pre-declared limit-up sensitivity (repro/e1_sensitivity.py -> data_evidence/e1_sensitivity_run1.json)
- Determinism: run 2 (same code + prediction dump) vs run 1: `"determinism_rows": [2334, 2334]`, `"determinism_max_abs_diff": 0.0`; e1_run1_decision.json and e1_run2_decision.json identical except wall_seconds. => F06 upgraded to CONFIRMED (run twice).
- Sensitivity (same run-2 predictions, no refit; test rows 1,715,860 of which 21,956 have a sealed limit-up t+1 entry bar):
  | arm | IC full | IC excl | spread full | spread excl | top-decile sealed share (full) |
  |---|---|---|---|---|---|
  | a IC-weighted | 0.0765 | 0.0758 | 87.4 bp | 89.1 bp | 0.47% |
  | b ridge | 0.0962 | 0.0946 | 137.9 | 114.5 | 1.23% |
  | c LGB raw | 0.1088 | 0.1073 | 171.7 | 149.9 | 1.36% |
  | d LGB ranked | 0.1052 | 0.1035 | 164.7 | 142.8 | 1.29% |
  R1 on the corrected domain: c - b IC diff +0.01269, t_HAC 3.50, 5/6 folds, spread c > b -> `"R1_holds": true`.
- Reading: F01 inflates the decile spread of the better models by 13-17% (ridge -23.4 bp, LGB -21.8 bp per 5 days) — the stronger the model, the more unbuyable limit-up names it puts in its top decile — but it does NOT explain the nonlinear-vs-linear IC gap. (The stumps arm's fold-0 NaN spread in run 1 came from tied predictions with `rank(pct=True)`; this script breaks ties with method="first".)


### R3_factor-F14 [P2] RD-Agent-style loop selects on a re-used validation window that is fed back to the LLM, with no embargo, no trial count and a raw-IC threshold: 53 of 96 proposals "selected", some at validation IC 0.0115
- Location: src/quantagent/factors/factor_synthesis.py:921-941 (`_chronological_split`: last 25% of dates, NO purge/embargo; if <4 dates or validation_fraction<=0 it silently returns train == validation); :1395-1445 (acceptance = `validation_ic >= min_validation_rank_ic`, optional ICIR, novelty |rho| vs accepted+reference — no t-stat, no deflation by attempts); src/quantagent/factors/factor_loop_memory.py:185-209 (`memory_digest` sends `validation_rank_ic` of the last 12 rejected and 8 accepted examples into the next prompt; memory persists across runs in runtime/state/factor_loop_memory_v89.jsonl).
- Reproduction (read-only): runtime/state/factor_loop_memory_v89.jsonl -> 96 records: `selected 53, SOTA duplicate gate failed 19, validation IC gate failed 11, duplicate expression 9, value gate failed 2, validation finite gate failed 2`; sources `llm 84, blueprint 12`; selected examples include `llm_price_above_rolling_min_rank 0.0115`, `llm_overnight_gap_reversion 0.0196`.
- Claim: the validation window acts as a training set for the generator (its scores are the optimisation signal returned to the LLM, across rounds and runs), and selections are never deflated by the number of expressions tried (HLZ/Chordia-Goyal-Saretto multiple testing). The fail-open split fallback would make "validation IC" an in-sample IC.
- What is CORRECT (verified): the LLM emits only DSL expressions + hypothesis text (ProposedFactor); it has no order/target-weight path; novelty is correlation-based on OOS values (not string-only); untradable names are dropped from the acceptance IC; generated expressions are size-capped (max_nodes).
- Impact: the 53 "selected" LLM factors are not evidence of edge; any downstream use inherits adaptive-selection bias.
- Proposed fix: purge/embargo >= label horizon between train and validation; raise on the degenerate split; feed back only pass/fail + reason (not the IC value) or hold out a third untouched window for the final verdict; count every evaluated expression into a cumulative trial counter used for DSR/BHY at promotion.
- Confidence: CONFIRMED (code + memory file counts). Severity P2 because no LLM factor is in the certified training panel.

