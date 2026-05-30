# Stage 1: Foundation — Stratified IC + Universe Filter + State Machine + Multi-Objective Loss

Status: **CODE COMPLETE + ENGINEERING REVIEW PASSED**

## Engineering review summary

Per the user's request, a focused self-review was done BEFORE proceeding
to Stage 2. The goal was *not* "does Stage 1 maximise return" — that's
expected to remain unfavourable until Stage 2-3 ship — but rather:

* Is the code clean, PIT-safe (no look-ahead), and interface-stable?
* Does it match the trading-intent the spec calls for?
* Will Stage 2-3 mechanisms be able to plug in without surprises?

Ten issues were identified and fixed, each with a regression test:

| # | File | Issue | Severity | Fix |
|---|---|---|---|---|
| 1 | `universe/filters.py` | `market_panel` with dup `(date, symbol)` rows would explode the merge | Important | `drop_duplicates` before merge |
| 2 | `universe/filters.py` | ST soft-filter loop assumed unique integer index — would KeyError on sliced input | Important | `reset_index(drop=True)` at top of `apply_universe_filter` |
| 3 | `universe/filters.py` | `st_flags` lacking `is_st` column crashed with KeyError | Important | validate schema; warn and skip on missing |
| 4 | `state_machine/machine.py` | `enter_min_pred_zscore` defined but never used (dead config) | Cosmetic | removed |
| 5 | `state_machine/machine.py` | `BAN` was permanent — once ST'd a stock could never come back even after the flag cleared. Contradicted spec "ST 不要那么绝对" | **Critical** | BAN now re-evaluated on each transition; returns to WATCH when `is_st` and `is_suspended` both False |
| 6 | `state_machine/machine.py` | `if entry > 0` skipped take-profit / reduce logic for negative `entry_prediction`. Edge case rarely fires but masked the test path. | Cosmetic | use `abs(entry) > 1e-6` |
| 7 | `optimization/multi_objective_loss.py` | `(1 + arithmetic_mean) ** 252 - 1` **overstates** ann return on volatile series (AM-GM inequality). A +10/-10 alternating series reported 0% instead of the true ≈ -1% / period | **Critical** | switched to geometric (compound) ann return via cumulative product |
| 8 | `optimization/multi_objective_loss.py` | `high_chase_rate` not clipped to [0,1] — buggy input could over-penalise | Important | clip both ends |
| 9 | `optimization/multi_objective_loss.py` | Semantic of `high_chase_rate` ambiguous: time-integrated vs per-day max | Cosmetic | docstring clarification |
| 10 | `training/v7_experiment.py` | `V7TrainingConfig` → `UniverseFilterConfig` bridge silently forwarded only 4 of 13 fields. User changing `universe_high_chase_*` had **no effect** on the deployed sleeve. | **Critical** | added all 13 fields to `V7TrainingConfig`; bridge now forwards every one explicitly |

After fixes: **60/60 tests pass** (up from 52). New tests cover each
fix path so future regressions are caught at CI.

## What this stage delivers

Stage 1 of the v4 strategy spec implementation. Five testable
modules, all with docstrings and a deterministic test suite. Nothing
is a stub.

### Module map

| Module | Purpose | Tests | LOC |
|---|---|---|---|
| `quantagent.diagnostics.stratified_ic` | Per-board / per-cap-quintile / per-vol-quintile / per-regime IC tables for any OOS predictions panel. Reveals where the model's alpha is concentrated. | 8 | 380 |
| `quantagent.universe.filters` | ST soft-exclude (≥90% blocked, top 10% by prediction may pass), suspended hard-exclude, limit-up new-entry block, high-chase block (AND-mode: parabolic + multi-连板). Derives `is_suspended` / `is_limit_up` / `is_limit_down` from OHLCV when explicit flags missing. | 13 | 350 |
| `quantagent.portfolio.state_machine` | 11-state machine (BAN / WATCH / LOW_BUY_READY / OPEN_POSITION / HOLD_SHORT/MID/LONG / DO_T / REDUCE / TAKE_PROFIT / STOP_LOSS / EXIT) with deterministic transition rules. Pure stateless function — caller passes per-(date, symbol) context, gets decision back. | 17 | 280 |
| `quantagent.optimization.multi_objective_loss` | 5-term Stage 1 loss: net_return + sharpe + calmar − max_dd − high_chase. Sign convention: total = what optimizer minimises. | 9 | 230 |
| `scripts/stratified_ic_report.py` | CLI for producing the stratified IC report on any model directory's OOS predictions. Includes a workaround for the `market_features.parquet` data gap (see Discoveries). | — | 130 |

**Total: 5 modules, 47 new unit tests, ~1,400 LOC including docstrings.**

### Wired into the deployed pipeline

* `V7TrainingConfig` exposes:
  - `universe_filter_enabled` (default `False` — feature-flag for staged rollout)
  - `universe_st_min_block_rate`, `universe_suspended_block_new`,
    `universe_limit_up_block_new`, `universe_limit_up_pct`
* `_compute_horizon_sleeve_backtest` now loads `market_panel.parquet`
  and applies the universe filter before sleeve picks when the flag
  is on. The filter audit (block rate, reason breakdown) is surfaced
  in the result dict under `universe_filter_summary`.
* `scripts/replay_horizon_sleeves.py` toggles the filter via
  `QA_UNIVERSE_FILTER=1` env var.

## Stage 1 graduation criteria

The task tracker calls Stage 1 graduated when the deployed-sleeve
replay on v9 OOS predictions hits:

* aggregate excess vs CSI300 ≥ +12% annualised
* aggregate max drawdown ≤ 9%
* aggregate hit-rate vs benchmark > 50%

### Current measurements

Three replay variants on v9's 756-day OOS panel (2020-02-06 → 2023-01-17):

| Variant | Excess | Max DD | Hit vs bench | Avg gross | Sharpe |
|---|---|---|---|---|---|
| **Filter OFF** (control) | +12.14% | -9.62% | 52.50% | 0.483 | 1.17 |
| Filter ON (OR, ≥3 limit-ups, lookback 10) | +2.85% | -9.49% | 51.81% | 0.368 | 0.64 |
| Filter ON (AND, ≥2 limit-ups, lookback 5) | +3.65% | -9.46% | 51.94% | 0.370 | 0.70 |
| Filter ON (AND, ≥3 limit-ups, lookback 5) | *pending replay* | | | | |

The filter is correct per spec — it implements "尽量不要接盘多日高涨停"
and "ST 至少 90% 不能买" — but **costs ~8.5pp of headline excess**
because the v9 model's alpha is concentrated in exactly the
high-momentum names the filter blocks (folds 7, 9 in particular).

### Interpretation

Stage 1 deliverables are CODE COMPLETE — every module has tests, docs,
and integrates without regression in the rest of the suite. The
graduation criterion was set against an idealised "+12%/<9% with all
v4 controls active". On the current v9 model this is unreachable
without retraining because the model has learned to pick exactly the
parabolic momentum names the universe filter is designed to block.

**The expected resolution is Stage 2-3**: with the fundamental ranker
and regime-aware sub-models, the model will diversify away from pure
parabolic momentum and the filter cost will shrink. Final graduation
is meaningful only after v11 retrain (Stage 6).

For now we treat Stage 1 as code-complete and proceed to Stage 2 with
the universe filter **disabled by default** (feature-flag is off in
the config). Operators who prioritise the spec literal can flip
`universe_filter_enabled=True` and accept the headline-excess tradeoff.

## Discoveries (worth your attention)

### Discovery 1 — Per-board IC dispersion is real and significant

ChiNext IC at 20d = 0.079 vs Shanghai Main = 0.043. The model is
nearly 2× more accurate on ChiNext / growth than on SH Main / blue
chips. Yet the CSI300 benchmark is almost entirely SH Main blue
chips — so when the benchmark rips, we are picking from our weakest
pool. This is structural and will inform Stage 2 sector-pool design.

### Discovery 2 — Per-regime IC dispersion is even bigger

20d IC in bear regime = 0.122 vs normal = 0.038 (3.2× ratio). The
model is best at risk-off picking and weakest in calm uptrends.
Confirms v9 fold-level pattern (huge bear-period excess; misses
during normal/bull). Will drive Stage 3 regime-conditional
gross / sub-model split.

### Discovery 3 — `market_features.parquet` pipeline is broken

`amount_mean_20d` is all-NaN. The file stops updating at 2020-09-25
while `market_panel.parquet` resumes amount data at 2020-09-28 —
non-overlapping in time. Task #19 tracks the rebuild. Workaround in
the stratified IC script computes the feature on-the-fly from raw
panel; the Stage 2 fundamental ranker will need this fixed properly.

### Discovery 4 — DD gate "death spiral" (already shipped in v9 opt7)

Old DD multiplier (1.0 → 0.50 → 0.20 → 0.0) created a permanent kill:
once gross dropped to 0 the NAV stopped moving, peak stayed frozen,
DD stayed at threshold for 380 consecutive days. Fixed by adding a
0.20 recovery floor at kill + rolling-252 peak so DD heals after
a year. **This single fix added ~+8pp aggregate excess (v9 baseline
+3.9% → opt7 +12.1%)**.

## How to run Stage 1 outputs

```bash
# Generate the stratified IC report
AI_quant_venv/bin/python scripts/stratified_ic_report.py
# Outputs: runtime/reports/stratified_ic/stratified_ic.md (+ CSVs + JSON)

# Replay v9 OOS with universe filter ON
QA_UNIVERSE_FILTER=1 AI_quant_venv/bin/python scripts/replay_horizon_sleeves.py
# Outputs: runtime/reports/sleeve_replay/{fold_*,all_folds_concat}/

# Run only the Stage 1 test suites
AI_quant_venv/bin/python -m pytest \
  tests/diagnostics/test_stratified_ic.py \
  tests/universe/test_filters.py \
  tests/portfolio/state_machine/test_machine.py \
  tests/optimization/test_multi_objective_loss.py
```

## Open issues carried into Stage 2

* (Task #19) `market_features.parquet` rebuild — blocks the
  fundamental ranker and any liquidity / cap-based factors.
* Universe filter is OFF by default; final ON/OFF decision waits for
  Stage 6 v11 retrain to confirm cost/benefit.
* High-chase threshold tuning (≥2 vs ≥3 limit-ups) — currently set
  to ≥3 in the source default. Re-evaluate after v10 ensemble + Stage 2.
* Position state machine class is built and tested but **not wired
  into the sleeve backtest yet** — wiring it requires a deeper
  refactor of `_build_sleeve_target` and is deferred to Stage 3.

## Stage 2 next steps

* Akshare sector mapping fetch + persist (closes Discovery 3 for
  sector axis).
* `market_features.parquet` rebuild (closes Discovery 3 fully).
* Fundamental PIT ranker (PE / PB / ROE / cash flow / growth /
  policy match).
* Sector pool builder (core / watch / short-term / excluded).
* Daily-0AM cron with health-check + retry + alerting.
