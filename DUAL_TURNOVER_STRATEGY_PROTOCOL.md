# DUAL_TURNOVER_STRATEGY_PROTOCOL — Track L vs Track H governed comparison

**Created 2026-07-07 (post-INC-E1). All evaluations use the corrected trusted
simulator (fix_cross_day_order_dedup=True, commit 7f09453). Strict variant-C,
zero retraining (frozen H-008 sleeve predictions), zero fresh-holdout contact.**

## 1. Purpose

The corrected evidence shows the old rank-blend family has weak net edge after
honest costs (family DSR≈0, every candidate negative median-excess vs benchmark,
turnover 0.57–1.35/day). This protocol governs a fair, net-metrics-only race
between two hypotheses about where robust absolute CAGR lives:

- **Track L (low-turnover):** slower, smoother, higher-capacity alpha survives
  costs better and compounds with lower drawdown.
- **Track H (high-turnover):** faster short-horizon alpha, if execution survives
  strict costs, captures more gross alpha and reacts to crashes better.

Neither is assumed correct. Candidates are accepted on **net** metrics only.

## 2. Common harness

`scripts/analysis/dual_track_eval.py` runs every candidate through one code path:

1. Build a carrier score from frozen sleeve predictions (rank-mean of the chosen
   sleeves, optional per-symbol EMA smoothing) — PIT-safe, warmup uses pre-OOS
   prediction days only.
2. Apply one book-construction rule (plain top-k / min-hold-N / keep-zone-buffer
   / rebalance-throttle-N / partial-adjust / score-hysteresis).
3. Delay-1 execution shift (identical to variant-C `_target_weights`).
4. `run_strict_backtest_v8` (corrected simulator) at **8 / 15 / 25 bps**.
5. Full net-metric suite + fold-block CSCV PBO + DSR across all candidates + the
   corrected C3_ema0.7 reference carrier.

Folds: H-008 F1–F4 (OOS 2023-07-03..2023-12-29 / 2024-01-02..2024-06-28 /
2024-07-01..2024-12-31 / 2025-01-02..2025-08-29). Benchmark: eqw-all-A per fold.
Quarantine guard armed (every fold asserts oos_end < 2025-09-01).

## 3. Metric suite (per candidate, aggregated over folds)

net CAGR (median & worst fold) · excess vs benchmark · Sharpe · Sortino · Calmar ·
max drawdown (worst fold) · F2 crash CAGR · turnover/day · avg holding period ·
cost drag (CAGR@8bps − CAGR@25bps) · 8/15/25 bps sensitivity · mean max sector
weight · PBO · DSR · peak RSS · disk delta. **Gross reported, net decides.**

## 4. Comparison tables (emitted every run)

1. Track L best vs Track H best (net CAGR, DD, turnover, F2, DSR).
2. Track L median vs Track H median candidate.
3. Best net-CAGR candidate. 4. Best drawdown-adjusted (Calmar).
5. Best capacity-adjusted (lowest turnover among net-positive).
6. Best crash-regime (F2). 7. Best production-readiness (most gates passed).

## 5. Track-specific acceptance gates

**Track L:** net median CAGR ≥ corrected carrier (+1.3%) ∧ max turnover materially
lower than carrier (≤0.5, i.e. <½ of 1.035) ∧ worst-fold DD ≤ carrier (33.9%) ∧
F2 ≥ carrier+5pp (−51.7%) ∧ CAGR@25bps still ≥ 0 on median ∧ PBO not adverse ∧
no leakage ∧ reproducible.

**Track H:** net median CAGR ≥ corrected carrier ∧ **CAGR@8 AND @15 AND @25bps all
beat carrier on median** (cost survival is the H bar) ∧ turnover high but capped
(≤1.5/day, no pathological >2) ∧ worst-fold DD ≤ carrier ∧ F2 ≥ carrier ∧ PBO not
adverse ∧ no hidden leverage ∧ no leakage ∧ reproducible.

Absolute production promises (turnover ≤0.10/day, no leverage, sector ≤0.33,
DSR≥0.95, zero fresh-holdout) still bind for any PRODUCTION_CANDIDATE_PROPOSAL.

## 6. Discipline

- Candidates are a-priori and interpretable; **no tiny parameter sweeps**.
- Every candidate counts as a trial; cumulative N tracked in the registry.
- The 4 H-008 folds are heavily mined (~65 prior looks) — **FRESH window (~2026-11)
  is the only uncontaminated arbiter**; fold results rank mechanisms, they do not
  crown a production winner.
- No production config change; no automatic promotion.
