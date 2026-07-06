# BOOK_CHURN_CONTROL_EXPERIMENT — EXP-011 / H-011 (Track A)

**Status: REGISTERED 2026-07-06 (candidates frozen at this commit, before any evaluation run).**

## 1. Motivation (from the closed overlay line)

EXP-008 established that the best blend (C3_ema0.7) fails exactly one economic gate — turnover
(max fold 0.259/day vs the 0.10/day production promise) — and that the whole family shares one
failure mode: the F2 crash fold (2024-H1 microcap collapse, bench −33.1%).

EXP-009/EXP-010 closed the exposure-overlay line with a structural conclusion: crash-regime
gross scaling works mechanically (R2a confirm-5: F2 −16.8%, worstDD 19.9% — best risk profile of
the cycle) but instant gross switching trades half the book per flip and breaks the turnover gate
(F1 0.362). **Churn must be solved in the book-construction layer, not the overlay layer.**

EXP-011 attacks churn where it is generated: the daily top-k re-selection.

## 2. Candidates (N=5, a-priori, frozen — no additions, no post-hoc tuning)

Carrier signal for all candidates: **C3_ema0.7** composite score, rebuilt per fold exactly as
EXP-008/009/010 (same `build_candidates`, same EMA warmup on pre-OOS prediction days).
Book size k=10, equal weight 1/k, long-only, gross ≤ 1.0 (construction-guaranteed, no leverage).
Eligibility semantics identical to variant-C `_target_weights`: a name must be in the day's
eligible set (not suspended / not ST / not limit-up) both to **enter** and to be **retained**
(same force-out semantics as the incumbent book, so comparisons are like-for-like).
Book state starts cold at each fold's OOS start. Delay-1 execution shift identical to
`_target_weights`.

| ID | Rule | Definition (exact) |
|----|------|--------------------|
| **B1_buffer30** | rank keep-zone | Enter: eligible rank < 10. Retain: held name stays while eligible rank < 30. Book = retained names + top-ranked non-held fills up to 10. |
| **B2_minhold10** | minimum holding period | Held names with age < 10 trading days are locked (retained if in eligible set). Free slots = 10 − locked; filled from eligible rank order (unlocked held names compete by rank). New entrant age = 1; age += 1 per day held. |
| **B3_partial30** | partial target adjustment | `w_t = 0.7·w_{t−1} + 0.3·target_t` where target_t = plain top-10 equal-weight book; weights < 0.005 pruned, then renormalized to 1.0. `w_0 = target_0`. |
| **B4_reb5d** | rebalance throttling | Recompute plain top-10 book only on fold-relative trading days 0, 5, 10, …; target weights held constant in between. |
| **B5_buffer_r2a_ramp** | keep-zone + gradual crash de-risking | B1_buffer30 book × ramped R2a gross: state machine identical to EXP-010 R2a (bench MA60, 5-day confirm both directions, states {1.0, 0.5}, observed t−1 → applied t), but gross moves toward the state target at most **0.1/day** (gradual scaling instead of instant switching). Scaled-out fraction is cash. |

B1–B4 isolate four churn-control mechanisms; B5 is the pre-registered combination that pairs the
lowest-churn book with the cycle's best crash mechanism, ramped to kill switch-churn.

## 3. Evaluation (identical to H-008 protocol)

- Folds: H-008 F1–F4 (OOS 2023-07-03..2023-12-29 / 2024-01-02..2024-06-28 / 2024-07-01..2024-12-31 / 2025-01-02..2025-08-29), frozen sleeve predictions, **zero retraining**.
- Strict variant-C: eligible-only, delay-1, k=10, 8 bps slippage, `run_strict_backtest_v8`, full A-share constraints (T+1, limit bands, suspension, ST).
- Quarantine guard armed; every fold asserts `oos_end < 2025-09-01`. **Zero fresh-holdout contact.**
- Cost sensitivity: every candidate re-run at 15 bps on all 4 folds (report-only, not selection).
- Statistics: fold-block CSCV PBO across {B1..B5 + C3_ema0.7 carrier}; DSR on stitched daily
  returns at cumulative trial count **N = 60** (55 prior + 5 here).
- Budget: CPU-only, 40 variant-C runs ≈ 10 min, RSS < 4 GiB, disk < 50 MB.

Command:

```
AI_quant_venv/bin/python3 scripts/analysis/exp011_book_churn.py
```

## 4. Frozen baselines (EXP-008 wf_summary.json, C3_ema0.7 carrier)

| Fold | bench | carrier CAGR | carrier DD | carrier turn | carrier sec-max |
|------|-------|--------------|-----------|--------------|-----------------|
| F1 | −2.6% | −6.9% | 14.5% | 0.259 | 0.279 |
| F2 (crash) | −33.1% | −29.9% | 25.0% | 0.185 | 0.272 |
| F3 | +69.4% | +73.0% | 10.6% | 0.153 | 0.244 |
| F4 | +46.5% | +77.8% | 17.1% | 0.106 | 0.257 |

Aggregates: median CAGR +33.0%, worstDD 25.03%, max turnover 0.2594.
Incumbent C2 for context: median +23.8%, worstDD 31.5%, max turnover 0.6985.
R2a (EXP-010, sealed): F2 −16.8%, worstDD 19.9%, but F1 turnover 0.362.

## 5. Acceptance gates (frozen; ALL required for acceptance)

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| G1 turnover | max fold turnover ≤ **0.10**/day | the production promise C3_ema0.7 failed — the gate this track exists to pass |
| G2 drawdown | worst fold maxDD ≤ **0.2503** | do-not-degrade vs carrier |
| G3 F2 crash | F2 CAGR ≥ **−0.249** | material improvement = baseline +5pp |
| G4 median | median fold CAGR ≥ **+0.2802** | baseline −5pp, no collapse |
| G5 sector | mean daily max sector weight ≤ **0.33** every fold | baseline max 0.279 + 0.05 margin |
| G6 leverage | gross ≤ 1.0 | construction guarantee, asserted |
| G7 quarantine | zero fresh-holdout contact | guard + per-fold assert |
| Stat (adoption) | DSR ≥ 0.95 at N=60 | unchanged H-008 convention; PBO reported |

Verdict rules (frozen): a candidate passing G1–G7 = **mechanism ACCEPTED** (still no automatic
production change; a `PRODUCTION_CANDIDATE_PROPOSAL` would additionally need the stat gate and a
FRESH-window read, which stays locked). Candidates passing G1+G2+G4+G5 but not G3 are recorded as
*churn-solved / crash-unsolved* — informative for layering, not accepted. No rule may be modified
after results are seen; the full trial count is ledgered regardless of outcome.

## 6. Results

*(filled after the run — this section intentionally empty at registration commit)*

## 7. Verdict

*(filled after the run)*
