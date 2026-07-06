# BOOK_CHURN_CONTROL_EXPERIMENT — EXP-011 / H-011 (Track A)

**Status: COMPLETE 2026-07-06 — all 5 candidates REJECTED under pre-registered gates (0/5).
Churn gate is solved mechanically (turnover 0.014–0.041 « 0.10 for B2/B3/B4), but every
churn-controlled book makes the F2 crash fold WORSE. See §6–§7.**

> ⚠ **pre-INC-E1 stamp (2026-07-06, same day, post-hoc):** the execution simulator was later
> found to silently drop all repeat (symbol, side) orders across a backtest
> (EVALUATOR_ORDER_DEDUP_BUG.md). The low-turnover numbers here are partly an artifact
> (incremental orders never reached the broker), and Finding 3's path-noise is largely that
> bug's cascade. All quantitative claims in this file must be re-run after the approved fix
> before being cited.

*(Registered at commit 1994cd4 before any evaluation run; candidates frozen there.)*

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

Run: `AI_quant_venv/bin/python3 scripts/analysis/exp011_book_churn.py` — 40 strict variant-C
backtests (20 @ 8bps + 20 @ 15bps), 327.6 s, peak RSS 2.02 GiB, artifacts in
`runtime/reports/v89_closed_loop/wf_h008/exp011_book_churn/`. One protocol-neutral
implementation fix mid-run: an over-strict anti-runaway assert (`len(book) ≤ 40`) contradicted
the registered B3 definition (no size cap) and was relaxed to 500; no candidate change.

### Fold CAGR (8bps) — carrier C3_ema0.7 baseline in brackets

| Fold | B1_buffer30 | B2_minhold10 | B3_partial30 | B4_reb5d | B5_buffer_r2a_ramp | carrier |
|------|------------|--------------|--------------|----------|--------------------|---------|
| F1 | −6.2% | **+14.5%** | +6.5% | **+15.0%** | +7.5% | [−6.9%] |
| F2 crash | −43.0% | −40.4% | −31.6% | −40.1% | −33.5% | [−29.9%] |
| F3 | +81.2% | +62.4% | +66.4% | +74.2% | +77.0% | [+73.0%] |
| F4 | +91.5% | +79.8% | +86.2% | +67.1% | +60.6% | [+77.8%] |

### Aggregates and gates

| Candidate | median | worstDD | maxTurn | G1 turn≤0.10 | G2 DD≤25.0% | G3 F2≥−24.9% | G4 med≥28.0% | G5 sec | all |
|-----------|--------|---------|---------|----|----|----|----|----|-----|
| B1_buffer30 | +37.5% | 32.9% | 0.153 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| B2_minhold10 | +38.4% | 32.9% | **0.041** | **✓** | ✗ | ✗ | ✓ | ✓ | ✗ |
| B3_partial30 | +36.4% | 35.0% | **0.015** | **✓** | ✗ | ✗ | ✓ | ✓ | ✗ |
| B4_reb5d | +41.1% | 37.4% | **0.034** | **✓** | ✗ | ✗ | ✓ | ✓ | ✗ |
| B5_buffer_r2a_ramp | +34.0% | 30.8% | 0.140 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |

- G6 no-leverage / G7 quarantine: construction-asserted, all ✓. Sector max ≤0.29 all folds.
- PBO (fold-block CSCV, 6 books) 0.833 (unchanged, coarse). DSR@N=60 stitched: B3 0.885
  (highest), B5 0.859, B2 0.853, B1 0.845, B4 0.845, carrier 0.872. None ≥ 0.95.
- Trial ledger: blend+overlay+book N = **60**.

### Finding 1 — churn is solved, and it was never the bottleneck outside crashes

B2/B3/B4 cut turnover 4–17× below the 0.10 production promise while *raising* median fold CAGR
(+36–41% vs +33.0%) and flipping the flat-market fold F1 from −6.9% to +14.5/+15.0%. The
turnover gate that killed C3_ema0.7 in EXP-008 is fully passable at the book layer with
one-line rules.

### Finding 2 — slow books die harder in the crash fold (the real rejection reason)

Every churn rule degraded F2 (−31.6% to −43.0% vs carrier −29.9%) and worstDD (30.8–37.4% vs
25.0%). Mechanism: the carrier's daily re-selection was providing *implicit crash defense* —
rotating out of collapsing microcaps as their scores decayed. Keep-zones let losers ride from
rank 10→30; min-hold locks them for 10 days; throttling waits 5 days — in 2024-H1 each of those
delays converts directly into deeper drawdown. Churn control and crash survivability are in
direct tension on this signal family; B5's ramped R2a de-risking (confirm-5 + 0.1/day) is too
slow to offset it (F2 −33.5%, DD 30.8%).

### Finding 3 — execution path-dependency noise quantified (evaluator robustness)

The 15bps sensitivity runs exposed non-monotone cost responses (B1/F1: −6.2% @8bps →
**+17.2%** @15bps). Verified deterministic (identical repeat at 8bps) and input-mutation-free
(simulator copies inputs); a bps-sweep probe on B1/F1 gives −6.2/−4.2/−8.4/−6.1/+17.2% at
8/9/10/12/15bps. **k=10 fold CAGRs carry ±3pp noise per 1–2bps perturbation with occasional
>20pp basin jumps** (lot-size → cash → T+1/limit feasibility cascades). Consequences: (a)
sub-5pp fold-level margins in EXP-008..011 are within execution-path noise; (b) 15bps columns
measure path perturbation more than cost drag at these turnover levels; (c) k=10 concentrated
books are structurally fragile — argues for wide-book variants and/or perturbation-averaged
evaluation in the next protocol. F2 failures above are NOT noise: 5/5 candidates degraded in
the same direction, 2–13pp deep.

## 7. Verdict

**REJECTED — 0/5 candidates pass the pre-registered gate set** (all fail G2+G3; B1/B5 also
fail G1). No production proposal. No candidate was modified after seeing results; trial count
ledgered at N=60.

**What survives:** the churn mechanisms themselves (B2/B3/B4) are proven, cheap, and
production-safe *outside crash regimes* — they are shelved as building blocks, not adopted.
R2a remains sealed pending FRESH-window arbitration (per EXP-010 stop-clause).

**Next recommended path (registered before results were seen? no — chosen after, so it needs
its own pre-registration):** the churn/crash tension plus Finding 3 says further mining of
these 4 folds at k=10 has hit diminishing returns (60 trials, PBO 0.833, path noise ≈ candidate
margins). The highest-EV next step is **structural, not parametric**: evaluate the existing
frontier configs at k=30 wide-book (capacity↑, path-noise↓, per-name crash impact↓ — EXP-004
already showed k30 improves worst quarters), with an explicit path-noise-band measurement, as
a small pre-registered batch (H-012). Crash-exit logic (absolute stop / sell-only stress mode)
is the other open lever but should wait for the wide-book base to avoid compounding
fold-mining.
