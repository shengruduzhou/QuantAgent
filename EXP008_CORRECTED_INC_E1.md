# EXP-008 CORRECTED under INC-E1 fix (2026-07-06)

**Trusted-evaluator order-dedup fix promoted (commit 7f09453). All 24 variant-C
fold evaluations re-run under the corrected simulator; pre-INC-E1 copies
preserved in `wf_h008/pre_inc_e1/`.** Runtime 209.2s,
peak RSS 1.92 GiB. CPU-only, zero retraining, zero
fresh-holdout contact (all folds OOS < 2025-09-01, quarantine guard armed).

## Headline: the pre-INC-E1 "low turnover" was an order-drop artifact

Every candidate's true turnover is 3–13× higher than recorded. The EMA books
that appeared to "solve" the 0.10/day turnover gate (0.05–0.19) actually churn
0.57–1.04/day. **EXP-011's core claim — "turnover gate is SOLVED at the book
layer" — is refuted: it was dropped incremental orders, not low churn.**

## Aggregate before → after (pre-INC-E1 → corrected)

| Candidate | median CAGR pre→post | worst fold pre→post | max turnover pre→post | DSR pre→post |
|-----------|----------------------|---------------------|-----------------------|--------------|
| C1_apriori_avg | +12.0% → **-15.8%** | -55.2% → -57.6% | 0.431 → **1.342** | 0.392 → **0.004** |
| C2_prod_rank110 | +23.8% → **-24.6%** | -33.0% → -70.9% | 0.699 → **1.347** | 0.651 → **0.000** |
| C3_rank_median | +15.8% → **-7.2%** | -29.7% → -60.9% | 0.509 → **1.336** | 0.610 → **0.004** |
| C3_ema0.3 | +26.2% → **+7.8%** | -35.7% → -53.7% | 0.145 → **0.643** | 0.555 → **0.049** |
| C3_ema0.5 | +34.8% → **+3.9%** | -48.4% → -59.3% | 0.180 → **0.848** | 0.628 → **0.029** |
| C3_ema0.7 | +33.0% → **+1.3%** | -29.9% → -56.6% | 0.259 → **1.035** | 0.736 → **0.026** |

- **fold-block PBO:** 0.833 → **0.167**
- **N (cumulative trials):** 50

## Per-fold before → after

| Fold | Candidate | CAGR pre→post | MaxDD pre→post | Turnover pre→post |
|------|-----------|---------------|----------------|-------------------|
| F1 | C1_apriori_avg | -5.2% → **-37.8%** | 11.2% → 23.5% | 0.431 → **1.342** |
| F1 | C2_prod_rank110 | -8.9% → **-32.7%** | 13.2% → 20.7% | 0.699 → **1.347** |
| F1 | C3_rank_median | -7.9% → **-22.9%** | 13.0% → 14.7% | 0.509 → **1.336** |
| F1 | C3_ema0.3 | +12.6% → **-1.0%** | 15.3% → 13.2% | 0.070 → **0.643** |
| F1 | C3_ema0.5 | -1.6% → **-10.3%** | 15.9% → 15.2% | 0.180 → **0.817** |
| F1 | C3_ema0.7 | -6.9% → **-18.7%** | 14.5% → 15.3% | 0.259 → **1.014** |
| F2 | C1_apriori_avg | -55.2% → **-57.6%** | 37.9% → 36.9% | 0.393 → **1.141** |
| F2 | C2_prod_rank110 | -33.0% → **-70.9%** | 31.5% → 45.2% | 0.352 → **1.188** |
| F2 | C3_rank_median | -29.7% → **-60.9%** | 26.9% → 38.4% | 0.411 → **1.224** |
| F2 | C3_ema0.3 | -35.7% → **-53.7%** | 32.4% → 39.0% | 0.145 → **0.611** |
| F2 | C3_ema0.5 | -48.4% → **-59.3%** | 31.1% → 36.8% | 0.149 → **0.802** |
| F2 | C3_ema0.7 | -29.9% → **-56.6%** | 25.0% → 33.9% | 0.185 → **0.987** |
| F3 | C1_apriori_avg | +29.1% → **+6.2%** | 14.1% → 11.0% | 0.296 → **1.245** |
| F3 | C2_prod_rank110 | +82.2% → **-16.5%** | 12.4% → 14.5% | 0.283 → **1.289** |
| F3 | C3_rank_median | +39.4% → **+8.4%** | 13.9% → 11.4% | 0.415 → **1.336** |
| F3 | C3_ema0.3 | +39.8% → **+29.9%** | 13.1% → 9.4% | 0.082 → **0.600** |
| F3 | C3_ema0.5 | +86.2% → **+35.3%** | 10.0% → 10.2% | 0.095 → **0.848** |
| F3 | C3_ema0.7 | +73.0% → **+33.4%** | 10.6% → 10.0% | 0.153 → **1.035** |
| F4 | C1_apriori_avg | +78.2% → **+29.0%** | 15.3% → 9.2% | 0.126 → **1.086** |
| F4 | C2_prod_rank110 | +56.6% → **+20.6%** | 9.1% → 10.0% | 0.316 → **1.142** |
| F4 | C3_rank_median | +71.1% → **+8.6%** | 16.0% → 12.1% | 0.386 → **1.229** |
| F4 | C3_ema0.3 | +46.1% → **+16.7%** | 13.0% → 11.0% | 0.050 → **0.572** |
| F4 | C3_ema0.5 | +71.2% → **+18.1%** | 15.8% → 13.0% | 0.080 → **0.770** |
| F4 | C3_ema0.7 | +77.8% → **+21.3%** | 17.1% → 12.5% | 0.106 → **0.934** |

## Corrected 15bps cost sensitivity (report-only, regenerated)

C2_prod_rank110: {"F1": -0.4658, "F2": -0.78, "F3": -0.3351, "F4": -0.1182}
C3_ema0.7:       {"F1": -0.314, "F2": -0.649, "F3": 0.0813, "F4": 0.0103}

## Interpretation

1. **DSR ≈ 0 for every blend** (max 0.0485, C3_ema0.3; was 0.55–0.74). After
   multiple-testing correction at N=50, **no blend
   has a statistically significant turnover-adjusted Sharpe.** The family's
   apparent edge was substantially an execution artifact.
2. **Median fold CAGR collapsed** across the board; the incumbent-style
   C2_prod_rank110 is now the worst of the set on median (-24.6%)
   and every candidate's median excess vs benchmark is negative
   (-34%..-24%).
3. **F2 crash is worse, not better** (−53.7%..−70.9% vs bench −33.1%);
   C3_ema0.7 remains least-bad at −56.7%. Crash exposure is signal-level.
4. **Direction preserved, magnitude destroyed:** EMA smoothing still dominates
   the fast daily-reselection books (C1/C2/median) on median CAGR and DD — the
   qualitative H-008 conclusion (smoothing helps; C2 incumbent is not a strong
   anchor) survives, but the economics are far poorer and the turnover gate is
   universally, badly violated.
5. **PBO fell to 0.167** only because everything is now
   consistently mediocre — with DSR ≈ 0 this is not a positive signal.

## Consequences for the mission

- **Track A (book-churn control) is re-opened, not closed.** Real churn is
  0.57–1.35/day, so genuine holding-period / partial-adjust / throttle rules may
  now *actually* help (their EXP-011 "success" was artifact). EXP-011's other
  conclusion ("every slow book deepens the F2 crash") must be re-tested under
  corrected fills before it is trusted.
- **Trust anchors (+17.3% / +17.25%) still pending re-run** — they pass through
  the same simulator and are expected to fall materially.
- **Production config unchanged** (red line); the incumbent looks weaker than
  believed but no auto-replacement. FRESH window (~2026-11) remains the arbiter.
