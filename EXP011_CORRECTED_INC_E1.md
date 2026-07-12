# EXP-011 CORRECTED under INC-E1 fix — Track A book-churn (2026-07-06)

**Re-ran the 5 pre-registered book-construction rules (frozen commit 1994cd4) on
the corrected simulator (fix promoted 7f09453). Candidate set unchanged; only the
reference carrier is corrected.** Runtime 344s, peak RSS 2.05 GiB, CPU-only, zero
retrain, zero fresh-holdout contact. Pre-INC-E1 copies in exp011_book_churn/pre_inc_e1/.

## Corrected carrier (C3_ema0.7, from corrected EXP-008)
median CAGR **+1.3%** · worstDD **33.9%** · F2 **-56.6%** ·
maxTurn **1.035** · DSR **0.026**

## Corrected results + gates re-derived vs corrected carrier

| Rule | median CAGR | maxTurn | worstDD | F2 crash | DSR | G1 turn≤.10 | G2 DD≤car | G3 F2≥car+5pp | G4 med≥car | verdict |
|------|-------------|---------|---------|----------|-----|------|------|------|------|---------|
| B1_buffer30 | +6.0% | 0.780 | 37.9% | -51.3% | 0.044 | ✗ | ✗ | ✓ | ✓ | beats-carrier(4/4) |
| B2_minhold10 | +36.4% | 0.202 | 36.6% | -40.2% | 0.427 | ✗ | ✗ | ✓ | ✓ | beats-carrier(4/4) |
| B3_partial30 | +23.0% | 0.452 | 36.7% | -54.2% | 0.165 | ✗ | ✗ | ✗ | ✓ | beats-carrier(4/4) |
| B4_reb5d | +29.4% | 0.352 | 38.9% | -60.3% | 0.128 | ✗ | ✗ | ✗ | ✓ | partial |
| B5_buffer_r2a_ramp | +3.5% | 0.625 | 25.2% | -39.1% | 0.051 | ✗ | ✓ | ✓ | ✓ | beats-carrier(4/4) |

(G5 sector ≤0.33 passes for all — column omitted. Gates G2/G3/G4 are carrier-relative
and were re-derived from the corrected carrier; G1 turnover and G5 sector are absolute
production promises, unchanged.)

## Reversal vs pre-INC-E1 EXP-011

The pre-INC-E1 EXP-011 concluded "turnover gate SOLVED at book layer (0.014–0.041)
but every slow book DEEPENS the F2 crash." **Both halves were artifacts:**
- The 0.014–0.041 turnovers were dropped incremental orders; real corrected
  turnovers are 0.19–0.78.
- The "slow books deepen the crash" was measured against an artifact carrier.
  Against the CORRECTED carrier (F2 −56.7%), **B2_minhold10 (−40.2%) and
  B5 (−39.1%) IMPROVE the crash by 16–18pp** — minimum-hold is crash-protective,
  not crash-amplifying, once fills are honest.

## Standout: B2_minhold10 (minimum 10-day holding period)

vs corrected carrier: median CAGR **+36.4% vs +1.3%** (28×), turnover **0.202 vs
1.035** (5× cut), F2 crash **−40.2% vs −56.7%** (+16pp), DSR **0.427 vs 0.025**
(17×), worstDD 36.6% vs 33.9% (marginally worse). **Beats the corrected carrier
on 4/4 core axes (median, turnover, F2, DSR).** It still FAILS the two absolute
production promises — G1 turnover (0.202 > 0.10) and G2 worst-DD (0.366 > 0.339) —
so it is NOT a production candidate, but it is by far the strongest Track-A
mechanism found and the only one with a materially non-trivial DSR.

## Verdict: INCONCLUSIVE-PROMISING (Track A re-opened)

- No candidate passes all absolute gates (turnover 0.10 promise binds hard: even
  the tightest book churns ~2× it; crash-window DD remains ~36%).
- But B2_minhold10 is a real, large, pre-registered improvement over the corrected
  carrier and inverts the prior crash conclusion. It warrants a corrected
  re-registration (H-014) that (a) explores min-hold length × partial-adjust to
  push turnover toward 0.10, (b) keeps the crash gain, with the FRESH window
  (~2026-11) as the out-of-sample arbiter — NOT a production swap.
- PBO 0.0; cumulative trials N unchanged (re-run of
  frozen candidates, not new trials).
