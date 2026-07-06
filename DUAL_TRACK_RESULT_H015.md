# DUAL-TRACK RESULT — H-015 (Track L low-turnover vs Track H high-turnover)

**Corrected simulator, strict variant-C, H-008 folds, 8/15/25 bps, net metrics
decide. N=73, PBO=0.0, runtime 804.6s,
peak RSS 1.96 GiB. Zero retrain, zero fresh-holdout contact.**

Reference: corrected C3_ema0.7 carrier — median +1.3%,
worstDD 33.9%, F2 -56.6%, maxTurn
1.035, DSR 0.0003. Benchmark per fold:
F1 -2.6% / F2 -33.1% /
F3 +69.4% / F4 +46.5%.

## Full candidate table (8 bps unless noted)

| Cand | Trk | medCAGR | worstFold | F2 | med@25bps | worstDD | Calmar | maxTurn | hold | medExcess | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1_c3ema07_minhold10 | L | +36.4% | -41.0% | -41.0% | +24.1% | 36.6% | 0.99 | 0.202 | 9.5d | +14.4% | 0.055 |
| L4_c3ema07_reb10 | L | +31.8% | -46.5% | -46.5% | +21.8% | 35.1% | 0.91 | 0.190 | 10.2d | +9.9% | 0.041 |
| L3_midlong_minhold10 | L | +27.4% | -33.0% | -33.0% | +16.9% | 37.2% | 0.74 | 0.200 | 9.5d | +5.5% | 0.040 |
| H4_short_minhold3 | H | +16.8% | -51.8% | -51.8% | -10.7% | 30.9% | 0.55 | 0.627 | 3.0d | -12.0% | 0.004 |
| L2_midlong_ema07 | L | +12.0% | -55.7% | -55.7% | -25.1% | 39.0% | 0.31 | 1.012 | 1.7d | -18.4% | 0.002 |
| H2_short_hyst | H | -9.1% | -53.6% | -53.6% | -47.7% | 31.6% | -0.29 | 1.382 | 1.3d | -31.1% | 0.000 |
| H1_short_fast | H | -15.2% | -63.7% | -63.7% | -51.3% | 38.8% | -0.39 | 1.443 | 1.2d | -38.6% | 0.000 |
| H3_c2_fast | H | -24.2% | -71.9% | -71.9% | -56.5% | 45.1% | -0.54 | 1.347 | 1.2d | -34.1% | 0.000 |

## The 7 governed comparisons

**1. Track L best vs Track H best**
| Cand | Trk | medCAGR | worstFold | F2 | med@25bps | worstDD | Calmar | maxTurn | hold | medExcess | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1_c3ema07_minhold10 | L | +36.4% | -41.0% | -41.0% | +24.1% | 36.6% | 0.99 | 0.202 | 9.5d | +14.4% | 0.055 |
| H4_short_minhold3 | H | +16.8% | -51.8% | -51.8% | -10.7% | 30.9% | 0.55 | 0.627 | 3.0d | -12.0% | 0.004 |

**2. Track L median vs Track H median candidate:** L median CAGR **+31.8%**
vs H median **-9.1%**.

**3. Best net-CAGR (8 bps):** **L1_c3ema07_minhold10** (+36.4%).
**4. Best drawdown-adjusted (Calmar):** **L1_c3ema07_minhold10** (Calmar 0.99).
**5. Best capacity-adjusted (lowest turnover among median@25bps>0):** **L4_c3ema07_reb10**
(turnover 0.190).
**6. Best crash-regime (F2):** **L3_midlong_minhold10** (F2 -33.0%).
**7. Best production-readiness (most gates):** see gate tables below — top-3 L
candidates pass 4/5 Track-L gates (fail only worst-DD by 1–3pp in the F2 crash);
no Track H candidate passes its cost-survival gate.

## Track-L gate table (vs corrected carrier + protocol §5)

| Cand | medCAGR≥carrier | turn≤0.5 | wDD≤33.9% | F2≥−51.7% | med@25bps≥0 | verdict |
|---|---|---|---|---|---|---|
| L1_c3ema07_minhold10 | ✓ | ✓ | ✗ | ✓ | ✓ | 4/5 |
| L2_midlong_ema07 | ✓ | ✗ | ✗ | ✗ | ✗ | 1/5 |
| L3_midlong_minhold10 | ✓ | ✓ | ✗ | ✓ | ✓ | 4/5 |
| L4_c3ema07_reb10 | ✓ | ✓ | ✗ | ✓ | ✓ | 4/5 |

## Track-H gate table (cost-survival is the bar)

| Cand | med@8≥carrier | med@15≥carrier | med@25≥carrier | turn≤1.5 | F2≥carrier | verdict |
|---|---|---|---|---|---|---|
| H1_short_fast | ✗ | ✗ | ✗ | ✓ | ✗ | 1/5 |
| H2_short_hyst | ✗ | ✗ | ✗ | ✓ | ✓ | 2/5 |
| H3_c2_fast | ✗ | ✗ | ✗ | ✓ | ✗ | 1/5 |
| H4_short_minhold3 | ✓ | ✓ | ✗ | ✓ | ✓ | 4/5 |

## Verdict: Track L VALIDATED as the robust path; Track H REJECTED on cost survival

1. **Low-turnover dominates.** The top-3 candidates are all Track L
   (L1 +36.4%,
   L4 +31.8%,
   L3 +27.4% median), each with
   turnover ≤0.20/day. Every fast (turn≥1.0) candidate — L2, H1, H2, H3 — is
   net-weak-to-negative.
2. **Cost survival is the discriminator.** At 25 bps the low-turnover L books stay
   strongly positive (L1 +24.1%,
   L4 +21.8%) while **no Track H
   candidate survives** (best H4 -10.7%).
   Track H fails its own gate.
3. **Turnover control, not horizon, is the lever.** L2 (plain mid+long, no hold)
   churns 1.01 and is weak; adding min-hold (L3) flips it to +27.4%. Even a fast
   short signal is rescued by min-hold-3 (H4 +16.8% vs H1 −15.2%). The mechanism
   is the book constraint.
4. **Best robust candidate = L1_c3ema07_minhold10** — median +36.4%, **median
   excess vs benchmark +14.4%** (the carrier was negative), turnover 0.20 (5× under
   carrier), survives 25 bps at +24.1%. **Best defensive = L3_midlong_minhold10**
   — best F2 crash (−33.0%), medium-horizon.
5. **Not production-ready.** All candidates fail worst-DD (~35–37% in the F2 crash)
   and DSR (<0.06 at N=73). The 4 folds are heavily mined; **FRESH (~2026-11) is
   the arbiter.** No production proposal.

**Next:** Track L is the focus. Highest-EV = attack the residual failures — the F2
crash worst-DD and the DSR bar — via (a) crash-regime cash buffer / R2a-ramp on the
L1 book, (b) PIT-safe low-turnover + defensive factor generation to lift signal
quality (a book constraint cannot fix a signal-level crash). Track H is closed
unless a materially cheaper execution assumption is justified.
