# EXP-010 CORRECTED under INC-E1 fix — R2a crash-defense overlay (2026-07-06)

**Re-ran the 2 pre-registered hysteresis overlays (R2a_confirm5, R2b_ema_gross)
on the frozen C3_ema0.7 carrier under the corrected simulator (fix promoted
7f09453).** 77s, peak RSS 1.9 GiB, CPU-only, zero retrain, zero fresh-holdout.
Pre-INC-E1 copy in exp010_hysteresis/pre_inc_e1/.

## The headline: R2a's crash advantage was largely an INC-E1 artifact

The memory flagged this as the most likely reversal ("R2a 的 F2 优势可能反转").
**Confirmed.** Pre-INC-E1, R2a's F2 crash read **−16.8%** (vs carrier −29.9%) —
the "best risk profile of the cycle." That number was an artifact: when R2a
de-risked (sold) into the crash, the bug **silently dropped every re-risk (rebuy)
order**, so the book stayed frozen-defensive for the rest of the fold. Under the
corrected simulator R2a re-risks properly and pays the real round-trip.

## Corrected per-fold (overlay on C3_ema0.7 carrier, 8 bps)

| Fold | Carrier C3_ema0.7 | +R2a_confirm5 | +R2b_ema_gross | R2a effect |
|------|-------------------|---------------|----------------|-----------|
| F1 (flat)  | −18.7% / DD 15.3% | −20.3% / DD 14.5% | −18.9% / DD 14.3% | slightly worse |
| **F2 (crash)** | **−56.7% / DD 33.9%** | **−48.5% / DD 28.2%** | −47.8% / DD 27.9% | **+8pp CAGR, +6pp DD** |
| F3 (bull)  | +33.4% / DD 10.0% | +16.7% / DD 9.6% | +36.5% / DD 7.8% | **−17pp (upside diluted)** |
| F4         | +21.3% / DD 12.5% | +14.7% / DD 12.6% | +16.2% / DD 10.5% | −7pp |

Turnover: R2a 0.74–0.86/day (carrier ~1.0), mean_gross 0.68–0.88.

## Interpretation

1. **R2a is a real but modest, costly crash hedge, not a free lunch.** Corrected,
   it improves the F2 crash by ~8pp (−56.7%→−48.5%) and worst-DD by ~6pp, but
   dilutes bull-market upside by 7–17pp (F3/F4). A genuine risk/return trade, far
   from the artifact's −16.8% "best profile."
2. **Book-layer beats overlay-layer.** B2_minhold10 (EXP-011 corrected) improves
   the crash MORE (F2 −40.2% vs R2a −48.5%) while *raising* median CAGR
   (+36.4%), whereas R2a improves the crash less and cuts return. **Churn/crash
   control belongs in book construction (min-hold), not gross-switching overlays**
   — the corrected data reverses the direction but confirms Track A over the
   overlay line more strongly than before.
3. **Consequence for Track D (RL overlay):** a gross-exposure RL overlay would be
   competing against a weak baseline mechanism (R2a) that the book layer already
   dominates. RL overlay value, if any, is as a *turnover-aware* controller on top
   of a min-hold book, not as a standalone gross scaler. Re-scope Track D
   accordingly.

Cumulative trials N unchanged (re-run of frozen candidates). No production
proposal. FRESH window (~2026-11) remains the arbiter.
