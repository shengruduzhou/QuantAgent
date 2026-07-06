# DUAL_TRACK_FACTOR_BATCH_PLAN — batch 1 (defensive / low-turnover)

**Created 2026-07-07. Motivation:** H-015 proved Track L (low-turnover) is the
robust path, but the residual failure is the **F2 crash (worst-DD ~36%)** which is
**signal-level** — no book constraint fixes it. This batch generates PIT-safe,
low-turnover, crash-resilient / defensive / liquidity factors to lift signal
quality on exactly that axis.

## Rules (mandate compliance)

- **PIT-safe constrained DSL only** (`quantagent.factors.expr`): Add/Sub/Mul/Div,
  Abs/Sign/Log/CsZscore, Ts{Mean,Std,Sum,Max,Min,Rank}, DecayLinear, Ts{Corr,Cov},
  Delay/Delta, Returns. No free-form Python. No future data (factor uses only
  past; label is forward return).
- **No fresh holdout, no burned holdout.** Evaluation window = union of the H-008
  folds, all pre-quarantine (2023-07-03..2025-08-29). Quarantine boundary asserted.
- A-priori, interpretable factors; **no parameter sweeps**. Batch capped at 7.

## Candidates (7, frozen)

| ID | class | expression (DSL, oriented high=good) | mechanism | exp. turnover |
|----|-------|--------------------------------------|-----------|---------------|
| D1_low_vol_20 | defensive | −TsStd(Returns(Close,1),20) | low realized vol = crash-resilient | low |
| D2_trend_quality_60 | low_turnover | Returns(Close,60) / (TsStd(Returns(Close,1),60)+ε) | risk-adjusted medium momentum (quality) | low |
| D3_near_high_120 | defensive | Close / TsMax(Close,120) | proximity to 6-mo high (52-wk-high effect) | low |
| D4_liquidity_amount_60 | liquidity | TsMean(Amount,60) | high-liquidity = defensive + capacity | low |
| D5_amihud_illiq_neg_20 | liquidity | −TsMean(\|Returns(Close,1)\|/(Amount+1),20) | Amihud illiquidity avoidance (liquidity stress) | low |
| D6_vol_compression | defensive | −TsStd(Returns,5)/(TsStd(Returns,60)+ε) | vol compression (calm recent vs long) | medium |
| D7_downside_range_neg_20 | defensive | −TsMean((High−Low)/Close,20) | tight intraday range = calm | low |

## Metrics recorded (FACTOR_CANDIDATE_LEDGER.csv)

rank_IC & rank_ICIR (h10 primary, h20 secondary) · positive_ratio · top-quantile
daily turnover · quintile long-short cost-adjusted @8/15/25 bps · **F2-crash-window
rank_IC (2024-01-02..2024-06-28)** · max abs decorrelation vs other candidates +
vs 20d-momentum ref + vs liquidity ref · capacity_rmb (5% participation).

## A-priori acceptance gates (record accept/reject/discard)

- `|rank_IC(h10 or h20)| ≥ 0.015` AND `|rank_ICIR| ≥ 0.20` (predictive + stable);
- top-quantile turnover ≤ 0.15 (low-turnover mandate);
- cost-adjusted LS @25bps keeps the sign of @8bps (cost-survival);
- **defensive_candidate additionally: F2-crash rank_IC ≥ 0** (crash resilience);
- max abs decorrelation vs any other candidate < 0.90 (not redundant).

**Survivors** → materialize as `synth_*` (reviewed) and queue for a Track-L
book/model integration test. **Not added to production automatically.**

---

## RESULT (2026-07-07, corrected verdict logic, commit pending)

Ran `scripts/analysis/dual_track_factor_batch.py` on 2023-07-03..2025-08-29
(pre-quarantine, asserted). 116s, peak RSS 2.70 GiB, CPU-only. Verdict logic
requires **oriented-positive IC** (no silent sign-flip: a negative-IC factor's
long side here is illiquid/reversal = capacity trap) + low turnover + cost
survival + (defensive) crash IC ≥0 + decorrelation cluster keep-best.

| Factor | IC10 | ICIR10 | turn | F2-crash IC | LS@25bps | verdict |
|--------|------|--------|------|-------------|----------|---------|
| **D1_low_vol_20** | **+0.080** | **+0.35** | **0.074** | **+0.080** | **+0.0045** | **ACCEPT** |
| D7_downside_range_neg_20 | +0.078 | +0.32 | 0.044 | +0.087 | +0.0036 | redundant (0.91 corr D1) |
| D6_vol_compression | +0.049 | +0.48 | 0.329 | +0.009 | +0.0063 | reject (turnover >0.15) |
| D2_trend_quality_60 | −0.071 | −0.47 | 0.114 | −0.062 | −0.0094 | reject (neg IC = 60d reversal) |
| D4_liquidity_amount_60 | −0.088 | −0.48 | 0.008 | +0.001 | −0.0105 | reject (illiq premium, long side = capacity trap) |
| D5_amihud_illiq_neg_20 | −0.070 | −0.34 | 0.025 | +0.041 | −0.0105 | reject (illiq premium) |
| D3_near_high_120 | −0.040 | −0.20 | 0.120 | −0.009 | −0.0088 | reject (weak, neg) |

**Survivor: D1_low_vol_20** = −TsStd(Returns(Close,1),20). A low-turnover
(0.074/day ≈ 13.5-day hold), crash-resilient (positive IC *inside* the F2 crash),
cost-surviving (positive long-short at 25 bps) defensive factor — exactly the
signal-level lever the H-015 residual crash failure needs (book constraints
cannot fix a signal-level crash).

**Notes:** D6 (vol-compression) has the best ICIR (0.48) but 0.33 turnover →
queued for a future *medium*-turnover batch. The illiquidity/reversal premia
(D2/D4/D5) are real but not tradable in a capacity-aware long book (long side =
small/illiquid/loser names) → not materialized.

## Materialization plan (survivor)

- Register `D1_low_vol_20` as reviewed `synth_low_vol_20` (FactorDefinition,
  `quantagent.factors.expr`, formula frozen above). **Not added to production.**
- **Integration test (next):** tilt the corrected C3_ema0.7 carrier by D1's
  per-date rank (a-priori weight 0.3) under the L1 min-hold-10 book on the H-008
  folds; accept only if it improves the F2 crash / worst-DD without wrecking
  median or turnover. FRESH (~2026-11) remains the OOS arbiter.
- Dataset rebuild (adding the column to the 8 GB training set) deferred until the
  integration test justifies it.
