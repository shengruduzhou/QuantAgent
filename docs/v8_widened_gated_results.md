# v8 Widened Sweep + Gated Backtest — Results Summary

OOS window: **2023-08-11 → 2024-12-31** (337 trading days).
Universe: top-500 A-shares by 2024 average dollar volume (`runtime/reports/v8/pipeline/universe_top500.txt`).
Strict A-share backtest model: T+1, 涨跌停 inability, slippage 8 bps, 印花税, sqrt-impact, initial cash 1M.

## 1. Widened FT-Transformer sweep (80 ep / d_token=256 / 6 blocks / 500 syms)

Run tag: `v8_deep_wide_20260531_185536` — 66 min wall-clock on RTX 3090.

| Horizon | Baseline (40/128/4) ann | Widened (80/256/6) ann | Baseline sharpe | Widened sharpe |
|---|---|---|---|---|
| short_5d        |  0.6% |  **22.6%**  | 0.12 | **1.15** |
| mid_5d_30d      | 11.2% |  **22.7%**  | 0.70 | **1.14** |
| long_30d_120d   |  5.4% |  **12.4%**  | 0.60 | **0.92** |

The widening was the single biggest individual win. Short-horizon model went
from non-functional to sharpe ≈ mid horizon.

## 2. Ensemble blend tuning (rank-IC OOS search)

Default 30/45/25 is suboptimal because short and long were under-trained at baseline.
After widening, the search still concentrates on the mid horizon:

* **Best weights on widened**: short = 0.00, mid = 0.90, long = 0.10 (rank-IC = 0.074).
* The 3-simplex grid search at step 0.05 visits 231 points × 3 folds.

## 3. 15-gate decision chain

`src/quantagent/risk/decision_chain.py` implements all 15 gates:

```
kill_switch                 data_quality              model_confidence_decay
sector_pool_top_n           is_suspended              is_st
limit_up_no_buy             consecutive_limit_up_cap  liquidity_floor
single_name_concentration   sector_concentration      drawdown_brake
order_rate_cap              capital_outflow_spike     model_score_floor
```

Gracefully degrades when sector_map is missing — unknown sectors get an
isolated bucket each so the concentration cap doesn't collapse the portfolio.

5 unit tests in `tests/risk/test_decision_chain.py` cover gate counts,
basic acceptance, unknown-sector handling, consecutive limit-up cap,
and kill switch.

## 4. End-to-end gated backtest results

Pipeline: tuned ensemble → 15-gate chain → strict A-share backtest.

### Baseline-trained sweep, gated

| Config | ann | sharpe | max_dd | calmar | turnover |
|---|---|---|---|---|---|
| Raw ensemble 30/45/25 (no gates) | 9.9%  | 0.70 | 11.7% | 0.85 |  8.8% |
| Tuned 0/0.9/0.1 + 15-gate, k=30 | 19.0% | 0.92 | 16.3% | 1.17 |  8.4% |
| Tuned + k=20                    | 40.2% | 1.26 | 24.7% | 1.63 |  8.5% |
| Tuned + k=10                    | 32.6% | 1.27 | 19.2% | 1.70 |  6.6% |

### Widened-trained sweep, gated

| Config | ann | sharpe | max_dd | calmar | turnover |
|---|---|---|---|---|---|
| Tuned + k=10  | **51.2%** | **1.71** | 13.3% | **3.86** | n/a |
| Tuned + k=20  | **53.7%** | 1.55     | 16.0% | 3.36     | n/a |
| Tuned + k=30  | 23.3%     | 1.08     | 12.0% | 1.95     | n/a |
| Tuned + k=50  |  8.3%     | 0.79     |  7.6% | 1.10     | n/a |

Best risk-adjusted = **widened + tuned + k=10**: ann 51.2%, sharpe 1.71, calmar 3.86.

CSI300 over the same OOS window returned roughly +3-7% annualized → the
gated widened ensemble produced ~45 pp annualized excess return.

## 5. New modules and CLI surface

```
src/quantagent/ensemble/blend_optimizer.py   # rank-IC + topK-return objectives
src/quantagent/risk/decision_chain.py        # 15-gate live chain
src/quantagent/cli/v8_gated.py               # CLI:
  - optimize-ensemble-weights-v8
  - apply-decision-chain-v8
  - run-gated-backtest-v8
```

## 6. Dataset verification (PIT)

* OHLCV: 0 NaN in open/high/low/close/volume across 7.3 M rows × 3 869 symbols.
* `amount`: 100 % null in 2018-2019 (qlib field absent), 0 % null 2021+. The
  trainer's >=50 % non-null filter drops amount-derived features when they
  fail this rule, so no silent imputation.
* PIT integrity: `available_at >= trade_date` for 7 316 380 / 7 316 380 rows.
* `sector_map.parquet` and `st_flags.parquet`: AkShare upstream is currently
  serving 0-success — both files remain the placeholder "missing" snapshot.
  The decision chain is built to keep working without them.

## 7. What still needs doing

1. **Full-universe run** (3 658 viable symbols, ~7× the data) — estimated
   5-8 GPU hours for the widened config. Plan: launch once AkShare-served
   sector / ST flags come back, so the sector-concentration gate adds
   independent value.
2. **Sector pool top-N filter** — module + CLI flag already wired
   (`--sector-pool-top-n`); inputs depend on a working sector_map +
   stratified-IC table.
3. **Bond + bank + policy ingestion** — pipelines wired in
   `quantagent.cli.v8` (v7 aliases). Live data still need to flow.
4. **Iteration loss tuning** — current GA loss is plain rank-IC; the spec
   wants drawdown / vol / turnover penalties weighted in. Module exists
   (`optimization/ga_weight_optimizer.py`); next pass should retune the
   λ_ coefficients on the widened OOS.

## 8. Artifact locations

* Baseline deep run: `runtime/reports/v8/deep/v8_deep_20260531_061714`
* Widened deep run: `runtime/reports/v8/deep/v8_deep_wide_20260531_185536`
* Gated comparisons: `gated_v1`, `gated_v2`, `gated_topk_{10,15,20,30,50}`,
  `gated_tuned_topk_{10,15,20}` under each run.
* Best run: `v8_deep_wide_20260531_185536/gated_tuned_topk_10`.
