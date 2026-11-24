# ENG_RISK30 pre-registration — exposure policy (drawdown ladder + vol targeting)

Written and committed BEFORE any real-data backtest of any exposure policy arm
(the commit timestamp of this file is the evidence). Nothing below is tuned
after results are seen. Both parameter sets are reported whichever way they go.

## Hypothesis

H1: applying `quantagent.risk.exposure_policy` to the gross of a long-only
weekly top-50 A-share book reduces the all-time-peak maximum drawdown by a
material amount without lowering the Calmar ratio and without a material loss
of Sharpe. Prior (Kaminski & Lo 2014): under a random walk any stop-loss rule
lowers expected return; value arises only with momentum / volatility
clustering. Prior (Cederburg et al. 2020): vol management does not
systematically beat the unmanaged book in real-time direct comparisons. So the
decision is judged on risk, and H1 may well be rejected.

## Strategies (exactly as in round 29; predictions rebuilt from the same scripts)

- **S1 canonical** — transparent weekly low-vol + 1-month reversal top-50:
  `score = 0.5·pct_rank(−vol_20d) + 0.5·pct_rank(−ret_20d)`, signal builder
  `round29/ENG_DATA/make_lowvol_rev_signal.py` (copied unchanged to the role
  repro dir, output path argument only), r29 execution panel. Evaluator:
  `scripts/baseline_protocol.py --panel runtime/data/gold/full_universe_r29/execution_panel.parquet
  --top-k 50 --start 2017-01-03 --end 2025-08-21 --slippage-bps 8 --variants C_flags_eligible_delay1`.
- **S2 R7 low-vol** — weekly (W-FRI) `score = −vol_60d`, universe
  `turnover_20d ≥ 20M CNY`, builder `round29/R7_expert/repro/build_predictions.py`
  part (b) (copied; only the output directory changed). Evaluator as above with
  `--start 2018-01-01 --end 2025-08-20`.

Both windows end on or before 2025-08-29 (configs/quarantined_windows.json is
enforced by the evaluator itself, fail closed).

## Arms

| arm | `--exposure-policy` | role |
|---|---|---|
| A0 | (flag absent) | baseline = canonical evaluator, unchanged |
| AI | `identity` | wiring inertness check (multiplier always 1.0); NOT a trial |
| P1 | `vt20` | trial 1: vol target only |
| P2 | `vt20_ddl` | trial 2: vol target + drawdown ladder with re-anchor |

Parameter sets (fixed a priori from the literature / common practice; the
presets in `src/quantagent/risk/exposure_policy.py` at the commit that adds
this file are binding, their `policy_sha256` is recorded in every output):

- **vt20**: realised vol of unit-exposure (capital-at-risk) daily returns,
  EWMA zero-mean half-life 20 sessions (Harvey et al. 2018), trailing window
  126 sessions, ≥ 20 measured observations else declared warm-up multiplier
  0.5, target 20% annualised (≈ long-run A-share broad-market vol; NOT fitted to
  either book), annualisation 244, cap 1.0 (cash account, no leverage),
  multiplier floored to a 0.05 grid.
- **vt20_ddl**: vt20 × drawdown ladder on the account's all-time peak (initial
  capital included): drawdown ≥ 8% → 0.75, ≥ 12% → 0.50, ≥ 16% → 0.25 (all
  strictly below the paper venue's hard 20% latch, which stays as the backstop);
  step down immediately; step up one tier only after drawdown ≤ threshold − 4pp
  AND ≥ 10 sessions in the tier; re-anchor (anchor := current NAV, up one
  tier) after 63 sessions in a reduced tier with NAV strictly above its value 63
  sessions earlier.

Mechanics (both trials): the multiplier is decided at each signal date `t` from
account observations with date ≤ `t` (the NAV at the close of `t`), scales that
signal's target weights, and the strict simulator executes them at the next
session close. The ladder/vol state advances on every session close.

Trials counted: 2 parameter sets × 2 strategies = 4 policy backtests (plus 2
baselines and 2 inertness checks, which are not trials). No other parameter set
will be run on real data in this round.

## Metrics (per arm, from the evaluator's own outputs)

CAGR (`ann`), max drawdown vs all-time peak including initial capital
(`maxDD`, strict_v8 metric), Calmar = CAGR / MaxDD, Sharpe, PSR vs 0 and the
stationary-bootstrap 95% CI already published by the evaluator, annualised
volatility, turnover, total / slippage cost (CNY), time in market (mean of the
daily invested fraction = positions value / NAV; mean applied multiplier on
signal dates), number of re-anchors, first session on which the all-time-peak
drawdown reaches 20% (where the paper venue's hard latch would fire) and the
share of sessions at or beyond it. Paired comparison vs A0: daily returns of
the two arms on their common sessions, Politis-Romano stationary bootstrap
(Politis-White block length of the paired difference series, 2000 resamples,
seed 0, same indices for both arms; Ledoit-Wolf 2008 logic), 95% CI of
ΔSharpe = Sharpe(P) − Sharpe(A0).

## Validity conditions (any failure voids the run, nothing is reported as a result)

V1 every arm run twice; `nav.csv` sha256 identical between runs.
V2 AI `nav.csv` sha256 == A0 `nav.csv` sha256 for both strategies (wiring inert at 1.0).
V3 A0 run on this branch has the same `nav.csv` sha256 as A0 run on `main`
   (2c4e296) code — the canonical evaluator is bit-identical with the flag off.

## Decision rule

A policy P is adopted as the paper-account default (DailyPaperLoopConfig
default) only if, for BOTH S1 and S2:

1. MaxDD(P) ≤ MaxDD(A0) − 0.05 (≥ 5 percentage points less all-time-peak drawdown), AND
2. Calmar(P) ≥ Calmar(A0), AND
3. the lower bound of the paired 95% CI of ΔSharpe > −0.10 (no material Sharpe loss).

P2 (`vt20_ddl`, the deliverable) is evaluated first; if it passes it is
adopted. Otherwise P1 is adopted if it passes. Otherwise the policy stays
opt-in (flag / config only) and the default exposure is unchanged. The
canonical evaluator's default (`baseline_protocol.py` without the flag) is
never changed by this decision. Results are reported as measured whichever way
they go; no parameter is changed after the first real-data run.
