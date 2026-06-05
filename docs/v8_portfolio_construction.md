# v8 Portfolio Construction — turnover control & the hedge gap

Authoritative record of the construction analysis behind
`quantagent.portfolio.alpha_portfolio` and the
`build-alpha-portfolio-v8` CLI. Consolidates what used to be scattered
across the `v8_*_results.md` notes.

## Problem

The v8 deep pipeline (`train-v8-deep`) built target weights as a naive
`top_k` equal weight **re-emitted every trading day**. The model signal is
strong — per-date `rank_IC ≈ 0.12` on the 20-day horizon, stable across
regimes — but the executed book *under*performed a costless equal-weight
all-A benchmark by ~5 %/yr. The leak is construction, not signal.

## Finding 1 — turnover is the dominant cost leak

A 20-day-horizon signal rebalanced daily churns 40–80 % of the book per
day; realistic A-share costs then erase the edge. Frictionless net-of-cost
grid (v8 mid_5d_30d OOS, cost 15 bps/side):

| construction | BULL ann | BULL excess | BEAR ann | BEAR excess | turnover |
|---|---|---|---|---|---|
| equal-weight all-A (bench) | +47.9 % | — | +4.5 % | — | 0.00 |
| top decile, **daily** rebal | +50.4 % | +2.5 % | −10.3 % | −14.8 % | 0.4–0.8 |
| top decile, **hold 20d** | +59.1 % | **+11.3 %** | +18.8 % | **+14.3 %** | 0.04 |
| top decile, smoothed | +61.4 % | +13.6 % | +14.5 % | +10.0 % | 0.06 |
| **market-neutral, hold 20d** | +6.9 % | −41 % | **+20.7 %** | **+16.1 %** | 0.04 |

Holding the signal for its horizon (emit a target only every
`rebalance_interval` days, simulator holds between) collapses turnover to
~4 % and recovers **+11 to +14 %/yr excess in both regimes**. This is the
single biggest lever and is implemented by `AlphaPortfolioConfig`.

## Finding 2 — the long book wins bulls, the neutral book wins bears

Directional long decile dominates trending markets; the market-neutral
(+top / −bottom) book dominates choppy/bear markets (frictionless Sharpe
~3, max-DD ~4 %). The institutional structure is a **regime-aware blend**:
turnover-controlled long core, tilting to hedged/neutral as the regime
turns bear (compose via `gross_scale` / the existing regime detector).

## Finding 3 — the execution engine is long-only (the blocking gap)

Validated through the real `strict_v8` simulator at institutional capital
(50 M, so lot-sizing is not the binding constraint):

| config | ann | Sharpe | maxDD | turnover | excess |
|---|---|---|---|---|---|
| BULL decile hold-20 long | +43.5 % | 1.54 | 17.3 % | 0.003 | −2.5 % |
| BEAR decile hold-20 long | +4.4 % | 0.32 | 21.5 % | 0.003 | +0.0 % |
| BEAR "market-neutral" hold-20 | +7.1 % | 0.46 | 17.5 % | 0.004 | +2.7 % |

Turnover control is confirmed robust (0.27 → 0.003; the bull long book now
matches benchmark beta cleanly instead of bleeding cost). **But the
market-neutral max-DD is 17.5 %, not the frictionless ~4 %**: a negative
target weight with no existing position routes to a sell-what-you-don't-own
→ skipped, so the short leg never executes. A-share single-stock shorting
is infeasible; realizing the drawdown-minimizing neutral alpha requires an
**index-futures / ETF hedge leg** (`portfolio.hedge_decision_engine`) wired
into the strict backtest. That is the next required step.

## Finding 4 — regime timing helps; index-hedge reveals the real ceiling

`build-alpha-portfolio-v8` now carries `--regime` (per-date gross scaling
from the market regime detector) and `--hedge-ratio` (short index-future
overlay, `portfolio/index_hedge.py`). Bear OOS (decile hold-20, 50 M):

| overlay | ann | Sharpe | maxDD | excess vs EW |
|---|---|---|---|---|
| none | +4.4 % | 0.32 | 21.5 % | +0.0 % |
| **--regime** | **+8.5 %** | 0.56 | 15.7 % | **+4.1 %** |
| --hedge-ratio 1.0 | −0.5 % | −0.11 | **7.6 %** | — |

* **Regime scaling is a real win**: cutting exposure in the worst bear
  stretches beat the benchmark by +4.1 % and lowered max-DD. Recommended
  **default-on** in the production target-weight builder.
* **The index hedge cuts drawdown hard** (21.5 % → 7.6 %) but the hedged
  return collapses to ~0. Reason: hedging shorts the *equal-weight index*,
  and the long-only, liquidity/ST-gated decile book's realised total return
  only *matches* that index (it does not beat it) — so removing the market
  removes nearly all the return.

### The real ceiling (now triply confirmed)

The cross-sectional alpha is genuine in IC / decile-spread terms
(rank_IC 0.12; frictionless decile-spread +11–16 %), but it is **not
realised as benchmark-beating total return** by the long-only book, because
the liquidity / ST gates and round-lot constraints strip exactly the
micro-cap names that drive the equal-weight all-A basket. To monetise it you
must pick one:

1. **True single-name long-short** — infeasible for A-share retail.
2. **Go down-cap / relax the liquidity floor** — captures what EW captures,
   at the cost of capacity and tail risk (a strategic choice).
3. **Accept beta-matching + regime timing** — the realistic edge the engine
   can execute today (bull: match EW at Sharpe 1.5 with turnover 0.003;
   bear: +4 % excess via regime).

## Capacity note

A ~290-name decile needs enough capital that each name clears one round lot
(`lot_size=100`). At 1 M the book is forced partly into cash
(under-invested, artificially low drawdown); ≥ ~30–50 M deploys fully.
Narrower books fit small capital but churn more and carry more
idiosyncratic noise.
