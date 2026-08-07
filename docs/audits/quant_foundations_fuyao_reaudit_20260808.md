# Quant foundations + Fuyao functional re-audit — 2026-08-08

## Scope

This audit covers QuantAgent backtesting, risk controls, nonlinear factor mixing, formulaic-alpha safety, Qlib governance, research-to-production promotion, and the 16 public Fuyao/HiThink research examples.

Primary research references: Sharpe (1964); Black & Scholes (1973); Fama & French (1992); Carhart (1997); Kakushadze (2016) *101 Formulaic Alphas*; Gu, Kelly & Xiu (2020); Yang et al. (2020) Qlib plus current Qlib docs; Bailey & López de Prado's probabilistic/deflated Sharpe work.

## Preserved because already correct

- purged expanding walk-forward folds and trailing holdout excluded from selection;
- factor interactions selected inside training folds;
- explicit linear/additive control beside interactions, GBM and OOF stacking;
- paired OOS Rank-IC/HAC comparisons;
- existing CAPM → FF3 → Carhart4 attribution with PIT caveats;
- full Alpha101 implementation and governed Qlib bridge;
- selection-governance PBO/DSR/SPA framework.

## Defects fixed

1. **PSR/DSR kurtosis convention.** `pandas.Series.kurt()` is excess kurtosis. The Bailey/López-de-Prado denominator is expressed with Pearson kurtosis as `(gamma4 - 1)/4`; in pandas convention the coefficient is `(excess_kurtosis + 2)/4`. The old implementation used `excess_kurtosis/4` and could overstate promotion probabilities.
2. **Nonlinear promotion.** Raw model comparison no longer suffices for production promotion. The canonical final gate requires PBO ≤ 0.25, DSR probability ≥ 0.95 and SPA p-value ≤ 0.05 after a nonlinear champion is frozen.
3. **Benchmark contract.** The governed `search-factor-fusion` CLI now requires both an explicit benchmark file and benchmark symbol. Equal-weight fallback is exploratory only.
4. **Formula-expression leakage.** Qlib-style negative `Ref/Shift` offsets and future/label tokens are rejected from feature expressions.
5. **Black-Scholes.** A tested European option price/Greeks/implied-volatility primitive is added for derivatives/convexity risk. It is deliberately not injected into cash-equity alpha blending.

## Fuyao workbench depth

The previous 16/16 mapping described navigation coverage; it did not mean every example had a dedicated, computed workbench. This change adds a real `/market-playbooks` runtime and `/api/market/playbooks/{id}` service for all 01–16 examples. Native surfaces remain native; computed surfaces carry source endpoints, PIT/T+1 assumptions and no-synthetic-data failure states.

Notable playbooks:

- 02 financial health: growth, margin, cash conversion, leverage, `report_date_ms` PIT key;
- 05 watchlist anomalies: snapshot + official event/reason data;
- 06 local market DB: validated full-market parquet, breadth/trend/liquidity, `raw_unadjusted` label until reliable as-of adjustment materialisation;
- 09 industry strength: 5d/20d strength, acceleration and current breadth;
- 10 cash-flow quality: conversion/FCF proxy/completeness with disclosure dates;
- 11 attention-price: same-day stock/benchmark/rank alignment and Spearman, no causal claim;
- 13/14/15: explicit T+1/cost-aware backtests/experiments;
- 16: current-constituent constrained dragon-tiger capital-flow topology.

## Fund / ETF / REITs

`/funds` and `/api/market/funds/overview` aggregate fund profile, disclosed holdings, NAV, interval returns and holder structure. Exchange-fund market snapshot/history are requested only under the upstream ETF-market contract. Holdings and holders are labelled periodic disclosures, never realtime positions.

## Reports

The Evidence Center retains JSON export and now also emits a self-contained offline HTML report. Computed playbooks and fund research also export single-file HTML. API credentials are never embedded in browser output.

## Upstream limits kept fail-closed

- current index constituents are not historical constituents;
- incomplete/coming-soon stock-basic, historical constituent/weight and stock-to-THS membership capabilities are not fabricated;
- a complete REIT universe is not claimed where the current meta enumeration cannot prove it;
- actual live conclusions require runtime data and `HITHINK_FINANCE_API_KEY`; unit tests are not presented as live-market validation.
