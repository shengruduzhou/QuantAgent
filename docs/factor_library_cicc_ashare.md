# CICC-Style A-Share High-Frequency Factor Library

The module implements daily-compatible approximations first and exposes minute/level2 hooks through the intraday schema. If intraday bars are unavailable, intraday-only factors are returned in the `unavailable` field instead of raising.

## Categories

1. `momentum_reversal`: close-to-VWAP proxy for late-day return, top-volume-bar return when minute bars exist.
2. `volatility`: intraday skew and kurtosis when minute bars exist.
3. `high_order_shape`: intraday return shape placeholders from skew/kurtosis.
4. `liquidity`: daily Amihud fallback and minute Amihud when available.
5. `price_volume_correlation`: close-volume correlation and lead-lag price-volume correlation.
6. `chip_distribution`: turnover concentration.
7. `crowding`: FFT volume concentration when intraday volume exists.
8. `money_flow`: amount z-score, money-flow strength, opening and closing flow ratios.

## A-Share Assumptions

Daily inputs must include `trade_date, symbol, open, high, low, close, volume, amount`. Intraday inputs add `datetime`. Suspensions, ST flags, T+1, lot size, and price-limit rules should be applied in the downstream tradability and backtest layers.

## Daily Fallback Logic

- `last_30min_return` falls back to `close / daily_vwap - 1`.
- `daily_amihud` uses absolute daily return divided by RMB amount.
- `opening_flow_ratio` uses open gap times amount intensity.
- `closing_flow_ratio` uses close-to-VWAP displacement times amount intensity.
- `top_volume_bar_return`, `intraday_skew`, `intraday_kurtosis`, `amihud_1min`, and `crowding_fft_ratio` are marked unavailable without intraday bars.

## Future Extension

Minute and level2 adapters should preserve the same long-form output: `trade_date, symbol, factor_name, factor_value`. Level2 fields can add order-book imbalance, large-order sweep pressure, cancel-rate crowding, and queue-position liquidity without changing the registry contract.

