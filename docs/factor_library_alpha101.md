# Alpha101 Daily Factor Library

This library implements `alpha001` to `alpha030` as point-in-time daily A-share approximations using OHLCV, amount, VWAP, returns, and rolling average volume. `adv` fields are computed from trailing volume. No future bars are used.

| Factor | Formula approximation | Required fields | Intuition | Horizon | Direction | A-share caveats |
| --- | --- | --- | --- | --- | --- | --- |
| alpha001 | `-rank(ts_argmax(where(ret<0,std(ret,20),close)^2,5))` | OHLCV, amount | Downside-volatility reversal | 5d | + | Limit-down names may be hard to exit |
| alpha002 | `-corr(rank(delta(log(volume),2)),rank((close-open)/open),6)` | OHLCV | Volume shock versus intraday move | 5d | + | Volume spikes around halts need masking |
| alpha003 | `-corr(rank(open),rank(volume),10)` | OHLCV | Crowded high-open names reverse | 5d | + | ST names should be excluded |
| alpha004 | `-ts_rank(rank(low),9)` | OHLCV | Low-price rank reversal | 5d | + | Board-specific limits affect lows |
| alpha005 | `rank(open-mean(vwap,10))*-rank(abs(close-vwap))` | OHLCV, amount | VWAP location reversal | 5d | + | VWAP is daily amount/volume fallback |
| alpha006 | `-corr(open,volume,10)` | OHLCV | Price-volume crowding reversal | 5d | + | Suspensions distort rolling windows |
| alpha007 | `if volume>adv20 then -ts_rank(abs(delta(close,7)),60)*sign(delta(close,7)) else -1` | OHLCV | Volume-confirmed reversal | 5d | + | Uses trailing volume as `adv20` |
| alpha008 | `-rank(sum(open,5)*sum(ret,5)-delay(...,10))` | OHLCV | Lagged open-return interaction | 5d | + | Needs enough history |
| alpha009 | Conditional one-day close delta reversal | OHLCV | Reversal only outside short trend filters | 5d | + | Price limits can cap deltas |
| alpha010 | `rank(alpha009_raw)` | OHLCV | Cross-sectional alpha009 | 5d | + | Same as alpha009 |
| alpha011 | `(rank(max(vwap-close,3))+rank(min(vwap-close,3)))*rank(delta(volume,3))` | OHLCV, amount | VWAP dislocation plus volume change | 5d | + | Daily VWAP approximation |
| alpha012 | `-sign(delta(volume,1))*delta(close,1)` | OHLCV | Volume direction reversal | 5d | + | Volume zero requires suspension mask |
| alpha013 | `-rank(cov(rank(close),rank(volume),5))` | OHLCV | Price-volume covariance reversal | 5d | + | Needs cross-section per date |
| alpha014 | `-rank(delta(ret,3))*corr(open,volume,10)` | OHLCV | Return acceleration and crowding | 5d | + | Robust to missing market cap |
| alpha015 | `-sum(rank(corr(rank(high),rank(volume),3)),3)` | OHLCV | High-volume rank correlation reversal | 5d | + | Short windows are noisy |
| alpha016 | `-rank(cov(rank(high),rank(volume),5))` | OHLCV | High-price volume covariance reversal | 5d | + | Price-limit highs can bunch |
| alpha017 | Composite of close rank, second derivative, and volume intensity | OHLCV | Acceleration plus liquidity | 5d | + | Uses `volume/adv20` |
| alpha018 | `-rank(std(abs(close-open),5)+close-open+corr(close,open,10))` | OHLCV | Intraday dispersion reversal | 5d | + | Auction effects can dominate open |
| alpha019 | `-sign(close-delay(close,7)+delta(close,7))*(1+rank(sum(ret,60)))` | OHLCV | Trend sign reversal | 5d | + | Medium-term window shortened from original |
| alpha020 | `-rank(open-delay(high,1))*rank(open-delay(close,1))*rank(open-delay(low,1))` | OHLCV | Gap reversal | 5d | + | Opens can be limit-locked |
| alpha021 | Mean-reversion state rule using mean2, mean8, std8, and volume/adv20 | OHLCV | Trend regime classifier | 5d | + | Binary output |
| alpha022 | `-delta(corr(high,volume,5),5)*rank(std(close,20))` | OHLCV | Correlation deterioration with volatility | 5d | + | Volatility inflated around limit moves |
| alpha023 | `if mean(high,20)<high then -delta(high,2) else 0` | OHLCV | Breakout reversal | 5d | + | Breakouts can persist in policy themes |
| alpha024 | Slow trend filter with `-delta(close,3)` or range pullback | OHLCV | Slow trend plus reversal | 5d | + | Uses 20-day proxy for long original window |
| alpha025 | `rank((-ret*adv20*vwap)*(high-close))` | OHLCV, amount | Liquidity-weighted pressure reversal | 5d | + | Uses daily VWAP |
| alpha026 | `-max(corr(ts_rank(volume,5),ts_rank(high,5),5),3)` | OHLCV | High-volume rank crowding | 5d | + | Needs intraday extension for microstructure |
| alpha027 | `if rank(mean(corr(rank(volume),rank(vwap),6),2))>0.5 then -1 else 1` | OHLCV, amount | VWAP-volume state | 5d | + | Binary output |
| alpha028 | `scale(corr(adv20,low,5)+(high+low)/2-close)` | OHLCV | Liquidity-low relation and price location | 5d | + | Cross-sectional scaling is date-local |
| alpha029 | `rank(-delta(close,5))*rank(volume/adv20)` | OHLCV | Reversal with volume intensity | 5d | + | Approximation of a complex original formula |
| alpha030 | `(1-rank(sign ret persistence))*sum(volume,5)/sum(volume,20)` | OHLCV | Return persistence and volume concentration | 5d | + | Volume sums should exclude halted days |

All factors should be evaluated with IC, Rank IC, ICIR, group returns, turnover, decay, capacity, neutralization, and transaction-cost-adjusted performance before use.

