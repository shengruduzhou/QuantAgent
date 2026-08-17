# Interior-bar NAV misalignment — diagnosis, attempted fix, and why it was reverted

Status: **OPEN**. Diagnosed and reproduced; a fix was attempted and reverted
because it broke a stronger invariant. This document exists so the next attempt
starts from the evidence rather than rediscovering it.

## The defect

`src/quantagent/backtest/engine.py` executes a next-day fill at `dates[i+1]` but
marks and stamps NAV under `dates[i]` at `close(dates[i])`. The NAV recorded for
day *t* therefore reflects a book that includes a position acquired at *t+1*.

Measured on a tape flat at 10.00 except one gap bar:

| metric | engine | honest |
|---|---|---|
| max drawdown | −28.56% | −0.046% |
| annual vol | 3.87 | 0.0033 |
| **Sharpe** | **+1.4768** | **−7.10** |

A losing strategy reports a positive Sharpe. Total return can self-correct
because the phantom mark unwinds on the following bar; the *path* does not, so
drawdown, volatility, Sharpe and Calmar are all affected.

The terminal-bar case of this same mechanism WAS fixed (commit `6027705`) and is
covered by `tests/backtest/test_terminal_bar_no_phantom_fill.py`. Only the
interior-bar case remains.

## Attempted fix and why it was reverted

The change moved mark-to-market to before the trading block, so each bar is
valued on the book held during it. `tests/backtest`, `tests/test_golden_backtest_scenarios.py`
and most of `tests/domain` passed, but two composite-replay tests failed:

```
('fast_backtest_native', 'fast_backtest_replay', 'nav', 1000492.89997, 1000480.446031)
```

A 12.45 divergence (~1.2 bps) between the native NAV and the NAV reconstructed
from the canonical ledger. That is not a rounding artefact — it means the two
paths disagree about *which* fills belong to the final bar's book.

Reverted rather than shipped. A core-engine semantic change that breaks the
ledger-replay invariant is exactly the class of change this repo has been burned
by, and the invariant is more valuable than a partial fix.

## The question the next attempt must answer first

Does `NAV(t)` include the fill executed *at* t (scheduled from t−1's signal)?

Both are defensible conventions, but the engine and the replay must agree, and
right now it is not established which one the replay assumes. Resolve that
first, in `domain/ledger.py` and the composite replay path, and only then move
the marking. Changing the engine without pinning the convention will keep
producing this divergence.

## Guard rails for whoever picks this up

* `tests/domain/test_composite_replay.py` is the binding constraint. It caught
  this immediately; do not weaken it to make a fix pass.
* The golden scenarios **cannot see this defect by construction**:
  `:179` asserts only `trades.iloc[0]` under a repeated signal, and `:268`
  builds every bar with `open == close` — the single case where the mismatch
  vanishes. Passing them is not evidence.
* Reproduce with `scratchpad/audit_r20/exp_engine_nav.py` (scenario B), which
  exercises the interior-bar path on a gap tape.
* Related and also open: the engine marks positions with
  `daily_prices["close"].get(sym, 0.0)`, so a symbol missing a close is valued
  at ZERO rather than being excluded. That manufactures losses which look real
  and should be fixed in the same pass.
