# Interior-bar NAV misalignment — diagnosis, attempted fix, and why it was reverted

Status: **CLOSED** (2026-08-18, round 21 — see the RESOLVED section at the end). Diagnosed and reproduced; a fix was attempted and reverted
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

---

## RESOLVED — 2026-08-18 (round 21)

Status is now **CLOSED**. The fix shipped with
`tests/backtest/test_interior_bar_nav_clock.py` (8 tests, 4 of which fail
against the pre-fix engine — verified by stashing the engine change).

### The convention, now pinned

> **`NAV(t)` contains exactly the fills whose `fill_date <= t`.**

That single rule resolves the question this document said had to be answered
first, and it resolves it the same way for both fill policies:

| policy | fill_date | belongs in NAV(t)? | where the mark happens |
|---|---|---|---|
| `next_day_fill=True` | `dates[i+1]` | no | **before** the trading block |
| `next_day_fill=False` | `dates[i]` | yes | **after** the trading block |

### Why the previous attempt broke the replay

It moved the mark before the trading block **unconditionally**. The composite
ledger-replay path (`reconciliation/composite.py:run_fast_path`) runs with
`next_day_fill=False`, where the fill genuinely does belong in `NAV(t)`.
Marking before trading there drops the same-bar fill from the final bar's book,
which is precisely the 12.45 (~1.2 bps) native-vs-replay divergence recorded
above. The convention was never wrong for that path — applying one policy's
answer to the other path was.

Making the mark conditional on the fill policy keeps both paths on the same
rule. `tests/domain/test_composite_replay.py` passes unchanged; it was not
weakened.

### What the numbers actually were

The earlier framing ("Sharpe +1.4768 where the honest answer is −7.10") compared
two *different tapes* — a gap tape against a flat `open == close` tape. The
accurate statement is narrower and still damning: on the gap tape the entire
11.05% move was stamped on the bar **before** the position existed. After the
fix the same tape puts that move on the bar where the shares were bought at
9.00 and marked at 10.00, which is a real intraday gain. Nothing is held during
the signal bar, so its NAV now sits exactly on the opening cash.

The consequence for reported statistics is unchanged: every path metric —
drawdown, volatility, Sharpe, Calmar — was previously computed from a portfolio
that was not held on the days it was measured.

### Also fixed in the same pass

`daily_prices["close"].get(sym, 0.0)` valued a held symbol with no close at
**zero**, manufacturing a loss that looks real while every accounting identity
still balances. Marking now excludes an unpriceable holding and records it in
`EventDrivenBacktester._unpriced_marks`, so the gap is visible instead of buried
inside NAV.

### Still open, related

* Limit-up/down flags are computed on `close` while fills execute at `open`
  (R1 F-03) — the judgement and the execution price are not the same
  observation, and a one-word board has no dedicated test.
* `BacktestConfig.cost.slippage_bps` (5.0) is still serialised into configs
  while `FillModelConfig.slippage_bps` (2.0) is what the engine applies
  (R1 F-04).
