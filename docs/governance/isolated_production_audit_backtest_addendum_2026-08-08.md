# Isolated production audit — backtest addendum

## BT-003 — first economic return was omitted from StrictBacktest metrics

**Author role:** `backtest_expert`  
**Severity:** P0/P1 evaluator integrity  
**Base evidence:** `src/quantagent/backtest/ashare_execution_simulator.py` and
`src/quantagent/backtest/strict_v8.py`.

### Finding

The A-share simulator stores the first NAV observation **after** the first day's
rebalance. The legacy strict metrics then calculated:

`total_return = last_post_trade_nav / first_post_trade_nav - 1`

and `pct_change()` dropped the first observation. As a result, first-day
slippage, explicit fees and mark-to-market PnL were normalised away. A one-day
simulation could report zero total return even when the first trade consumed
cash through execution costs.

### Independent cross-review

- `testing_expert`: **APPROVE**. Golden reproduction: one NAV observation at
  990,000 from 1,000,000 initial capital must be -1%, not 0%.
- `risk_expert`: **APPROVE**. Omitting the initial loss also understates the
  drawdown path.
- `quant_expert_tester`: **APPROVE** with a compatibility requirement: historical
  metrics generated under the old semantics must not silently satisfy future
  production model trust.

### Main-role repair

Accepted and implemented as metric semantics
`strict_v8_nav_v2_initial_cash`:

1. `run_strict_backtest_v8` passes configured `initial_cash` into metric
   computation.
2. First daily return is
   `first_post_trade_nav / configured_initial_cash - 1`.
3. Total return is `last_nav / configured_initial_cash - 1`.
4. Drawdown path prepends configured initial capital.
5. `daily_pnl` contains the real first economic return instead of a synthetic
   zero.
6. Metrics and artifact config carry the semantics version.
7. `evaluate_live_model_trust` requires exactly this evaluator semantics before
   a certificate can be live-accepted.
8. The current blocked model manifest is explicitly marked as legacy evaluator
   evidence and requires regeneration.

### Acceptance tests

- `tests/backtest/test_strict_v8_initial_nav.py`
- `tests/execution/test_live_model_trust.py`

### Governance consequence

Historical StrictBacktest metrics that predate
`strict_v8_nav_v2_initial_cash` are **legacy evidence**. They may remain useful
for forensic comparison, but cannot be mixed with the corrected evaluator as if
they were produced under identical definitions, and they cannot satisfy the
live-model-trust evaluator-semantics gate.
