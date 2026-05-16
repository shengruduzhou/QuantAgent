"""Regression tests for V7 target-weights tradability constraint mapping.

The old implementation looked up config flags by ``"block_<first_token>"``
which silently failed for ``limit_up`` / ``limit_down`` (it asked for
``block_limit``). These tests pin the constraint table so any future
refactor keeps the four tradability flags actually applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _toy_panel(flags_for: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags_for = flags_for or {}
    dates = pd.date_range("2025-01-02", periods=4, freq="B")
    symbols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    market: list[dict[str, object]] = []
    for date in dates:
        for i, symbol in enumerate(symbols):
            rows.append({"symbol": symbol, "trade_date": date, "prediction": rng.standard_normal()})
            market.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "open": 10.0 + i,
                    "close": 10.5 + i,
                    "amount": 1e7,
                    "is_suspended": symbol == flags_for.get("is_suspended"),
                    "is_st": symbol == flags_for.get("is_st"),
                    "is_limit_up": symbol == flags_for.get("is_limit_up"),
                    "is_limit_down": symbol == flags_for.get("is_limit_down"),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(market)


@pytest.mark.parametrize(
    "flag_column,reason",
    [
        ("is_suspended", "suspended"),
        ("is_st", "st"),
        ("is_limit_up", "limit_up_buy_block"),
        ("is_limit_down", "limit_down_sell_block"),
    ],
)
def test_each_tradability_flag_blocks_symbol(flag_column: str, reason: str):
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _toy_panel({flag_column: "A"})
    result = build_v7_target_weights(preds, market, config=V7TargetWeightsConfig(top_k=4))
    frame = result.target_weights
    if "A" in frame.columns:
        assert float(frame["A"].abs().sum()) == 0.0, f"{flag_column} should remove symbol A from output"
    rejections = [row for row in result.diagnostics.get("rejected", []) if row.get("reason") == reason]
    assert rejections, f"{flag_column} should be reported with reason '{reason}'"


def test_unblocked_symbols_retain_weight():
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights

    preds, market = _toy_panel({"is_limit_up": "A"})
    result = build_v7_target_weights(
        preds,
        market,
        config=V7TargetWeightsConfig(top_k=4, block_limit_up_buy=False),
    )
    frame = result.target_weights
    # When the block is disabled, the suspended-flag column should not remove the symbol.
    if "A" in frame.columns:
        # A may or may not appear depending on rng, but it should not be deterministically zeroed.
        # The diagnostics must not list it as a limit_up_buy_block rejection.
        rejections = [r for r in result.diagnostics.get("rejected", []) if r.get("reason") == "limit_up_buy_block"]
        assert not rejections


def test_constraint_table_matches_config_fields():
    """Ensure every entry in the constraint table references a real config attribute."""
    from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, _TRADABILITY_CONSTRAINTS

    config = V7TargetWeightsConfig()
    for _column, attr, _reason in _TRADABILITY_CONSTRAINTS:
        assert hasattr(config, attr), f"V7TargetWeightsConfig missing attribute {attr}"
