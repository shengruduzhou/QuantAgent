from __future__ import annotations

import pandas as pd

from quantagent.risk.regime_family import RegimeFamilyConfig, compute_regime_family


def test_regime_family_emits_bull_neutral_bear():
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    for symbol in ["A", "B", "C"]:
        price = 10.0
        for i, date in enumerate(dates):
            if i < 30:
                drift = 0.004
            elif i < 60:
                drift = 0.0
            else:
                drift = -0.004
            price *= 1.0 + drift
            rows.append({"trade_date": date, "symbol": symbol, "close": price})
    panel = pd.DataFrame(rows)

    labels = compute_regime_family(
        panel,
        config=RegimeFamilyConfig(
            lookback_return_days=10,
            short_return_days=5,
            bull_return_threshold=0.01,
            bear_return_threshold=-0.01,
            bull_breadth_threshold=0.50,
            bear_breadth_threshold=0.60,
        ),
    )

    assert "bull" in set(labels)
    assert "neutral" in set(labels)
    assert "bear" in set(labels)
