import numpy as np
import pandas as pd

from quantagent.factors.alpha181 import (
    ALPHA181_CICC_COUNT,
    ALPHA181_CICC_NAME_MAP,
    ALPHA181_NAMES,
    alpha181_source_map,
    compute_alpha181,
)
from quantagent.factors.cicc_ashare80 import cicc_ashare80_names


def _ohlcv(days: int = 160, symbols: tuple[str, ...] = ("A", "B", "C", "D")) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-02", periods=days, freq="B")
    rows = []
    for j, symbol in enumerate(symbols):
        close = 20.0 + np.cumsum(rng.normal(0.02 + j * 0.001, 0.25, len(dates)))
        volume = 1_000_000 + np.abs(rng.normal(0, 50_000, len(dates))).cumsum() + j * 20_000
        for i, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close[i] * 0.995,
                    "high": close[i] * 1.012,
                    "low": close[i] * 0.985,
                    "close": close[i],
                    "volume": max(volume[i], 1_000.0),
                    "amount": max(volume[i], 1_000.0) * close[i],
                }
            )
    return pd.DataFrame(rows)


def test_alpha181_constants_are_consistent():
    assert len(ALPHA181_NAMES) == 181
    assert ALPHA181_CICC_COUNT == 80
    assert len(ALPHA181_CICC_NAME_MAP) == 80
    # alpha102..alpha181 = 80 names, exactly the image of the CICC map.
    assert set(ALPHA181_CICC_NAME_MAP.values()) == {f"alpha{i:03d}" for i in range(102, 182)}
    # Source names are exactly the CICC ashare80 universe (preserves order).
    assert tuple(ALPHA181_CICC_NAME_MAP.keys()) == cicc_ashare80_names()


def test_compute_alpha181_returns_all_181_factors():
    frame = _ohlcv()
    factors = compute_alpha181(frame)
    assert {"trade_date", "symbol", "factor_name", "factor_value"}.issubset(factors.columns)
    produced = set(factors["factor_name"].unique())
    # Every fixed-base name must appear at least once.
    assert produced == set(ALPHA181_NAMES), (
        f"missing {set(ALPHA181_NAMES) - produced}, extra {produced - set(ALPHA181_NAMES)}"
    )


def test_alpha181_source_map_covers_all_names():
    mapping = alpha181_source_map()
    assert set(mapping.keys()) == set(ALPHA181_NAMES)
    # 1..101 routed to alpha101.* and 102..181 routed to cicc_ashare80.*.
    for i in range(1, 102):
        assert mapping[f"alpha{i:03d}"].startswith("alpha101.")
    for i in range(102, 182):
        assert mapping[f"alpha{i:03d}"].startswith("cicc_ashare80.")


def test_compute_alpha181_respects_name_subset():
    frame = _ohlcv()
    subset = ["alpha001", "alpha050", "alpha101", "alpha102", "alpha150", "alpha181"]
    factors = compute_alpha181(frame, names=subset)
    assert set(factors["factor_name"].unique()) == set(subset)


def test_compute_alpha181_handles_missing_synth_path(tmp_path):
    frame = _ohlcv()
    missing = tmp_path / "does_not_exist.json"
    factors = compute_alpha181(frame, synthesized_definitions_path=missing)
    # 181 base factors still produced even with absent synth definitions file.
    assert set(factors["factor_name"].unique()) == set(ALPHA181_NAMES)
