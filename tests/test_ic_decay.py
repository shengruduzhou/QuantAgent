"""Tests for IC-decay diagnostic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantagent.training.diagnostics import compute_factor_ic_decay, render_ic_decay_heatmap


def _synthetic_panel(n_symbols: int = 30, n_months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for month in range(n_months):
        base = pd.Timestamp("2024-01-01") + pd.DateOffset(months=month)
        for day_offset in range(20):  # ~20 trading days/month, well above min_monthly_observations
            date = base + pd.Timedelta(days=day_offset)
            shock = rng.normal(0.0, 0.01, size=n_symbols)
            # alpha_real has stable IC ~ +0.5
            alpha_real = shock + rng.normal(0.0, 0.002, size=n_symbols)
            # alpha_noise has IC ~ 0
            alpha_noise = rng.normal(0.0, 1.0, size=n_symbols)
            for s in range(n_symbols):
                rows.append({
                    "symbol": f"S{s:03d}",
                    "trade_date": date,
                    "forward_return_5d": float(shock[s]),
                    "alpha_real": float(alpha_real[s]),
                    "alpha_noise": float(alpha_noise[s]),
                })
    return pd.DataFrame(rows)


def test_compute_factor_ic_decay_shape_and_signal():
    panel = _synthetic_panel()
    decay = compute_factor_ic_decay(
        panel,
        feature_columns=["alpha_real", "alpha_noise"],
        label_column="forward_return_5d",
        min_monthly_observations=20,
    )
    assert not decay.empty
    assert "alpha_real" in decay.index
    assert "alpha_noise" in decay.index
    # alpha_real should have consistently high positive IC.
    assert decay.loc["alpha_real"].mean() > 0.4
    # alpha_noise should hover near zero.
    assert abs(decay.loc["alpha_noise"].mean()) < 0.20


def test_render_ic_decay_heatmap_writes_artifacts(tmp_path):
    panel = _synthetic_panel(n_symbols=20, n_months=6)
    decay = compute_factor_ic_decay(
        panel, ["alpha_real", "alpha_noise"], "forward_return_5d",
        min_monthly_observations=20,
    )
    out = tmp_path / "diagnostics" / "ic_decay"
    paths = render_ic_decay_heatmap(decay, out)
    assert "json" in paths
    json_path = out.with_suffix(".json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert set(payload["factors"]) == {"alpha_real", "alpha_noise"}
    assert len(payload["months"]) == decay.shape[1]


def test_compute_factor_ic_decay_handles_missing_label():
    panel = pd.DataFrame({"symbol": ["A"], "trade_date": ["2024-01-02"], "alpha_real": [0.1]})
    import pytest
    with pytest.raises(KeyError):
        compute_factor_ic_decay(panel, ["alpha_real"], "forward_return_5d")


def test_compute_factor_ic_decay_empty_input():
    assert compute_factor_ic_decay(pd.DataFrame(), [], "forward_return_5d").empty
