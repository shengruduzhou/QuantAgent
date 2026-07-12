"""Stage B: production blend materializer equivalence tests (synthetic data only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_production_composite import blend  # noqa: E402
import ensemble_weight_search as ews  # noqa: E402


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    symbols = [f"s{i}" for i in range(6)]
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({"trade_date": d, "symbol": s,
                         "short_5d_score": rng.normal(), "mid_5d_30d_score": rng.normal(),
                         "long_30d_120d_score": rng.normal()})
    return pd.DataFrame(rows)


def test_blend_matches_original_search_ranked_sleeves():
    frame = _synthetic_frame()
    weights = {"short_5d": 1.0, "mid_5d_30d": 1.0, "long_30d_120d": 0.0}
    ours = blend(frame, weights)
    ranked = ews._ranked_sleeves(frame)
    theirs = (1 * ranked["short_5d_score"] + 1 * ranked["mid_5d_30d_score"]
              + 0 * ranked["long_30d_120d_score"]).to_numpy()
    assert np.array_equal(ours["composite_score"].to_numpy(), theirs)


def test_blend_nonzero_long_weight_and_missing_column():
    frame = _synthetic_frame()
    full = blend(frame, {"short_5d": 2.0, "mid_5d_30d": 1.0, "long_30d_120d": 0.5})
    ranked = ews._ranked_sleeves(frame)
    expected = (2.0 * ranked["short_5d_score"] + 1.0 * ranked["mid_5d_30d_score"]
                + 0.5 * ranked["long_30d_120d_score"]).to_numpy()
    assert np.allclose(full["composite_score"].to_numpy(), expected, atol=0)
    # missing sleeve column behaves like the original (score contribution 0)
    partial = blend(frame.drop(columns=["long_30d_120d_score"]),
                    {"short_5d": 1.0, "mid_5d_30d": 1.0, "long_30d_120d": 5.0})
    base = blend(frame, {"short_5d": 1.0, "mid_5d_30d": 1.0, "long_30d_120d": 0.0})
    assert np.array_equal(partial["composite_score"].to_numpy(), base["composite_score"].to_numpy())


def test_rank_is_per_date_cross_section():
    frame = _synthetic_frame()
    out = blend(frame, {"short_5d": 1.0, "mid_5d_30d": 0.0, "long_30d_120d": 0.0})
    for _, g in out.groupby("trade_date"):
        scores = sorted(g["composite_score"].to_numpy())
        assert np.allclose(scores, np.arange(1, len(g) + 1) / len(g))
