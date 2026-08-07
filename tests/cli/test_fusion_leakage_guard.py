from __future__ import annotations

import pytest
import typer

from quantagent.cli.fusion import _validate_factor_names_for_leakage


@pytest.mark.parametrize(
    "name",
    [
        "forward_return",
        "forward_return_5d",
        "future_return_20d",
        "label",
        "label_5d",
        "target",
        "target_alpha",
        "y",
    ],
)
def test_rejects_label_like_factor_names(name: str):
    with pytest.raises(typer.BadParameter, match="blocked fail-closed"):
        _validate_factor_names_for_leakage(("momentum_20d", name))


def test_allows_observable_return_features():
    _validate_factor_names_for_leakage(
        ("return_5d_lag1", "momentum_20d", "volatility_20d", "quality_score")
    )
