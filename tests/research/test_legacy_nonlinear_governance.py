from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import quantagent.research.governed_model_comparison as governed_module


def test_legacy_helper_is_explicitly_research_only(monkeypatch) -> None:
    comparison = SimpleNamespace()
    promotion = SimpleNamespace(accepted=True)
    monkeypatch.setattr(
        governed_module,
        "run_model_comparison",
        lambda *args, **kwargs: comparison,
    )
    monkeypatch.setattr(
        governed_module,
        "evaluate_nonlinear_promotion",
        lambda *args, **kwargs: promotion,
    )

    result = governed_module.run_governed_model_comparison(
        pd.DataFrame(),
        ("factor_a", "factor_b"),
    )

    assert result.governance["stage4Governed"] is False
    assert result.governance["researchOnly"] is True
    assert result.governance["economicBacktestCertified"] is False
    assert result.governance["productionBlockers"]
    assert result.production_eligible is False
