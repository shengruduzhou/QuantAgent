from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.research.governed_model_comparison import GovernedModelComparison


def test_legacy_nonlinear_cli_never_claims_production_eligibility(tmp_path, monkeypatch) -> None:
    import quantagent.cli.nonlinear as nonlinear_cli

    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            {
                "trade_date": "2026-01-05",
                "symbol": "000001.SZ",
                "factor_a": 1.0,
                "factor_b": 2.0,
                "forward_return_5d": 0.01,
            }
        ]
    ).to_csv(panel_path, index=False)

    comparison = SimpleNamespace(as_dict=lambda: {"verdict": "production_accepted"})
    promotion = SimpleNamespace(
        accepted=True,
        champion="gbm",
        pbo=0.1,
        dsr_probability=0.99,
        spa_pvalue=0.01,
        rejection_reasons=(),
        to_dict=lambda: {"accepted": True},
    )
    governed = GovernedModelComparison(
        comparison=comparison,
        promotion=promotion,
        governance={
            "stage4Governed": False,
            "researchOnly": True,
            "economicBacktestCertified": False,
            "productionBlockers": ["legacy path is research-only"],
        },
    )

    monkeypatch.setattr(
        nonlinear_cli,
        "run_governed_model_comparison",
        lambda *args, **kwargs: governed,
    )
    monkeypatch.setattr(
        nonlinear_cli,
        "save_comparison_report",
        lambda *args, **kwargs: tmp_path / "model_comparison.json",
    )

    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "audit-nonlinear-factors",
            "--panel-path",
            str(panel_path),
            "--factor-names",
            "factor_a,factor_b",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "research_promotion=PASS" in result.stdout
    assert "production_eligible=NO" in result.stdout
    assert "production_blocker: legacy path is research-only" in result.stdout
    payload = (output_dir / "promotion_gate.json").read_text(encoding="utf-8")
    assert '"productionEligible": false' in payload
