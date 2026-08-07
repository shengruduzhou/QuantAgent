"""Governed research entrypoints for literature-backed model promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root
from quantagent.research.model_comparison import ComparisonConfig, run_model_comparison, save_comparison_report
from quantagent.research.nonlinear_promotion import NonlinearPromotionConfig, evaluate_nonlinear_promotion


@app.command("audit-nonlinear-promotion")
def audit_nonlinear_promotion(
    panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    factor_names: str = typer.Option(..., help="comma-separated pre-declared feature columns"),
    label_column: str = typer.Option("forward_return_5d"),
    horizon_days: int = typer.Option(5, min=1, max=252),
    top_k: int = typer.Option(30, min=1, max=500),
    cost_bps: float = typer.Option(12.0, min=0.0, max=500.0),
    output_dir: Path = typer.Option(default_reports_root() / "research" / "nonlinear-promotion"),
) -> Path:
    """Compare nonlinear arms, freeze a champion, then enforce PBO/DSR/SPA."""
    factors = tuple(name.strip() for name in factor_names.split(",") if name.strip())
    if len(factors) < 2:
        raise typer.BadParameter("at least two pre-declared factor names are required")
    panel = pd.read_csv(panel_path) if panel_path.suffix.lower() == ".csv" else pd.read_parquet(panel_path)
    missing = [name for name in factors if name not in panel.columns]
    if missing:
        raise typer.BadParameter(f"factor columns missing from panel: {missing}")
    if label_column not in panel.columns:
        raise typer.BadParameter(f"label column missing from panel: {label_column}")

    comparison = run_model_comparison(
        panel,
        factors,
        config=ComparisonConfig(
            label_column=label_column,
            horizon_days=horizon_days,
            top_k=top_k,
            cost_bps=cost_bps,
            max_pbo=0.25,
        ),
        progress=lambda stage, done, total: typer.echo(f"[{done} / {total}] {stage}"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_report(comparison, output_dir)
    promotion = evaluate_nonlinear_promotion(
        comparison,
        config=NonlinearPromotionConfig(max_pbo=0.25, min_dsr_probability=0.95, max_spa_pvalue=0.05),
    )
    path = output_dir / "nonlinear_promotion_gate.json"
    path.write_text(json.dumps(promotion.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(
        f"raw={promotion.raw_verdict} final={promotion.final_verdict} champion={promotion.champion or 'none'} "
        f"pbo={promotion.pbo:.4f} dsr={promotion.dsr_probability:.4f} spa={promotion.spa_pvalue:.4f}"
    )
    for reason in promotion.rejection_reasons:
        typer.echo(f"blocked: {reason}")
    typer.echo(f"wrote {path}")
    return path
