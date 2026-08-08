from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root
from quantagent.research.governed_model_comparison import run_governed_model_comparison
from quantagent.research.model_comparison import ComparisonConfig, save_comparison_report
from quantagent.research.nonlinear_promotion import NonlinearPromotionConfig


def _read_panel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


@app.command("audit-nonlinear-factors")
def audit_nonlinear_factors(
    panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    factor_names: str = typer.Option(..., help="Comma-separated PIT-safe factor columns."),
    label_column: str = typer.Option("forward_return_5d"),
    horizon_days: int = typer.Option(5, min=1, max=252),
    top_k: int = typer.Option(30, min=1, max=500),
    cost_bps: float = typer.Option(12.0, min=0.0, max=500.0),
    n_folds: int = typer.Option(6, min=4, max=20),
    holdout_folds: int = typer.Option(2, min=1, max=6),
    valid_size_days: int = typer.Option(40, min=10, max=252),
    min_train_days: int = typer.Option(250, min=60),
    embargo_days: int = typer.Option(5, min=0, max=120),
    output_dir: Path = typer.Option(default_reports_root() / "nonlinearity" / "audit"),
    enforce_promotion_gates: bool = typer.Option(
        False,
        help="Exit non-zero unless PBO<=0.25, DSR>=0.95 and SPA<=0.05 all pass.",
    ),
) -> Path:
    """Compare nonlinear challengers with the linear control, then gate promotion.

    The raw model comparison is always written for research diagnosis. The
    separately written ``promotion_gate.json`` is the sole production verdict.
    Missing statistical evidence fails closed.
    """
    names = tuple(name.strip() for name in factor_names.split(",") if name.strip())
    if len(names) < 2:
        raise typer.BadParameter("factor_names must resolve to at least two columns")
    if holdout_folds >= n_folds:
        raise typer.BadParameter("holdout_folds must be smaller than n_folds")

    panel = _read_panel(panel_path)
    missing = [name for name in ("trade_date", "symbol", label_column, *names) if name not in panel.columns]
    if missing:
        raise typer.BadParameter(f"panel is missing required columns: {missing}")

    comparison_config = ComparisonConfig(
        label_column=label_column,
        horizon_days=horizon_days,
        top_k=top_k,
        cost_bps=cost_bps,
        n_folds=n_folds,
        holdout_folds=holdout_folds,
        valid_size_days=valid_size_days,
        min_train_days=min_train_days,
        embargo_days=embargo_days,
        # The raw comparison retains its internal research threshold. The
        # production gate below is deliberately stricter and immutable here.
        max_pbo=0.25,
    )
    promotion_config = NonlinearPromotionConfig(
        max_pbo=0.25,
        min_dsr_probability=0.95,
        max_spa_pvalue=0.05,
    )

    def _progress(stage: str, completed: int, total: int) -> None:
        typer.echo(f"[{completed}/{total}] {stage}")

    governed = run_governed_model_comparison(
        panel,
        names,
        comparison_config=comparison_config,
        promotion_config=promotion_config,
        progress=_progress,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_report(governed.comparison, output_dir)
    gate_path = output_dir / "promotion_gate.json"
    gate_path.write_text(
        json.dumps(governed.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    promotion = governed.promotion
    typer.echo(
        "promotion=" + ("ELIGIBLE" if promotion.accepted else "BLOCKED")
        + f" champion={promotion.champion or 'none'}"
        + f" pbo={promotion.pbo}"
        + f" dsr={promotion.dsr_probability}"
        + f" spa={promotion.spa_pvalue}"
    )
    for reason in promotion.rejection_reasons:
        typer.echo(f"promotion_blocker: {reason}")
    typer.echo(f"wrote governed nonlinear audit to {output_dir}")

    if enforce_promotion_gates and not promotion.accepted:
        raise typer.Exit(code=2)
    return output_dir


__all__ = ["audit_nonlinear_factors"]
