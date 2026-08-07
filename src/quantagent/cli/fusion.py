"""Governed CLI for the factor-fusion search (ATLAS layer L2).

One command, ``search-factor-fusion``, so the web workstation has exactly one
allowlisted entrypoint into the search. It reads a factor panel and a forward
return panel, runs the purged walk-forward protocol over every enumerated blend
scheme, and writes the candidate set, the Pareto frontier and a hashed manifest.

The command deliberately exposes no ``--n-trials`` flag: the trial count is a
property of the enumerated search space, and letting an operator declare it
would make the deflated Sharpe ratio meaningless.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root


# Factor panels often carry labels beside features for convenience. A typo in
# --factor-names must not silently promote those future values into predictors.
# Keep this deliberately narrow: reject semantic label/target names, not every
# feature containing words such as "return" (momentum/lagged return is valid).
_LEAKAGE_FACTOR_NAME = re.compile(
    r"^(?:forward[_-]?return(?:s)?(?:_|$)|future[_-]?return(?:s)?(?:_|$)|label(?:_|$)|target(?:_|$)|y(?:_|$))",
    re.IGNORECASE,
)


def _read_panel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _resolve_forward_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    candidates = [column for column in frame.columns if column.startswith("forward_return")]
    if not candidates:
        raise typer.BadParameter(
            f"forward panel has no '{requested}' column and no forward_return* fallback; "
            f"available columns: {sorted(frame.columns)[:20]}"
        )
    chosen = sorted(candidates)[0]
    typer.echo(f"forward column '{requested}' not found; using '{chosen}'")
    return chosen


def _validate_factor_names_for_leakage(names: tuple[str, ...]) -> None:
    blocked = sorted(name for name in names if _LEAKAGE_FACTOR_NAME.search(name))
    if blocked:
        raise typer.BadParameter(
            "factor_names contains label/forward-looking columns and is blocked fail-closed: "
            f"{blocked}. Use only features observable at the decision timestamp."
        )


@app.command("search-factor-fusion")
def search_factor_fusion(
    factor_panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    forward_returns_path: Path = typer.Option(..., exists=True, dir_okay=False),
    factor_names: str = typer.Option(..., help="comma-separated factor column names"),
    output_dir: Path = typer.Option(default_reports_root() / "fusion" / "search"),
    forward_column: str = typer.Option("forward_return"),
    horizon_days: int = typer.Option(5, min=1, max=252),
    top_k: int = typer.Option(30, min=1, max=500),
    n_folds: int = typer.Option(4, min=1, max=12),
    embargo_days: int = typer.Option(5, min=0, max=120),
    min_train_days: int = typer.Option(120, min=20),
    min_test_days: int = typer.Option(40, min=5),
    transaction_cost_bps: float = typer.Option(8.0, min=0.0, max=500.0),
    include_genetic: bool = typer.Option(True),
    random_controls: int = typer.Option(8, min=0, max=64),
    single_factor_baselines: int = typer.Option(6, min=0, max=64),
    seed: int = typer.Option(17),
    benchmark_path: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    benchmark_symbol: str = typer.Option(""),
    preference_excess_return: float = typer.Option(0.40, min=0.0, max=1.0),
    preference_annual_return: float = typer.Option(0.20, min=0.0, max=1.0),
    preference_drawdown_control: float = typer.Option(0.25, min=0.0, max=1.0),
    preference_robustness: float = typer.Option(0.15, min=0.0, max=1.0),
):
    """Search factor blends and rank the out-of-sample Pareto frontier."""
    from quantagent.fusion import (
        FusionSearchConfig,
        ObjectivePreference,
        run_fusion_search,
        save_fusion_artifacts,
    )

    names = tuple(name.strip() for name in factor_names.split(",") if name.strip())
    if not names:
        raise typer.BadParameter("factor_names resolved to an empty list")
    _validate_factor_names_for_leakage(names)

    factor_panel = _read_panel(factor_panel_path)
    forward_panel = _read_panel(forward_returns_path)
    resolved_column = _resolve_forward_column(forward_panel, forward_column)
    forward_panel = forward_panel.rename(columns={resolved_column: "forward_return"})

    missing = [name for name in names if name not in factor_panel.columns]
    if missing:
        raise typer.BadParameter(
            f"factor columns missing from the panel: {missing}"
        )

    benchmark_returns = None
    if benchmark_path is not None:
        benchmark_frame = _read_panel(benchmark_path)
        if "trade_date" not in benchmark_frame.columns:
            raise typer.BadParameter("benchmark file must contain a trade_date column")
        value_column = next(
            (
                column
                for column in ("benchmark_return", "forward_return", "return")
                if column in benchmark_frame.columns
            ),
            None,
        )
        if value_column is None:
            raise typer.BadParameter(
                "benchmark file must contain benchmark_return, forward_return or return"
            )
        benchmark_returns = (
            benchmark_frame.assign(trade_date=pd.to_datetime(benchmark_frame["trade_date"]))
            .set_index("trade_date")[value_column]
            .astype(float)
            .sort_index()
        )

    config = FusionSearchConfig(
        factor_names=names,
        horizon_days=horizon_days,
        top_k=top_k,
        n_folds=n_folds,
        embargo_days=embargo_days,
        min_train_days=min_train_days,
        min_test_days=min_test_days,
        transaction_cost_bps=transaction_cost_bps,
        include_genetic=include_genetic,
        random_controls=random_controls,
        single_factor_baselines=single_factor_baselines,
        seed=seed,
        benchmark_symbol=benchmark_symbol,
        preference=ObjectivePreference(
            excess_return=preference_excess_return,
            annual_return=preference_annual_return,
            drawdown_control=preference_drawdown_control,
            robustness=preference_robustness,
        ),
    )

    def _progress(stage: str, completed: int, total: int) -> None:
        # Bracket form is what the web JobRunner's progress parser understands.
        typer.echo(f"[{completed} / {total}] {stage}")

    result = run_fusion_search(
        factor_panel=factor_panel,
        forward_panel=forward_panel,
        config=config,
        benchmark_returns=benchmark_returns,
        progress=_progress,
    )
    paths = save_fusion_artifacts(result, output_dir=output_dir)

    typer.echo(
        f"n_trials={result.n_trials} frontier={len(result.frontier)} "
        f"pbo={result.pbo:.4f} benchmark={result.benchmark_mode}"
    )
    preferred = result.preferred
    if preferred is not None:
        metrics = preferred.metrics
        typer.echo(
            f"preferred={preferred.candidate_id} "
            f"excess={metrics['excessReturn']:+.4f} "
            f"annual={metrics['annualReturn']:+.4f} "
            f"max_dd={metrics['maxDrawdown']:.4f} "
            f"robustness={metrics['robustness']:.3f}"
        )
    else:
        typer.echo("no candidate produced usable out-of-sample observations")
    typer.echo(f"wrote {len(paths)} artifacts to {output_dir}")
    return output_dir
