"""V7 optimization CLI: parameter search over alpha training hyperparameters."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quantagent.cli._utils import (
    app,
    default_reports_root,
    json_dump,
    read_frame,
)


@app.command("optimize-alpha-v7")
def optimize_alpha_v7(
    dataset_path: Path = typer.Option(..., "--dataset"),
    search_space_path: Path = typer.Option(..., "--search-space", help="JSON file mapping param name → list of values."),
    output_dir: Path = typer.Option(None, "--output-dir"),
    objective: str = typer.Option("rank_ic_mean", "--objective"),
    mode: str = typer.Option("max", "--mode", help="max or min"),
    sampler: str = typer.Option("grid", "--sampler", help="grid or random"),
    n_trials: int | None = typer.Option(None, "--n-trials"),
    seed: int = typer.Option(1729, "--seed"),
    split_mode: str = typer.Option("expanding", "--split-mode"),
    valid_size_days: int = typer.Option(5, "--valid-size-days"),
    min_train_days: int = typer.Option(20, "--min-train-days"),
    rolling_train_days: int = typer.Option(252, "--rolling-train-days"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    purge_days: int | None = typer.Option(None, "--purge-days"),
    min_folds: int = typer.Option(1, "--min-folds"),
    stability_threshold: float = typer.Option(float("-inf"), "--stability-threshold"),
) -> None:
    """Grid / random search over alpha training hyperparameters.

    Reads a JSON file describing the search space (e.g. ``{"model": ["ridge", "elastic_net"], "min_train_rows": [100, 500]}``)
    and writes ``optimization_report.json`` plus per-trial training
    artefacts under ``--output-dir``. Live trading remains disabled.
    """
    from quantagent.training.optimize import OptimizationConfig, run_alpha_param_search

    space = json.loads(Path(search_space_path).read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise typer.BadParameter("search space JSON must be an object of param → list[values]")

    resolved_dir = Path(output_dir) if output_dir is not None else default_reports_root() / "optimization"
    config = OptimizationConfig(
        parameter_space=space,
        objective=objective,
        mode=mode,
        sampler=sampler,
        n_trials=n_trials,
        seed=seed,
        output_dir=str(resolved_dir),
        min_folds=min_folds,
        stability_threshold=stability_threshold,
        train_kwargs={
            "split_mode": split_mode,
            "valid_size_days": valid_size_days,
            "min_train_days": min_train_days,
            "rolling_train_days": rolling_train_days,
            "embargo_days": embargo_days,
            "purge_days": purge_days,
        },
    )
    result = run_alpha_param_search(read_frame(dataset_path), config)
    typer.echo(json_dump(result.to_dict()))
