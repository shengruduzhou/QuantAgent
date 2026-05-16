"""V7 training CLI: alpha training, evaluation, and real-data orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump, parse_csv_tuple, read_frame


@app.command("train-alpha-v7")
def train_alpha_v7(
    dataset_path: Path = typer.Option(..., "--dataset"),
    output_dir: Path = typer.Option(Path("artifacts/v7_alpha"), "--output-dir"),
    model: str = typer.Option("ridge", "--model", help="ridge | elastic_net | lightgbm | xgboost"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    mark_production_ready: bool = typer.Option(False, "--mark-production-ready"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
    experiment_name: str | None = typer.Option(None, "--experiment-name"),
    registry_root: Path = typer.Option(Path("artifacts/v7_alpha/registry"), "--registry-root"),
) -> None:
    """Train Ridge/ElasticNet alpha with purged walk-forward CV and acceptance gates."""
    from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment

    result = run_v7_training_experiment(
        read_frame(dataset_path),
        V7TrainingConfig(
            model=model,
            min_train_rows=min_train_rows,
            output_dir=str(output_dir),
            mark_production_ready=mark_production_ready,
            paper_report_path=str(paper_report) if paper_report else None,
            experiment_name=experiment_name,
            registry_root=str(registry_root),
        ),
    )
    typer.echo(json_dump(result))


@app.command("evaluate-alpha-v7")
def evaluate_alpha_v7(
    metrics_path: Path = typer.Option(..., "--metrics"),
    acceptance_path: Path | None = typer.Option(None, "--acceptance-report"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
    output_path: Path = typer.Option(Path("artifacts/v7_alpha/evaluation_report.json"), "--output"),
) -> None:
    """Re-evaluate an existing metrics.json against the acceptance gates without retraining."""
    from quantagent.data.v7_quality_gates import (
        V7ModelAcceptanceGateConfig,
        evaluate_model_acceptance_gates,
    )

    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if acceptance_path and acceptance_path.exists():
        prior = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
        metrics.setdefault("prior_acceptance_passed", bool(prior.get("passed", False)))
    config = V7ModelAcceptanceGateConfig()
    report = evaluate_model_acceptance_gates(metrics, config, paper_report_path=paper_report)
    payload = report.to_dict()
    payload["metrics_path"] = str(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dump(payload), encoding="utf-8")
    typer.echo(json_dump(payload))


@app.command("train-deep-alpha-v7")
def train_deep_alpha_v7(
    dataset_path: Path = typer.Option(..., "--dataset"),
    output_dir: Path = typer.Option(Path("artifacts/v7_alpha/deep"), "--output-dir"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    hidden_sizes: str = typer.Option("64,32", "--hidden-sizes"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate"),
    weight_decay: float = typer.Option(1e-4, "--weight-decay"),
    batch_size: int = typer.Option(1024, "--batch-size"),
    max_epochs: int = typer.Option(30, "--max-epochs"),
    early_stopping_patience: int = typer.Option(5, "--early-stopping-patience"),
    rank_loss_weight: float = typer.Option(0.5, "--rank-loss-weight"),
    utility_loss_weight: float = typer.Option(0.0, "--utility-loss-weight"),
    device: str = typer.Option("auto", "--device"),
    feature_columns: str = typer.Option("", "--feature-columns"),
    use_torch: bool = typer.Option(True, "--use-torch/--no-use-torch"),
    seed: int = typer.Option(1729, "--seed"),
    validation_dataset: Path | None = typer.Option(None, "--validation-dataset"),
) -> None:
    """Train the V7 deep alpha model (PyTorch if installed, numpy ridge head otherwise)."""
    from quantagent.training.v7_deep_trainer import V7DeepAlphaTrainer, V7DeepAlphaTrainerConfig

    config = V7DeepAlphaTrainerConfig(
        horizons=tuple(int(h) for h in parse_csv_tuple(horizons)),
        hidden_sizes=tuple(int(h) for h in parse_csv_tuple(hidden_sizes)),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        rank_loss_weight=rank_loss_weight,
        utility_loss_weight=utility_loss_weight,
        device=device,
        feature_columns=parse_csv_tuple(feature_columns),
        seed=seed,
        output_dir=str(output_dir),
        use_torch=use_torch,
    )
    trainer = V7DeepAlphaTrainer(config)
    train_frame = read_frame(dataset_path)
    val_frame = read_frame(validation_dataset) if validation_dataset else None
    state = trainer.fit(train_frame, validation_dataset=val_frame)
    saved = trainer.save(output_dir)
    typer.echo(
        json_dump(
            {
                "backend": state.backend,
                "horizons": state.horizons,
                "feature_count": len(state.feature_columns),
                "training_history": state.training_history,
                "state_path": str(saved),
                "config": asdict(config),
            }
        )
    )


@app.command("run-real-training-v7")
def run_real_training_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    labels_path: Path = typer.Option(..., "--labels"),
    output_dir: Path = typer.Option(Path("artifacts/v7_alpha"), "--output-dir"),
    fundamentals_root: Path | None = typer.Option(None, "--fundamentals-root"),
    valuation_path: Path | None = typer.Option(None, "--valuation"),
    disclosures_path: Path | None = typer.Option(None, "--disclosures"),
    training_dataset_path: Path = typer.Option(
        Path("data/v7/gold/training_dataset/training_dataset.parquet"), "--training-dataset"
    ),
    model: str = typer.Option("ridge", "--model"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    min_rows: int = typer.Option(100, "--min-rows"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    min_symbols: int = typer.Option(2, "--min-symbols"),
    min_dates: int = typer.Option(5, "--min-dates"),
    mark_production_ready: bool = typer.Option(False, "--mark-production-ready"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
) -> None:
    """Compose build-training-dataset-v7 + train-alpha-v7 into one auditable real-data run."""
    from quantagent.cli._utils import parse_csv_tuple
    from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
    from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment

    horizons_tuple = tuple(int(item) for item in parse_csv_tuple(horizons))
    dataset_result = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(market_panel_path),
            labels_path=str(labels_path),
            output_path=str(training_dataset_path),
            fundamentals_root=str(fundamentals_root) if fundamentals_root else None,
            valuation_path=str(valuation_path) if valuation_path else None,
            disclosures_path=str(disclosures_path) if disclosures_path else None,
            horizons=horizons_tuple,
            min_rows=min_rows,
            min_symbols=min_symbols,
            min_dates=min_dates,
        )
    )
    training_result = run_v7_training_experiment(
        read_frame(dataset_result.output_path),
        V7TrainingConfig(
            model=model,
            horizons=horizons_tuple,
            min_train_rows=min_train_rows,
            output_dir=str(output_dir),
            mark_production_ready=mark_production_ready,
            paper_report_path=str(paper_report) if paper_report else None,
        ),
    )
    typer.echo(
        json_dump(
            {
                "training_dataset": dataset_result.summary,
                "training": training_result,
            }
        )
    )
