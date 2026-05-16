"""V7 training CLI: alpha training, evaluation, and real-data orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import typer

from quantagent.cli._utils import (
    app,
    default_artifact_root,
    default_v7_lake_root,
    json_dump,
    parse_csv_tuple,
    read_frame,
    write_frame,
)


@app.command("train-alpha-v7")
def train_alpha_v7(
    dataset_path: Path = typer.Option(..., "--dataset"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    model: str = typer.Option("ridge", "--model", help="ridge | elastic_net | lightgbm | xgboost"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    mark_production_ready: bool = typer.Option(False, "--mark-production-ready"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
    experiment_name: str | None = typer.Option(None, "--experiment-name"),
    registry_root: Path = typer.Option(Path("artifacts/v7_alpha/registry"), "--registry-root"),
    allow_model_downgrade: bool = typer.Option(
        False,
        "--allow-model-downgrade",
        help="If lightgbm/xgboost are not installed, fall back to ridge instead of failing.",
    ),
) -> None:
    """Train alpha with purged walk-forward CV and acceptance gates.

    LightGBM / XGBoost are real implementations when installed. If they
    are missing the command fails loudly unless --allow-model-downgrade
    is passed.
    """
    from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment

    resolved_output = Path(output_dir) if output_dir is not None else default_artifact_root()
    resolved_registry = (
        Path(registry_root) if registry_root != Path("artifacts/v7_alpha/registry") else resolved_output / "registry"
    )
    result = run_v7_training_experiment(
        read_frame(dataset_path),
        V7TrainingConfig(
            model=model,
            min_train_rows=min_train_rows,
            output_dir=str(resolved_output),
            mark_production_ready=mark_production_ready,
            paper_report_path=str(paper_report) if paper_report else None,
            experiment_name=experiment_name,
            registry_root=str(resolved_registry),
            allow_model_downgrade=allow_model_downgrade,
        ),
    )
    typer.echo(json_dump(result))


@app.command("evaluate-alpha-v7")
def evaluate_alpha_v7(
    metrics_path: Path = typer.Option(..., "--metrics"),
    acceptance_path: Path | None = typer.Option(None, "--acceptance-report"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
    output_path: Path = typer.Option(None, "--output"),
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
    resolved_output = (
        Path(output_path) if output_path is not None else default_artifact_root() / "evaluation_report.json"
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json_dump(payload), encoding="utf-8")
    payload["output_path"] = str(resolved_output)
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

    resolved_output = Path(output_dir) if output_dir is not None else default_artifact_root() / "deep"
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
        output_dir=str(resolved_output),
        use_torch=use_torch,
    )
    trainer = V7DeepAlphaTrainer(config)
    train_frame = read_frame(dataset_path)
    val_frame = read_frame(validation_dataset) if validation_dataset else None
    state = trainer.fit(train_frame, validation_dataset=val_frame)
    saved = trainer.save(resolved_output)
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
    output_dir: Path = typer.Option(None, "--output-dir"),
    fundamentals_root: Path | None = typer.Option(None, "--fundamentals-root"),
    valuation_path: Path | None = typer.Option(None, "--valuation"),
    disclosures_path: Path | None = typer.Option(None, "--disclosures"),
    training_dataset_path: Path = typer.Option(None, "--training-dataset"),
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

    resolved_output = Path(output_dir) if output_dir is not None else default_artifact_root()
    resolved_training_dataset = (
        Path(training_dataset_path)
        if training_dataset_path is not None
        else default_v7_lake_root() / "gold" / "training_dataset" / "training_dataset.parquet"
    )
    horizons_tuple = tuple(int(item) for item in parse_csv_tuple(horizons))
    dataset_result = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(market_panel_path),
            labels_path=str(labels_path),
            output_path=str(resolved_training_dataset),
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
            output_dir=str(resolved_output),
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


@app.command("predict-alpha-v7")
def predict_alpha_v7(
    model_dir: Path = typer.Option(..., "--model-dir"),
    feature_dataset: Path = typer.Option(..., "--feature-dataset"),
    output_path: Path = typer.Option(None, "--output"),
    primary_horizon: int | None = typer.Option(None, "--primary-horizon"),
) -> None:
    """Run inference against a trained V7 alpha artifact directory.

    Supports both classical (ridge / elastic_net / lightgbm / xgboost)
    and deep alpha artifact layouts. Writes a wide ``alpha_*d`` +
    ``prediction`` frame and a sidecar JSON summary.
    """
    from quantagent.training.v7_predictor import predict_v7_alpha

    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_artifact_root() / "predictions" / "predictions.parquet"
    )
    result = predict_v7_alpha(
        model_dir,
        read_frame(feature_dataset),
        primary_horizon=primary_horizon,
    )
    written = write_frame(result.predictions, resolved_output)
    summary = {
        "model_kind": result.model_kind,
        "horizons": list(result.horizons),
        "feature_count": len(result.feature_columns),
        "row_count": int(len(result.predictions)),
        "output": str(written),
        "model_dir": result.artifact_dir,
    }
    summary_path = written.with_suffix(".summary.json")
    summary_path.write_text(json_dump(summary), encoding="utf-8")
    typer.echo(json_dump(summary))


@app.command("build-target-weights-v7")
def build_target_weights_v7(
    predictions_path: Path = typer.Option(..., "--predictions"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    sector_map_path: Path | None = typer.Option(None, "--sector-map"),
    output_path: Path = typer.Option(None, "--output"),
    top_k: int = typer.Option(30, "--top-k"),
    max_weight_per_name: float = typer.Option(0.10, "--max-weight"),
    max_sector_weight: float = typer.Option(0.30, "--max-sector"),
    max_turnover: float = typer.Option(0.50, "--max-turnover"),
    cost_bps: float = typer.Option(12.0, "--cost-bps"),
    long_short: bool = typer.Option(False, "--long-short/--long-only"),
    horizon_column: str | None = typer.Option(None, "--horizon-column"),
    min_amount_yuan: float = typer.Option(0.0, "--min-amount-yuan"),
) -> None:
    """Convert per-symbol predictions into a constrained target-weights panel.

    Applies tradability filters (ST / suspension / limit), liquidity cap,
    top-K selection, sector cap projection, and turnover cap. Writes
    both the wide target_weights frame and a diagnostics JSON.
    """
    from quantagent.portfolio.v7_target_weights import (
        V7TargetWeightsConfig,
        build_v7_target_weights,
        write_v7_target_weights,
    )

    sector_frame = read_frame(sector_map_path) if sector_map_path else None
    config = V7TargetWeightsConfig(
        long_short=long_short,
        top_k=top_k,
        max_weight_per_name=max_weight_per_name,
        max_sector_weight=max_sector_weight,
        max_turnover=max_turnover,
        cost_bps=cost_bps,
        horizon_column=horizon_column,
        min_amount_yuan=min_amount_yuan,
    )
    result = build_v7_target_weights(
        read_frame(predictions_path),
        read_frame(market_panel_path),
        sector_map=sector_frame,
        config=config,
    )
    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_artifact_root() / "target_weights" / "target_weights.parquet"
    )
    written = write_v7_target_weights(result, resolved_output)
    diagnostics_path = Path(written).with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json_dump(result.diagnostics), encoding="utf-8")
    typer.echo(
        json_dump(
            {
                "status": result.diagnostics.get("status", "passed"),
                "rows": int(len(result.target_weights)),
                "output": str(written),
                "diagnostics": str(diagnostics_path),
            }
        )
    )


@app.command("run-full-real-training-v7")
def run_full_real_training_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    labels_path: Path = typer.Option(..., "--labels"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    fundamentals_root: Path | None = typer.Option(None, "--fundamentals-root"),
    valuation_path: Path | None = typer.Option(None, "--valuation"),
    disclosures_path: Path | None = typer.Option(None, "--disclosures"),
    sector_map_path: Path | None = typer.Option(None, "--sector-map"),
    training_dataset_path: Path = typer.Option(None, "--training-dataset"),
    model: str = typer.Option("ridge", "--model"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    primary_horizon: int = typer.Option(5, "--primary-horizon"),
    min_rows: int = typer.Option(100, "--min-rows"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    min_symbols: int = typer.Option(2, "--min-symbols"),
    min_dates: int = typer.Option(5, "--min-dates"),
    top_k: int = typer.Option(30, "--top-k"),
    max_weight_per_name: float = typer.Option(0.10, "--max-weight"),
    max_sector_weight: float = typer.Option(0.30, "--max-sector"),
    max_turnover: float = typer.Option(0.50, "--max-turnover"),
    mark_production_ready: bool = typer.Option(False, "--mark-production-ready"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
    allow_model_downgrade: bool = typer.Option(False, "--allow-model-downgrade"),
) -> None:
    """End-to-end real-data pipeline: dataset → train → predict → target weights → backtest.

    Live trading remains disabled. Backtest runs through the existing
    OrderManager → VirtualBroker dry-run path.
    """
    from quantagent.backtest.ashare_execution_simulator import simulate_ashare_target_weights
    from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
    from quantagent.portfolio.v7_target_weights import (
        V7TargetWeightsConfig,
        build_v7_target_weights,
        write_v7_target_weights,
    )
    from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment
    from quantagent.training.v7_predictor import predict_v7_alpha

    horizons_tuple = tuple(int(item) for item in parse_csv_tuple(horizons))
    output_dir = Path(output_dir) if output_dir is not None else default_artifact_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_training_dataset = (
        Path(training_dataset_path)
        if training_dataset_path is not None
        else default_v7_lake_root() / "gold" / "training_dataset" / "training_dataset.parquet"
    )

    dataset_result = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(market_panel_path),
            labels_path=str(labels_path),
            output_path=str(resolved_training_dataset),
            fundamentals_root=str(fundamentals_root) if fundamentals_root else None,
            valuation_path=str(valuation_path) if valuation_path else None,
            disclosures_path=str(disclosures_path) if disclosures_path else None,
            horizons=horizons_tuple,
            min_rows=min_rows,
            min_symbols=min_symbols,
            min_dates=min_dates,
        )
    )

    training_dataset = read_frame(dataset_result.output_path)
    training_result = run_v7_training_experiment(
        training_dataset,
        V7TrainingConfig(
            model=model,
            horizons=horizons_tuple,
            min_train_rows=min_train_rows,
            output_dir=str(output_dir),
            mark_production_ready=mark_production_ready,
            paper_report_path=str(paper_report) if paper_report else None,
            allow_model_downgrade=allow_model_downgrade,
        ),
    )

    predictions = predict_v7_alpha(output_dir, training_dataset, primary_horizon=primary_horizon)
    predictions_path = output_dir / "predictions" / "predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    written_predictions = write_frame(predictions.predictions, predictions_path)

    sector_frame = read_frame(sector_map_path) if sector_map_path else None
    weights_result = build_v7_target_weights(
        predictions.predictions,
        read_frame(market_panel_path),
        sector_map=sector_frame,
        config=V7TargetWeightsConfig(
            top_k=top_k,
            max_weight_per_name=max_weight_per_name,
            max_sector_weight=max_sector_weight,
            max_turnover=max_turnover,
        ),
    )
    weights_path = output_dir / "target_weights" / "target_weights.parquet"
    written_weights = write_v7_target_weights(weights_result, weights_path)

    backtest_path = output_dir / "walk_forward_backtest.json"
    backtest_status: dict[str, object] = {"status": "skipped", "reason": "no_target_weights"}
    if not weights_result.target_weights.empty:
        weights_frame = weights_result.target_weights.copy()
        if "trade_date" in weights_frame.columns:
            weights_frame = weights_frame.set_index("trade_date")
        sim = simulate_ashare_target_weights(weights_frame, read_frame(market_panel_path))
        backtest_path.write_text(
            json_dump(
                {
                    "nav": sim.nav.to_dict(),
                    "orders": sim.order_audit.to_dict("records"),
                    "failed_orders": sim.failed_order_audit.to_dict("records"),
                    "config": sim.config,
                }
            ),
            encoding="utf-8",
        )
        backtest_status = {
            "status": "ok",
            "output": str(backtest_path),
            "failed_orders": int(len(sim.failed_order_audit)),
        }

    pipeline_report = {
        "training_dataset": dataset_result.summary,
        "training": training_result,
        "predictions": {
            "output": str(written_predictions),
            "horizons": list(predictions.horizons),
            "model_kind": predictions.model_kind,
        },
        "target_weights": {
            "output": str(written_weights),
            "diagnostics": weights_result.diagnostics,
        },
        "backtest": backtest_status,
    }
    pipeline_report_path = output_dir / "full_pipeline_report.json"
    pipeline_report_path.write_text(json_dump(pipeline_report), encoding="utf-8")
    typer.echo(json_dump(pipeline_report))
