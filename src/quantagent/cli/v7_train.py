"""V7 training CLI: alpha training, evaluation, and real-data orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import (
    app,
    default_artifact_root,
    default_predictions_root,
    default_reports_root,
    default_target_weights_root,
    default_v7_lake_root,
    json_dump,
    merge_symbols,
    parse_csv_tuple,
    read_frame,
    write_frame,
)
from quantagent.config.paths import quant_paths
from quantagent.data.lake import v7_lake_paths
from quantagent.data.v7_auto_range import (
    list_qlib_feature_symbols,
    read_qlib_calendar_range,
)


@app.command("train-alpha-v7")
def train_alpha_v7(
    dataset_path: Path = typer.Option(..., "--dataset"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    model: str = typer.Option("ridge", "--model", help="ridge | elastic_net | lightgbm | xgboost | ft_transformer"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    split_mode: str = typer.Option("expanding", "--split-mode", help="expanding | rolling | purged | chronological"),
    valid_size_days: int = typer.Option(5, "--valid-size-days"),
    min_train_days: int = typer.Option(20, "--min-train-days"),
    rolling_train_days: int = typer.Option(252, "--rolling-train-days"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    purge_days: int | None = typer.Option(None, "--purge-days", help="Defaults to max configured label horizon."),
    mark_production_ready: bool = typer.Option(False, "--mark-production-ready"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
    experiment_name: str | None = typer.Option(None, "--experiment-name"),
    registry_root: Path | None = typer.Option(None, "--registry-root"),
    ft_max_epochs: int = typer.Option(60, "--ft-max-epochs"),
    ft_batch_size: int = typer.Option(8192, "--ft-batch-size"),
    ft_device: str = typer.Option("auto", "--ft-device", help="auto | cuda | cuda:0 | cpu for ft_transformer."),
    require_gpu: bool = typer.Option(False, "--require-gpu", help="Fail if ft_transformer cannot train on CUDA."),
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
    resolved_registry = Path(registry_root) if registry_root is not None else resolved_output / "registry"
    result = run_v7_training_experiment(
        read_frame(dataset_path),
        V7TrainingConfig(
            model=model,
            min_train_rows=min_train_rows,
            split_mode=split_mode,
            valid_size_days=valid_size_days,
            min_train_days=min_train_days,
            rolling_train_days=rolling_train_days,
            embargo_days=embargo_days,
            purge_days=purge_days,
            output_dir=str(resolved_output),
            mark_production_ready=mark_production_ready,
            paper_report_path=str(paper_report) if paper_report else None,
            experiment_name=experiment_name,
            registry_root=str(resolved_registry),
            allow_model_downgrade=allow_model_downgrade,
            ft_max_epochs=ft_max_epochs,
            ft_batch_size=ft_batch_size,
            ft_device=ft_device,
            require_gpu=require_gpu,
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


@app.command("auto-train-v7")
def auto_train_v7(
    symbols: str = typer.Option(
        "auto",
        "--symbols",
        help="Comma-separated A-share symbols, or 'auto' to use local Qlib feature instruments.",
    ),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    max_symbols: int = typer.Option(0, "--max-symbols", help="0 means no cap when --symbols=auto."),
    include_indices: bool = typer.Option(False, "--include-indices"),
    provider_uri: Path | None = typer.Option(None, "--provider-uri", help="Local Qlib provider_uri for calendar and symbol discovery."),
    market_panel_path: Path | None = typer.Option(None, "--market-panel"),
    refresh_akshare_market: bool = typer.Option(False, "--refresh-akshare-market"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    as_of_date: str | None = typer.Option(None, "--as-of-date"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    model: str = typer.Option("ridge", "--model"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    primary_horizon: int = typer.Option(5, "--primary-horizon"),
    min_rows: int = typer.Option(100, "--min-rows"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    split_mode: str = typer.Option("rolling", "--split-mode"),
    valid_size_days: int = typer.Option(20, "--valid-size-days"),
    min_train_days: int = typer.Option(120, "--min-train-days"),
    rolling_train_days: int = typer.Option(756, "--rolling-train-days"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    purge_days: int | None = typer.Option(None, "--purge-days"),
    min_symbols: int = typer.Option(2, "--min-symbols"),
    min_dates: int = typer.Option(5, "--min-dates"),
    top_k: int = typer.Option(30, "--top-k"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash"),
    ft_device: str = typer.Option("auto", "--ft-device", help="auto | cuda | cuda:0 | cpu for ft_transformer."),
    require_gpu: bool = typer.Option(False, "--require-gpu", help="Fail if ft_transformer cannot train on CUDA."),
    allow_model_downgrade: bool = typer.Option(False, "--allow-model-downgrade"),
) -> None:
    """Auto-resolve local data range and run the V7 end-to-end training path.

    The command is intended for repeated production-like research runs. It
    never enables live trading; the terminal step remains target_weights plus
    paper/backtest reporting through the V7 safe execution simulator.
    """

    from quantagent.data.bootstrap.akshare_market_bootstrap import AkShareMarketPanelConfig, build_akshare_market_panel
    from quantagent.data.bootstrap.qlib_bootstrap import QlibBootstrapConfig, build_qlib_market_panel
    from quantagent.data.v7_label_builder import build_forward_return_labels

    lake = v7_lake_paths(default_v7_lake_root()).ensure()
    resolved_provider_uri = Path(provider_uri) if provider_uri else quant_paths().raw / "qlib" / "cn_data"
    if symbols.strip().lower() == "auto":
        symbol_tuple = list_qlib_feature_symbols(
            resolved_provider_uri,
            include_indices=include_indices,
            max_symbols=max_symbols,
        )
    else:
        symbol_tuple = merge_symbols(symbols, symbols_file)
    if symbols.strip().lower() == "auto" and symbols_file is not None:
        symbol_tuple = tuple(dict.fromkeys([*symbol_tuple, *merge_symbols("", symbols_file)]))
    if not symbol_tuple:
        raise typer.BadParameter(
            "No symbols resolved. Pass --symbols explicitly or prepare local Qlib features under provider_uri/features."
        )

    stages: dict[str, object] = {
        "symbols_mode": symbols,
        "symbol_count": len(symbol_tuple),
        "provider_uri": str(resolved_provider_uri),
    }

    if refresh_akshare_market:
        market_result = build_akshare_market_panel(
            AkShareMarketPanelConfig(
                symbols=symbol_tuple,
                output_root=str(lake.root),
                allow_network=allow_network,
                provider_uri_for_range=str(resolved_provider_uri),
                as_of_date=as_of_date,
            )
        )
        if market_result["status"] != "passed":
            typer.echo(json_dump({"status": "failed", "stage": "akshare_market", "market": market_result}))
            raise typer.Exit(code=1)
        resolved_market_panel = Path(str(market_result["output"]))
        stages["market"] = market_result
    elif market_panel_path is not None:
        resolved_market_panel = Path(market_panel_path)
        stages["market"] = {"status": "existing_path", "output": str(resolved_market_panel)}
    elif _market_manifest_is_usable(lake.manifests / "market_panel.json"):
        resolved_market_panel = _existing_table_path(lake.silver_market_panel / "market_panel.parquet")
        stages["market"] = {"status": "existing_lake", "output": str(resolved_market_panel)}
    else:
        qlib_range = read_qlib_calendar_range(resolved_provider_uri)
        if qlib_range is None:
            raise typer.BadParameter(
                "No usable market panel and no Qlib calendar found. Pass --market-panel or --refresh-akshare-market --allow-network."
            )
        market_result = build_qlib_market_panel(
            QlibBootstrapConfig(
                provider_uri=str(resolved_provider_uri),
                start_date=qlib_range.start_date,
                end_date=qlib_range.end_date,
                symbols=symbol_tuple,
                output_root=str(lake.root),
                metadata={"auto_train": True},
            )
        )
        resolved_market_panel = Path(str(market_result["market_path"]))
        stages["market"] = market_result

    labels_path = lake.root / "labels.parquet"
    label_result = build_forward_return_labels(read_frame(resolved_market_panel), tuple(int(item) for item in parse_csv_tuple(horizons)))
    written_labels = write_frame(label_result.frame, labels_path)
    stages["labels"] = {
        "status": "passed",
        "output": str(written_labels),
        "rows": int(len(label_result.frame)),
        "label_schema": label_result.label_schema,
    }

    run_full_real_training_v7(
        market_panel_path=resolved_market_panel,
        labels_path=written_labels,
        output_dir=output_dir,
        fundamentals_root=quant_paths().data_root / "v7" / "raw" / "akshare" / "fundamentals",
        valuation_path=None,
        disclosures_path=None,
        sector_map_path=None,
        training_dataset_path=None,
        symbols=",".join(symbol_tuple),
        symbols_file=None,
        model=model,
        horizons=horizons,
        primary_horizon=primary_horizon,
        min_rows=min_rows,
        min_train_rows=min_train_rows,
        split_mode=split_mode,
        valid_size_days=valid_size_days,
        min_train_days=min_train_days,
        rolling_train_days=rolling_train_days,
        embargo_days=embargo_days,
        purge_days=purge_days,
        ft_max_epochs=60,
        ft_batch_size=8192,
        ft_device=ft_device,
        require_gpu=require_gpu,
        min_symbols=min_symbols,
        min_dates=min_dates,
        top_k=top_k,
        top_k_ratio=0.10,
        min_selection_pressure=3.0,
        fail_if_top_k_covers_universe=True,
        max_weight_per_name=0.10,
        max_sector_weight=0.30,
        max_turnover=0.50,
        optimizer_backend="auto",
        objective="max_expected_alpha",
        cash_floor=0.0,
        initial_cash=initial_cash,
        min_order_value_yuan=100.0,
        benchmark_symbol=None,
        paper_report_output_dir=None,
        mark_production_ready=False,
        paper_report=None,
        allow_model_downgrade=allow_model_downgrade,
    )
    typer.echo(
        json_dump(
            {
                "status": "started_and_completed",
                "safe_execution": "target_weights_only; live trading disabled",
                "stages": stages,
            }
        )
    )


@app.command("train-deep-alpha-v7")
def train_deep_alpha_v7(
    dataset_path: Path = typer.Option(..., "--dataset"),
    output_dir: Path = typer.Option(None, "--output-dir"),
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
    split_mode: str = typer.Option("expanding", "--split-mode"),
    valid_size_days: int = typer.Option(5, "--valid-size-days"),
    min_train_days: int = typer.Option(20, "--min-train-days"),
    rolling_train_days: int = typer.Option(252, "--rolling-train-days"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    purge_days: int | None = typer.Option(None, "--purge-days"),
    ft_device: str = typer.Option("auto", "--ft-device", help="auto | cuda | cuda:0 | cpu for ft_transformer."),
    require_gpu: bool = typer.Option(False, "--require-gpu", help="Fail if ft_transformer cannot train on CUDA."),
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
            split_mode=split_mode,
            valid_size_days=valid_size_days,
            min_train_days=min_train_days,
            rolling_train_days=rolling_train_days,
            embargo_days=embargo_days,
            purge_days=purge_days,
            ft_device=ft_device,
            require_gpu=require_gpu,
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
        else default_predictions_root() / "predictions.parquet"
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
    top_k_ratio: float | None = typer.Option(0.10, "--top-k-ratio"),
    min_selection_pressure: float = typer.Option(3.0, "--min-selection-pressure"),
    fail_if_top_k_covers_universe: bool = typer.Option(
        True,
        "--fail-if-top-k-covers-universe/--allow-top-k-covers-universe",
    ),
    selection_mode: str = typer.Option("ai_threshold", "--selection-mode", help="ai_threshold | top_k"),
    alpha_threshold: float = typer.Option(0.0, "--alpha-threshold"),
    confidence_floor: float = typer.Option(0.55, "--confidence-floor"),
    selection_top_k_min: int = typer.Option(5, "--selection-top-k-min"),
    selection_top_k_max: int = typer.Option(100, "--selection-top-k-max"),
    max_weight_per_name: float = typer.Option(0.10, "--max-weight"),
    max_sector_weight: float = typer.Option(0.30, "--max-sector"),
    max_turnover: float = typer.Option(0.50, "--max-turnover"),
    cost_bps: float = typer.Option(12.0, "--cost-bps"),
    long_short: bool = typer.Option(False, "--long-short/--long-only"),
    horizon_column: str | None = typer.Option(None, "--horizon-column"),
    min_amount_yuan: float = typer.Option(0.0, "--min-amount-yuan"),
    optimizer_backend: str = typer.Option("auto", "--optimizer-backend", help="auto | deterministic | cvxpy"),
    objective: str = typer.Option("max_expected_alpha", "--objective"),
    cash_floor: float = typer.Option(0.0, "--cash-floor"),
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
        top_k_ratio=top_k_ratio,
        min_selection_pressure=min_selection_pressure,
        fail_if_top_k_covers_universe=fail_if_top_k_covers_universe,
        selection_mode=selection_mode,
        alpha_threshold=alpha_threshold,
        confidence_floor=confidence_floor,
        selection_top_k_min=selection_top_k_min,
        selection_top_k_max=selection_top_k_max,
        max_weight_per_name=max_weight_per_name,
        max_sector_weight=max_sector_weight,
        max_turnover=max_turnover,
        cost_bps=cost_bps,
        horizon_column=horizon_column,
        min_amount_yuan=min_amount_yuan,
        optimizer_backend=optimizer_backend,
        objective=objective,
        cash_floor=cash_floor,
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
        else default_target_weights_root() / "target_weights.parquet"
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
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    model: str = typer.Option("ridge", "--model"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    primary_horizon: int = typer.Option(5, "--primary-horizon"),
    min_rows: int = typer.Option(100, "--min-rows"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    split_mode: str = typer.Option("rolling", "--split-mode"),
    valid_size_days: int = typer.Option(20, "--valid-size-days"),
    min_train_days: int = typer.Option(120, "--min-train-days"),
    rolling_train_days: int = typer.Option(756, "--rolling-train-days"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    purge_days: int | None = typer.Option(None, "--purge-days"),
    n_splits: int = typer.Option(4, "--n-splits", help="Walk-forward fold count; raise to cover the full OOS span."),
    ft_max_epochs: int = typer.Option(60, "--ft-max-epochs"),
    ft_batch_size: int = typer.Option(8192, "--ft-batch-size"),
    ft_device: str = typer.Option("auto", "--ft-device", help="auto | cuda | cuda:0 | cpu for ft_transformer."),
    require_gpu: bool = typer.Option(False, "--require-gpu", help="Fail if ft_transformer cannot train on CUDA."),
    min_symbols: int = typer.Option(2, "--min-symbols"),
    min_dates: int = typer.Option(5, "--min-dates"),
    top_k: int = typer.Option(30, "--top-k"),
    top_k_ratio: float | None = typer.Option(0.10, "--top-k-ratio"),
    min_selection_pressure: float = typer.Option(3.0, "--min-selection-pressure"),
    fail_if_top_k_covers_universe: bool = typer.Option(
        True,
        "--fail-if-top-k-covers-universe/--allow-top-k-covers-universe",
    ),
    selection_mode: str = typer.Option("ai_threshold", "--selection-mode", help="ai_threshold | top_k"),
    alpha_threshold: float = typer.Option(0.0, "--alpha-threshold"),
    confidence_floor: float = typer.Option(0.55, "--confidence-floor"),
    selection_top_k_min: int = typer.Option(5, "--selection-top-k-min"),
    selection_top_k_max: int = typer.Option(100, "--selection-top-k-max"),
    max_weight_per_name: float = typer.Option(0.10, "--max-weight"),
    max_sector_weight: float = typer.Option(0.30, "--max-sector"),
    max_turnover: float = typer.Option(0.50, "--max-turnover"),
    optimizer_backend: str = typer.Option("auto", "--optimizer-backend", help="auto | deterministic | cvxpy"),
    objective: str = typer.Option("max_expected_alpha", "--objective"),
    cash_floor: float = typer.Option(0.0, "--cash-floor"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash"),
    min_order_value_yuan: float = typer.Option(100.0, "--min-order-value-yuan"),
    benchmark_symbol: str | None = typer.Option(None, "--benchmark-symbol"),
    paper_report_output_dir: Path | None = typer.Option(None, "--paper-report-output-dir"),
    mark_production_ready: bool = typer.Option(False, "--mark-production-ready"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
    allow_model_downgrade: bool = typer.Option(False, "--allow-model-downgrade"),
    # Phase 3 dynamic-portfolio knobs.
    multi_horizon_blend: bool = typer.Option(
        True,
        "--multi-horizon-blend/--no-multi-horizon-blend",
        help="Blend multi-horizon predictions instead of filtering to --primary-horizon.",
    ),
    dynamic_top_k: bool = typer.Option(
        False,
        "--dynamic-top-k/--no-dynamic-top-k",
        help="Resolve top_k per-date from lifecycle / alpha signals.",
    ),
    top_k_min: int = typer.Option(8, "--top-k-min"),
    top_k_max: int = typer.Option(50, "--top-k-max"),
    timing_gate: bool = typer.Option(
        False,
        "--timing-gate/--no-timing-gate",
        help="Enable ATR-based entry/exit gate before optimisation.",
    ),
    holding_period_mode: str = typer.Option(
        "off",
        "--holding-period-mode",
        help="off | soft. Soft locks per-name |Δw| while age < expected_horizon.",
    ),
    holding_period_max_delta: float = typer.Option(0.02, "--holding-period-max-delta"),
    capital_tier: str = typer.Option(
        "",
        "--capital-tier",
        help="Capital-tier ladder, e.g. '1e6:0.10,1e7:0.05,1e8:0.02'. Empty disables tiering.",
    ),
) -> None:
    """End-to-end real-data pipeline: dataset -> train -> predict -> target weights -> paper report.

    Live trading remains disabled. Backtest runs through the existing
    OrderManager -> VirtualBroker dry-run path.
    """
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig, simulate_ashare_target_weights
    from quantagent.backtest.paper_report import PaperReportConfig, write_paper_report
    from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
    from quantagent.data.v7_quality_gates import V7ModelAcceptanceGateConfig, evaluate_model_acceptance_gates
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
            symbols=merge_symbols(symbols, symbols_file),
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
            n_splits=n_splits,
            split_mode=split_mode,
            valid_size_days=valid_size_days,
            min_train_days=min_train_days,
            rolling_train_days=rolling_train_days,
            embargo_days=embargo_days,
            purge_days=purge_days,
            output_dir=str(output_dir),
            mark_production_ready=mark_production_ready,
            paper_report_path=str(paper_report) if paper_report else None,
            allow_model_downgrade=allow_model_downgrade,
            ft_max_epochs=ft_max_epochs,
            ft_batch_size=ft_batch_size,
            ft_device=ft_device,
            require_gpu=require_gpu,
        ),
    )

    raw_predictions = read_frame(Path(training_result.artifact_paths["predictions"]))
    if multi_horizon_blend and "horizon" in raw_predictions.columns and raw_predictions["horizon"].nunique() > 1:
        from quantagent.portfolio.multi_horizon_blender import (
            MultiHorizonBlendConfig,
            blend_multi_horizon_predictions,
        )

        blend_result = blend_multi_horizon_predictions(
            raw_predictions,
            config=MultiHorizonBlendConfig(primary_horizon=primary_horizon),
        )
        predictions_frame = blend_result.blended.copy()
        predictions_frame["sample_role"] = "validation"
        predictions_frame["fold_id"] = 0
        blender_diagnostics = blend_result.diagnostics
    else:
        predictions_frame = _load_oos_predictions(
            Path(training_result.artifact_paths["predictions"]),
            primary_horizon=primary_horizon,
        )
        blender_diagnostics = {"status": "skipped", "reason": "single_horizon_or_disabled"}
    predictions_path = default_predictions_root() / "predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    written_predictions = write_frame(predictions_frame, predictions_path)

    sector_frame = read_frame(sector_map_path) if sector_map_path else None

    capital_tier_overrides: tuple[tuple[float, float], ...] = ()
    if capital_tier.strip():
        parsed: list[tuple[float, float]] = []
        for item in capital_tier.split(","):
            piece = item.strip()
            if not piece or ":" not in piece:
                continue
            threshold_str, rate_str = piece.split(":", 1)
            parsed.append((float(threshold_str), float(rate_str)))
        capital_tier_overrides = tuple(parsed)

    timing_plan_frame = None
    if timing_gate:
        from quantagent.agents.technical_timing_agent import compute_technical_timing

        timing_plan_frame = compute_technical_timing(read_frame(market_panel_path))

    position_state_path = (
        default_target_weights_root() / "position_state.parquet"
        if holding_period_mode != "off"
        else None
    )

    weights_result = build_v7_target_weights(
        predictions_frame,
        read_frame(market_panel_path),
        sector_map=sector_frame,
        config=V7TargetWeightsConfig(
            top_k=top_k,
            top_k_ratio=top_k_ratio,
            min_selection_pressure=min_selection_pressure,
            fail_if_top_k_covers_universe=fail_if_top_k_covers_universe,
            selection_mode=selection_mode,
            alpha_threshold=alpha_threshold,
            confidence_floor=confidence_floor,
            selection_top_k_min=selection_top_k_min,
            selection_top_k_max=selection_top_k_max,
            max_weight_per_name=max_weight_per_name,
            max_sector_weight=max_sector_weight,
            max_turnover=max_turnover,
            optimizer_backend=optimizer_backend,
            objective=objective,
            cash_floor=cash_floor,
            capital_yuan=initial_cash,
            dynamic_top_k_enabled=dynamic_top_k,
            top_k_min=top_k_min,
            top_k_max=top_k_max,
            timing_gate_enabled=timing_gate,
            holding_period_mode=holding_period_mode,
            holding_period_max_delta=holding_period_max_delta,
            capital_tier_overrides=capital_tier_overrides,
        ),
        timing_plan=timing_plan_frame,
        position_state_path=position_state_path,
    )
    weights_result.diagnostics["multi_horizon_blend"] = blender_diagnostics
    weights_result.diagnostics["training_dataset_symbol_count"] = (
        int(training_dataset["symbol"].nunique()) if "symbol" in training_dataset.columns else 0
    )
    weights_result.diagnostics["training_dataset_row_count"] = int(len(training_dataset))
    weights_path = default_target_weights_root() / "target_weights.parquet"
    written_weights = write_v7_target_weights(weights_result, weights_path)

    reports_root = default_reports_root()
    reports_root.mkdir(parents=True, exist_ok=True)
    backtest_path = reports_root / "walk_forward_backtest.json"
    acceptance_report_path = reports_root / "acceptance_report.json"
    backtest_status: dict[str, object] = {"status": "skipped", "reason": "no_target_weights"}
    paper_report_status: dict[str, object] = {"status": "skipped", "reason": "no_target_weights"}
    quant_acceptance_status = "not_evaluated"
    failure_reasons: list[str] = []
    if not weights_result.target_weights.empty:
        weights_frame = weights_result.target_weights.copy()
        if "trade_date" in weights_frame.columns:
            weights_frame = weights_frame.set_index("trade_date")
        market_frame = _restrict_market_for_paper(
            read_frame(market_panel_path),
            weights_frame,
            benchmark_symbol=benchmark_symbol,
        )
        paper_dir = Path(paper_report_output_dir) if paper_report_output_dir is not None else reports_root / "paper_report"
        report_weights_path = write_frame(weights_result.target_weights, paper_dir / "target_weights.parquet")
        sim = simulate_ashare_target_weights(
            weights_frame,
            market_frame,
            AShareExecutionSimulationConfig(
                initial_cash=initial_cash,
                min_order_value_yuan=min_order_value_yuan,
                audit_log_dir=str(paper_dir / "audit"),
            ),
        )
        backtest_path.write_text(
            json_dump(
                {
                    "nav": _series_to_json_dict(sim.nav),
                    "orders": sim.order_audit.to_dict("records"),
                    "failed_orders": sim.failed_order_audit.to_dict("records"),
                    "skipped_orders": sim.skipped_order_audit.to_dict("records"),
                    "holdings": sim.position_history.to_dict("records"),
                    "config": sim.config,
                }
            ),
            encoding="utf-8",
        )
        paper_result = write_paper_report(
            sim,
            market_panel=market_frame,
            config=PaperReportConfig(
                initial_cash=initial_cash,
                benchmark_symbol=benchmark_symbol,
                output_dir=paper_dir,
                target_weights_path=str(report_weights_path),
            ),
        )
        acceptance_metrics = _build_full_pipeline_acceptance_metrics(
            training_result.metrics,
            paper_result.summary,
            weights_result.diagnostics,
            training_dataset,
            predictions_frame,
            benchmark_symbol,
        )
        acceptance = evaluate_model_acceptance_gates(
            acceptance_metrics,
            V7ModelAcceptanceGateConfig(
                require_paper_report=mark_production_ready,
                require_benchmark=mark_production_ready,
                min_training_symbols=max(50 if mark_production_ready else 1, int(min_symbols)),
                min_prediction_symbols=50 if mark_production_ready else 1,
                min_effective_universe_by_date=50 if mark_production_ready else 1,
                min_selection_pressure=min_selection_pressure,
            ),
            paper_report_path=paper_dir / "paper_report.json",
        )
        acceptance_report_path.write_text(json_dump(acceptance.to_dict()), encoding="utf-8")
        paper_result = write_paper_report(
            sim,
            market_panel=market_frame,
            config=PaperReportConfig(
                initial_cash=initial_cash,
                benchmark_symbol=benchmark_symbol,
                output_dir=paper_dir,
                target_weights_path=str(report_weights_path),
                acceptance_report_path=acceptance_report_path,
            ),
        )
        quant_acceptance_status = paper_result.quant_acceptance_status
        failure_reasons = list(acceptance.failures)
        backtest_status = {
            "status": "ok",
            "output": str(backtest_path),
            "failed_orders": int(len(sim.failed_order_audit)),
            "skipped_orders": int(len(sim.skipped_order_audit)),
        }
        paper_report_status = {
            "status": paper_result.status,
            "report_generation_status": "passed",
            "quant_acceptance_status": paper_result.quant_acceptance_status,
            "output_dir": paper_result.output_dir,
            "summary": paper_result.summary,
            "files": paper_result.files,
        }

    pipeline_report = {
        "training_dataset": dataset_result.summary,
        "training": training_result,
        "predictions": {
            "output": str(written_predictions),
            "horizons": [primary_horizon],
            "model_kind": model,
            "sample_role": "validation",
        },
        "target_weights": {
            "output": str(written_weights),
            "diagnostics": weights_result.diagnostics,
        },
        "backtest": backtest_status,
        "paper_report": paper_report_status,
        "acceptance_report": str(acceptance_report_path) if acceptance_report_path.exists() else None,
        "TRAINING_STATUS": training_result.status,
        "PAPER_REPORT_STATUS": paper_report_status.get("report_generation_status", paper_report_status.get("status")),
        "QUANT_ACCEPTANCE_STATUS": quant_acceptance_status,
        "FAILURE_REASONS": failure_reasons,
    }
    pipeline_report_path = reports_root / "full_pipeline_report.json"
    pipeline_report_path.write_text(json_dump(pipeline_report), encoding="utf-8")
    typer.echo(json_dump(pipeline_report))


def _series_to_json_dict(series: "pd.Series") -> dict[str, object]:
    return {str(key.date() if hasattr(key, "date") else key): value for key, value in series.to_dict().items()}


def _build_full_pipeline_acceptance_metrics(
    training_metrics: dict[str, object],
    paper_summary: dict[str, object],
    weight_diagnostics: dict[str, object],
    training_dataset: "pd.DataFrame",
    predictions: "pd.DataFrame",
    benchmark_symbol: str | None,
) -> dict[str, object]:
    metrics = dict(training_metrics)
    metrics.update(
        {
            "turnover_adjusted_net_return": paper_summary.get("turnover_adjusted_net_return", paper_summary.get("net_return_after_estimated_costs", 0.0)),
            "max_drawdown": paper_summary.get("max_drawdown", 0.0),
            "benchmark_symbol": benchmark_symbol,
            "benchmark_return": paper_summary.get("benchmark_return"),
            "excess_return": paper_summary.get("excess_return"),
            "excess_return_after_costs": paper_summary.get("excess_return_after_costs", paper_summary.get("excess_return", 0.0)),
            "benchmark_excess_return": paper_summary.get("excess_return"),
            "selection_pressure_min": weight_diagnostics.get("selection_pressure_min", 0.0),
            "selection_pressure_mean": weight_diagnostics.get("selection_pressure_mean", 0.0),
            "prediction_symbol_count": int(predictions["symbol"].nunique()) if "symbol" in predictions.columns else 0,
            "training_dataset_symbol_count": int(training_dataset["symbol"].nunique()) if "symbol" in training_dataset.columns else 0,
            "training_dataset_rows": int(len(training_dataset)),
            "training_dataset_date_count": int(training_dataset["trade_date"].nunique()) if "trade_date" in training_dataset.columns else 0,
        }
    )
    eligible = weight_diagnostics.get("eligible_symbol_count_by_date", {})
    if isinstance(eligible, dict) and eligible:
        metrics["eligible_symbol_count_min"] = int(min(int(value) for value in eligible.values()))
        metrics["effective_universe_min"] = metrics["eligible_symbol_count_min"]
    else:
        metrics["eligible_symbol_count_min"] = 0
        metrics["effective_universe_min"] = 0
    if "pit_violation_count" not in metrics:
        metrics["pit_violation_count"] = 0
    return metrics


def _restrict_market_for_paper(
    market_frame: "pd.DataFrame",
    weights_frame: "pd.DataFrame",
    benchmark_symbol: str | None = None,
) -> "pd.DataFrame":
    if market_frame is None or market_frame.empty or weights_frame is None or weights_frame.empty:
        return market_frame
    data = market_frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    dates = pd.to_datetime(pd.Index(weights_frame.index), errors="coerce")
    dates = dates[~pd.isna(dates)]
    symbols = {str(column) for column in weights_frame.columns if str(column) != "trade_date"}
    if benchmark_symbol:
        symbols.add(str(benchmark_symbol))
    mask = data["trade_date"].isin(set(dates)) & data["symbol"].astype(str).isin(symbols)
    return data.loc[mask].reset_index(drop=True)


def _existing_table_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.with_suffix(".csv")
    return fallback if fallback.exists() else path


def _market_manifest_is_usable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("quality_status") in {"passed", "warning"} and int(payload.get("row_count") or 0) > 0


def _load_oos_predictions(path: Path, primary_horizon: int) -> "pd.DataFrame":
    import pandas as pd

    frame = read_frame(path)
    if "sample_role" not in frame.columns or set(frame["sample_role"].astype(str)) != {"validation"}:
        raise ValueError("run-full-real-training-v7 requires validation-only out-of-sample predictions")
    if "horizon" not in frame.columns:
        raise ValueError("walk-forward predictions are missing horizon")
    selected = frame[frame["horizon"].astype(int) == int(primary_horizon)].copy()
    if selected.empty:
        raise ValueError(f"no out-of-sample predictions found for horizon {primary_horizon}")
    required = {"symbol", "trade_date", "prediction"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"out-of-sample predictions missing columns {sorted(missing)}")
    selected["trade_date"] = pd.to_datetime(selected["trade_date"], errors="coerce")
    if selected["trade_date"].isna().any():
        raise ValueError("out-of-sample predictions contain invalid trade_date values")
    return selected[["symbol", "trade_date", "prediction", "sample_role", "fold_id"]].reset_index(drop=True)


@app.command("hp-search")
def hp_search(
    dataset_path: Path = typer.Option(None, "--dataset"),
    n_trials: int = typer.Option(100, "--n-trials"),
    gpu: bool = typer.Option(False, "--gpu/--no-gpu"),
    study_name: str = typer.Option("v7_alpha", "--study-name"),
    model: str = typer.Option("ft_transformer", "--model"),
    ft_batch_size: int = typer.Option(8192, "--ft-batch-size"),
    ft_max_epochs: int = typer.Option(60, "--ft-max-epochs"),
    require_gpu: bool = typer.Option(True, "--require-gpu/--no-require-gpu"),
) -> None:
    """Layer A: Optuna HP search over FT-Transformer and portfolio knobs."""
    from quantagent.optimization.optuna_search import OptunaSearchConfig, run_optuna_hp_search

    resolved_dataset = _default_training_dataset_path(dataset_path)
    result = run_optuna_hp_search(
        read_frame(resolved_dataset),
        OptunaSearchConfig(
            study_name=study_name,
            n_trials=n_trials,
            model=model,
            ft_device="cuda" if gpu else "auto",
            require_gpu=require_gpu if gpu else False,
            ft_batch_size=ft_batch_size,
            ft_max_epochs=ft_max_epochs,
        ),
    )
    typer.echo(json_dump(result.to_dict()))


@app.command("evolve-factors")
def evolve_factors(
    dataset_path: Path = typer.Option(None, "--dataset"),
    generations: int = typer.Option(30, "--generations"),
    population: int = typer.Option(60, "--population"),
    seed_from_optuna: str = typer.Option("v7_alpha", "--seed-from-optuna"),
    model: str = typer.Option("ridge", "--model"),
) -> None:
    """Layer B: GA search over factor mask, horizon blend, and ensemble weights."""
    from quantagent.optimization.factor_evolution import FactorEvolutionConfig, run_factor_evolution

    resolved_dataset = _default_training_dataset_path(dataset_path)
    result = run_factor_evolution(
        read_frame(resolved_dataset),
        FactorEvolutionConfig(
            generations=generations,
            population=population,
            seed_from_optuna=seed_from_optuna,
            model=model,
        ),
    )
    typer.echo(json_dump(result.to_dict()))


@app.command("train-rl-agent")
def train_rl_agent(
    predictions_path: Path = typer.Option(None, "--predictions"),
    market_panel_path: Path = typer.Option(None, "--market-panel"),
    timesteps: int = typer.Option(2_000_000, "--timesteps"),
    device: str = typer.Option("cuda", "--device"),
    env_config: Path | None = typer.Option(None, "--env-config"),
    n_envs: int = typer.Option(4, "--n-envs"),
    require_gpu: bool = typer.Option(True, "--require-gpu/--no-require-gpu"),
) -> None:
    """Layer C: train a PPO portfolio delta policy on paper/backtest data."""
    from quantagent.rl.portfolio_env import PortfolioEnvConfig
    from quantagent.rl.train_ppo import PPOTrainingConfig, train_ppo_policy

    resolved_predictions = predictions_path or (quant_paths().predictions / "predictions.parquet")
    resolved_market = market_panel_path or (quant_paths().data_root / "v7" / "silver" / "market_panel" / "market_panel.parquet")
    env_kwargs = _load_env_config(env_config)
    result = train_ppo_policy(
        read_frame(resolved_predictions),
        read_frame(resolved_market),
        PPOTrainingConfig(
            timesteps=timesteps,
            device=device,
            n_envs=n_envs,
            require_gpu=require_gpu,
            env=PortfolioEnvConfig(**env_kwargs),
        ),
    )
    typer.echo(json_dump(result))


@app.command("autopilot")
def autopilot(
    dataset_path: Path = typer.Option(None, "--dataset"),
    market_panel_path: Path | None = typer.Option(None, "--market-panel"),
    predictions_path: Path | None = typer.Option(None, "--predictions"),
    n_trials: int = typer.Option(100, "--n-trials"),
    generations: int = typer.Option(30, "--generations"),
    timesteps: int = typer.Option(2_000_000, "--timesteps"),
    study_name: str = typer.Option("v7_alpha", "--study-name"),
    require_gpu: bool = typer.Option(True, "--require-gpu/--no-require-gpu"),
    report_out: Path | None = typer.Option(None, "--report-out"),
) -> None:
    """Run Layer A -> B -> C and write a unified research report."""
    result = _run_autopilot_impl(
        dataset_path=_default_training_dataset_path(dataset_path),
        market_panel_path=market_panel_path,
        predictions_path=predictions_path,
        n_trials=n_trials,
        generations=generations,
        timesteps=timesteps,
        study_name=study_name,
        require_gpu=require_gpu,
        report_out=report_out,
    )
    typer.echo(json_dump(result))


@app.command("run-full-ai-quant-v7")
def run_full_ai_quant_v7(
    symbols: str = typer.Option(
        "auto",
        "--symbols",
        help="Comma-separated A-share symbols, or 'auto' to use local Qlib features / AkShare universe.",
    ),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    max_symbols: int = typer.Option(0, "--max-symbols", help="0 means no cap for the resolved universe."),
    provider_uri: Path | None = typer.Option(None, "--provider-uri", help="Local Qlib provider_uri for symbol/date discovery."),
    market_panel_path: Path | None = typer.Option(None, "--market-panel"),
    allow_network: bool = typer.Option(False, "--allow-network", help="Enable AkShare online loading explicitly."),
    refresh_akshare_market: bool = typer.Option(False, "--refresh-akshare-market"),
    refresh_fundamentals: bool = typer.Option(False, "--refresh-fundamentals"),
    refresh_valuation: bool = typer.Option(False, "--refresh-valuation"),
    refresh_sector_map: bool = typer.Option(False, "--refresh-sector-map"),
    start_date: str | None = typer.Option(None, "--start-date"),
    end_date: str | None = typer.Option(None, "--end-date"),
    as_of_date: str | None = typer.Option(None, "--as-of-date"),
    model: str = typer.Option("ft_transformer", "--model"),
    require_gpu: bool = typer.Option(True, "--require-gpu/--no-require-gpu"),
    ft_device: str = typer.Option("cuda", "--ft-device"),
    ft_max_epochs: int = typer.Option(60, "--ft-max-epochs"),
    ft_batch_size: int = typer.Option(8192, "--ft-batch-size"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    primary_horizon: int = typer.Option(5, "--primary-horizon"),
    split_mode: str = typer.Option("rolling", "--split-mode"),
    valid_size_days: int = typer.Option(20, "--valid-size-days"),
    min_train_days: int = typer.Option(120, "--min-train-days"),
    rolling_train_days: int = typer.Option(756, "--rolling-train-days"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    purge_days: int | None = typer.Option(None, "--purge-days", help="Defaults to max configured label horizon."),
    min_rows: int = typer.Option(1000, "--min-rows"),
    min_train_rows: int = typer.Option(1000, "--min-train-rows"),
    min_symbols: int = typer.Option(50, "--min-symbols"),
    min_dates: int = typer.Option(252, "--min-dates"),
    top_k: int = typer.Option(30, "--top-k"),
    top_k_ratio: float | None = typer.Option(0.10, "--top-k-ratio"),
    min_selection_pressure: float = typer.Option(3.0, "--min-selection-pressure"),
    selection_mode: str = typer.Option("ai_threshold", "--selection-mode", help="ai_threshold | top_k"),
    alpha_threshold: float = typer.Option(0.0, "--alpha-threshold"),
    confidence_floor: float = typer.Option(0.55, "--confidence-floor"),
    selection_top_k_min: int = typer.Option(5, "--selection-top-k-min"),
    selection_top_k_max: int = typer.Option(100, "--selection-top-k-max"),
    max_weight_per_name: float = typer.Option(0.10, "--max-weight"),
    max_sector_weight: float = typer.Option(0.30, "--max-sector"),
    max_turnover: float = typer.Option(0.40, "--max-turnover"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash"),
    min_order_value_yuan: float = typer.Option(100.0, "--min-order-value-yuan"),
    dynamic_top_k: bool = typer.Option(True, "--dynamic-top-k/--no-dynamic-top-k"),
    timing_gate: bool = typer.Option(True, "--timing-gate/--no-timing-gate"),
    holding_period_mode: str = typer.Option("soft", "--holding-period-mode"),
    capital_tier: str = typer.Option("1000000:0.10,10000000:0.05,100000000:0.02", "--capital-tier"),
    run_autopilot_search: bool = typer.Option(True, "--run-autopilot-search/--skip-autopilot-search"),
    n_trials: int = typer.Option(100, "--n-trials"),
    generations: int = typer.Option(30, "--generations"),
    rl_timesteps: int = typer.Option(5_000_000, "--rl-timesteps"),
    allow_model_downgrade: bool = typer.Option(False, "--allow-model-downgrade"),
) -> None:
    """Full V7 AI quant research autopilot.

    This is the opinionated "full data, full dates" entrypoint. It
    ingests/refreshes requested AkShare layers, rebuilds PIT labels and the
    gold dataset, trains all configured horizons, builds A-share-safe target
    weights, runs the T+1 / 100-share / liquidity paper simulator, and then
    optionally launches Optuna + GA + RL research search.

    It never enables live trading and never emits broker orders.
    """

    stages = _prepare_full_ai_quant_inputs(
        symbols=symbols,
        symbols_file=symbols_file,
        max_symbols=max_symbols,
        provider_uri=provider_uri,
        market_panel_path=market_panel_path,
        allow_network=allow_network,
        refresh_akshare_market=refresh_akshare_market,
        refresh_fundamentals=refresh_fundamentals,
        refresh_valuation=refresh_valuation,
        refresh_sector_map=refresh_sector_map,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        horizons=horizons,
    )

    run_full_real_training_v7(
        market_panel_path=Path(str(stages["market_panel_path"])),
        labels_path=Path(str(stages["labels_path"])),
        output_dir=quant_paths().models / "v7_alpha_full_ai",
        fundamentals_root=Path(str(stages["fundamentals_root"])) if stages.get("fundamentals_root") else None,
        valuation_path=Path(str(stages["valuation_path"])) if stages.get("valuation_path") else None,
        disclosures_path=None,
        sector_map_path=Path(str(stages["sector_map_path"])) if stages.get("sector_map_path") else None,
        training_dataset_path=Path(str(stages["training_dataset_path"])),
        symbols=",".join(stages["symbols"]),
        symbols_file=None,
        model=model,
        horizons=horizons,
        primary_horizon=primary_horizon,
        min_rows=min_rows,
        min_train_rows=min_train_rows,
        split_mode=split_mode,
        valid_size_days=valid_size_days,
        min_train_days=min_train_days,
        rolling_train_days=rolling_train_days,
        embargo_days=embargo_days,
        purge_days=purge_days,
        n_splits=4,
        ft_max_epochs=ft_max_epochs,
        ft_batch_size=ft_batch_size,
        ft_device=ft_device,
        require_gpu=require_gpu,
        min_symbols=min_symbols,
        min_dates=min_dates,
        top_k=top_k,
        top_k_ratio=top_k_ratio,
        min_selection_pressure=min_selection_pressure,
        fail_if_top_k_covers_universe=True,
        selection_mode=selection_mode,
        alpha_threshold=alpha_threshold,
        confidence_floor=confidence_floor,
        selection_top_k_min=selection_top_k_min,
        selection_top_k_max=selection_top_k_max,
        max_weight_per_name=max_weight_per_name,
        max_sector_weight=max_sector_weight,
        max_turnover=max_turnover,
        optimizer_backend="auto",
        objective="max_expected_alpha",
        cash_floor=0.0,
        initial_cash=initial_cash,
        min_order_value_yuan=min_order_value_yuan,
        benchmark_symbol=None,
        paper_report_output_dir=None,
        mark_production_ready=False,
        paper_report=None,
        allow_model_downgrade=allow_model_downgrade,
        multi_horizon_blend=True,
        dynamic_top_k=dynamic_top_k,
        top_k_min=8,
        top_k_max=50,
        timing_gate=timing_gate,
        holding_period_mode=holding_period_mode,
        holding_period_max_delta=0.02,
        capital_tier=capital_tier,
    )

    autopilot_status: dict[str, object]
    if run_autopilot_search:
        autopilot_status = _run_autopilot_impl(
            dataset_path=Path(str(stages["training_dataset_path"])),
            market_panel_path=Path(str(stages["market_panel_path"])),
            predictions_path=quant_paths().predictions / "predictions.parquet",
            n_trials=n_trials,
            generations=generations,
            timesteps=rl_timesteps,
            study_name="v7_full_ai",
            require_gpu=require_gpu,
            report_out=quant_paths().reports / "autopilot" / "v7_full_ai.html",
        )
    else:
        autopilot_status = {"status": "skipped"}

    typer.echo(
        json_dump(
            {
                "status": "passed",
                "safe_execution": "target_weights_and_paper_simulation_only; live_trading_disabled",
                "ashare_constraints": {
                    "t_plus_1": True,
                    "lot_size": 100,
                    "min_order_value_yuan": min_order_value_yuan,
                    "limit_up_down_blocks": True,
                    "suspension_and_st_blocks": True,
                },
                "stages": stages,
                "autopilot": autopilot_status,
            }
        )
    )


def _run_autopilot_impl(
    *,
    dataset_path: Path,
    market_panel_path: Path | None,
    predictions_path: Path | None,
    n_trials: int,
    generations: int,
    timesteps: int,
    study_name: str,
    require_gpu: bool,
    report_out: Path | None = None,
) -> dict[str, object]:
    from datetime import datetime

    from quantagent.optimization.factor_evolution import FactorEvolutionConfig, run_factor_evolution
    from quantagent.optimization.optuna_search import OptunaSearchConfig, run_optuna_hp_search
    from quantagent.rl.train_ppo import PPOTrainingConfig, train_ppo_policy

    dataset = read_frame(dataset_path)
    stages: dict[str, object] = {"dataset": str(dataset_path)}
    hp = run_optuna_hp_search(
        dataset,
        OptunaSearchConfig(
            study_name=study_name,
            n_trials=n_trials,
            ft_device="cuda" if require_gpu else "auto",
            require_gpu=require_gpu,
        ),
    )
    stages["layer_a_optuna"] = hp.to_dict()
    ga = run_factor_evolution(
        dataset,
        FactorEvolutionConfig(
            generations=generations,
            population=max(8, min(60, generations * 20)),
            seed_from_optuna=study_name,
        ),
    )
    stages["layer_b_factor_evolution"] = ga.to_dict()
    rl_status: dict[str, object]
    resolved_predictions = predictions_path or (quant_paths().predictions / "predictions.parquet")
    resolved_market = market_panel_path or (quant_paths().data_root / "v7" / "silver" / "market_panel" / "market_panel.parquet")
    if Path(resolved_predictions).exists() and Path(resolved_market).exists() and timesteps > 0:
        rl_status = train_ppo_policy(
            read_frame(Path(resolved_predictions)),
            read_frame(Path(resolved_market)),
            PPOTrainingConfig(timesteps=timesteps, require_gpu=require_gpu, device="cuda" if require_gpu else "auto"),
        )
    else:
        rl_status = {
            "status": "skipped",
            "reason": "predictions_or_market_panel_missing_or_timesteps_zero",
            "predictions": str(resolved_predictions),
            "market_panel": str(resolved_market),
        }
    stages["layer_c_rl"] = rl_status
    report_path = report_out or (
        quant_paths().reports / "autopilot" / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.html"
    )
    _write_autopilot_report(Path(report_path), stages)
    stages["report_out"] = str(report_path)
    stages["safe_execution"] = "research_only_target_weights_downstream; live_trading_disabled"
    return stages


def _prepare_full_ai_quant_inputs(
    *,
    symbols: str,
    symbols_file: Path | None,
    max_symbols: int,
    provider_uri: Path | None,
    market_panel_path: Path | None,
    allow_network: bool,
    refresh_akshare_market: bool,
    refresh_fundamentals: bool,
    refresh_valuation: bool,
    refresh_sector_map: bool,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
    horizons: str,
) -> dict[str, object]:
    from quantagent.data.bootstrap.akshare_bootstrap import AkShareBootstrapConfig, build_akshare_financial_cache
    from quantagent.data.bootstrap.akshare_market_bootstrap import AkShareMarketPanelConfig, build_akshare_market_panel
    from quantagent.data.bootstrap.valuation_bootstrap import ValuationBootstrapConfig, build_valuation_cache
    from quantagent.data.lake import v7_lake_paths
    from quantagent.data.providers.akshare_valuation_provider import AkShareUniverseProvider
    from quantagent.data.v7_label_builder import build_forward_return_labels

    paths = quant_paths().ensure()
    lake = v7_lake_paths(default_v7_lake_root()).ensure()
    resolved_provider_uri = Path(provider_uri) if provider_uri else paths.raw / "qlib" / "cn_data"
    resolved_symbols = _resolve_full_ai_symbols(
        symbols=symbols,
        symbols_file=symbols_file,
        provider_uri=resolved_provider_uri,
        allow_network=allow_network,
        max_symbols=max_symbols,
        universe_provider=AkShareUniverseProvider,
    )
    if not resolved_symbols:
        raise typer.BadParameter("No symbols resolved. Prepare Qlib features, pass --symbols/--symbols-file, or use --allow-network.")

    stages: dict[str, object] = {
        "symbols": list(resolved_symbols),
        "symbol_count": len(resolved_symbols),
        "provider_uri": str(resolved_provider_uri),
    }
    if (refresh_fundamentals or refresh_valuation) and not allow_network:
        requested = ", ".join(
            name
            for name, enabled in (
                ("--refresh-fundamentals", refresh_fundamentals),
                ("--refresh-valuation", refresh_valuation),
            )
            if enabled
        )
        raise typer.BadParameter(
            f"{requested} can pull AkShare data into the PIT cache and requires explicit --allow-network."
        )

    if refresh_akshare_market:
        market_result = build_akshare_market_panel(
            AkShareMarketPanelConfig(
                symbols=resolved_symbols,
                start_date=start_date,
                end_date=end_date,
                output_root=str(lake.root),
                allow_network=allow_network,
                provider_uri_for_range=str(resolved_provider_uri),
                as_of_date=as_of_date,
            )
        )
        if market_result["status"] != "passed":
            raise RuntimeError(f"AkShare market refresh failed or empty: {market_result}")
        resolved_market_panel = Path(str(market_result["output"]))
        stages["market_refresh"] = market_result
    elif market_panel_path is not None:
        resolved_market_panel = Path(market_panel_path)
    else:
        resolved_market_panel = _existing_table_path(lake.silver_market_panel / "market_panel.parquet")
    if not resolved_market_panel.exists():
        raise typer.BadParameter(
            f"market panel not found: {resolved_market_panel}. Pass --market-panel or enable --refresh-akshare-market --allow-network."
        )

    fundamentals_root = lake.silver_fundamentals
    if refresh_fundamentals:
        financial_result = build_akshare_financial_cache(
            AkShareBootstrapConfig(
                start_date=start_date or "1990-01-01",
                end_date=end_date or as_of_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
                symbols=resolved_symbols,
                allow_network=allow_network,
                lake_root=str(lake.root),
            )
        )
        stages["fundamentals_refresh"] = financial_result
    has_fundamentals = any((fundamentals_root / name).exists() for name in ("income.parquet", "income.csv"))

    valuation_path = _existing_table_path(lake.silver_valuation / "valuation.parquet")
    if refresh_valuation:
        valuation_result = build_valuation_cache(
            ValuationBootstrapConfig(
                as_of_dates=parse_csv_tuple(as_of_date or end_date or pd.Timestamp.today().strftime("%Y-%m-%d")),
                symbols=resolved_symbols,
                lake_root=str(lake.root),
                allow_network=allow_network,
            )
        )
        valuation_path = Path(str(valuation_result["output_path"]))
        stages["valuation_refresh"] = valuation_result

    sector_map_path = _existing_table_path(lake.root / "silver" / "sector_map" / "sector_map.parquet")
    if refresh_sector_map:
        sector_map_path = _build_akshare_sector_map(
            symbols=resolved_symbols,
            lake_root=lake.root,
            allow_network=allow_network,
            as_of_date=as_of_date or end_date,
        )
        stages["sector_map_refresh"] = {"status": "passed", "output": str(sector_map_path)}

    label_result = build_forward_return_labels(read_frame(resolved_market_panel), tuple(int(item) for item in parse_csv_tuple(horizons)))
    labels_path = write_frame(label_result.frame, lake.root / "labels.parquet")
    training_dataset_path = lake.gold_training_dataset / "training_dataset.parquet"
    stages.update(
        {
            "market_panel_path": str(resolved_market_panel),
            "labels_path": str(labels_path),
            "training_dataset_path": str(training_dataset_path),
            "fundamentals_root": str(fundamentals_root) if has_fundamentals or refresh_fundamentals else None,
            "valuation_path": str(valuation_path) if valuation_path.exists() else None,
            "sector_map_path": str(sector_map_path) if sector_map_path.exists() else None,
            "labels": {
                "rows": int(len(label_result.frame)),
                "horizons": list(parse_csv_tuple(horizons)),
            },
        }
    )
    return stages


def _resolve_full_ai_symbols(
    *,
    symbols: str,
    symbols_file: Path | None,
    provider_uri: Path,
    allow_network: bool,
    max_symbols: int,
    universe_provider: object,
) -> tuple[str, ...]:
    if symbols.strip().lower() != "auto":
        resolved = merge_symbols(symbols, symbols_file)
    else:
        resolved = list_qlib_feature_symbols(provider_uri, include_indices=False, max_symbols=max_symbols)
        extra = merge_symbols("", symbols_file)
        if extra:
            resolved = tuple(dict.fromkeys([*resolved, *extra]))
        if not resolved and allow_network:
            provider = universe_provider(allow_network=True)
            result = provider.list_universe()
            resolved = tuple(result.frame["symbol"].astype(str).tolist()) if not result.frame.empty else ()
    if max_symbols and len(resolved) > max_symbols:
        return tuple(resolved[:max_symbols])
    return tuple(resolved)


def _build_akshare_sector_map(
    *,
    symbols: tuple[str, ...],
    lake_root: Path,
    allow_network: bool,
    as_of_date: str | None,
) -> Path:
    from quantagent.data.manifest import build_manifest_for_frame
    from quantagent.data.providers.akshare_valuation_provider import (
        AKSHARE_SECTOR_REQUIRED_COLUMNS,
        AkShareSectorProvider,
    )
    from quantagent.data.providers.base import ProviderRequest

    output_path = lake_root / "silver" / "sector_map" / "sector_map.parquet"
    result = AkShareSectorProvider(allow_network=allow_network).industry_classification(
        ProviderRequest("", as_of_date or "", symbols=symbols),
        as_of_date=as_of_date,
    )
    written = write_frame(result.frame, output_path)
    manifest = build_manifest_for_frame(
        dataset_name="sector_map",
        vendor="akshare",
        frame=result.frame,
        output_paths=[written],
        symbols=symbols,
        required_columns=AKSHARE_SECTOR_REQUIRED_COLUMNS,
        warnings=result.warnings,
        extra={"source": result.source, "schema_report": result.metadata.get("schema_report", {})},
    )
    manifest.write(lake_root / "manifests" / "sector_map.json")
    return written


def _default_training_dataset_path(path: Path | None) -> Path:
    resolved = path or (quant_paths().data_root / "v7" / "gold" / "training_dataset" / "training_dataset.parquet")
    if not Path(resolved).exists():
        raise typer.BadParameter(
            f"training dataset not found: {resolved}. Build it with build-training-dataset-v7 or auto-train-v7 first."
        )
    return Path(resolved)


def _load_env_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise typer.BadParameter("YAML env config requires pyyaml") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise typer.BadParameter("env config must be a YAML object")
    env = payload.get("rl_env", payload.get("env", payload))
    if not isinstance(env, dict):
        raise typer.BadParameter("env config must contain an object")
    allowed = set(PortfolioEnvConfig.__dataclass_fields__) if "PortfolioEnvConfig" in globals() else {
        "top_n",
        "max_delta",
        "max_weight_per_name",
        "max_gross",
        "max_turnover",
        "cost_bps",
        "drawdown_lambda",
        "drawdown_limit",
        "kill_switch_drawdown",
        "initial_nav",
    }
    return {str(k): v for k, v in env.items() if str(k) in allowed}


def _write_autopilot_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json_dump(payload)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>V7 Autopilot Report</title></head>"
        "<body><h1>V7 Autopilot Report</h1><p>Live trading disabled; research artefacts only.</p>"
        f"<pre>{body}</pre></body></html>",
        encoding="utf-8",
    )
