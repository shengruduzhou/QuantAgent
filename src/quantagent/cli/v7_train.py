"""V7 training CLI: alpha training, evaluation, and real-data orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
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
from quantagent.research.verdict import (
    RETURN_DIFFERENCING_DAYS,
    ResearchRejection,
    reject_insufficient_oos,
    required_oos_days,
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


@app.command("synthesize-factors-v7")
def synthesize_factors_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    labels_path: Path | None = typer.Option(None, "--labels"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    rd_agent: bool = typer.Option(True, "--rd-agent/--legacy-ga", help="Use RD-Agent-style factor R&D loop by default; pass --legacy-ga for the old symbolic GA."),
    label_column: str = typer.Option("forward_return_5d", "--label-column"),
    rounds: int = typer.Option(4, "--rounds", help="RD-Agent-style proposal/evaluation loop count."),
    factors_per_round: int = typer.Option(3, "--factors-per-round", help="RD-Agent-style 1-5 factor tasks per loop."),
    population: int = typer.Option(80, "--population"),
    generations: int = typer.Option(20, "--generations"),
    top_k: int = typer.Option(20, "--top-k"),
    max_depth: int = typer.Option(4, "--max-depth"),
    validation_fraction: float = typer.Option(0.25, "--validation-fraction"),
    min_validation_rank_ic: float = typer.Option(0.0, "--min-validation-rank-ic"),
    fitness_sample_dates: int = typer.Option(400, "--fitness-sample-dates", help="0 disables date subsampling."),
    fitness_sample_symbols: int = typer.Option(500, "--fitness-sample-symbols", help="0 disables symbol subsampling."),
    seed: int = typer.Option(1729, "--seed"),
    warm_start_fraction: float = typer.Option(0.4, "--warm-start-fraction", help="Initial population fraction seeded from Alpha101-style templates."),
    icir_weight: float = typer.Option(0.05, "--icir-weight", help="Weight of |daily ICIR| in fitness."),
    reference_columns: str = typer.Option("", "--reference-columns", help="Comma-separated existing factor columns (from --labels) to decorrelate against."),
    max_reference_correlation: float = typer.Option(0.7, "--max-reference-correlation"),
    max_sota_correlation: float = typer.Option(0.99, "--max-sota-correlation", help="RD-Agent-style duplicate gate against accepted/reference factors."),
    use_llm: bool = typer.Option(False, "--use-llm/--no-use-llm", help="RD-Agent closed loop: after the blueprint warm-start round, let an LLM propose NEW DSL factor tasks from the accumulated trace + memory."),
    allow_network: bool = typer.Option(False, "--allow-network", help="Permit the LLM proposer to reach the network (required for --use-llm to actually call a model)."),
    llm_model: str = typer.Option("", "--llm-model", help="Override the proposer model id (else QUANTAGENT_LLM_MODEL / provider default)."),
    llm_start_round: int = typer.Option(1, "--llm-start-round", help="First round (0-based) that uses LLM proposals; round 0 always warm-starts from blueprints."),
    llm_candidates_per_round: int = typer.Option(4, "--llm-candidates-per-round", help="How many factors the LLM proposes per LLM round."),
    rag_escalation_round: int = typer.Option(3, "--rag-escalation-round", help="Round after which the research directive escalates from easy to higher-IC factors."),
    llm_timeout_seconds: float = typer.Option(360.0, "--llm-timeout-seconds", help="Per-call LLM timeout (thinking models need 300s+)."),
    memory_path: Path | None = typer.Option(None, "--memory-path", help="Persistent JSONL of accept/reject knowledge digested into each LLM prompt and across runs."),
    train_end: str = typer.Option("", "--train-end", help="Trustworthy-OOS cutoff: the search (GA/LLM) only sees trade_date <= this; later dates stay clean for evaluate-discovered-factors. Empty = no cutoff."),
    exclude_st: bool = typer.Option(False, "--exclude-st", help="Tradability guard: also drop ST names from the validation/fitness IC (limit-up/down/suspended are always dropped when those flags are in the panel). Prevents accepting phantom edge over untradable names."),
    min_validation_icir: float = typer.Option(0.0, "--min-validation-icir", help="Stability floor: reject factors whose tradable validation ICIR is below this (rd-agent loop only). 0 disables."),
) -> None:
    """Discover factors in the safe expression DSL.

    By default this uses an RD-Agent-style loop: hypothesis -> factor tasks
    -> implementation/value gate -> validation -> SOTA library feedback. Pass
    ``--legacy-ga`` to run the original one-shot symbolic GA.

    The command writes definitions that can be fed back into
    ``build-training-dataset-v7 --factor-library alpha181 --synthesized-factors``.
    """
    from quantagent.factors.factor_synthesis import (
        RDAgentFactorLoopConfig,
        SymbolicGAConfig,
        save_result,
        synthesize_factors,
        synthesize_factors_rd_agent,
    )

    resolved_output = Path(output_dir) if output_dir is not None else default_reports_root() / "v7" / "factor_synthesis"
    panel = read_frame(market_panel_path)
    labels = read_frame(labels_path) if labels_path else None
    if train_end.strip():
        import pandas as _pd

        cutoff = _pd.Timestamp(train_end.strip())
        before = len(panel)
        panel = panel[_pd.to_datetime(panel["trade_date"], errors="coerce") <= cutoff].reset_index(drop=True)
        if labels is not None and "trade_date" in labels.columns:
            labels = labels[_pd.to_datetime(labels["trade_date"], errors="coerce") <= cutoff].reset_index(drop=True)
        typer.echo(
            json_dump({"event": "train_end_cutoff", "train_end": train_end.strip(),
                       "panel_rows_before": int(before), "panel_rows_after": int(len(panel))})
        )
    refs = tuple(c.strip() for c in reference_columns.split(",") if c.strip())
    if rd_agent:
        config = RDAgentFactorLoopConfig(
            rounds=rounds,
            factors_per_round=factors_per_round,
            top_k=top_k,
            label_column=label_column,
            validation_fraction=validation_fraction,
            min_validation_rank_ic=min_validation_rank_ic,
            min_finite_ratio=0.3,
            fitness_sample_dates=fitness_sample_dates,
            fitness_sample_symbols=fitness_sample_symbols,
            seed=seed,
            icir_weight=icir_weight,
            reference_columns=refs,
            max_reference_correlation=max_reference_correlation,
            max_sota_correlation=max_sota_correlation,
            exclude_st=exclude_st,
            min_validation_icir=min_validation_icir,
            use_llm=use_llm,
            allow_network=allow_network,
            llm_model=(llm_model or None),
            llm_start_round=llm_start_round,
            llm_candidates_per_round=llm_candidates_per_round,
            rag_escalation_round=rag_escalation_round,
            llm_timeout_seconds=llm_timeout_seconds,
            memory_path=(str(memory_path) if memory_path is not None else None),
        )
        result = synthesize_factors_rd_agent(panel, labels=labels, config=config)
    else:
        config = SymbolicGAConfig(
            population=population,
            generations=generations,
            max_depth=max_depth,
            top_k=top_k,
            label_column=label_column,
            validation_fraction=validation_fraction,
            min_validation_rank_ic=min_validation_rank_ic,
            fitness_sample_dates=fitness_sample_dates,
            fitness_sample_symbols=fitness_sample_symbols,
            seed=seed,
            warm_start_fraction=warm_start_fraction,
            icir_weight=icir_weight,
            reference_columns=refs,
            max_reference_correlation=max_reference_correlation,
            exclude_st=exclude_st,
        )
        result = synthesize_factors(panel, labels=labels, config=config)
    paths = save_result(result, resolved_output)
    typer.echo(
        json_dump(
            {
                "status": "passed",
                "mode": "rd-agent" if rd_agent else "legacy-ga",
                "definitions": paths["definitions"],
                "leaderboard": paths["leaderboard"],
                "history": paths["history"],
                "rd_agent_trace": paths.get("rd_agent_trace"),
                "rd_agent_task_feedback": paths.get("rd_agent_task_feedback"),
                "selected": int(len(result.definitions)),
                "config": asdict(config),
            }
        )
    )


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
    top_k_candidates: str = typer.Option(
        "10,20,30,50,80",
        "--top-k-candidates",
        help="Comma-separated exact Top-K values evaluated on the portfolio-selection OOS segment.",
    ),
    stock_selection_modes: str = typer.Option(
        "none,fundamental",
        "--stock-selection-modes",
        help="Comma-separated candidates: none,fundamental.",
    ),
    fundamental_selection_threshold: float = typer.Option(
        0.50, "--fundamental-selection-threshold", min=0.0, max=1.0
    ),
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
        acceptance_max_drawdown=0.10,
        acceptance_min_sharpe=None,
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
    factor_library: str = typer.Option("all_reviewed", "--factor-library", help="all_reviewed | basic | alpha101 | alpha181 | cicc_ashare80"),
    synthesized_factors_path: Path | None = typer.Option(None, "--synthesized-factors"),
    factor_min_finite_ratio: float = typer.Option(0.30, "--factor-min-finite-ratio"),
    macro_root: Path | None = typer.Option(None, "--macro-root"),
    flow_root: Path | None = typer.Option(None, "--flow-root"),
    index_root: Path | None = typer.Option(None, "--index-root"),
    enable_macro: bool = typer.Option(True, "--enable-macro/--no-enable-macro"),
    enable_flow: bool = typer.Option(True, "--enable-flow/--no-enable-flow"),
    enable_index: bool = typer.Option(True, "--enable-index/--no-enable-index"),
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
            factor_library=factor_library,
            synthesized_factors_path=str(synthesized_factors_path) if synthesized_factors_path else None,
            factor_min_finite_ratio=factor_min_finite_ratio,
            macro_root=str(macro_root) if macro_root else None,
            flow_root=str(flow_root) if flow_root else None,
            index_root=str(index_root) if index_root else None,
            enable_macro=enable_macro,
            enable_flow=enable_flow,
            enable_index=enable_index,
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
    objective_excess_weight: float = typer.Option(0.45, "--objective-excess-weight"),
    objective_annual_weight: float = typer.Option(0.30, "--objective-annual-weight"),
    objective_drawdown_weight: float = typer.Option(0.25, "--objective-drawdown-weight"),
    cash_floor: float = typer.Option(0.0, "--cash-floor"),
    weighting: str = typer.Option("rank", "--weighting", help="equal | rank | softmax"),
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
        weighting=weighting,
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


#: Peak resident memory the dataset build costs per symbol-day row, in bytes.
#: MEASURED, not estimated: a 400-symbol x 2,562-day build (1.02M rows) of the
#: `all_reviewed` factor library peaked at 15.3 GiB RSS on 2026-08-02. The
#: builder materialises many intermediate factor frames at once, which is why
#: the figure is far above the size of the finished parquet.
DATASET_BUILD_BYTES_PER_ROW = 15.3 * (2**30) / 1_020_000

#: Leave headroom so a build that fits on paper does not push the box into swap.
DATASET_BUILD_MEMORY_HEADROOM = 0.80


def _assert_dataset_build_fits_in_memory(
    *,
    labels_path: Path,
    symbols: list[str] | tuple[str, ...] | None,
    output_dir: Path,
) -> dict[str, object]:
    """Refuse a dataset build that the machine cannot hold.

    A full-universe build is offered in the web UI, but on a 62 GiB box the
    5,790-symbol panel needs several times that. Left unchecked the process runs
    for tens of minutes and is then OOM-killed, which surfaces as an opaque
    engineering failure rather than "this machine is too small for this scope".
    """
    import psutil

    from quantagent.research.verdict import ResearchRejection

    try:
        import pyarrow.compute as compute
        import pyarrow.dataset as parquet_dataset

        dataset = parquet_dataset.dataset(labels_path, format="parquet")
        names = dataset.schema.names
        if "symbol" not in names or "trade_date" not in names:
            return {"status": "unknown", "reason": "labels panel lacks symbol/trade_date"}
        table = dataset.to_table(columns=["symbol", "trade_date"])
        total_days = int(compute.count_distinct(table.column("trade_date")).as_py())
        panel_symbols = int(compute.count_distinct(table.column("symbol")).as_py())
        panel_rows = int(table.num_rows)
    except Exception as exc:  # pragma: no cover - the build will report it too
        return {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}

    selected = len({str(item) for item in (symbols or []) if str(item).strip()})
    symbol_count = selected or panel_symbols
    # Scale the panel's real row count rather than assuming every symbol trades
    # every day — listings, suspensions and delistings make symbols x days
    # overstate a full-universe panel by a third.
    projected_rows = (
        int(round(panel_rows * (selected / panel_symbols)))
        if selected and panel_symbols
        else panel_rows
    )
    projected_bytes = projected_rows * DATASET_BUILD_BYTES_PER_ROW
    available = int(psutil.virtual_memory().available)
    budget = available * DATASET_BUILD_MEMORY_HEADROOM

    report = {
        "status": "pass" if projected_bytes <= budget else "blocked",
        "symbolCount": symbol_count,
        "tradingDays": total_days,
        "projectedRows": projected_rows,
        "projectedPeakGiB": round(projected_bytes / 2**30, 1),
        "availableGiB": round(available / 2**30, 1),
        "usableGiB": round(budget / 2**30, 1),
        "basis": "measured 15.3 GiB peak RSS for 1.02M rows (400 symbols x 2,562 days)",
    }
    typer.echo(json_dump({
        "progress": 0.04,
        "stage": "resources",
        "message": (
            f"dataset build needs ~{report['projectedPeakGiB']} GiB for "
            f"{symbol_count} symbols x {total_days} days; "
            f"{report['usableGiB']} GiB usable"
        ),
        "resources": report,
    }))
    if projected_bytes > budget:
        raise ResearchRejection(
            code="insufficient_memory_for_scope",
            title="本机内存不足以构建该股票池的训练数据集，运行已在开始前中止",
            reasons=(
                f"{symbol_count} 只股票、{total_days} 个交易日，实际约 {projected_rows:,} 行，"
                f"预计峰值约 {report['projectedPeakGiB']} GiB",
                f"当前可用内存 {report['availableGiB']} GiB，"
                f"扣除余量后可用 {report['usableGiB']} GiB",
            ),
            stage="preflight",
            verdict="blocked",
            remediation=(
                "缩小股票池（--symbols-file），或复用已构建的数据集（--training-dataset），"
                "或换用内存更大的机器。继续运行只会在数十分钟后被 OOM 杀掉，"
                "并且看起来像是工程故障而不是规模问题。"
            ),
            metrics=report,
            output_dir=output_dir,
        )
    return report


def _assert_oos_budget_is_feasible(
    training_dataset: "pd.DataFrame",
    training_config: "V7TrainingConfig",
    *,
    primary_horizon: int,
    min_selection_days: int,
    min_holdout_days: int,
    output_dir: Path,
) -> dict[str, object]:
    """Abort before training when the fold budget cannot meet the protocol.

    The dataset is built at this point, so the usable date span per horizon —
    after feature warm-up and the label tail — is exactly known. Running the
    walk-forward first and discovering the shortfall in the portfolio stage
    costs the entire training run and reports a configuration error as if a
    research gate had refused the candidate.
    """
    import pandas as pd

    from quantagent.research.verdict import block_infeasible_oos_budget, required_oos_days
    from quantagent.training.splitters import plan_walk_forward
    from quantagent.training.v7_experiment import (
        _auto_feature_columns,
        shared_oos_anchor,
        walk_forward_split_config,
    )

    label_column = f"forward_return_{primary_horizon}d"
    if label_column not in training_dataset.columns:
        return {"status": "unknown", "reason": f"missing {label_column}"}
    # Resolved exactly as the trainer will resolve it: a pre-flight that counts
    # a different feature set counts a different number of usable dates, and
    # then disagrees with the run it was supposed to predict.
    feature_columns = list(
        training_config.feature_columns or _auto_feature_columns(training_dataset)
    )
    usable = training_dataset.dropna(subset=[label_column, *feature_columns])
    anchor_end = shared_oos_anchor(training_dataset, feature_columns, training_config)
    if anchor_end is not None:
        usable = usable[pd.to_datetime(usable["trade_date"], errors="coerce") <= anchor_end]
    usable_days = int(pd.to_datetime(usable["trade_date"], errors="coerce").dropna().nunique())

    split_cfg = walk_forward_split_config(usable, training_config)
    plan = plan_walk_forward(usable_days, split_cfg)
    required = required_oos_days(min_selection_days, min_holdout_days)

    budget = {
        "status": "pass" if plan.oos_days >= required else "blocked",
        "primaryHorizon": int(primary_horizon),
        "usableTradingDays": usable_days,
        "requiredOosDays": required,
        **plan.as_dict(),
    }
    typer.echo(json_dump({
        "progress": 0.2,
        "stage": "oos_budget",
        "message": (
            f"OOS budget: {plan.achievable_splits}/{plan.requested_splits} folds "
            f"= {plan.oos_days} days (protocol needs {required})"
        ),
        "budget": budget,
    }))
    if plan.oos_days < required:
        raise block_infeasible_oos_budget(
            achievable_oos_days=plan.oos_days,
            requested_splits=plan.requested_splits,
            achievable_splits=plan.achievable_splits,
            valid_size_days=plan.valid_size_days,
            min_selection_days=min_selection_days,
            min_holdout_days=min_holdout_days,
            trading_days_available=usable_days,
            trading_days_required=plan.days_needed_for(
                -(-required // max(1, plan.valid_size_days))
            ),
            output_dir=output_dir,
        )
    return budget


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
    top_k_candidates: str = typer.Option(
        "10,20,30,50,80",
        "--top-k-candidates",
        help="Bounded Top-K candidates evaluated on the early OOS selection segment.",
    ),
    stock_selection_modes: str = typer.Option(
        "none,fundamental",
        "--stock-selection-modes",
        help="none,fundamental keeps a no-fundamental ablation baseline.",
    ),
    fundamental_selection_mode: str = typer.Option(
        "auto",
        "--fundamental-selection-mode",
        help="auto | fixed | off",
    ),
    fundamental_selection_threshold: float = typer.Option(
        0.50,
        "--fundamental-selection-threshold",
        min=0.0,
        max=1.0,
    ),
    fundamental_blend_weight: float = typer.Option(
        0.40,
        "--fundamental-blend-weight",
        min=0.0,
        max=1.0,
    ),
    fundamental_threshold_candidates: str = typer.Option(
        "0.25,0.40,0.55",
        "--fundamental-threshold-candidates",
    ),
    fundamental_blend_candidates: str = typer.Option(
        "0.20,0.40,0.60",
        "--fundamental-blend-candidates",
    ),
    selection_max_candidates: int = typer.Option(
        64,
        "--selection-max-candidates",
        min=2,
        max=128,
    ),
    selection_min_oos_days: int = typer.Option(
        80,
        "--selection-min-oos-days",
        min=20,
    ),
    selection_min_holdout_days: int = typer.Option(
        20,
        "--selection-min-holdout-days",
        min=10,
    ),
    max_pbo: float = typer.Option(0.25, "--max-pbo", min=0.0, max=1.0),
    min_dsr_probability: float = typer.Option(
        0.95,
        "--min-dsr-probability",
        min=0.0,
        max=1.0,
    ),
    max_spa_pvalue: float = typer.Option(
        0.05,
        "--max-spa-pvalue",
        min=0.0,
        max=1.0,
    ),
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
    objective_excess_weight: float = typer.Option(
        0.45, "--objective-excess-weight", min=0.0, max=1.0
    ),
    objective_annual_weight: float = typer.Option(
        0.30, "--objective-annual-weight", min=0.0, max=1.0
    ),
    objective_drawdown_weight: float = typer.Option(
        0.25, "--objective-drawdown-weight", min=0.0, max=1.0
    ),
    cash_floor: float = typer.Option(0.0, "--cash-floor"),
    weighting: str = typer.Option("rank", "--weighting", help="equal | rank | softmax"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash"),
    min_order_value_yuan: float = typer.Option(100.0, "--min-order-value-yuan"),
    benchmark_symbol: str | None = typer.Option(None, "--benchmark-symbol"),
    acceptance_max_drawdown: float = typer.Option(
        0.10,
        "--acceptance-max-drawdown",
        min=0.0,
        max=0.80,
        help="Maximum absolute drawdown allowed by the research acceptance gate.",
    ),
    acceptance_min_sharpe: float | None = typer.Option(
        None,
        "--acceptance-min-sharpe",
        min=-5.0,
        max=10.0,
        help="Optional minimum paper/backtest Sharpe required by the research acceptance gate.",
    ),
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
    horizon_blend_method: str = typer.Option(
        "adaptive_oos",
        "--horizon-blend-method",
        help=(
            "adaptive_oos | balanced | short_tactical | long_fundamental | "
            "primary_only"
        ),
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
    factor_library: str = typer.Option(
        "all_reviewed",
        "--factor-library",
        help="all_reviewed | basic | alpha101 | alpha181 | cicc_ashare80",
    ),
    factor_screening_mode: str = typer.Option(
        "off",
        "--factor-screening-mode",
        help="off | evaluate_only | pretrain",
    ),
    factor_calibration_days: int = typer.Option(252, "--factor-calibration-days"),
    factor_holdout_days: int = typer.Option(60, "--factor-holdout-days"),
    synthesized_factors_path: Path | None = typer.Option(None, "--synthesized-factors"),
    factor_min_finite_ratio: float = typer.Option(0.30, "--factor-min-finite-ratio"),
    factor_min_abs_rank_ic: float = typer.Option(0.005, "--factor-min-abs-rank-ic"),
    factor_min_abs_rank_icir: float = typer.Option(0.10, "--factor-min-abs-rank-icir"),
    factor_min_abs_monotonicity: float = typer.Option(0.15, "--factor-min-abs-monotonicity"),
    factor_max_pairwise_correlation: float = typer.Option(0.85, "--factor-max-pairwise-correlation"),
    do_t_mode: str = typer.Option(
        "off",
        "--do-t-mode",
        help="off | intraday | daily_swing | both",
    ),
    minute_panel_path: Path | None = typer.Option(None, "--minute-panel-path"),
    macro_root: Path | None = typer.Option(None, "--macro-root"),
    flow_root: Path | None = typer.Option(None, "--flow-root"),
    index_root: Path | None = typer.Option(None, "--index-root"),
    enable_macro: bool = typer.Option(True, "--enable-macro/--no-enable-macro"),
    enable_flow: bool = typer.Option(True, "--enable-flow/--no-enable-flow"),
    enable_index: bool = typer.Option(True, "--enable-index/--no-enable-index"),
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
    if primary_horizon not in horizons_tuple:
        raise typer.BadParameter(
            "primary-horizon must be included in horizons; "
            f"received primary={primary_horizon}, horizons={list(horizons_tuple)}"
        )
    resolved_horizon_blend_method = horizon_blend_method.strip().lower()
    if resolved_horizon_blend_method not in {
        "adaptive_oos",
        "balanced",
        "short_tactical",
        "long_fundamental",
        "primary_only",
    }:
        raise typer.BadParameter(
            "horizon-blend-method must be adaptive_oos, balanced, "
            "short_tactical, long_fundamental, or primary_only"
        )
    top_k_values = sorted({
        int(item)
        for item in top_k_candidates.split(",")
        if item.strip()
    } or {int(top_k)})
    if any(value < 5 or value > 500 for value in top_k_values):
        raise typer.BadParameter("top-k candidates must remain between 5 and 500")
    selection_modes = list(dict.fromkeys(
        item.strip().lower()
        for item in stock_selection_modes.split(",")
        if item.strip()
    ))
    if not selection_modes or set(selection_modes) - {"none", "fundamental"}:
        raise typer.BadParameter("stock-selection-modes must contain none and/or fundamental")
    resolved_fundamental_mode = fundamental_selection_mode.strip().lower()
    if resolved_fundamental_mode not in {"auto", "fixed", "off"}:
        raise typer.BadParameter(
            "fundamental-selection-mode must be auto, fixed, or off"
        )
    if (
        resolved_fundamental_mode == "off"
        and "fundamental" in selection_modes
    ):
        raise typer.BadParameter(
            "stock-selection-modes cannot include fundamental when "
            "fundamental-selection-mode is off"
        )
    threshold_values = sorted({
        float(item)
        for item in fundamental_threshold_candidates.split(",")
        if item.strip()
    })
    blend_values = sorted({
        float(item)
        for item in fundamental_blend_candidates.split(",")
        if item.strip()
    })
    if any(value <= 0.0 or value >= 1.0 for value in threshold_values):
        raise typer.BadParameter(
            "fundamental-threshold-candidates must remain strictly between 0 and 1"
        )
    if any(value <= 0.0 or value > 1.0 for value in blend_values):
        raise typer.BadParameter(
            "fundamental-blend-candidates must remain in (0, 1]"
        )
    fundamental_grid = (
        [(threshold, weight) for threshold in threshold_values for weight in blend_values]
        if resolved_fundamental_mode == "auto"
        else [(float(fundamental_selection_threshold), float(fundamental_blend_weight))]
        if resolved_fundamental_mode == "fixed"
        else []
    )
    if "fundamental" in selection_modes and not fundamental_grid:
        raise typer.BadParameter(
            "fundamental selection requires at least one threshold/weight candidate"
        )
    search_variants = (
        (1 if "none" in selection_modes else 0)
        + (len(fundamental_grid) if "fundamental" in selection_modes else 0)
    )
    candidate_count = len(top_k_values) * search_variants
    if candidate_count < 2:
        raise typer.BadParameter(
            "portfolio selection requires at least two candidates for overfitting governance"
        )
    if candidate_count > selection_max_candidates:
        raise typer.BadParameter(
            "portfolio search exceeds selection-max-candidates: "
            f"{candidate_count} > {selection_max_candidates}"
        )
    if selection_min_holdout_days >= selection_min_oos_days:
        raise typer.BadParameter(
            "selection-min-holdout-days must be smaller than selection-min-oos-days"
        )
    screening_mode = factor_screening_mode.strip().lower()
    if screening_mode not in {"off", "evaluate_only", "pretrain"}:
        raise typer.BadParameter("factor-screening-mode must be off, evaluate_only, or pretrain")
    resolved_do_t_mode = do_t_mode.strip().lower()
    if resolved_do_t_mode not in {"off", "intraday", "daily_swing", "both"}:
        raise typer.BadParameter("do-t-mode must be off, intraday, daily_swing, or both")
    if resolved_do_t_mode in {"intraday", "both"} and minute_panel_path is None:
        raise typer.BadParameter("minute-panel-path is required for intraday Do-T")
    objective_weights = (
        float(objective_excess_weight),
        float(objective_annual_weight),
        float(objective_drawdown_weight),
    )
    if any(value < 0 for value in objective_weights) or abs(sum(objective_weights) - 1.0) > 0.001:
        raise typer.BadParameter("objective weights must be non-negative and sum to 1")
    if resolved_do_t_mode in {"daily_swing", "both"}:
        timing_gate = True
        holding_period_mode = "soft"

    output_dir = Path(output_dir) if output_dir is not None else default_artifact_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(json_dump({"progress": 0.03, "stage": "contract", "message": "strategy contract accepted"}))
    resolved_training_dataset = (
        Path(training_dataset_path)
        if training_dataset_path is not None
        else output_dir / "dataset" / "training_dataset.parquet"
    )
    if training_dataset_path is None:
        _assert_dataset_build_fits_in_memory(
            labels_path=labels_path,
            symbols=merge_symbols(symbols, symbols_file),
            output_dir=output_dir,
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
            factor_library=factor_library,
            synthesized_factors_path=str(synthesized_factors_path) if synthesized_factors_path else None,
            factor_min_finite_ratio=factor_min_finite_ratio,
            macro_root=str(macro_root) if macro_root else None,
            flow_root=str(flow_root) if flow_root else None,
            index_root=str(index_root) if index_root else None,
            enable_macro=enable_macro,
            enable_flow=enable_flow,
            enable_index=enable_index,
        )
    )
    typer.echo(json_dump({"progress": 0.18, "stage": "dataset", "message": "PIT dataset persisted"}))

    training_dataset = read_frame(dataset_result.output_path)
    factor_evaluation: dict[str, object] = {
        "status": "skipped",
        "mode": screening_mode,
    }
    if screening_mode != "off":
        from quantagent.factors.experiment import (
            FactorScreeningConfig,
            chronological_calibration_slice,
            evaluate_factor_library,
            factor_columns_from_report,
        )

        calibration, calibration_evidence = chronological_calibration_slice(
            training_dataset,
            calibration_days=factor_calibration_days,
            holdout_days=factor_holdout_days,
        )
        factor_columns = factor_columns_from_report(
            dataset_result.summary.get("factor_report")
            if isinstance(dataset_result.summary.get("factor_report"), dict)
            else None
        )
        screening = evaluate_factor_library(
            calibration,
            factor_columns,
            f"forward_return_{primary_horizon}d",
            output_dir / "factor_evaluation",
            config=FactorScreeningConfig(
                min_finite_ratio=factor_min_finite_ratio,
                min_abs_rank_ic=factor_min_abs_rank_ic,
                min_abs_rank_icir=factor_min_abs_rank_icir,
                min_abs_monotonicity=factor_min_abs_monotonicity,
                max_pairwise_correlation=factor_max_pairwise_correlation,
            ),
        )
        factor_evaluation = {
            **screening.to_dict(),
            "mode": screening_mode,
            "calibration": calibration_evidence,
        }
        if screening_mode == "pretrain":
            rejected = [
                column
                for column in screening.rejected_factors
                if column in training_dataset.columns
            ]
            training_dataset = training_dataset.drop(columns=rejected)
            factor_evaluation["droppedBeforeTraining"] = rejected
            if not screening.selected_factors:
                raise ResearchRejection(
                    code="no_factor_passed_screening",
                    title="没有因子通过冻结的预训练筛选闸门",
                    reasons=(
                        f"evaluated {len(screening.summary)} factors; 0 passed the "
                        "frozen calibration gate, so there is nothing to train on",
                    ),
                    stage="factor_screening",
                    remediation=(
                        "放宽 factor 门槛（min-abs-rank-ic / icir / monotonicity）、"
                        "更换 factorLibrary，或改用 factorScreeningMode=evaluate_only "
                        "先观察证据再决定。"
                    ),
                    metrics={
                        "evaluatedFactors": len(screening.summary),
                        "selectedFactors": 0,
                        "droppedBeforeTraining": len(rejected),
                    },
                    output_dir=output_dir,
                )
        typer.echo(
            json_dump({
                "progress": 0.26,
                "stage": "factor_screening",
                "message": (
                    f"evaluated {len(screening.summary)} factors; "
                    f"selected {len(screening.selected_factors)}"
                ),
            })
        )

    training_config = V7TrainingConfig(
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
        output_dir=str(output_dir / "training"),
        mark_production_ready=mark_production_ready,
        paper_report_path=str(paper_report) if paper_report else None,
        allow_model_downgrade=allow_model_downgrade,
        ft_max_epochs=ft_max_epochs,
        ft_batch_size=ft_batch_size,
        ft_device=ft_device,
        require_gpu=require_gpu,
    )
    _assert_oos_budget_is_feasible(
        training_dataset,
        training_config,
        primary_horizon=primary_horizon,
        min_selection_days=selection_min_oos_days,
        min_holdout_days=selection_min_holdout_days,
        output_dir=output_dir,
    )

    training_result = run_v7_training_experiment(training_dataset, training_config)
    typer.echo(json_dump({"progress": 0.48, "stage": "training", "message": "walk-forward training completed"}))

    raw_predictions = read_frame(Path(training_result.artifact_paths["predictions"]))
    if (
        multi_horizon_blend
        and resolved_horizon_blend_method != "primary_only"
        and "horizon" in raw_predictions.columns
        and raw_predictions["horizon"].nunique() > 1
    ):
        from quantagent.portfolio.multi_horizon_blender import (
            blend_multi_horizon_predictions,
            resolve_horizon_blend_config,
        )

        blend_config, blend_policy = resolve_horizon_blend_config(
            raw_predictions,
            method=resolved_horizon_blend_method,
            primary_horizon=primary_horizon,
            holdout_days=selection_min_holdout_days,
        )
        blend_result = blend_multi_horizon_predictions(
            raw_predictions,
            config=blend_config,
        )
        predictions_frame = blend_result.blended.copy()
        predictions_frame["sample_role"] = "validation"
        predictions_frame["fold_id"] = 0
        blender_diagnostics = {**blend_result.diagnostics, **blend_policy}
    else:
        predictions_frame = _load_oos_predictions(
            Path(training_result.artifact_paths["predictions"]),
            primary_horizon=primary_horizon,
        )
        blender_diagnostics = {
            "status": "skipped",
            "reason": "single_horizon_or_primary_only",
            "method": resolved_horizon_blend_method,
            "primaryHorizon": int(primary_horizon),
        }
    predictions_path = output_dir / "predictions" / "predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    written_predictions = write_frame(predictions_frame, predictions_path)
    typer.echo(json_dump({"progress": 0.62, "stage": "prediction", "message": "OOS predictions persisted"}))

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
        output_dir / "portfolio" / "position_state.parquet"
        if holding_period_mode != "off"
        else None
    )

    market_panel_frame = read_frame(market_panel_path)
    if benchmark_symbol and benchmark_symbol not in set(
        market_panel_frame["symbol"].astype(str)
    ):
        raise ValueError(
            f"benchmark {benchmark_symbol} is absent from the market panel; "
            "excess-return optimisation cannot be verified"
        )

    def build_candidate_weights(
        candidate_predictions: "pd.DataFrame",
        candidate_top_k: int,
        *,
        candidate_state_path: Path | None = None,
    ):
        return build_v7_target_weights(
            candidate_predictions,
            market_panel_frame,
            sector_map=sector_frame,
            config=V7TargetWeightsConfig(
                top_k=candidate_top_k,
                top_k_ratio=None,
                min_selection_pressure=min_selection_pressure,
                fail_if_top_k_covers_universe=fail_if_top_k_covers_universe,
                selection_mode="top_k",
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
                weighting=weighting,
                capital_yuan=initial_cash,
                dynamic_top_k_enabled=False,
                top_k_min=candidate_top_k,
                top_k_max=candidate_top_k,
                timing_gate_enabled=timing_gate,
                holding_period_mode=holding_period_mode,
                holding_period_max_delta=holding_period_max_delta,
                capital_tier_overrides=capital_tier_overrides,
            ),
            timing_plan=timing_plan_frame,
            position_state_path=candidate_state_path,
        )

    try:
        portfolio_selection_predictions, portfolio_holdout_predictions, portfolio_split = (
            _split_portfolio_selection_holdout(
                predictions_frame,
                minimum_selection_dates=selection_min_oos_days,
                minimum_holdout_dates=selection_min_holdout_days,
            )
        )
    except ResearchRejection as rejection:
        # The helper cannot know where this run files its evidence.
        rejection.output_dir = output_dir
        raise
    candidates: list[dict[str, object]] = []
    candidate_errors: list[dict[str, object]] = []
    for stock_mode in selection_modes:
        variants: list[tuple[float, float]] = (
            [(0.0, 0.0)]
            if stock_mode == "none"
            else list(fundamental_grid)
        )
        for threshold_value, blend_weight in variants:
            variant_id = (
                "none"
                if stock_mode == "none"
                else (
                    f"fund_q{round(threshold_value * 100):02d}"
                    f"_w{round(blend_weight * 100):02d}"
                )
            )
            try:
                selected_predictions, selection_evidence = _apply_stock_selection_mode(
                    portfolio_selection_predictions,
                    training_dataset,
                    stock_mode,
                    threshold=threshold_value,
                    blend_weight=blend_weight,
                )
            except ValueError as exc:
                candidate_errors.append({
                    "selectionMode": stock_mode,
                    "fundamentalThreshold": threshold_value,
                    "fundamentalBlendWeight": blend_weight,
                    "status": "failed",
                    "error": str(exc),
                })
                continue
            for candidate_top_k in top_k_values:
                candidate_id = f"{variant_id}_top_{candidate_top_k}"
                candidate_dir = output_dir / "portfolio_search" / candidate_id
                try:
                    candidate_state_path = (
                        candidate_dir / "position_state.parquet"
                        if holding_period_mode != "off"
                        else None
                    )
                    candidate_weights = build_candidate_weights(
                        selected_predictions,
                        candidate_top_k,
                        candidate_state_path=candidate_state_path,
                    )
                    if candidate_weights.target_weights.empty:
                        raise ValueError("candidate produced no target weights")
                    weights_frame = candidate_weights.target_weights.copy()
                    if "trade_date" in weights_frame.columns:
                        weights_frame = weights_frame.set_index("trade_date")
                    candidate_market = _restrict_market_for_paper(
                        market_panel_frame,
                        weights_frame,
                        benchmark_symbol=benchmark_symbol,
                    )
                    candidate_sim = simulate_ashare_target_weights(
                        weights_frame,
                        candidate_market,
                        AShareExecutionSimulationConfig(
                            initial_cash=initial_cash,
                            min_order_value_yuan=min_order_value_yuan,
                            audit_log_dir=str(candidate_dir / "audit"),
                        ),
                    )
                    candidate_report = write_paper_report(
                        candidate_sim,
                        market_panel=candidate_market,
                        config=PaperReportConfig(
                            initial_cash=initial_cash,
                            benchmark_symbol=benchmark_symbol,
                            output_dir=candidate_dir,
                        ),
                    )
                    candidate_returns = (
                        pd.to_numeric(candidate_sim.nav, errors="coerce")
                        .sort_index()
                        .pct_change()
                        .replace([float("inf"), float("-inf")], float("nan"))
                        .dropna()
                    )
                    candidates.append({
                        "id": candidate_id,
                        "status": "evaluated",
                        "selectionMode": stock_mode,
                        "fundamentalThreshold": threshold_value,
                        "fundamentalBlendWeight": blend_weight,
                        "topK": candidate_top_k,
                        "metrics": candidate_report.summary,
                        "selectionEvidence": selection_evidence,
                        "_returns": candidate_returns,
                    })
                except (ValueError, RuntimeError) as exc:
                    candidate_errors.append({
                        "id": candidate_id,
                        "selectionMode": stock_mode,
                        "fundamentalThreshold": threshold_value,
                        "fundamentalBlendWeight": blend_weight,
                        "topK": candidate_top_k,
                        "status": "failed",
                        "error": str(exc),
                    })
    if not candidates:
        # Every candidate was refused by a portfolio-construction constraint
        # (selection pressure, universe coverage, weight limits). The pipeline
        # worked; the research conditions could not support a portfolio.
        failure_path = output_dir / "portfolio_search" / "candidate_failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json_dump(candidate_errors), encoding="utf-8")
        raise ResearchRejection(
            code="no_viable_portfolio_candidate",
            title="没有任何组合候选满足构建约束",
            reasons=tuple(
                f"{item.get('id')}: {item.get('error')}" for item in candidate_errors
            )[:8],
            stage="portfolio_construction",
            remediation=(
                "常见原因是可交易宇宙相对 Top-K 太小（选股压力不足或 Top-K 覆盖了整个宇宙）。"
                "扩大研究范围、降低 Top-K，或放宽单票/行业上限后重试。"
            ),
            metrics={"candidateCount": len(candidate_errors)},
            evidence_paths=(str(failure_path),),
            output_dir=output_dir,
        )
    champion, portfolio_frontier = _select_portfolio_frontier(
        candidates,
        objective_weights=objective_weights,
    )
    from quantagent.research.selection_governance import (
        NestedSelectionConfig,
        evaluate_frozen_candidate,
    )

    candidate_returns = pd.concat(
        {
            str(candidate["id"]): candidate["_returns"]
            for candidate in candidates
            if isinstance(candidate.get("_returns"), pd.Series)
        },
        axis=1,
        join="inner",
    ).dropna(how="all")
    governance = evaluate_frozen_candidate(
        candidate_returns,
        selected_candidate=str(champion["id"]),
        config=NestedSelectionConfig(
            max_pbo=max_pbo,
            min_dsr_probability=min_dsr_probability,
            max_spa_pvalue=max_spa_pvalue,
        ),
        cumulative_trials=len(candidates),
        minimum_observed_days=selection_min_oos_days,
    )
    governance_path = output_dir / "portfolio_search" / "selection_governance.json"
    governance_path.parent.mkdir(parents=True, exist_ok=True)
    governance_path.write_text(
        json_dump(governance.to_dict()),
        encoding="utf-8",
    )
    if not governance.accepted:
        # The gate is unchanged; only its reporting is. The champion, the whole
        # Pareto frontier and the governance evidence are persisted first so a
        # rejection is auditable instead of vanishing with the exception.
        frontier_path = output_dir / "portfolio_search" / "rejected_frontier.json"
        frontier_path.write_text(
            json_dump({
                "champion": _public_portfolio_candidate(champion),
                "paretoFrontier": portfolio_frontier,
                "candidateFailures": candidate_errors,
                "split": portfolio_split,
            }),
            encoding="utf-8",
        )
        raise ResearchRejection(
            code="overfitting_governance_rejected",
            title="候选组合被过拟合治理闸门否决",
            reasons=tuple(governance.rejection_reasons),
            stage="portfolio_selection",
            remediation=(
                "PBO/DSR/SPA 是预先声明的闸门，不允许事后放宽来通过。"
                "减少候选数量、延长 OOS 观测窗口，或更换研究设计后重新预注册。"
            ),
            metrics=governance.to_dict(),
            evidence_paths=(str(governance_path), str(frontier_path)),
            output_dir=output_dir,
        )
    final_predictions, final_selection_evidence = _apply_stock_selection_mode(
        portfolio_holdout_predictions,
        training_dataset,
        str(champion["selectionMode"]),
        threshold=float(champion.get("fundamentalThreshold", 0.0)),
        blend_weight=float(champion.get("fundamentalBlendWeight", 0.0)),
    )
    predictions_frame = final_predictions
    weights_result = build_candidate_weights(
        predictions_frame,
        int(champion["topK"]),
        candidate_state_path=position_state_path,
    )
    weights_result.diagnostics["portfolio_selection"] = {
        "split": portfolio_split,
        "champion": _public_portfolio_candidate(champion),
        "paretoFrontier": portfolio_frontier,
        "overfittingGovernance": governance.to_dict(),
        "failures": candidate_errors,
        "finalSelectionEvidence": final_selection_evidence,
    }
    weights_result.diagnostics["multi_horizon_blend"] = blender_diagnostics
    weights_result.diagnostics["training_dataset_symbol_count"] = (
        int(training_dataset["symbol"].nunique()) if "symbol" in training_dataset.columns else 0
    )
    weights_result.diagnostics["training_dataset_row_count"] = int(len(training_dataset))
    weights_path = output_dir / "portfolio" / "target_weights.parquet"
    written_weights = write_v7_target_weights(weights_result, weights_path)
    typer.echo(json_dump({"progress": 0.76, "stage": "portfolio", "message": "constrained target weights persisted"}))
    do_t_status: dict[str, object] = {
        "mode": resolved_do_t_mode,
        "dailySwing": (
            {
                "status": "enabled",
                "timingGate": "ATR",
                "holdingPeriod": "soft",
            }
            if resolved_do_t_mode in {"daily_swing", "both"}
            else {"status": "not_requested"}
        ),
        "intraday": {"status": "not_requested"},
    }
    if resolved_do_t_mode in {"intraday", "both"}:
        from quantagent.cli.v8_intraday import run_do_t_overlay_v8

        intraday_dir = run_do_t_overlay_v8(
            target_weights_path=Path(written_weights),
            market_panel_path=market_panel_path,
            base_nav_path=None,
            minute_panel_path=minute_panel_path,
            provider_uri=None,
            output_dir=output_dir / "do_t_overlay",
            initial_cash=initial_cash,
            trade_fraction=0.30,
            min_edge_pct=0.025,
            min_minutes_between_legs=5,
            max_trades_per_day=50,
            symbol_batch_size=50,
        )
        do_t_status["intraday"] = {
            "status": "completed",
            "outputDir": str(intraday_dir),
            "inventoryPolicy": "yesterday-settled inventory only",
        }

    reports_root = output_dir / "reports"
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
                max_drawdown=acceptance_max_drawdown,
                min_sharpe=acceptance_min_sharpe,
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
        typer.echo(json_dump({"progress": 0.94, "stage": "risk", "message": "A-share simulation and acceptance gates completed"}))

    pipeline_report = {
        "training_dataset": dataset_result.summary,
        "factor_evaluation": factor_evaluation,
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
        "portfolio_frontier": {
            "split": portfolio_split,
            "champion": _public_portfolio_candidate(champion),
            "paretoFrontier": portfolio_frontier,
            "overfittingGovernance": governance.to_dict(),
            "failures": candidate_errors,
        },
        "do_t": do_t_status,
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
    typer.echo(json_dump({"progress": 0.98, "stage": "evidence", "message": "pipeline evidence persisted"}))
    typer.echo(json_dump(pipeline_report))


def _split_portfolio_selection_holdout(
    predictions: "pd.DataFrame",
    *,
    holdout_fraction: float = 0.30,
    minimum_selection_dates: int = 5,
    minimum_holdout_dates: int = 5,
) -> tuple["pd.DataFrame", "pd.DataFrame", dict[str, object]]:
    if "trade_date" not in predictions.columns:
        raise ValueError("portfolio selection split requires trade_date")
    dates = (
        pd.to_datetime(predictions["trade_date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if len(dates) < required_oos_days(minimum_selection_dates, minimum_holdout_dates):
        # A protocol that cannot be executed is a research-condition failure,
        # not a crash: the walk-forward simply did not produce enough OOS days
        # for a nested selection plus a frozen holdout.
        raise reject_insufficient_oos(
            observed_days=len(dates),
            min_selection_days=minimum_selection_dates,
            min_holdout_days=minimum_holdout_dates,
        )
    holdout_count = max(
        minimum_holdout_dates,
        int(math.ceil(len(dates) * holdout_fraction)),
    )
    # The selection segment has to keep one extra day: governance measures the
    # segment in daily returns, and differencing the NAV consumes the first day.
    holdout_count = min(
        holdout_count,
        len(dates) - minimum_selection_dates - RETURN_DIFFERENCING_DAYS,
    )
    split_at = len(dates) - holdout_count
    selection_end = pd.Timestamp(dates[split_at - 1])
    holdout_start = pd.Timestamp(dates[split_at])
    normalized = pd.to_datetime(predictions["trade_date"], errors="coerce")
    selection = predictions.loc[normalized <= selection_end].copy()
    holdout = predictions.loc[normalized >= holdout_start].copy()
    return selection, holdout, {
        "policy": "nested chronological OOS: select on early segment, report acceptance on frozen later segment",
        "selectionStart": str(pd.Timestamp(dates[0]).date()),
        "selectionEnd": str(selection_end.date()),
        "selectionDates": int(split_at),
        "holdoutStart": str(holdout_start.date()),
        "holdoutEnd": str(pd.Timestamp(dates[-1]).date()),
        "holdoutDates": int(holdout_count),
    }


def _apply_stock_selection_mode(
    predictions: "pd.DataFrame",
    training_dataset: "pd.DataFrame",
    mode: str,
    *,
    threshold: float,
    blend_weight: float,
) -> tuple["pd.DataFrame", dict[str, object]]:
    if mode == "none":
        return predictions.copy(), {
            "mode": "none",
            "fundamentalBlendWeight": 0.0,
            "rowsBefore": int(len(predictions)),
            "rowsAfter": int(len(predictions)),
            "selectionApplied": False,
        }
    if mode != "fundamental":
        raise ValueError(f"unsupported stock selection mode: {mode}")
    fundamental_inputs = {
        "pe_ttm", "pb", "ps_ttm", "roe", "gross_margin",
        "operating_cf_to_net_income", "revenue_yoy", "net_income_yoy",
        "quality_score", "growth_score",
    }
    available_inputs = [
        column
        for column in fundamental_inputs
        if column in training_dataset.columns
        and pd.to_numeric(training_dataset[column], errors="coerce").notna().any()
    ]
    if not available_inputs:
        raise ValueError(
            "fundamental stock selection requested but no PIT fundamental metrics are available"
        )
    if "available_at" not in training_dataset.columns:
        raise ValueError("fundamental stock selection requires available_at for PIT ranking")

    from quantagent.data.fundamental.ranker import (
        FundamentalRankerConfig,
        build_fundamental_ranker,
    )

    dates = pd.to_datetime(predictions["trade_date"], errors="coerce").dropna().unique()
    ranked = build_fundamental_ranker(
        training_dataset,
        as_of_dates=dates,
        config=FundamentalRankerConfig(
            min_universe_per_bucket=3,
            source_version="strategy_studio_pit",
        ),
    )
    ranks = ranked.frame[
        ["symbol", "as_of_date", "composite_rank", "metric_completeness"]
    ].copy()
    if ranks.empty:
        raise ValueError("fundamental stock selection produced no eligible PIT ranks")
    ranks = ranks.rename(columns={"as_of_date": "trade_date"})
    ranks["trade_date"] = pd.to_datetime(ranks["trade_date"], errors="coerce")
    selected = predictions.copy()
    selected["trade_date"] = pd.to_datetime(selected["trade_date"], errors="coerce")
    selected["symbol"] = selected["symbol"].astype(str)
    ranks["symbol"] = ranks["symbol"].astype(str)
    selected = selected.merge(ranks, on=["trade_date", "symbol"], how="inner")
    selected = selected[
        (pd.to_numeric(selected["composite_rank"], errors="coerce") >= threshold)
        & (pd.to_numeric(selected["metric_completeness"], errors="coerce") > 0.0)
    ].copy()
    if selected.empty:
        raise ValueError("fundamental stock selection removed the entire OOS universe")
    if not 0.0 <= float(blend_weight) <= 1.0:
        raise ValueError("fundamental blend weight must remain in [0, 1]")
    prediction_source = (
        "risk_adjusted_prediction"
        if "risk_adjusted_prediction" in selected.columns
        else "prediction"
        if "prediction" in selected.columns
        else next(
            (
                column
                for column in selected.columns
                if str(column).startswith("alpha_")
            ),
            None,
        )
    )
    if prediction_source is None:
        raise ValueError(
            "fundamental blending requires prediction, risk_adjusted_prediction, "
            "or alpha_*"
        )
    selected[prediction_source] = pd.to_numeric(
        selected[prediction_source], errors="coerce"
    )
    model_rank = selected.groupby("trade_date")[prediction_source].rank(
        method="average",
        pct=True,
    )
    fundamental_rank = pd.to_numeric(
        selected["composite_rank"], errors="coerce"
    )
    blended_prediction = (
        (1.0 - float(blend_weight)) * model_rank
        + float(blend_weight) * fundamental_rank
    )
    selected["prediction"] = blended_prediction
    if "risk_adjusted_prediction" in selected.columns:
        selected["risk_adjusted_prediction"] = blended_prediction
    selected = selected.dropna(subset=["prediction"])
    if selected.empty:
        raise ValueError("fundamental blending produced no finite predictions")
    selected = selected.drop(columns=["composite_rank", "metric_completeness"])
    return selected, {
        "mode": "fundamental",
        "selectionApplied": True,
        "threshold": float(threshold),
        "fundamentalBlendWeight": float(blend_weight),
        "modelRankWeight": float(1.0 - blend_weight),
        "predictionSource": str(prediction_source),
        "blendPolicy": "cross-sectional model rank × PIT within-sector fundamental rank",
        "inputMetrics": sorted(available_inputs),
        "rowsBefore": int(len(predictions)),
        "rowsAfter": int(len(selected)),
        "symbolsBefore": int(predictions["symbol"].nunique()),
        "symbolsAfter": int(selected["symbol"].nunique()),
        "pitPolicy": "latest available_at <= prediction trade_date",
    }


def _select_portfolio_frontier(
    candidates: list[dict[str, object]],
    *,
    objective_weights: tuple[float, float, float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    def metrics(candidate: dict[str, object]) -> tuple[float, float, float]:
        raw = candidate.get("metrics")
        values = raw if isinstance(raw, dict) else {}
        excess = _finite_metric(
            values.get("excess_return_after_costs", values.get("excess_return")),
            default=float("-inf"),
        )
        annual = _finite_metric(values.get("annualized_return"), default=float("-inf"))
        drawdown = abs(_finite_metric(values.get("max_drawdown"), default=float("inf")))
        return excess, annual, drawdown

    frontier: list[dict[str, object]] = []
    for candidate in candidates:
        current = metrics(candidate)
        dominated = False
        for challenger in candidates:
            if challenger is candidate:
                continue
            other = metrics(challenger)
            no_worse = other[0] >= current[0] and other[1] >= current[1] and other[2] <= current[2]
            strictly_better = other[0] > current[0] or other[1] > current[1] or other[2] < current[2]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    if not frontier:
        frontier = list(candidates)
    excess_values = [value for value in (metrics(item)[0] for item in frontier) if pd.notna(value) and value != float("-inf")]
    annual_values = [value for value in (metrics(item)[1] for item in frontier) if pd.notna(value) and value != float("-inf")]
    drawdown_values = [value for value in (metrics(item)[2] for item in frontier) if pd.notna(value) and value != float("inf")]
    scored: list[tuple[float, dict[str, object]]] = []
    public_frontier: list[dict[str, object]] = []
    for candidate in frontier:
        excess, annual, drawdown = metrics(candidate)
        score = (
            objective_weights[0] * _normalize_frontier_metric(excess, excess_values, True)
            + objective_weights[1] * _normalize_frontier_metric(annual, annual_values, True)
            + objective_weights[2] * _normalize_frontier_metric(drawdown, drawdown_values, False)
        )
        scored.append((score, candidate))
        public = _public_portfolio_candidate(candidate)
        public["preferenceScore"] = score
        public["paretoOptimal"] = True
        public_frontier.append(public)
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], public_frontier


def _public_portfolio_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in candidate.items() if not str(key).startswith("_")}


def _finite_metric(value: object, *, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    return resolved if pd.notna(resolved) else default


def _normalize_frontier_metric(
    value: float,
    finite_values: list[float],
    higher_is_better: bool,
) -> float:
    if not finite_values or value in {float("inf"), float("-inf")} or pd.isna(value):
        return 0.0
    low, high = min(finite_values), max(finite_values)
    if abs(high - low) <= 1e-12:
        return 1.0
    normalized = (value - low) / (high - low)
    return float(normalized if higher_is_better else 1.0 - normalized)


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
            "sharpe": paper_summary.get("sharpe", training_metrics.get("sharpe", 0.0)),
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
    # `metrics["pit_violation_count"] = 0` used to be injected here whenever the
    # training metrics carried none. That is the whole DEF-023 hardening undone in
    # two lines: the acceptance gate stopped defaulting the measurement, and this
    # supplied the default instead, so `no_pit_violations` reported a *measured*
    # pass on a run where no PIT comparison had ever been made. A run that was not
    # audited must reach the gate as unaudited.
    #
    # The dataset audits (`pit_violation_count`, `uses_mock_or_synthetic`,
    # `label_alignment_status`) travel with `training_metrics`, forwarded by
    # `_dataset_audit_metrics` from the gates that actually measured them.
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
    weighting: str = typer.Option("rank", "--weighting", help="equal | rank | softmax"),
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
    factor_library: str = typer.Option("alpha181", "--factor-library", help="basic | alpha101 | alpha181 | cicc_ashare80"),
    synthesized_factors_path: Path | None = typer.Option(None, "--synthesized-factors"),
    run_symbolic_ga: bool = typer.Option(False, "--run-symbolic-ga/--skip-symbolic-ga"),
    symbolic_ga_population: int = typer.Option(80, "--symbolic-ga-population"),
    symbolic_ga_generations: int = typer.Option(20, "--symbolic-ga-generations"),
    symbolic_ga_top_k: int = typer.Option(50, "--symbolic-ga-top-k"),
    macro_root: Path | None = typer.Option(None, "--macro-root"),
    flow_root: Path | None = typer.Option(None, "--flow-root"),
    index_root: Path | None = typer.Option(None, "--index-root"),
    enable_macro: bool = typer.Option(True, "--enable-macro/--no-enable-macro"),
    enable_flow: bool = typer.Option(True, "--enable-flow/--no-enable-flow"),
    enable_index: bool = typer.Option(True, "--enable-index/--no-enable-index"),
    refresh_macro: bool = typer.Option(False, "--refresh-macro/--no-refresh-macro"),
    refresh_flow: bool = typer.Option(False, "--refresh-flow/--no-refresh-flow"),
    refresh_index: bool = typer.Option(False, "--refresh-index/--no-refresh-index"),
    run_synth_ablation: bool = typer.Option(True, "--run-synth-ablation/--no-run-synth-ablation"),
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
        refresh_macro=refresh_macro,
        refresh_flow=refresh_flow,
        refresh_index=refresh_index,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        horizons=horizons,
    )

    resolved_synthesized_factors = synthesized_factors_path
    symbolic_ga_status: dict[str, object] = {"status": "skipped"}
    if run_symbolic_ga:
        from quantagent.factors.factor_synthesis import SymbolicGAConfig, save_result, synthesize_factors

        ga_output = quant_paths().reports / "v7" / "factor_synthesis"
        ga_result = synthesize_factors(
            read_frame(Path(str(stages["market_panel_path"]))),
            labels=read_frame(Path(str(stages["labels_path"]))),
            config=SymbolicGAConfig(
                population=symbolic_ga_population,
                generations=symbolic_ga_generations,
                top_k=symbolic_ga_top_k,
                label_column="forward_return_5d",
            ),
        )
        ga_paths = save_result(ga_result, ga_output)
        resolved_synthesized_factors = Path(ga_paths["definitions"])
        symbolic_ga_status = {
            "status": "passed",
            "selected": int(len(ga_result.definitions)),
            "definitions": ga_paths["definitions"],
            "leaderboard": ga_paths["leaderboard"],
            "history": ga_paths["history"],
        }

    run_full_real_training_v7(
        market_panel_path=Path(str(stages["market_panel_path"])),
        labels_path=Path(str(stages["labels_path"])),
        output_dir=quant_paths().models / "v7_alpha_full_ai",
        fundamentals_root=Path(str(stages["fundamentals_root"])) if stages.get("fundamentals_root") else None,
        valuation_path=Path(str(stages["valuation_path"])) if stages.get("valuation_path") else None,
        disclosures_path=None,
        sector_map_path=Path(str(stages["sector_map_path"])) if stages.get("sector_map_path") else None,
        training_dataset_path=Path(str(stages["training_dataset_path"])),
        macro_root=Path(str(stages["macro_root"])) if stages.get("macro_root") else macro_root,
        flow_root=Path(str(stages["flow_root"])) if stages.get("flow_root") else flow_root,
        index_root=Path(str(stages["index_root"])) if stages.get("index_root") else index_root,
        enable_macro=enable_macro,
        enable_flow=enable_flow,
        enable_index=enable_index,
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
        weighting=weighting,
        optimizer_backend="auto",
        objective="max_expected_alpha",
        cash_floor=0.0,
        initial_cash=initial_cash,
        min_order_value_yuan=min_order_value_yuan,
        benchmark_symbol=None,
        acceptance_max_drawdown=0.10,
        acceptance_min_sharpe=None,
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
        factor_library=factor_library,
        synthesized_factors_path=resolved_synthesized_factors,
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
                "symbolic_ga": symbolic_ga_status,
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
    refresh_macro: bool = False,
    refresh_flow: bool = False,
    refresh_index: bool = False,
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
        # Let auto-range bridge from the qlib calendar's last date so AkShare only
        # fetches the gap (e.g. 2020-09-26 → end_date) rather than trying to refetch
        # 20+ years that qlib already covers. The user's --start-date governs the
        # training window, not the AkShare fetch window.
        market_result = build_akshare_market_panel(
            AkShareMarketPanelConfig(
                symbols=resolved_symbols,
                start_date=None,
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

    macro_root = lake.root / "raw" / "akshare" / "macro"
    flow_root = lake.root / "raw" / "akshare" / "flow"
    index_root = lake.root / "raw" / "akshare" / "index"
    if refresh_macro:
        from quantagent.data.providers.akshare_macro_provider import AkShareMacroProvider
        provider = AkShareMacroProvider(allow_network=allow_network, root=str(macro_root))
        macro_result = provider.fetch_all(start_date=start_date, end_date=end_date)
        stages["macro_refresh"] = {
            name: {"rows": int(len(res.frame)), "warnings": list(res.warnings)}
            for name, res in macro_result.items()
        }
    if refresh_flow:
        from quantagent.data.providers.akshare_flow_provider import AkShareFlowProvider
        provider = AkShareFlowProvider(allow_network=allow_network, root=str(flow_root))
        flow_result = provider.fetch_all()
        stages["flow_refresh"] = {
            name: {"rows": int(len(res.frame)), "warnings": list(res.warnings)}
            for name, res in flow_result.items()
        }
    if refresh_index:
        from quantagent.data.providers.akshare_index_provider import AkShareIndexProvider
        provider = AkShareIndexProvider(allow_network=allow_network, root=str(index_root))
        index_result = provider.fetch_all(start_date=start_date, end_date=end_date)
        stages["index_refresh"] = {
            name: {"rows": int(len(res.frame)), "warnings": list(res.warnings)}
            for name, res in index_result.items()
        }

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
            "macro_root": str(macro_root) if macro_root.exists() else None,
            "flow_root": str(flow_root) if flow_root.exists() else None,
            "index_root": str(index_root) if index_root.exists() else None,
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
