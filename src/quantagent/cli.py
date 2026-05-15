from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, is_dataclass

import pandas as pd
import typer
import yaml

from quantagent.factors.evaluation import factor_summary_table, forward_return_labels

app = typer.Typer(help="QuantAgent V7 research, fundamentals, and execution CLI.")


@app.command("build-factors")
def build_factors(
    input_path: Path,
    output_path: Path,
    library: str = "alpha101",
) -> None:
    """Compute Alpha101 or CICC high-frequency factors from a price panel."""
    frame = pd.read_csv(input_path)
    if library == "alpha101":
        from quantagent.factors.alpha101 import compute_alpha101

        result = compute_alpha101(frame)
    elif library == "cicc_ashare":
        from quantagent.factors.cicc_high_freq import compute_cicc_high_freq_factors

        result = compute_cicc_high_freq_factors(frame).factors
    else:
        raise typer.BadParameter("library must be alpha101 or cicc_ashare")
    result.to_csv(output_path, index=False)
    typer.echo(f"wrote {output_path}")


@app.command("evaluate-factors")
def evaluate_factors(
    price_path: Path,
    factor_path: Path,
    output_path: Path,
    horizon_days: int = 5,
) -> None:
    prices = forward_return_labels(pd.read_csv(price_path), horizons=(horizon_days,))
    factors = pd.read_csv(factor_path)
    if {"factor_name", "factor_value"}.issubset(factors.columns):
        wide = factors.pivot_table(index=["trade_date", "symbol"], columns="factor_name", values="factor_value", aggfunc="last").reset_index()
    else:
        wide = factors
    data = prices.merge(wide, on=["trade_date", "symbol"], how="inner")
    factor_columns = [column for column in wide.columns if column not in {"trade_date", "symbol"}]
    summary = factor_summary_table(data, factor_columns, f"forward_return_{horizon_days}d")
    summary.to_csv(output_path, index=False)
    typer.echo(f"wrote {output_path}")


@app.command("build-flow-features")
def build_flow_features(
    config_path: Path,
    output_path: Path,
) -> None:
    from quantagent.ashare.fund_flow import build_flow_feature_frame

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sources = {name: pd.read_csv(path) for name, path in config.get("sources", {}).items()}
    result = build_flow_feature_frame(sources, window=int(config.get("window", 20)))
    result.frame.to_csv(output_path, index=False)
    typer.echo(f"wrote {output_path}")


@app.command("build-sector-rotation")
def build_sector_rotation(
    input_path: Path,
    output_path: Path,
    sector_column: str = "sector",
) -> None:
    from quantagent.factors.sector_rotation import compute_sector_rotation_factors

    result = compute_sector_rotation_factors(pd.read_csv(input_path), sector_column=sector_column)
    result.to_csv(output_path, index=False)
    typer.echo(f"wrote {output_path}")


@app.command("validate-v7")
def validate_v7_cli(
    config: Path = Path("configs/v7.default.yaml"),
) -> None:
    from quantagent.services.v7_pipeline_service import validate_v7

    typer.echo(_json(validate_v7(config)))


@app.command("run-daily-v7")
def run_daily_v7_cli(
    config: Path = Path("configs/v7.default.yaml"),
    date: str = typer.Option("2026-05-15", "--date"),
    output_dir: Path = Path("reports/v7"),
) -> None:
    from quantagent.services.v7_pipeline_service import run_daily_v7_research

    result = run_daily_v7_research(config, as_of_date=date)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v7_daily_research_report.json"
    output_path.write_text(_json(result), encoding="utf-8")
    typer.echo(f"status=ok themes={len(result['theme_ranking'])} targets={len(result['portfolio_plan']['target_weights'])} output={output_path}")


@app.command("build-fundamentals-v7")
def build_fundamentals_v7_cli(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated A-share symbols (e.g. 600519.SH,000858.SZ)"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    provider: str = typer.Option("tushare", "--provider", help="tushare or akshare"),
    fundamentals_root: Path = Path("data/v7/fundamentals"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    token_env: str = typer.Option("TUSHARE_TOKEN", "--token-env"),
) -> None:
    """Pull PIT-aware financial statements from TuShare/AkShare and write them to the V7 cache."""
    from quantagent.data.providers.akshare_financial_provider import AkShareFinancialProvider
    from quantagent.data.providers.base import ProviderRequest
    from quantagent.data.providers.financial_cache import FinancialCacheConfig, FinancialStatementCache
    from quantagent.data.providers.tushare_financial_provider import TuShareFinancialProvider

    request = ProviderRequest(
        start_date=start_date,
        end_date=end_date,
        symbols=tuple(item.strip() for item in symbols.split(",") if item.strip()),
    )
    if provider == "tushare":
        adapter = TuShareFinancialProvider(allow_network=allow_network, token_env=token_env)
    elif provider == "akshare":
        adapter = AkShareFinancialProvider(allow_network=allow_network)
    else:
        raise typer.BadParameter("provider must be tushare or akshare")
    statements = adapter.all_statements(request)
    cache = FinancialStatementCache(FinancialCacheConfig(root=str(fundamentals_root)))
    summary: dict[str, dict[str, object]] = {}
    for name, result in statements.items():
        path = cache.upsert(name, result.frame)
        summary[name] = {
            "rows": int(0 if result.frame is None else len(result.frame)),
            "source": result.source,
            "path": str(path),
            "warnings": list(result.warnings),
        }
    typer.echo(_json({"provider": provider, "statements": summary}))


@app.command("check-qlib-v7")
def check_qlib_v7_cli(
    provider_uri: str = typer.Option(..., "--provider-uri"),
    start_date: str = typer.Option("2026-05-01", "--start-date"),
    end_date: str = typer.Option("2026-05-15", "--end-date"),
    symbols: str = typer.Option("", "--symbols", help="Optional comma-separated symbols for a schema probe."),
    universe: str = typer.Option("", "--universe", help="Optional qlib universe name."),
    region: str = typer.Option("cn", "--region"),
) -> None:
    """Check local Qlib CN provider readiness and PIT market schema."""
    from quantagent.data.providers.base import ProviderRequest
    from quantagent.data.providers.qlib_provider import QlibProvider

    request = None
    symbol_tuple = tuple(item.strip() for item in symbols.split(",") if item.strip())
    if symbol_tuple or universe:
        request = ProviderRequest(
            start_date=start_date,
            end_date=end_date,
            symbols=symbol_tuple,
            universe=universe or None,
        )
    result = QlibProvider(provider_uri=provider_uri, region=region).health_check(request)
    typer.echo(_json(result))


@app.command("download-qlib-v7")
def download_qlib_v7_cli(
    target_dir: str = typer.Option("~/.qlib/qlib_data/cn_data", "--target-dir"),
    region: str = typer.Option("cn", "--region"),
) -> None:
    """Print the official Qlib CN data command; run it in a Qlib checkout."""
    command = f"python scripts/get_data.py qlib_data --target_dir {target_dir} --region {region}"
    typer.echo(_json({"status": "manual_step_required", "command": command, "note": "run this inside the official Qlib scripts directory"}))


@app.command("build-market-panel-v7")
def build_market_panel_v7_cli(
    provider_uri: str = typer.Option(..., "--provider-uri"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    output_root: Path = typer.Option(Path("data/v7"), "--output-root"),
    symbols: str = typer.Option("", "--symbols"),
    universe: str = typer.Option("", "--universe"),
    region: str = typer.Option("cn", "--region"),
) -> None:
    """Export a PIT market panel and close-available-next-day features from local Qlib CN data."""
    from quantagent.data.bootstrap.qlib_bootstrap import QlibBootstrapConfig, build_qlib_market_panel

    result = build_qlib_market_panel(
        QlibBootstrapConfig(
            provider_uri=provider_uri,
            start_date=start_date,
            end_date=end_date,
            symbols=tuple(item.strip() for item in symbols.split(",") if item.strip()),
            universe=universe or None,
            region=region,
            output_root=str(output_root),
        )
    )
    typer.echo(_json(result))


@app.command("build-akshare-v7")
def build_akshare_v7_cli(
    symbols: str = typer.Option(..., "--symbols"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    fundamentals_root: Path = typer.Option(Path("data/v7/fundamentals"), "--fundamentals-root"),
    allow_network: bool = typer.Option(False, "--allow-network"),
) -> None:
    """Download AkShare statements into the PIT financial cache."""
    from quantagent.data.bootstrap.akshare_bootstrap import AkShareBootstrapConfig, build_akshare_financial_cache

    result = build_akshare_financial_cache(
        AkShareBootstrapConfig(
            start_date=start_date,
            end_date=end_date,
            symbols=tuple(item.strip() for item in symbols.split(",") if item.strip()),
            fundamentals_root=str(fundamentals_root),
            allow_network=allow_network,
        )
    )
    typer.echo(_json(result))


@app.command("build-labels-v7")
def build_labels_v7_cli(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("data/v7/labels.parquet"), "--output"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
) -> None:
    """Build future-return labels for training; labels must never be used for inference."""
    from quantagent.data.v7_label_builder import build_forward_return_labels

    frame = _read_frame(market_panel_path)
    result = build_forward_return_labels(frame, tuple(int(item.strip()) for item in horizons.split(",") if item.strip()))
    actual = _write_frame(result.frame, output_path)
    typer.echo(_json({"status": "passed", "output": str(actual), "rows": len(result.frame), "label_schema": result.label_schema}))


@app.command("train-alpha-v7")
def train_alpha_v7_cli(
    dataset_path: Path = typer.Option(..., "--dataset"),
    output_dir: Path = typer.Option(Path("artifacts/v7_alpha"), "--output-dir"),
    model: str = typer.Option("ridge", "--model"),
    min_train_rows: int = typer.Option(100, "--min-train-rows"),
    mark_production_ready: bool = typer.Option(False, "--mark-production-ready"),
    paper_report: Path | None = typer.Option(None, "--paper-report"),
) -> None:
    """Train Ridge/ElasticNet alpha with purged walk-forward CV and acceptance gates."""
    from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment

    result = run_v7_training_experiment(
        _read_frame(dataset_path),
        V7TrainingConfig(
            model=model,
            min_train_rows=min_train_rows,
            output_dir=str(output_dir),
            mark_production_ready=mark_production_ready,
            paper_report_path=str(paper_report) if paper_report else None,
        ),
    )
    typer.echo(_json(result))


@app.command("walk-forward-backtest-v7")
def walk_forward_backtest_v7_cli(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("reports/v7/walk_forward_backtest.json"), "--output"),
) -> None:
    """Replay target_weights through the A-share OrderManager execution simulator."""
    from quantagent.backtest.ashare_execution_simulator import simulate_ashare_target_weights

    weights = _read_frame(target_weights_path)
    if "trade_date" in weights.columns:
        weights = weights.set_index("trade_date")
    result = simulate_ashare_target_weights(weights, _read_frame(market_panel_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nav": result.nav.to_dict(),
        "orders": result.order_audit.to_dict("records"),
        "failed_orders": result.failed_order_audit.to_dict("records"),
        "config": result.config,
    }
    output_path.write_text(_json(payload), encoding="utf-8")
    typer.echo(f"status=ok output={output_path} failed_orders={len(result.failed_order_audit)}")


@app.command("paper-trade-v7")
def paper_trade_v7_cli(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("reports/v7/paper_trade_report.json"), "--output"),
) -> None:
    """Run the same dry-run VirtualBroker path used by backtests; no live submit is possible here."""
    walk_forward_backtest_v7_cli(target_weights_path, market_panel_path, output_path)


@app.command("v7-live-readiness-report")
def v7_live_readiness_report_cli(
    metrics_path: Path = typer.Option(..., "--metrics"),
    paper_report: Path = typer.Option(..., "--paper-report"),
    output_path: Path = typer.Option(Path("reports/v7/live_readiness_report.json"), "--output"),
) -> None:
    """Evaluate live-readiness gates without enabling live trading."""
    from quantagent.data.v7_quality_gates import evaluate_model_acceptance_gates

    import json

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = evaluate_model_acceptance_gates(metrics, paper_report_path=paper_report).to_dict()
    report["safety_defaults"] = {"live_trading_enabled": False, "dry_run": True, "virtual_broker_only": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json(report), encoding="utf-8")
    typer.echo(_json(report))


@app.command("walk-forward-v7")
def walk_forward_v7_cli(
    sleeve_returns_path: Path = typer.Option(..., "--sleeve-returns"),
    output_path: Path = Path("reports/v7/walk_forward_sleeve_allocation.json"),
    grid_step: float = typer.Option(0.05, "--grid-step"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    walk_forward_splits: int = typer.Option(4, "--splits"),
    drawdown_penalty: float = typer.Option(0.50, "--drawdown-penalty"),
) -> None:
    """Run the walk-forward sleeve allocator on a daily sleeve-returns panel."""
    from quantagent.portfolio.walk_forward_sleeve_allocator import (
        WalkForwardSleeveConfig,
        allocate_sleeves_walk_forward,
    )

    frame = pd.read_csv(sleeve_returns_path)
    if "trade_date" not in frame.columns:
        raise typer.BadParameter("sleeve-returns CSV must contain a trade_date column")
    panel = frame.set_index("trade_date")
    result = allocate_sleeves_walk_forward(
        panel,
        config=WalkForwardSleeveConfig(
            walk_forward_splits=walk_forward_splits,
            embargo_days=embargo_days,
            grid_step=grid_step,
            drawdown_penalty=drawdown_penalty,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json(result), encoding="utf-8")
    typer.echo(f"status=ok windows={result.diagnostics.get('walk_forward_windows', 0)} cash_weight={result.cash_weight:.3f} output={output_path}")


@app.command("generate-factor-report")
def generate_factor_report(
    input_path: Path,
    output_path: Path,
    return_column: str,
) -> None:
    from quantagent.reports.factor_report import build_factor_report

    frame = pd.read_csv(input_path)
    factor_columns = [column for column in frame.select_dtypes("number").columns if column != return_column]
    report = build_factor_report(frame, factor_columns, return_column)
    report.rank_ic_table.to_csv(output_path, index=False)
    typer.echo(f"wrote {output_path}")


@app.command("generate-valuation-report")
def generate_valuation_report(
    config_path: Path,
    output_path: Path,
) -> None:
    from quantagent.fundamental.target_price import final_target_price_band
    from quantagent.fundamental.valuation import DCFInputs

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dcf = DCFInputs(**config["dcf_inputs"])
    estimate = final_target_price_band(
        symbol=str(config["symbol"]),
        current_price=float(config["current_price"]),
        dcf_inputs=dcf,
        relative_price=config.get("relative_price"),
        fraud_risk=float(config.get("fraud_risk", 0.0)),
        quality_score=float(config.get("quality_score", 50.0)),
    )
    pd.DataFrame([estimate.__dict__]).to_csv(output_path, index=False)
    typer.echo(f"wrote {output_path}")


def build_factors_entry() -> None:
    typer.run(build_factors)


def evaluate_factors_entry() -> None:
    typer.run(evaluate_factors)


def build_flow_features_entry() -> None:
    typer.run(build_flow_features)


def build_sector_rotation_entry() -> None:
    typer.run(build_sector_rotation)


def generate_factor_report_entry() -> None:
    typer.run(generate_factor_report)


def generate_valuation_report_entry() -> None:
    typer.run(generate_valuation_report)


def _json(value: object) -> str:
    import json

    def default(obj: object) -> object:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        return str(obj)

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=default)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    app()
