from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, is_dataclass

import pandas as pd
import typer
import yaml

from quantagent.domain.schemas import ModelScores
from quantagent.factors.evaluation import factor_summary_table, forward_return_labels
from quantagent.strategy.decision_engine import decide_trade

app = typer.Typer(help="QuantAgent research and decision CLI.")


@app.command()
def demo_decision(
    ticker: str = "NVDA",
    short_score: float = 82.0,
    long_score: float = 86.0,
    news_score: float = 70.0,
    llm_score: float = 68.0,
    risk_score: float = 32.0,
    confidence: float = 0.72,
) -> None:
    """Run the deterministic decision layer with normalized scores."""
    decision = decide_trade(
        ModelScores(
            ticker=ticker,
            short_score=short_score,
            long_score=long_score,
            news_score=news_score,
            llm_score=llm_score,
            risk_score=risk_score,
            confidence=confidence,
        )
    )
    typer.echo(decision)


@app.command("build-factors")
def build_factors(
    input_path: Path,
    output_path: Path,
    library: str = "alpha101",
) -> None:
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


@app.command("build-features-v4")
def build_features_v4(
    output_path: Path = Path("data/processed/v4_features.csv"),
) -> None:
    from quantagent.services.build_features_service import build_features_v4 as build_service

    result = build_service()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(output_path, index=False)
    typer.echo(f"wrote {len(result.frame)} v4 feature rows to {output_path}")


@app.command("train-v4")
def train_v4() -> None:
    from quantagent.services.train_v4_service import train_v4_synthetic

    metadata = train_v4_synthetic()
    typer.echo(metadata)


@app.command("infer-v4")
def infer_v4(
    feature_path: Path | None = None,
    output_path: Path = Path("data/processed/v4_signals.csv"),
) -> None:
    from quantagent.services.build_features_service import build_features_v4 as build_service
    from quantagent.services.daily_signal_service import infer_v4_alpha

    features = pd.read_csv(feature_path) if feature_path else build_service().frame
    signals = infer_v4_alpha(features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path, index=False)
    typer.echo(f"wrote {len(signals)} v4 signals to {output_path}")


@app.command("build-portfolio-v4")
def build_portfolio_v4(
    signal_path: Path | None = None,
    output_path: Path = Path("data/processed/v4_target_weights.csv"),
    mode: str = "long_only_enhancement",
) -> None:
    from quantagent.services.build_features_service import build_features_v4 as build_service
    from quantagent.services.daily_signal_service import infer_v4_alpha
    from quantagent.services.portfolio_build_service import build_portfolio_v4 as portfolio_service

    signals = pd.read_csv(signal_path) if signal_path else infer_v4_alpha(build_service().frame)
    result = portfolio_service(signals, mode=mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.target_weights.rename("target_weight").reset_index().rename(columns={"index": "symbol"}).to_csv(output_path, index=False)
    typer.echo(f"wrote {len(result.target_weights)} target weights to {output_path}")


@app.command("backtest-v4")
def backtest_v4(
    output_path: Path = Path("data/processed/v4_backtest_report.csv"),
) -> None:
    from quantagent.backtest.engine import EventDrivenBacktester
    from quantagent.services.build_features_service import build_features_v4 as build_service
    from quantagent.services.daily_signal_service import infer_v4_alpha
    from quantagent.services.portfolio_build_service import build_portfolio_v4 as portfolio_service

    features = build_service().frame
    signals = infer_v4_alpha(features)
    portfolio = portfolio_service(signals)
    dates = sorted(pd.to_datetime(features["trade_date"]).drop_duplicates())
    weights = pd.DataFrame(0.0, index=dates, columns=portfolio.target_weights.index)
    weights.iloc[-10:] = portfolio.target_weights
    result = EventDrivenBacktester().run(weights, features[["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.report | result.diagnostics]).to_csv(output_path, index=False)
    typer.echo(f"wrote v4 backtest report to {output_path}")


@app.command("paper-trade-v4")
def paper_trade_v4(
    dry_run: bool = True,
) -> None:
    from quantagent.services.build_features_service import build_features_v4 as build_service
    from quantagent.services.daily_signal_service import infer_v4_alpha
    from quantagent.services.paper_trading_service import generate_dry_run_order_intents
    from quantagent.services.portfolio_build_service import build_portfolio_v4 as portfolio_service

    features = build_service().frame
    signals = infer_v4_alpha(features)
    portfolio = portfolio_service(signals)
    latest = features.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")
    intents = generate_dry_run_order_intents(portfolio.target_weights, latest["close"]) if dry_run else []
    typer.echo(f"generated {len(intents)} dry-run order intents")


@app.command("build-features-v6")
def build_features_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    start_date: str | None = None,
    end_date: str | None = None,
    universe: str | None = None,
    output_dir: Path = Path("data/processed"),
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import build_features_v6

    del dry_run
    result = build_features_v6(config, start_date=start_date, end_date=end_date, universe=universe)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v6_features.csv"
    result.frame.to_csv(output_path, index=False)
    typer.echo(f"status=ok rows={len(result.frame)} feature_version={result.feature_version} output={output_path}")


@app.command("train-v6")
def train_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    output_dir: Path = Path("artifacts/models/v6"),
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import build_features_v6, train_v6_multitower
    import yaml as _yaml

    cfg = _yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    cfg.setdefault("safety", {})["dry_run"] = dry_run
    features = build_features_v6(cfg)
    result = train_v6_multitower(features.frame, cfg.get("model", {}), output_dir=output_dir, dry_run=dry_run)
    typer.echo(_json(result))


@app.command("validate-v6")
def validate_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    output_dir: Path | None = None,
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import load_v6_config, validate_v6

    cfg = load_v6_config(config)
    if output_dir is not None:
        cfg.setdefault("reporting", {})["output_dir"] = str(output_dir)
    cfg.setdefault("safety", {})["dry_run"] = dry_run
    typer.echo(_json(validate_v6(cfg)))


@app.command("infer-v6")
def infer_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    date: str | None = typer.Option(None, "--date"),
    output_dir: Path = Path("data/processed"),
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import infer_v6, load_v6_config

    cfg = load_v6_config(config)
    cfg.setdefault("safety", {})["dry_run"] = dry_run
    result = infer_v6(cfg, trade_date=date)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v6_inference.csv"
    result.to_csv(output_path, index=False)
    typer.echo(f"status=ok rows={len(result)} output={output_path}")


@app.command("build-portfolio-v6")
def build_portfolio_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    date: str | None = typer.Option(None, "--date"),
    output_dir: Path = Path("data/processed"),
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import build_portfolio_v6, load_v6_config

    cfg = load_v6_config(config)
    cfg.setdefault("safety", {})["dry_run"] = dry_run
    result = build_portfolio_v6(cfg, trade_date=date)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v6_target_weights.csv"
    result["target_weights"].rename("target_weight").reset_index().rename(columns={"index": "symbol"}).to_csv(output_path, index=False)
    typer.echo(f"status=ok weights={len(result['target_weights'])} output={output_path}")


@app.command("backtest-v6")
def backtest_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: Path = Path("reports/v6"),
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import load_v6_config, run_backtest_v6

    cfg = load_v6_config(config)
    cfg.setdefault("safety", {})["dry_run"] = dry_run
    result = run_backtest_v6(cfg, start_date=start_date, end_date=end_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v6_backtest_report.json"
    output_path.write_text(_json(result), encoding="utf-8")
    typer.echo(f"status=ok output={output_path}")


@app.command("replay-v6")
def replay_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    scenario: str = "mock_recent_replay",
    output_dir: Path | None = None,
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import load_v6_config, run_historical_live_replay_v6

    cfg = load_v6_config(config)
    cfg.setdefault("safety", {})["dry_run"] = dry_run
    if output_dir is not None:
        cfg.setdefault("reporting", {})["output_dir"] = str(output_dir)
    result = run_historical_live_replay_v6(cfg, scenario_name=scenario)
    typer.echo(_json(result))


@app.command("paper-trade-v6")
def paper_trade_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    date: str | None = typer.Option(None, "--date"),
    output_dir: Path | None = None,
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import load_v6_config, run_paper_trade_v6

    cfg = load_v6_config(config)
    cfg.setdefault("execution", {})["dry_run"] = dry_run
    if output_dir is not None:
        cfg.setdefault("reporting", {})["output_dir"] = str(output_dir)
    result = run_paper_trade_v6(cfg, trade_date=date)
    typer.echo(f"status=ok orders={len(result['order_states'])} account_value={result['account_value']:.2f} audit={result['audit_path']}")


@app.command("audit-replay-v6")
def audit_replay_v6_cli(
    config: Path = Path("configs/v6.default.yaml"),
    audit_path: Path = Path("logs/v6/virtual_broker_audit.jsonl"),
    output_dir: Path | None = None,
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import audit_replay_v6

    del config, output_dir, dry_run
    typer.echo(_json(audit_replay_v6(audit_path)))


@app.command("generate-v6-report")
def generate_v6_report_cli(
    config: Path = Path("configs/v6.default.yaml"),
    output_dir: Path = Path("reports/v6"),
    dry_run: bool = True,
) -> None:
    from quantagent.services.v6_pipeline_service import generate_v6_report, load_v6_config

    cfg = load_v6_config(config)
    cfg.setdefault("safety", {})["dry_run"] = dry_run
    result = generate_v6_report(cfg, output_dir=output_dir)
    typer.echo(_json(result))


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


if __name__ == "__main__":
    app()
