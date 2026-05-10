from __future__ import annotations

from pathlib import Path

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


@app.command("run-v3-backtest")
def run_v3_backtest(
    weights_path: Path,
    prices_path: Path,
    output_path: Path,
) -> None:
    from quantagent.backtest.engine import EventDrivenBacktester

    weights = pd.read_csv(weights_path)
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    target_weights = weights.set_index("trade_date")
    result = EventDrivenBacktester().run(target_weights, pd.read_csv(prices_path))
    diagnostics = pd.DataFrame([result.diagnostics])
    diagnostics.to_csv(output_path, index=False)
    typer.echo(f"wrote {output_path}")


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


def run_v3_backtest_entry() -> None:
    typer.run(run_v3_backtest)


def generate_factor_report_entry() -> None:
    typer.run(generate_factor_report)


def generate_valuation_report_entry() -> None:
    typer.run(generate_valuation_report)


if __name__ == "__main__":
    app()
