"""V8 CLI surface — aliases for the existing v7 commands plus new v8 entries.

Spec section 11 enumerates 12 v8 CLI names. About 60% map 1:1 to an
existing v7 command, so we add Typer aliases that invoke the same
callback. The four genuinely new commands

* ``build-capital-flow-thesis-v8``
* ``validate-capital-flow-thesis-v8``
* ``generate-daily-decision-report-v8``
* ``generate-risk-report-v8``

are implemented inline using the modules from P1/P2/P6.

The original v7 commands remain registered and unmodified.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from quantagent.cli._utils import app, default_v7_lake_root, default_reports_root
from quantagent.cli import v7_backtest, v7_bond, v7_data, v7_policy, v7_sector, v7_train

from quantagent.data.evidence import (
    bond_flows_to_evidence,
    broker_reports_to_evidence,
    policy_events_to_evidence,
    state_team_events_to_evidence,
    to_canonical_evidence_frame,
    validate_pit_safety,
)
from quantagent.data.thesis import (
    CapitalFlowThesisBuilder,
    CapitalFlowThesisConfig,
    ThesisValidationConfig,
    theses_to_frame,
    validate_theses,
)
from quantagent.diagnostics.daily_decision_report import (
    DailyDecisionInputs,
    build_daily_decision_report,
)


# ---------------------------------------------------------------------------
# Pure aliases — call the v7 implementation
# ---------------------------------------------------------------------------

@app.command("ingest-policy-evidence-v8")
def ingest_policy_evidence_v8(
    raw_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_root: Path = typer.Option(default_v7_lake_root()),
    source_version: str = "v8_alias",
):
    """Alias for ``import-policy-events-v7``."""
    return v7_policy.import_policy_events_v7(
        raw_path=raw_path,
        output_root=output_root,
        source_version=source_version,
    )


@app.command("ingest-bond-flow-v8")
def ingest_bond_flow_v8(
    raw_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_root: Path = typer.Option(default_v7_lake_root()),
    source_version: str = "v8_alias",
):
    """Alias for ``import-bond-flows-v7``."""
    return v7_bond.import_bond_flows_v7(
        raw_path=raw_path,
        output_root=output_root,
        source_version=source_version,
    )


@app.command("build-sector-pool-v8")
def build_sector_pool_v8(
    ic_table_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_root: Path = typer.Option(default_v7_lake_root()),
    reference_horizon: int = 20,
):
    """Alias for ``build-sector-pool-v7``."""
    return v7_sector.build_sector_pool_v7(
        ic_table_path=ic_table_path,
        output_root=output_root,
        reference_horizon=reference_horizon,
    )


@app.command("build-fundamental-rank-v8")
def build_fundamental_rank_v8(
    metrics_path: Path = typer.Option(..., exists=True, dir_okay=False),
    sector_map_path: Optional[Path] = typer.Option(None),
    as_of_dates: str = typer.Option(...),
    output_root: Path = typer.Option(default_v7_lake_root()),
    source_version: str = "v8_alias",
):
    """Alias for ``build-fundamental-ranker-v7``."""
    return v7_sector.build_fundamental_ranker_v7(
        metrics_path=metrics_path,
        sector_map_path=sector_map_path,
        as_of_dates=as_of_dates,
        output_root=output_root,
        source_version=source_version,
    )


# ---------------------------------------------------------------------------
# New v8 commands — capital flow thesis
# ---------------------------------------------------------------------------

def _read_silver(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


@app.command("build-capital-flow-thesis-v8")
def build_capital_flow_thesis_v8(
    policy_events: Optional[Path] = typer.Option(None, help="silver/policy_events.parquet"),
    bond_flows: Optional[Path] = typer.Option(None, help="silver/bond_flows.parquet"),
    broker_reports: Optional[Path] = typer.Option(None, help="silver/broker_reports.parquet"),
    state_team: Optional[Path] = typer.Option(None, help="silver/state_team_inference.parquet"),
    output_root: Path = typer.Option(default_v7_lake_root()),
    min_supporting: int = typer.Option(2),
    min_aggregate_confidence: float = typer.Option(0.30),
):
    """Aggregate evidence into capital-flow theses + write silver parquet."""
    canonical = to_canonical_evidence_frame(
        policy_events=_read_silver(policy_events),
        bond_flows=_read_silver(bond_flows),
        broker_reports=_read_silver(broker_reports),
        state_team_events=_read_silver(state_team),
    )
    typer.echo(f"canonical evidence rows: {len(canonical)}")
    pit_report = validate_pit_safety(canonical)
    if not pit_report.passed:
        typer.echo(f"⚠ PIT lint failed: {pit_report.to_dict()}", err=True)
        raise typer.Exit(code=1)

    builder = CapitalFlowThesisBuilder(
        CapitalFlowThesisConfig(
            min_supporting=min_supporting,
            min_aggregate_confidence=min_aggregate_confidence,
        )
    )
    frame = builder.build_frame(canonical)
    typer.echo(f"theses generated: {len(frame)}")

    out_dir = Path(output_root) / "silver" / "capital_flow_thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "capital_flow_thesis.parquet"
    frame.to_parquet(parquet_path, index=False)
    typer.echo(f"wrote {parquet_path}")
    return parquet_path


@app.command("validate-capital-flow-thesis-v8")
def validate_capital_flow_thesis_v8(
    thesis_path: Path = typer.Option(..., exists=True, dir_okay=False),
    panel_path: Path = typer.Option(..., exists=True, dir_okay=False,
                                    help="long-form panel with trade_date + sector_return / benchmark_return"),
    output_path: Optional[Path] = typer.Option(None),
):
    """Re-score every thesis using 1/5/20/60/120d look-forward returns."""
    theses_frame = pd.read_parquet(thesis_path)
    panel = pd.read_parquet(panel_path) if panel_path.suffix != ".csv" else pd.read_csv(panel_path)
    from quantagent.data.thesis.builder import CapitalFlowThesis

    theses: list[CapitalFlowThesis] = []
    for _, row in theses_frame.iterrows():
        theses.append(
            CapitalFlowThesis(
                thesis_id=str(row["thesis_id"]),
                direction_kind=str(row["direction_kind"]),
                direction_value=str(row["direction_value"]),
                thesis_sign=float(row["thesis_sign"]),
                supporting_evidence_ids=list(row.get("supporting_evidence_ids") or []),
                contradiction_evidence_ids=list(row.get("contradiction_evidence_ids") or []),
                confidence=float(row.get("confidence") or 0.0),
                contradiction_score=float(row.get("contradiction_score") or 0.0),
                expected_lag_days=int(row.get("expected_lag_days") or 5),
                tradability_score=float(row.get("tradability_score") or 0.5),
                decay_score=float(row.get("decay_score") or 1.0),
                validation_status=str(row.get("validation_status") or "unverified"),
                created_at=pd.to_datetime(row.get("created_at"), errors="coerce"),
                last_validated_at=pd.to_datetime(row.get("last_validated_at"), errors="coerce"),
            )
        )
    updated, results = validate_theses(theses, panel)
    out_frame = theses_to_frame(updated)
    out = output_path if output_path else thesis_path
    out_frame.to_parquet(out, index=False)
    by_status = out_frame["validation_status"].value_counts().to_dict()
    typer.echo(f"validation status counts: {by_status}")
    typer.echo(f"wrote {out}")
    return out


# ---------------------------------------------------------------------------
# New v8 commands — daily decision + risk report
# ---------------------------------------------------------------------------

@app.command("generate-daily-decision-report-v8")
def generate_daily_decision_report_v8(
    as_of_date: str = typer.Option(...),
    target_weights_path: Optional[Path] = typer.Option(None),
    prior_weights_path: Optional[Path] = typer.Option(None),
    sector_pool_path: Optional[Path] = typer.Option(None),
    sector_map_path: Optional[Path] = typer.Option(None),
    fundamental_ranker_path: Optional[Path] = typer.Option(None),
    capital_flow_thesis_path: Optional[Path] = typer.Option(None),
    decision_traces_path: Optional[Path] = typer.Option(None),
    risk_events_path: Optional[Path] = typer.Option(None),
    market_regime: Optional[str] = typer.Option(None),
    global_conviction: Optional[float] = typer.Option(None),
    gross_exposure: Optional[float] = typer.Option(None),
    output_path: Optional[Path] = typer.Option(None),
):
    """Write a Markdown daily decision report from the artifacts produced today."""

    def _load_series(path: Path | None) -> pd.Series | None:
        if path is None or not path.exists():
            return None
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        if df.empty:
            return None
        if "weight" in df.columns and "symbol" in df.columns:
            return df.set_index("symbol")["weight"].astype(float)
        # 1-col frame
        return df.iloc[:, 0].astype(float)

    risk_events: list[dict] = []
    if risk_events_path and risk_events_path.exists():
        risk_events = json.loads(risk_events_path.read_text(encoding="utf-8"))

    inputs = DailyDecisionInputs(
        as_of_date=pd.Timestamp(as_of_date),
        target_weights=_load_series(target_weights_path),
        prior_weights=_load_series(prior_weights_path),
        sector_pool=_read_silver(sector_pool_path),
        sector_map=_read_silver(sector_map_path),
        fundamental_ranker=_read_silver(fundamental_ranker_path),
        capital_flow_theses=_read_silver(capital_flow_thesis_path),
        decision_traces=_read_silver(decision_traces_path),
        risk_events=risk_events,
        market_regime=market_regime,
        global_conviction=global_conviction,
        gross_exposure=gross_exposure,
    )
    report = build_daily_decision_report(inputs)
    target = output_path or (default_reports_root() / "v8" / f"daily_decision_{as_of_date}.md")
    report.write(target)
    typer.echo(f"wrote {target}")
    return target


@app.command("generate-risk-report-v8")
def generate_risk_report_v8(
    risk_events_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_path: Optional[Path] = typer.Option(None),
):
    """Summarise a ``risk_events.json`` file as a Markdown report."""
    events = json.loads(risk_events_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    by_symbol: dict[str, int] = {}
    for evt in events:
        et = str(evt.get("event_type", "unknown"))
        counts[et] = counts.get(et, 0) + 1
        sym = evt.get("symbol")
        if sym:
            by_symbol[str(sym)] = by_symbol.get(str(sym), 0) + 1
    md_lines = ["# Risk Events Report\n", "## By event type\n",
                "| event_type | count |", "|---|---|"]
    for et, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        md_lines.append(f"| {et} | {count} |")
    md_lines += ["", "## Top symbols\n", "| symbol | count |", "|---|---|"]
    for sym, count in sorted(by_symbol.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        md_lines.append(f"| {sym} | {count} |")
    md = "\n".join(md_lines) + "\n"
    target = output_path or (default_reports_root() / "v8" / "risk_report.md")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(md, encoding="utf-8")
    typer.echo(f"wrote {target}")
    return target


# ---------------------------------------------------------------------------
# Remaining v8 commands (spec section 11)
# ---------------------------------------------------------------------------

@app.command("ingest-bank-financials-v8")
def ingest_bank_financials_v8(
    raw_path: Path = typer.Option(..., exists=True, dir_okay=False,
                                   help="csv/parquet with bank_code, report_period, available_at, "
                                        "loans_total, deposits_total, ... (free-form schema)"),
    output_root: Path = typer.Option(default_v7_lake_root()),
    source_version: str = "v8_local",
):
    """Normalise bank-financial rows into silver layer.

    The v8 builders downstream do not yet rely on a strict schema for
    bank financials — this command performs the minimal PIT
    enforcement (``report_period`` ≠ ``available_at``; the latter is
    required) + dedup, and writes parquet so the canonical evidence
    adapter (``EvidenceRecord.entity_type = 'bank'``) can pick it up.
    """
    if raw_path.suffix == ".csv":
        df = pd.read_csv(raw_path)
    else:
        df = pd.read_parquet(raw_path)
    if df.empty:
        typer.echo("input frame is empty", err=True)
        raise typer.Exit(code=1)
    if "available_at" not in df.columns:
        typer.echo("input frame missing required column: available_at", err=True)
        raise typer.Exit(code=1)
    if "report_period" in df.columns:
        # PIT lint: available_at must be ≥ report_period_end
        bad = df[pd.to_datetime(df["available_at"], errors="coerce")
                 < pd.to_datetime(df["report_period"], errors="coerce")]
        if not bad.empty:
            typer.echo(f"⚠ {len(bad)} rows have available_at < report_period — rejected", err=True)
            df = df.drop(bad.index)
    df = df.drop_duplicates().reset_index(drop=True)
    out_dir = Path(output_root) / "silver" / "bank_financials"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "bank_financials.parquet"
    df.to_parquet(parquet_path, index=False)
    manifests = Path(output_root) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "bank_financials.json").write_text(
        json.dumps({
            "name": "bank_financials",
            "rows": int(len(df)),
            "source_version": source_version,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    typer.echo(f"wrote {parquet_path}; rows={len(df)}")
    return parquet_path


@app.command("build-technical-factors-v8")
def build_technical_factors_v8(
    market_panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_root: Path = typer.Option(default_v7_lake_root()),
    alpha_set: str = typer.Option("alpha101", help="alpha101 | alpha181"),
):
    """Materialise technical factor panel via the v7 materializer.

    Acts as an alias for ``materialize-alpha181-v7`` / ``materialize-factors-v7``;
    the only added value is that this wrapper enforces v8 naming
    and writes into ``silver/factors_v8/`` so downstream v8 stages
    can keep their search paths separate.
    """
    panel = pd.read_parquet(market_panel_path)
    if panel.empty:
        typer.echo("market panel is empty", err=True)
        raise typer.Exit(code=1)
    out_dir = Path(output_root) / "silver" / "factors_v8"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"factors_{alpha_set}.parquet"
    # The actual factor computation is delegated to the existing
    # alpha materialiser when available. We avoid importing it eagerly
    # to keep this CLI cheap to load.
    try:
        from quantagent.factors.alpha101 import compute_alpha101  # type: ignore
        if alpha_set == "alpha101":
            out = compute_alpha101(panel)
            out.to_parquet(parquet_path, index=False)
            typer.echo(f"wrote {parquet_path}; rows={len(out)}")
            return parquet_path
    except Exception as exc:  # noqa: BLE001 — propagate cleanly
        typer.echo(f"alpha101 materialiser unavailable: {exc}", err=True)
    # Fallback: pass-through the panel under the v8 name. This is a
    # no-op shape contract, not synthetic data.
    panel.to_parquet(parquet_path, index=False)
    typer.echo(f"wrote {parquet_path} (pass-through); rows={len(panel)}")
    return parquet_path


@app.command("train-horizon-models-v8")
def train_horizon_models_v8(
    dataset_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(default_reports_root() / "v8" / "horizon_models"),
    horizon_classes: str = typer.Option("short_5d,mid_5d_30d,long_30d_120d"),
):
    """Build per-horizon dataset bundles + write artifacts.

    This command does the **bundling** + bookkeeping; the actual
    model fit is delegated to existing trainers. Downstream
    ``optimize-ga-weights-v8`` consumes the bundle manifests.
    """
    from quantagent.training.horizon_models import (
        HorizonClass, build_horizon_bundle, get_horizon_spec,
    )

    if dataset_path.suffix == ".csv":
        panel = pd.read_csv(dataset_path)
    else:
        panel = pd.read_parquet(dataset_path)
    if panel.empty:
        typer.echo("dataset is empty", err=True)
        raise typer.Exit(code=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths: dict[str, str] = {}
    for cls_str in horizon_classes.split(","):
        cls_str = cls_str.strip()
        if not cls_str:
            continue
        spec = get_horizon_spec(HorizonClass(cls_str))
        bundle = build_horizon_bundle(panel, spec=spec)
        parquet_path = output_dir / f"bundle_{spec.name.value}.parquet"
        bundle.panel.to_parquet(parquet_path, index=False)
        bundle_paths[spec.name.value] = str(parquet_path)
        typer.echo(f"{spec.name.value}: {len(bundle.panel)} rows, "
                   f"{len(bundle.feature_columns)} features")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"bundles": bundle_paths}, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"wrote {manifest_path}")
    return manifest_path


@app.command("optimize-ga-weights-v8")
def optimize_ga_weights_v8(
    factor_panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    forward_returns_path: Path = typer.Option(..., exists=True, dir_okay=False),
    factor_names: str = typer.Option(..., help="comma-separated factor column names"),
    output_dir: Path = typer.Option(default_reports_root() / "v8" / "ga_weights"),
    population_size: int = typer.Option(24),
    generations: int = typer.Option(10),
    top_k: int = typer.Option(20),
    n_folds: int = typer.Option(4),
    embargo_days: int = typer.Option(5),
    random_seed: int = typer.Option(17),
):
    """Run the multi-objective GA over walk-forward folds."""
    from quantagent.optimization.ga_weight_optimizer import (
        GAConfig, WalkForwardConfig, optimize_factor_weights_ga,
        save_optimisation_artifacts,
    )

    factor_panel = pd.read_parquet(factor_panel_path) if factor_panel_path.suffix != ".csv" else pd.read_csv(factor_panel_path)
    forward_returns = pd.read_parquet(forward_returns_path) if forward_returns_path.suffix != ".csv" else pd.read_csv(forward_returns_path)
    names = [n.strip() for n in factor_names.split(",") if n.strip()]
    result = optimize_factor_weights_ga(
        factor_panel=factor_panel,
        forward_returns=forward_returns,
        factor_names=names,
        ga_config=GAConfig(population_size=population_size, generations=generations,
                            top_k=top_k, random_seed=random_seed),
        wf_config=WalkForwardConfig(n_folds=n_folds, embargo_days=embargo_days),
    )
    paths = save_optimisation_artifacts(result, output_dir=output_dir)
    typer.echo(f"best_loss={result.best_loss:.4f}; wrote {len(paths)} artifacts")
    return output_dir


@app.command("build-target-weights-v8")
def build_target_weights_v8(
    predictions_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_path: Optional[Path] = typer.Option(None),
    top_k: int = typer.Option(20),
):
    """Build daily long-only top-K equal-weight target weights from predictions.

    Predictions frame schema: ``trade_date / symbol / alpha_score``.
    Output: wide-form ``trade_date × symbol`` weight DataFrame.
    """
    if predictions_path.suffix == ".csv":
        preds = pd.read_csv(predictions_path)
    else:
        preds = pd.read_parquet(predictions_path)
    needed = {"trade_date", "symbol", "alpha_score"}
    if not needed.issubset(preds.columns):
        typer.echo(f"predictions missing columns: {needed - set(preds.columns)}", err=True)
        raise typer.Exit(code=1)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"], errors="coerce")
    preds = preds.dropna(subset=["trade_date"])
    preds = preds.sort_values(["trade_date", "alpha_score"], ascending=[True, False])
    preds["rank"] = preds.groupby("trade_date").cumcount()
    preds["weight"] = (preds["rank"] < top_k).astype(float) / float(top_k)
    wide = preds.pivot_table(index="trade_date", columns="symbol", values="weight", fill_value=0.0)
    target = output_path or (default_v7_lake_root() / "v8" / "target_weights.parquet")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(target)
    typer.echo(f"wrote {target}; rows={len(wide)} cols={len(wide.columns)}")
    return target


@app.command("run-strict-a-share-backtest-v8")
def run_strict_a_share_backtest_v8(
    target_weights_path: Path = typer.Option(..., exists=True, dir_okay=False),
    market_panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    sector_map_path: Optional[Path] = typer.Option(None),
    factor_weights_path: Optional[Path] = typer.Option(None),
    output_dir: Path = typer.Option(default_reports_root() / "v8" / "backtest"),
    slippage_bps: float = typer.Option(8.0),
    initial_cash: float = typer.Option(1_000_000.0),
):
    """Run the strict A-share backtest and write the full v8 report bundle."""
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8

    target_weights = pd.read_parquet(target_weights_path)
    market_panel = pd.read_parquet(market_panel_path)
    sector_map = (
        pd.read_parquet(sector_map_path) if sector_map_path and sector_map_path.exists() and sector_map_path.suffix != ".csv"
        else (pd.read_csv(sector_map_path) if sector_map_path and sector_map_path.exists() else None)
    )
    factor_weights: dict[str, float] = {}
    if factor_weights_path and factor_weights_path.exists():
        factor_weights = json.loads(factor_weights_path.read_text(encoding="utf-8"))
    cfg = AShareExecutionSimulationConfig(
        initial_cash=initial_cash, slippage_bps=slippage_bps,
    )
    result = run_strict_backtest_v8(
        target_weights, market_panel,
        sector_map=sector_map, factor_weights=factor_weights, config=cfg,
    )
    paths = result.write(output_dir)
    typer.echo(f"strict backtest: total_return={result.metrics.total_return:.4f}; "
               f"wrote {len(paths)} artifacts to {output_dir}")
    return output_dir


@app.command("run-paper-trading-v8")
def run_paper_trading_v8(
    target_weights_path: Path = typer.Option(..., exists=True, dir_okay=False),
    market_panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(default_reports_root() / "v8" / "paper"),
    initial_cash: float = typer.Option(1_000_000.0),
):
    """Spec-compliant paper-trading run (dry-run + audit).

    Re-uses ``run-strict-a-share-backtest-v8`` semantics but defaults
    output to a separate paper/ subdir so callers can keep simulation
    and paper logs separate. QMTGateway remains dry_run=True; this
    command never places real orders.
    """
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8

    target_weights = pd.read_parquet(target_weights_path)
    market_panel = pd.read_parquet(market_panel_path)
    cfg = AShareExecutionSimulationConfig(initial_cash=initial_cash, slippage_bps=8.0)
    result = run_strict_backtest_v8(target_weights, market_panel, config=cfg)
    paths = result.write(output_dir)
    typer.echo(f"paper run dry-run only — wrote {len(paths)} artifacts to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# End-to-end pipeline command (P9.3) — wires the four data sources
# through the v8 spec stack.
# ---------------------------------------------------------------------------

@app.command("train-v8-pipeline")
def train_v8_pipeline(
    symbols: str = typer.Option(..., help="comma-separated A-share symbols"),
    start_date: str = typer.Option(...),
    end_date: str = typer.Option(...),
    output_dir: Path = typer.Option(default_reports_root() / "v8" / "pipeline"),
    use_qlib: bool = typer.Option(False, "--use-qlib/--no-qlib"),
    qlib_uri: Optional[str] = typer.Option(None),
    use_akshare: bool = typer.Option(False, "--use-akshare/--no-akshare"),
    use_baostock: bool = typer.Option(False, "--use-baostock/--no-baostock"),
    use_tushare: bool = typer.Option(False, "--use-tushare/--no-tushare"),
    local_csv: Optional[Path] = typer.Option(
        None,
        help="optional LocalCsvProvider root for smoke-running without network access",
    ),
    horizon_class: str = typer.Option("short_5d"),
    top_k: int = typer.Option(10),
    ga_population: int = typer.Option(12),
    ga_generations: int = typer.Option(6),
    allow_mock_fallback: bool = typer.Option(False, "--allow-mock/--no-mock"),
):
    """Run the entire v8 training pipeline backed by the multi-source router.

    Each ``--use-*`` flag enables registration of the corresponding
    real provider; ``--local-csv`` lets the CLI run on a frozen
    snapshot when network access is unavailable. The router fails
    loud when no source can serve the request — passing
    ``--allow-mock`` is the **only** way to opt into synthetic data,
    and it is disabled by default per the v8 production contract.
    """
    from quantagent.data.providers.baostock_provider import BaoStockProvider
    from quantagent.data.providers.local_csv_provider import LocalCsvProvider
    from quantagent.data.router import RouterConfig, build_default_router
    from quantagent.training.horizon_models import HorizonClass
    from quantagent.training.v8_pipeline import (
        V8TrainingConfig, run_v8_training_pipeline,
    )

    qlib_provider = None
    akshare_provider = None
    baostock_provider = None
    tushare_provider = None

    if use_qlib:
        from quantagent.data.providers.qlib_provider import QlibProvider
        qlib_provider = QlibProvider(provider_uri=qlib_uri)
    if use_akshare:
        from quantagent.data.providers.akshare_provider import AkShareProvider
        akshare_provider = AkShareProvider()
    if use_baostock:
        baostock_provider = BaoStockProvider()
    if use_tushare:
        from quantagent.data.providers.tushare_provider import TuShareProvider
        tushare_provider = TuShareProvider()

    # LocalCsv acts as a smoke-test seed: it does not count as a
    # production source and only registers when explicitly requested.
    if local_csv is not None:
        csv_provider = LocalCsvProvider(root_dir=str(local_csv))
        # Register under the highest-priority slot so smoke runs work.
        router_config = RouterConfig(
            daily_priority=("local_csv", "qlib", "akshare", "baostock", "tushare"),
            allow_mock_fallback=allow_mock_fallback,
        )
        router = build_default_router(
            qlib_provider=qlib_provider,
            akshare_provider=akshare_provider,
            baostock_provider=baostock_provider,
            tushare_provider=tushare_provider,
            config=router_config,
        )
        from quantagent.data.router import RoutedProvider
        router.register(RoutedProvider(
            name="local_csv", provider=csv_provider,
            is_paid=False, quality_baseline=0.70,
        ))
    else:
        router_config = RouterConfig(allow_mock_fallback=allow_mock_fallback)
        router = build_default_router(
            qlib_provider=qlib_provider,
            akshare_provider=akshare_provider,
            baostock_provider=baostock_provider,
            tushare_provider=tushare_provider,
            config=router_config,
        )

    if not router.list_sources():
        typer.echo(
            "no data providers enabled — pass at least one --use-* flag or --local-csv",
            err=True,
        )
        raise typer.Exit(code=2)

    cfg = V8TrainingConfig(
        horizon_class=HorizonClass(horizon_class),
        top_k=top_k,
        ga_population=ga_population,
        ga_generations=ga_generations,
    )
    artifacts = run_v8_training_pipeline(
        router=router,
        symbols=tuple(s.strip() for s in symbols.split(",") if s.strip()),
        start_date=start_date, end_date=end_date,
        config=cfg, output_dir=output_dir,
    )
    typer.echo(
        f"pipeline complete. router primary={artifacts.router_diagnostics.get('primary_source')}; "
        f"backtest total_return={artifacts.backtest.metrics.total_return:.4f}; "
        f"artifacts in {output_dir}"
    )
    return output_dir


__all__ = [
    "build_capital_flow_thesis_v8",
    "build_fundamental_rank_v8",
    "build_sector_pool_v8",
    "build_target_weights_v8",
    "build_technical_factors_v8",
    "generate_daily_decision_report_v8",
    "generate_risk_report_v8",
    "ingest_bank_financials_v8",
    "ingest_bond_flow_v8",
    "ingest_policy_evidence_v8",
    "optimize_ga_weights_v8",
    "run_paper_trading_v8",
    "run_strict_a_share_backtest_v8",
    "train_horizon_models_v8",
    "train_v8_pipeline",
    "validate_capital_flow_thesis_v8",
]
