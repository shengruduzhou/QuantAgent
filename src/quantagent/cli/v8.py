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
from typing import Any, Optional

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


@app.command("summarize-v8-results")
def summarize_v8_results(
    results_root: Path = typer.Option(
        Path("runtime/reports/v8"),
        exists=True,
        file_okay=False,
        help="root containing v8 headline_report.json/backtest/metrics.json artifacts",
    ),
    output_csv: Path = typer.Option(
        Path("runtime/reports/v8/result_table/v8_result_table.csv"),
        help="normalised result table CSV",
    ),
    output_md: Path = typer.Option(
        Path("runtime/reports/v8/result_table/v8_result_table.md"),
        help="normalised result table Markdown",
    ),
    max_drawdown_soft_cap: float = typer.Option(
        0.25, help="soft cap used only by return_first_score",
    ),
    drawdown_penalty: float = typer.Option(
        0.50, help="penalty per drawdown point beyond the soft cap",
    ),
):
    """Write a unified bull/bear v8 result table with one metric convention."""
    from quantagent.diagnostics.v8_result_table import (
        ResultScoreConfig,
        collect_v8_result_rows,
        write_v8_result_table,
    )

    table = collect_v8_result_rows(
        [results_root],
        score_config=ResultScoreConfig(
            max_drawdown_soft_cap=max_drawdown_soft_cap,
            drawdown_penalty=drawdown_penalty,
        ),
    )
    if table.empty:
        typer.echo(f"[warn] no headline_report.json found under {results_root}", err=True)
        raise typer.Exit(code=1)
    paths = write_v8_result_table(table, output_csv=output_csv, output_md=output_md)
    typer.echo(f"[ok] wrote {paths['csv']}")
    if "md" in paths:
        typer.echo(f"[ok] wrote {paths['md']}")

    # Print a compact top list by environment for quick operator triage.
    for env, g in table.groupby("market_env", sort=False):
        top = g.sort_values("return_first_score", ascending=False).head(5)
        typer.echo(f"\n[{env}] top return-first candidates")
        for _, r in top.iterrows():
            typer.echo(
                f"  {r['strategy']} | ann={float(r['annualized_return']):+.4f} "
                f"excess={float(r['excess_equal_weight_return']):+.4f} "
                f"maxDD={float(r['max_drawdown']):.4f} "
                f"score={float(r['return_first_score']):+.4f}"
            )
    return output_csv


@app.command("build-cicc-selection-dataset-v8")
def build_cicc_selection_dataset_v8(
    market_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True,
        dir_okay=False,
        help="silver market panel with OHLCV",
    ),
    dataset_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_intraday_cicc.parquet"),
        exists=True,
        dir_okay=False,
        help="gold training dataset to augment",
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
        help="optional sector map for CICC sector selection score",
    ),
    agent_scores_path: Optional[Path] = typer.Option(
        None,
        help="optional true agent scores with trade_date,symbol,agent_* columns",
    ),
    start_date: Optional[str] = typer.Option(None),
    end_date: Optional[str] = typer.Option(None),
    output_dataset_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_intraday_cicc_selection.parquet"),
        help="augmented training dataset output",
    ),
    selection_output_path: Path = typer.Option(
        Path("runtime/data/v7/silver/cicc_selection/cicc_selection_scores.parquet"),
        help="standalone CICC selection score output",
    ),
):
    """Compute CICC stock/sector selection scores and merge into gold dataset."""
    from quantagent.factors.cicc_ashare80 import compute_cicc_ashare80_factors
    from quantagent.factors.cicc_selection import compute_cicc_selection_scores

    market = pd.read_parquet(market_panel_path)
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    if start_date is not None:
        market = market[market["trade_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        market = market[market["trade_date"] <= pd.Timestamp(end_date)]
    if market.empty:
        typer.echo("[fatal] market panel is empty after date filtering", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[info] market rows={len(market):,} symbols={market['symbol'].nunique()} "
               f"{market['trade_date'].min()}→{market['trade_date'].max()}")

    sector_map = None
    if sector_map_path is not None and sector_map_path.exists():
        sector_map = pd.read_parquet(sector_map_path)
    typer.echo("[info] computing CICC A-share 80 proxy factors (wide) ...")
    cicc_wide = compute_cicc_ashare80_factors(market, wide=True)
    typer.echo(f"[info] CICC wide rows={len(cicc_wide):,} cols={len(cicc_wide.columns)}")
    selection = compute_cicc_selection_scores(cicc_wide, sector_map=sector_map)
    del cicc_wide

    if agent_scores_path is not None:
        if not agent_scores_path.exists():
            typer.echo(f"[fatal] agent scores missing: {agent_scores_path}", err=True)
            raise typer.Exit(code=1)
        agent = pd.read_parquet(agent_scores_path) if agent_scores_path.suffix != ".csv" else pd.read_csv(agent_scores_path)
        agent["trade_date"] = pd.to_datetime(agent["trade_date"], errors="coerce")
        agent["symbol"] = agent["symbol"].astype(str)
        keep = ["trade_date", "symbol"] + [
            c for c in agent.columns
            if c.startswith("agent_") or c.endswith("_agent_score")
        ]
        if len(keep) <= 2:
            typer.echo("[fatal] agent score file has no agent_* or *_agent_score columns", err=True)
            raise typer.Exit(code=1)
        selection = selection.merge(agent[keep], on=["trade_date", "symbol"], how="left")
        typer.echo(f"[info] merged true agent score columns={len(keep) - 2}")
    else:
        typer.echo("[info] no --agent-scores-path supplied; not fabricating agent scores")

    selection_output_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(selection_output_path, index=False)

    ds = pd.read_parquet(dataset_path)
    ds["trade_date"] = pd.to_datetime(ds["trade_date"], errors="coerce")
    ds["symbol"] = ds["symbol"].astype(str)
    merge_cols = [c for c in selection.columns if c not in {"trade_date", "symbol"}]
    merged = ds.merge(selection[["trade_date", "symbol", *merge_cols]], on=["trade_date", "symbol"], how="left")
    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_dataset_path, index=False)
    summary = {
        "selection_output_path": str(selection_output_path),
        "output_dataset_path": str(output_dataset_path),
        "dataset_rows": int(len(merged)),
        "selection_rows": int(len(selection)),
        "selection_columns": merge_cols,
        "coverage": {
            c: float(merged[c].notna().mean())
            for c in merge_cols
            if c in merged.columns
        },
    }
    (output_dataset_path.parent / f"{output_dataset_path.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


@app.command("build-core-factor-dataset-v8")
def build_core_factor_dataset_v8(
    dataset_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_intraday_cicc_selection.parquet"),
        exists=True,
        dir_okay=False,
        help="wide gold dataset with CICC/intraday/labels",
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
        help="optional sector map for sector resonance",
    ),
    fundamentals_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/fundamentals/metrics_panel.parquet"),
        help="optional PIT fundamentals metrics panel",
    ),
    evidence_path: Optional[Path] = typer.Option(
        None,
        help="optional canonical/evidence store parquet with policy/sentiment scores",
    ),
    agent_scores_path: Optional[Path] = typer.Option(
        None,
        help="optional LLM/agent scores with trade_date,symbol,agent_* columns",
    ),
    start_date: Optional[str] = typer.Option(None),
    end_date: Optional[str] = typer.Option(None),
    output_dataset_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"),
        help="core <=30 factor dataset output",
    ),
):
    """Build the <=30 core A-share factor dataset for regime experts."""
    from quantagent.factors.core_policy import aggregate_evidence_scores, build_core_factor_frame

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(dataset_path)
    labels = [c for c in pf.schema.names if c.startswith("forward_return_") or c.startswith("label_end_")]
    preferred = {
        "symbol", "trade_date", "available_at",
        "return_1d", "momentum_5d", "momentum_20d", "volatility_20d",
        "amount_mean_20d", "volume_mean_20d", "intraday_return",
        "first30_return", "last30_return", "vwap_deviation", "intraday_range_pos",
        "net_buy_pressure", "volume_concentration", "spike_minutes",
        "close30_volume_share", "flow_north_total", "flow_margin_sh", "idx_csi300_ret5",
        "cicc_stock_selection_score", "cicc_sector_selection_score",
        "cicc_aggressive_momentum_score", "cicc_defensive_quality_score",
        "cicc_liquidity_defense_score",
    }
    read_cols = [c for c in pf.schema.names if c in preferred or c in labels]
    typer.echo(f"[info] reading {len(read_cols)} columns from {dataset_path}")
    ds = pd.read_parquet(dataset_path, columns=read_cols)
    ds["trade_date"] = pd.to_datetime(ds["trade_date"], errors="coerce")
    if start_date is not None:
        ds = ds[ds["trade_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        ds = ds[ds["trade_date"] <= pd.Timestamp(end_date)]
    typer.echo(f"[info] core input rows={len(ds):,} symbols={ds['symbol'].nunique()}")

    sector_map = pd.read_parquet(sector_map_path) if sector_map_path is not None and sector_map_path.exists() else None
    fundamentals = pd.read_parquet(fundamentals_path) if fundamentals_path is not None and fundamentals_path.exists() else None
    evidence_scores = None
    if evidence_path is not None:
        if not evidence_path.exists():
            typer.echo(f"[fatal] evidence path missing: {evidence_path}", err=True)
            raise typer.Exit(code=1)
        evidence_scores = aggregate_evidence_scores(pd.read_parquet(evidence_path))
        typer.echo(f"[info] evidence score rows={len(evidence_scores):,}")
    agent_scores = None
    if agent_scores_path is not None:
        if not agent_scores_path.exists():
            typer.echo(f"[fatal] agent score path missing: {agent_scores_path}", err=True)
            raise typer.Exit(code=1)
        agent_scores = pd.read_parquet(agent_scores_path) if agent_scores_path.suffix != ".csv" else pd.read_csv(agent_scores_path)
        typer.echo(f"[info] agent score rows={len(agent_scores):,}")

    core, summary = build_core_factor_frame(
        ds,
        sector_map=sector_map,
        fundamentals=fundamentals,
        evidence_scores=evidence_scores,
        agent_scores=agent_scores,
    )
    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    core.to_parquet(output_dataset_path, index=False)
    summary_path = output_dataset_path.parent / f"{output_dataset_path.stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary.as_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    typer.echo(f"[done] wrote {output_dataset_path} rows={len(core):,}")
    typer.echo(f"[done] feature_count={len(summary.feature_columns)} old_dealer_block_rate={summary.old_dealer_block_rate:.4f}")
    typer.echo(f"[done] summary={summary_path}")
    return output_dataset_path


@app.command("run-llm-stock-selection-v8")
def run_llm_stock_selection_v8(
    predictions_path: Path = typer.Option(..., exists=True, dir_okay=False),
    core_dataset_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"),
        exists=True,
        dir_okay=False,
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
        help="optional sector map",
    ),
    as_of_date: Optional[str] = typer.Option(None, help="default: latest prediction date"),
    top_k: int = typer.Option(
        200,
        help="ranking candidate pool size sent to LLM/fallback; use >30 so factor and agent rankings can resonate",
    ),
    output_dir: Path = typer.Option(Path("runtime/reports/v8/llm_stock_selection")),
    allow_network: bool = typer.Option(False, "--allow-network/--no-allow-network"),
    require_llm: bool = typer.Option(
        False,
        "--require-llm/--allow-fallback",
        help="fail instead of writing deterministic fallback output when the LLM call is unavailable",
    ),
):
    """LLM-assisted stock-selection analysis over deterministic candidates.

    LLM/Agent output is evidence only. It never creates orders.
    """
    from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
    from quantagent.agents.skills import get_skill
    from quantagent.factors.core_policy import CORE_FACTOR_PRIOR_WEIGHTS

    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(predictions_path)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"], errors="coerce")
    pred["symbol"] = pred["symbol"].astype(str)
    score_col = "prediction" if "prediction" in pred.columns else (
        "composite_score" if "composite_score" in pred.columns else "alpha_score"
    )
    if score_col not in pred.columns:
        raise typer.BadParameter("predictions must include prediction, composite_score, or alpha_score")
    date = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(pred["trade_date"].max())
    day = pred[pred["trade_date"] == date].dropna(subset=[score_col]).copy()
    if day.empty:
        raise typer.BadParameter(f"no predictions for as_of_date={date.date()}")
    ranked = day.sort_values(score_col, ascending=False).head(int(top_k))[["trade_date", "symbol", score_col]]
    ranked = ranked.rename(columns={score_col: "prediction"})
    ranked["model_rank"] = range(1, len(ranked) + 1)

    core_cols = [
        "trade_date", "symbol",
        "core_policy_score", "core_sentiment_score", "fundamental_quality_score",
        "cicc_stock_selection_score", "cicc_sector_selection_score",
        "sector_resonance_score", "dip_buy_flow_score", "old_dealer_risk_score",
        "old_dealer_block", "trend_strength_score", "net_buy_pressure",
        "vwap_deviation", "intraday_range_pos", "volume_concentration",
    ]
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(core_dataset_path).schema.names)
    available = pd.read_parquet(core_dataset_path, columns=[c for c in core_cols if c in names])
    available["trade_date"] = pd.to_datetime(available["trade_date"], errors="coerce")
    available["symbol"] = available["symbol"].astype(str)
    available = available[available["trade_date"] == date]
    candidates = ranked.merge(available, on=["trade_date", "symbol"], how="left")

    if sector_map_path is not None and sector_map_path.exists():
        sector = pd.read_parquet(sector_map_path)
        if "sector_level_1" in sector.columns:
            sector["symbol"] = sector["symbol"].astype(str)
            candidates = candidates.merge(
                sector[["symbol", "sector_level_1"]].drop_duplicates("symbol"),
                on="symbol",
                how="left",
            )
    fallback = _fallback_stock_selection_analysis(candidates, CORE_FACTOR_PRIOR_WEIGHTS)
    skill = get_skill("stock_selection_analyst")
    env_cfg = LLMSkillConfig.from_env()
    client_cfg = LLMSkillConfig(
        provider=env_cfg.provider,
        enabled=env_cfg.enabled,
        allow_network=allow_network,
        endpoint=env_cfg.endpoint,
        model=env_cfg.model,
        api_key_env=env_cfg.api_key_env,
        timeout_seconds=env_cfg.timeout_seconds,
        max_input_chars=env_cfg.max_input_chars,
        temperature=env_cfg.temperature,
        response_format=env_cfg.response_format,
    )
    result = LLMSkillClient(client_cfg).invoke(
        skill.name,
        system_prompt=skill.system_prompt,
        user_text=json.dumps({
            "as_of_date": str(date.date()),
            "candidate_rows": candidates.replace({pd.NA: None}).to_dict("records"),
            "ranking_contract": {
                "candidate_pool_size": int(len(candidates)),
                "keep_model_rank": True,
                "do_not_limit_to_top30": True,
            },
            "core_factor_prior_weights": CORE_FACTOR_PRIOR_WEIGHTS,
            "constraints": {
                "no_live_orders": True,
                "t_plus_1": True,
                "avoid_old_dealer": True,
                "target": "maximize equal-weight all-A excess return with controlled drawdown",
            },
        }, ensure_ascii=False, default=str),
        fallback=fallback,
    )
    if require_llm and result.used_fallback:
        typer.echo(
            json.dumps({
                "status": "failed",
                "reason": "llm_required_but_fallback_used",
                "fallback_reason": result.fallback_reason,
                "provider": client_cfg.provider,
                "model": client_cfg.model,
                "api_key_env": client_cfg.api_key_env,
                "hint": "Set QUANTAGENT_LLM_PROVIDER=gemini, QUANTAGENT_LLM_ENABLED=1, QUANTAGENT_LLM_ALLOW_NETWORK=1, QUANTAGENT_LLM_MODEL, and GOOGLE_API_KEY/google_API_KEY in the runtime environment.",
            }, ensure_ascii=False, indent=2),
            err=True,
        )
        raise typer.Exit(code=2)
    analysis = result.output if result.output else fallback
    agent_scores = _agent_scores_from_analysis(candidates, analysis)
    agent_scores_path = output_dir / "agent_scores.parquet"
    analysis_path = output_dir / "stock_selection_analysis.json"
    md_path = output_dir / "stock_selection_analysis.md"
    agent_scores.to_parquet(agent_scores_path, index=False)
    analysis_path.write_text(
        json.dumps({
            "as_of_date": str(date.date()),
            "used_fallback": result.used_fallback,
            "fallback_reason": result.fallback_reason,
            "provider": client_cfg.provider,
            "model": client_cfg.model,
            "api_key_env": client_cfg.api_key_env,
            "analysis": analysis,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(_stock_selection_markdown(date, analysis, agent_scores, result), encoding="utf-8")
    typer.echo(json.dumps({
        "as_of_date": str(date.date()),
        "used_fallback": result.used_fallback,
        "fallback_reason": result.fallback_reason,
        "agent_scores_path": str(agent_scores_path),
        "analysis_path": str(analysis_path),
        "markdown_path": str(md_path),
        "rows": int(len(agent_scores)),
    }, ensure_ascii=False, indent=2))
    return output_dir


def _fallback_stock_selection_analysis(candidates: pd.DataFrame, weights: dict[str, float]) -> dict[str, object]:
    rows = []
    data = candidates.copy()
    for col in weights:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    data["old_dealer_risk_score"] = pd.to_numeric(data.get("old_dealer_risk_score", 0.0), errors="coerce").fillna(0.0)
    for _, row in data.iterrows():
        score = 50.0
        for col, weight in weights.items():
            score += 50.0 * float(weight) * float(row.get(col, 0.0))
        score += 10.0 * float(row.get("prediction", 0.0))
        score -= 25.0 * float(row.get("old_dealer_risk_score", 0.0))
        score = max(0.0, min(100.0, score))
        old_risk = float(row.get("old_dealer_risk_score", 0.0))
        bucket = "avoid" if bool(row.get("old_dealer_block", False)) or old_risk >= 0.70 else (
            "do_t_watch" if float(row.get("dip_buy_flow_score", 0.0)) > 0.15 else "core_watch"
        )
        rows.append({
            "symbol": str(row["symbol"]),
            "model_rank": int(row.get("model_rank", len(rows) + 1)),
            "agent_score": score,
            "conviction": round(score / 100.0, 4),
            "action_bucket": bucket,
            "key_positive_factors": _positive_factor_names(row, weights),
            "key_risks": ["old_dealer_risk"] if old_risk >= 0.55 else [],
            "regime_fit": "deterministic_core30_fallback",
            "do_t_suitability": float(max(0.0, min(1.0, 0.5 + float(row.get("dip_buy_flow_score", 0.0))))),
            "old_dealer_risk": old_risk,
            "rationale": "deterministic fallback from core30 priors; no LLM call used",
        })
    return {
        "summary": "deterministic fallback; configure QUANTAGENT_LLM_* and --allow-network to use LLM",
        "candidates": rows,
        "factor_weight_view": weights,
        "risk_flags": ["llm_not_used"],
        "next_research_steps": [
            "Backtest agent_scores as an additional rank feature.",
            "Run factor ranking/top-k selection search against strict excess-return objective.",
        ],
    }


def _positive_factor_names(row: pd.Series, weights: dict[str, float]) -> list[str]:
    positives = []
    for col, weight in weights.items():
        value = float(row.get(col, 0.0) or 0.0)
        if weight > 0 and value > 0.10:
            positives.append(col)
        if weight < 0 and value < 0.40:
            positives.append(f"low_{col}")
    return positives[:5]


def _agent_scores_from_analysis(candidates: pd.DataFrame, analysis: dict[str, object]) -> pd.DataFrame:
    by_symbol = {}
    for item in analysis.get("candidates", []) if isinstance(analysis, dict) else []:
        if isinstance(item, dict) and item.get("symbol"):
            by_symbol[str(item["symbol"])] = item
    rows = []
    for _, row in candidates.iterrows():
        item = by_symbol.get(str(row["symbol"]), {})
        score = float(item.get("agent_score", 50.0))
        conviction = float(item.get("conviction", score / 100.0))
        rows.append({
            "trade_date": row["trade_date"],
            "symbol": str(row["symbol"]),
            "model_rank": int(item.get("model_rank", row.get("model_rank", len(rows) + 1)) or len(rows) + 1),
            "agent_stock_score": score / 100.0,
            "agent_conviction_score": conviction,
            "agent_old_dealer_risk": float(item.get("old_dealer_risk", row.get("old_dealer_risk_score", 0.0) or 0.0)),
            "agent_do_t_suitability": float(item.get("do_t_suitability", 0.0)),
            "agent_action_bucket": str(item.get("action_bucket", "unknown")),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["agent_stock_score", "model_rank"], ascending=[False, True]).reset_index(drop=True)
        out["agent_rank"] = out.index + 1
    return out


def _stock_selection_markdown(
    date: pd.Timestamp,
    analysis: dict[str, object],
    scores: pd.DataFrame,
    result,
) -> str:
    lines = [
        f"# LLM Stock Selection Analysis - {date.date()}",
        "",
        "本报告是 research evidence，不是订单或投资建议。",
        f"- used_fallback: {result.used_fallback}",
        f"- fallback_reason: {result.fallback_reason}",
        f"- rows: {len(scores)}",
        "- ranking_mode: full candidate pool, not fixed top30",
        "",
        "## Summary",
        str(analysis.get("summary", "")) if isinstance(analysis, dict) else "",
        "",
        "## Top Agent Scores",
    ]
    if not scores.empty:
        top = scores.sort_values("agent_stock_score", ascending=False).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- #{int(row.get('agent_rank', 0))} {row['symbol']} "
                f"(model_rank={int(row.get('model_rank', 0))}): score={float(row['agent_stock_score']):.3f}, "
                f"bucket={row['agent_action_bucket']}, do_t={float(row['agent_do_t_suitability']):.3f}, "
                f"old_dealer={float(row['agent_old_dealer_risk']):.3f}"
            )
    return "\n".join(lines) + "\n"


@app.command("build-llm-hybrid-stock-pool-v8")
def build_llm_hybrid_stock_pool_v8(
    predictions_path: Path = typer.Option(..., exists=True, dir_okay=False),
    core_dataset_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"),
        exists=True,
        dir_okay=False,
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
        help="optional symbol -> sector_level_1 map",
    ),
    canonical_evidence_path: Optional[Path] = typer.Option(
        None,
        help="optional canonical EvidenceRecord parquet/csv; filtered by available_at <= as_of_date",
    ),
    policy_events: Optional[Path] = typer.Option(None, help="optional policy_events silver parquet/csv"),
    bond_flows: Optional[Path] = typer.Option(None, help="optional bond_flows silver parquet/csv"),
    broker_reports: Optional[Path] = typer.Option(None, help="optional broker_reports silver parquet/csv"),
    state_team: Optional[Path] = typer.Option(None, help="optional state_team_inference silver parquet/csv"),
    capital_flow_thesis_path: Optional[Path] = typer.Option(None, help="optional prebuilt capital_flow_thesis parquet/csv"),
    as_of_date: Optional[str] = typer.Option(None, help="default: latest prediction date"),
    candidate_pool_size: int = typer.Option(300, min=30, help="ranking rows sent into hybrid/LLM analysis"),
    stock_top_n: int = typer.Option(120, min=10, help="final stock-pool rows retained"),
    sector_top_n: int = typer.Option(20, min=1, help="sector/theme theses retained"),
    output_dir: Path = typer.Option(Path("runtime/reports/v8/llm_hybrid_stock_pool")),
    allow_network: bool = typer.Option(False, "--allow-network/--no-allow-network"),
    require_llm: bool = typer.Option(
        False,
        "--require-llm/--allow-fallback",
        help="fail instead of writing deterministic fallback output when capital-flow LLM is unavailable",
    ),
    capital: float = typer.Option(0.0, help="research-only capital amount used for allocation hints; 0 disables amount hints"),
    max_base_gross: float = typer.Option(0.60, help="research-only normal gross exposure ceiling"),
    max_high_conf_gross: float = typer.Option(0.80, help="research-only high-conviction gross exposure ceiling"),
    cash_reserve_min: float = typer.Option(0.20, help="research-only minimum cash reserve for Do-T/dip-buy inventory"),
):
    """Build a PIT LLM+factor hybrid stock pool from evidence, theses, and ranks.

    This produces evidence/ranking artifacts only. It never emits live orders.
    """
    from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
    from quantagent.agents.skills import get_skill
    from quantagent.factors.core_policy import CORE_FACTOR_PRIOR_WEIGHTS

    output_dir.mkdir(parents=True, exist_ok=True)
    pred = _read_table(predictions_path)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"], errors="coerce")
    pred["symbol"] = pred["symbol"].astype(str)
    score_col = _prediction_score_column(pred)
    date = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(pred["trade_date"].max())
    day = pred[pred["trade_date"] == date].dropna(subset=[score_col]).copy()
    if day.empty:
        raise typer.BadParameter(f"no predictions for as_of_date={date.date()}")
    ranked = day.sort_values(score_col, ascending=False).head(int(candidate_pool_size))[["trade_date", "symbol", score_col]]
    ranked = ranked.rename(columns={score_col: "prediction"})
    ranked["model_rank"] = range(1, len(ranked) + 1)

    sector_map = _read_table(sector_map_path) if sector_map_path is not None and sector_map_path.exists() else None
    candidates = _attach_core_and_sector_for_hybrid(ranked, core_dataset_path, sector_map, date)
    canonical = _load_hybrid_canonical_evidence(
        canonical_evidence_path=canonical_evidence_path,
        policy_events=policy_events,
        bond_flows=bond_flows,
        broker_reports=broker_reports,
        state_team=state_team,
        as_of_date=date,
    )
    pit_report = validate_pit_safety(canonical, as_of=date)
    if not pit_report.passed:
        typer.echo(json.dumps(pit_report.to_dict(), ensure_ascii=False, indent=2, default=str), err=True)
        raise typer.Exit(code=1)

    theses = _read_or_build_hybrid_theses(capital_flow_thesis_path, canonical)
    sector_pool = _sector_pool_from_theses(theses, sector_top_n=sector_top_n)
    candidates = _merge_hybrid_sector_scores(candidates, sector_pool)
    candidates = _add_hybrid_pre_llm_scores(candidates, CORE_FACTOR_PRIOR_WEIGHTS)
    fallback = _fallback_capital_flow_stock_pool(candidates, sector_pool, date, stock_top_n)

    skill = get_skill("capital_flow_sector_analyst")
    env_cfg = LLMSkillConfig.from_env()
    client_cfg = LLMSkillConfig(
        provider=env_cfg.provider,
        enabled=env_cfg.enabled,
        allow_network=allow_network,
        endpoint=env_cfg.endpoint,
        model=env_cfg.model,
        api_key_env=env_cfg.api_key_env,
        timeout_seconds=env_cfg.timeout_seconds,
        max_input_chars=env_cfg.max_input_chars,
        temperature=env_cfg.temperature,
        response_format=env_cfg.response_format,
    )
    result = LLMSkillClient(client_cfg).invoke(
        skill.name,
        system_prompt=skill.system_prompt,
        user_text=json.dumps(
            {
                "as_of_date": str(date.date()),
                "evidence_summary_rows": _summarize_evidence_for_llm(canonical),
                "capital_flow_thesis_rows": _records_for_llm(theses, limit=80),
                "deterministic_sector_pool": _records_for_llm(sector_pool, limit=sector_top_n),
                "candidate_rows": _records_for_llm(candidates, limit=int(candidate_pool_size)),
                "contract": {
                    "first_build_sector_pool_from_policy_bond_broker_state_team_flows": True,
                    "then_score_stock_pool_with_fundamental_rank_and_quant_factors": True,
                    "keep_full_ranking_pool_not_top30": True,
                    "no_live_orders": True,
                    "t_plus_1_inventory_for_do_t": True,
                    "avoid_old_dealer": True,
                    "target": "maximize equal-weight all-A excess return by bull/neutral/bear regime after strict backtest",
                    "position_hint": {
                        "capital": capital,
                        "normal_gross_ceiling": max_base_gross,
                        "high_confidence_gross_ceiling": max_high_conf_gross,
                        "minimum_cash_reserve_for_do_t": cash_reserve_min,
                    },
                },
            },
            ensure_ascii=False,
            default=str,
        ),
        fallback=fallback,
    )
    if require_llm and result.used_fallback:
        typer.echo(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "llm_required_but_fallback_used",
                    "fallback_reason": result.fallback_reason,
                    "provider": client_cfg.provider,
                    "model": client_cfg.model,
                    "api_key_env": client_cfg.api_key_env,
                    "hint": "Set QUANTAGENT_LLM_PROVIDER=gemini, QUANTAGENT_LLM_ENABLED=1, QUANTAGENT_LLM_ALLOW_NETWORK=1, QUANTAGENT_LLM_MODEL, and GOOGLE_API_KEY/google_API_KEY in the runtime environment.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            err=True,
        )
        raise typer.Exit(code=2)

    analysis = result.output if result.output else fallback
    stock_pass_result = None
    if not _analysis_has_stock_pool(analysis):
        stock_skill = get_skill("stock_selection_analyst")
        stock_pass_result = LLMSkillClient(client_cfg).invoke(
            stock_skill.name,
            system_prompt=stock_skill.system_prompt,
            user_text=json.dumps(
                {
                    "as_of_date": str(date.date()),
                    "candidate_rows": _records_for_llm(candidates, limit=int(candidate_pool_size)),
                    "capital_flow_context": analysis,
                    "contract": {
                        "must_return_all_candidates_as_ranking": True,
                        "candidate_count": int(len(candidates)),
                        "preserve_model_rank": True,
                        "output_candidates_field": True,
                        "no_live_orders": True,
                        "avoid_old_dealer": True,
                        "score_policy_sentiment_fundamental_sector_factor_dot_confluence": True,
                    },
                },
                ensure_ascii=False,
                default=str,
            ),
            fallback={"candidates": []},
        )
        if require_llm and stock_pass_result.used_fallback:
            typer.echo(
                json.dumps(
                    {
                        "status": "failed",
                        "reason": "llm_stock_ranking_required_but_fallback_used",
                        "fallback_reason": stock_pass_result.fallback_reason,
                        "provider": client_cfg.provider,
                        "model": client_cfg.model,
                        "api_key_env": client_cfg.api_key_env,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                err=True,
            )
            raise typer.Exit(code=2)
        stock_output = stock_pass_result.output if isinstance(stock_pass_result.output, dict) else {"candidates": []}
        extra_stock_passes = []
        missing_symbols = _missing_stock_selection_symbols(stock_output, candidates)
        if missing_symbols:
            missing_frame = candidates[candidates["symbol"].astype(str).isin(missing_symbols)].copy()
            for start in range(0, len(missing_frame), 20):
                chunk = missing_frame.iloc[start: start + 20]
                chunk_result = LLMSkillClient(client_cfg).invoke(
                    stock_skill.name,
                    system_prompt=stock_skill.system_prompt,
                    user_text=json.dumps(
                        {
                            "as_of_date": str(date.date()),
                            "candidate_rows": _records_for_llm(chunk, limit=len(chunk)),
                            "capital_flow_context": analysis,
                            "contract": {
                                "must_return_every_candidate_in_this_chunk": True,
                                "candidate_count": int(len(chunk)),
                                "preserve_model_rank": True,
                                "output_candidates_field": True,
                                "no_live_orders": True,
                                "avoid_old_dealer": True,
                            },
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                    fallback={"candidates": []},
                )
                extra_stock_passes.append(chunk_result)
                if require_llm and chunk_result.used_fallback:
                    typer.echo(
                        json.dumps(
                            {
                                "status": "failed",
                                "reason": "llm_stock_ranking_chunk_required_but_fallback_used",
                                "fallback_reason": chunk_result.fallback_reason,
                                "provider": client_cfg.provider,
                                "model": client_cfg.model,
                                "api_key_env": client_cfg.api_key_env,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        err=True,
                    )
                    raise typer.Exit(code=2)
                if isinstance(chunk_result.output, dict):
                    stock_output.setdefault("candidates", [])
                    stock_output["candidates"].extend(chunk_result.output.get("candidates", []) or [])
        stock_output["chunk_passes"] = len(extra_stock_passes)
        analysis = _merge_stock_selection_pass_into_hybrid_analysis(
            capital_flow_analysis=analysis,
            stock_selection_analysis=stock_output,
            candidates=candidates,
        )
    llm_stock_pool = _llm_stock_pool_from_analysis(candidates, analysis)
    hybrid_pool = _final_hybrid_stock_pool(
        candidates,
        llm_stock_pool,
        stock_top_n=stock_top_n,
        capital=capital,
        max_base_gross=max_base_gross,
        max_high_conf_gross=max_high_conf_gross,
        cash_reserve_min=cash_reserve_min,
    )
    agent_scores = hybrid_pool[[
        "trade_date", "symbol", "model_rank", "hybrid_rank",
        "hybrid_score", "llm_stock_score", "old_dealer_risk_score",
        "do_t_suitability_score",
    ]].rename(
        columns={
            "hybrid_rank": "agent_rank",
            "hybrid_score": "agent_stock_score",
            "llm_stock_score": "agent_conviction_score",
            "old_dealer_risk_score": "agent_old_dealer_risk",
            "do_t_suitability_score": "agent_do_t_suitability",
        }
    )
    agent_scores["agent_action_bucket"] = hybrid_pool["action_bucket"].to_numpy()

    paths = {
        "sector_pool": output_dir / "sector_pool.parquet",
        "llm_stock_pool": output_dir / "llm_stock_pool.parquet",
        "hybrid_stock_pool": output_dir / "hybrid_stock_pool.parquet",
        "agent_scores": output_dir / "agent_scores.parquet",
        "analysis": output_dir / "capital_flow_stock_pool_analysis.json",
        "summary": output_dir / "summary.json",
    }
    sector_pool.to_parquet(paths["sector_pool"], index=False)
    llm_stock_pool.to_parquet(paths["llm_stock_pool"], index=False)
    hybrid_pool.to_parquet(paths["hybrid_stock_pool"], index=False)
    agent_scores.to_parquet(paths["agent_scores"], index=False)
    paths["analysis"].write_text(
        json.dumps(
            {
                "as_of_date": str(date.date()),
                "used_fallback": result.used_fallback,
                "fallback_reason": result.fallback_reason,
                "provider": client_cfg.provider,
                "model": client_cfg.model,
                "api_key_env": client_cfg.api_key_env,
                "stock_selection_second_pass_used": stock_pass_result is not None,
                "stock_selection_second_pass_fallback": stock_pass_result.used_fallback if stock_pass_result else None,
                "analysis": analysis,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    summary = {
        "as_of_date": str(date.date()),
        "used_fallback": result.used_fallback,
        "fallback_reason": result.fallback_reason,
        "candidate_rows": int(len(candidates)),
        "sector_rows": int(len(sector_pool)),
        "final_stock_rows": int(len(hybrid_pool)),
        "top_symbols": hybrid_pool["symbol"].head(20).tolist(),
        "paths": {k: str(v) for k, v in paths.items()},
        "position_hint": {
            "capital": capital,
            "normal_gross_ceiling": max_base_gross,
            "high_confidence_gross_ceiling": max_high_conf_gross,
            "minimum_cash_reserve_for_do_t": cash_reserve_min,
            "no_orders_generated": True,
        },
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return output_dir


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _prediction_score_column(frame: pd.DataFrame) -> str:
    for col in ("prediction", "composite_score", "alpha_score", "score"):
        if col in frame.columns:
            return col
    raise typer.BadParameter("predictions must include prediction, composite_score, alpha_score, or score")


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema.names)
    available = [c for c in columns if c in names]
    if not available:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=available)


def _attach_core_and_sector_for_hybrid(
    ranked: pd.DataFrame,
    core_dataset_path: Path,
    sector_map: pd.DataFrame | None,
    date: pd.Timestamp,
) -> pd.DataFrame:
    core_cols = [
        "trade_date", "symbol",
        "core_policy_score", "core_sentiment_score", "fundamental_quality_score",
        "cicc_stock_selection_score", "cicc_sector_selection_score",
        "cicc_aggressive_momentum_score", "cicc_defensive_quality_score",
        "cicc_liquidity_defense_score", "sector_resonance_score", "dip_buy_flow_score",
        "old_dealer_risk_score", "old_dealer_block", "trend_strength_score",
        "net_buy_pressure", "vwap_deviation", "intraday_range_pos",
        "volume_concentration", "roe", "revenue_yoy", "net_income_yoy",
        "gross_margin", "operating_cash_to_revenue", "debt_to_asset_ratio",
        "pe_ttm", "pb", "ps_ttm",
    ]
    core = _read_parquet_columns(core_dataset_path, core_cols)
    if core.empty:
        out = ranked.copy()
    else:
        core["trade_date"] = pd.to_datetime(core["trade_date"], errors="coerce")
        core["symbol"] = core["symbol"].astype(str)
        out = ranked.merge(core[core["trade_date"] == date], on=["trade_date", "symbol"], how="left")
    if sector_map is not None and not sector_map.empty and "sector_level_1" in sector_map.columns:
        sm = sector_map[["symbol", "sector_level_1"]].drop_duplicates("symbol").copy()
        sm["symbol"] = sm["symbol"].astype(str)
        out = out.merge(sm, on="symbol", how="left")
    if "sector_level_1" not in out.columns:
        out["sector_level_1"] = "unknown"
    out["sector_level_1"] = out["sector_level_1"].fillna("unknown").astype(str)
    return out


def _load_hybrid_canonical_evidence(
    *,
    canonical_evidence_path: Path | None,
    policy_events: Path | None,
    bond_flows: Path | None,
    broker_reports: Path | None,
    state_team: Path | None,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if canonical_evidence_path is not None:
        canonical = _read_table(canonical_evidence_path)
    else:
        canonical = to_canonical_evidence_frame(
            policy_events=_read_silver(policy_events),
            bond_flows=_read_silver(bond_flows),
            broker_reports=_read_silver(broker_reports),
            state_team_events=_read_silver(state_team),
        )
    if canonical.empty:
        return canonical
    canonical = canonical.copy()
    canonical["available_at"] = pd.to_datetime(canonical["available_at"], errors="coerce")
    cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)
    return canonical[canonical["available_at"] < cutoff].reset_index(drop=True)


def _read_or_build_hybrid_theses(path: Path | None, canonical: pd.DataFrame) -> pd.DataFrame:
    if path is not None and path.exists():
        return _read_table(path)
    if canonical is None or canonical.empty:
        return pd.DataFrame(columns=["direction_kind", "direction_value", "thesis_sign", "confidence"])
    builder = CapitalFlowThesisBuilder(
        CapitalFlowThesisConfig(min_supporting=1, min_aggregate_confidence=0.15)
    )
    return builder.build_frame(canonical)


def _sector_pool_from_theses(theses: pd.DataFrame, *, sector_top_n: int) -> pd.DataFrame:
    if theses is None or theses.empty:
        return pd.DataFrame(
            columns=[
                "direction_kind", "direction_value", "sector_level_1", "theme",
                "sector_pool_score", "sector_pool_signed_score", "confidence",
                "expected_lag_days", "n_supporting", "validation_status",
                "supporting_evidence_ids",
            ]
        )
    frame = theses.copy()
    frame = frame[frame["direction_kind"].isin(["sector", "theme", "province", "symbol"])].copy()
    if frame.empty:
        return _sector_pool_from_theses(pd.DataFrame(), sector_top_n=sector_top_n)
    for col in ("thesis_sign", "confidence", "contradiction_score", "tradability_score", "decay_score"):
        if col not in frame.columns:
            frame[col] = 0.0 if col in {"thesis_sign", "contradiction_score"} else 1.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    strength = frame["confidence"].clip(0, 1) * (1.0 - frame["contradiction_score"].clip(0, 1))
    strength = strength * frame["tradability_score"].replace(0, 1.0).clip(0, 1) * frame["decay_score"].replace(0, 1.0).clip(0, 1)
    frame["sector_pool_signed_score"] = frame["thesis_sign"].clip(-1, 1) * strength
    frame["sector_pool_score"] = frame["sector_pool_signed_score"].clip(lower=0.0)
    frame["sector_level_1"] = frame.apply(
        lambda r: str(r["direction_value"]) if str(r.get("direction_kind")) == "sector" else None,
        axis=1,
    )
    frame["theme"] = frame.apply(
        lambda r: str(r["direction_value"]) if str(r.get("direction_kind")) != "sector" else None,
        axis=1,
    )
    keep = [
        "direction_kind", "direction_value", "sector_level_1", "theme",
        "sector_pool_score", "sector_pool_signed_score", "confidence",
        "expected_lag_days", "n_supporting", "validation_status",
        "supporting_evidence_ids",
    ]
    for col in keep:
        if col not in frame.columns:
            frame[col] = None
    return frame.sort_values("sector_pool_signed_score", ascending=False)[keep].head(int(sector_top_n)).reset_index(drop=True)


def _merge_hybrid_sector_scores(candidates: pd.DataFrame, sector_pool: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if sector_pool is None or sector_pool.empty or "sector_level_1" not in sector_pool.columns:
        out["sector_pool_score"] = 0.0
        out["sector_pool_signed_score"] = 0.0
        out["sector_policy_confidence"] = 0.0
        return out
    sector_scores = (
        sector_pool.dropna(subset=["sector_level_1"])
        .groupby("sector_level_1", as_index=False)[["sector_pool_score", "sector_pool_signed_score", "confidence"]]
        .max()
        .rename(columns={"confidence": "sector_policy_confidence"})
    )
    out = out.merge(sector_scores, on="sector_level_1", how="left")
    for col in ("sector_pool_score", "sector_pool_signed_score", "sector_policy_confidence"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _add_hybrid_pre_llm_scores(candidates: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = candidates.copy()
    n = max(1, len(out))
    out["model_rank_score"] = 1.0 - ((pd.to_numeric(out["model_rank"], errors="coerce").fillna(n) - 1.0) / max(1.0, n - 1.0))
    out["factor_prior_score"] = 0.0
    for col, weight in weights.items():
        values = pd.to_numeric(out[col], errors="coerce").fillna(0.0) if col in out.columns else 0.0
        out["factor_prior_score"] = out["factor_prior_score"] + float(weight) * values
    out["factor_prior_score"] = out["factor_prior_score"].clip(-1.0, 1.0)
    out["factor_rank_score"] = (0.65 * out["model_rank_score"] + 0.35 * (out["factor_prior_score"] + 1.0) / 2.0).clip(0, 1)
    out["fundamental_rank_score"] = _fundamental_rank_for_hybrid(out)
    out["do_t_suitability_score"] = _do_t_score_for_hybrid(out)
    if "old_dealer_risk_score" in out.columns:
        out["old_dealer_risk_score"] = pd.to_numeric(out["old_dealer_risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        out["old_dealer_risk_score"] = 0.0
    out["pre_llm_score"] = (
        0.42 * out["factor_rank_score"]
        + 0.20 * out["sector_pool_score"].clip(0, 1)
        + 0.18 * out["fundamental_rank_score"]
        + 0.10 * out["do_t_suitability_score"]
        + 0.10 * (0.5 + out["sector_pool_signed_score"].clip(-0.5, 0.5))
        - 0.20 * out["old_dealer_risk_score"]
    ).clip(0, 1)
    return out


def _fundamental_rank_for_hybrid(frame: pd.DataFrame) -> pd.Series:
    if "fundamental_quality_score" in frame.columns:
        base = pd.to_numeric(frame["fundamental_quality_score"], errors="coerce").fillna(0.0)
        return (base.rank(pct=True) if base.nunique(dropna=True) > 1 else (base + 0.5)).clip(0, 1)
    parts = []
    for col in ("roe", "revenue_yoy", "net_income_yoy", "gross_margin", "operating_cash_to_revenue"):
        if col in frame.columns:
            parts.append(pd.to_numeric(frame[col], errors="coerce").rank(pct=True))
    for col in ("debt_to_asset_ratio", "pe_ttm", "pb", "ps_ttm"):
        if col in frame.columns:
            parts.append(1.0 - pd.to_numeric(frame[col], errors="coerce").rank(pct=True))
    if not parts:
        return pd.Series(0.5, index=frame.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.5).clip(0, 1)


def _do_t_score_for_hybrid(frame: pd.DataFrame) -> pd.Series:
    parts = []
    if "dip_buy_flow_score" in frame.columns:
        parts.append((pd.to_numeric(frame["dip_buy_flow_score"], errors="coerce").fillna(0.0) + 0.5).clip(0, 1))
    if "vwap_deviation" in frame.columns:
        dev = -pd.to_numeric(frame["vwap_deviation"], errors="coerce").fillna(0.0)
        parts.append(dev.rank(pct=True).fillna(0.5))
    if "intraday_range_pos" in frame.columns:
        parts.append((1.0 - pd.to_numeric(frame["intraday_range_pos"], errors="coerce").fillna(0.5)).clip(0, 1))
    if not parts:
        return pd.Series(0.5, index=frame.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.5).clip(0, 1)


def _fallback_capital_flow_stock_pool(
    candidates: pd.DataFrame,
    sector_pool: pd.DataFrame,
    date: pd.Timestamp,
    stock_top_n: int,
) -> dict[str, Any]:
    stocks = []
    ranked = candidates.sort_values("pre_llm_score", ascending=False).head(int(stock_top_n))
    for _, row in ranked.iterrows():
        score = float(row.get("pre_llm_score", 0.5))
        stocks.append(
            {
                "symbol": str(row["symbol"]),
                "sector_level_1": str(row.get("sector_level_1", "unknown")),
                "llm_stock_score": round(score, 6),
                "confidence": round(max(0.10, min(0.85, 0.45 + 0.40 * score)), 6),
                "horizon_bucket": "short" if float(row.get("do_t_suitability_score", 0.5)) >= 0.65 else "mid",
                "key_positive_factors": _hybrid_positive_factor_names(row),
                "key_risks": ["old_dealer_risk"] if float(row.get("old_dealer_risk_score", 0.0)) >= 0.55 else [],
                "rationale": "deterministic fallback from policy/flow thesis, factor rank, fundamentals, and Do-T proxies",
            }
        )
    return {
        "summary": f"deterministic fallback hybrid pool for {date.date()}; no real LLM call used",
        "capital_flow_thesis": [],
        "sector_pool": _records_for_llm(sector_pool, limit=50),
        "stock_pool": stocks,
        "risk_flags": ["llm_not_used"],
        "data_gaps": ["configure QUANTAGENT_LLM_* and --allow-network for true Gemma/Gemini analysis"],
    }


def _hybrid_positive_factor_names(row: pd.Series) -> list[str]:
    names = []
    for col in (
        "core_policy_score", "core_sentiment_score", "sector_pool_score",
        "fundamental_quality_score", "cicc_stock_selection_score",
        "sector_resonance_score", "dip_buy_flow_score", "trend_strength_score",
    ):
        try:
            if float(row.get(col, 0.0)) > 0.10:
                names.append(col)
        except (TypeError, ValueError):
            continue
    return names[:6]


def _summarize_evidence_for_llm(canonical: pd.DataFrame, limit: int = 120) -> list[dict[str, Any]]:
    if canonical is None or canonical.empty:
        return []
    keep = [
        "evidence_id", "source_name", "source_type", "available_at", "entity_type",
        "entities", "extracted_claims", "sentiment_score", "policy_direction_score",
        "capital_flow_direction_score", "confidence", "contradiction_score",
    ]
    frame = canonical[[c for c in keep if c in canonical.columns]].copy()
    score = pd.Series(0.0, index=frame.index)
    for col in ("confidence", "policy_direction_score", "capital_flow_direction_score", "sentiment_score"):
        if col in frame.columns:
            score = score + pd.to_numeric(frame[col], errors="coerce").fillna(0.0).abs()
    frame["_importance"] = score
    return _records_for_llm(frame.sort_values("_importance", ascending=False).drop(columns=["_importance"]), limit=limit)


def _records_for_llm(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    clean = frame.head(int(limit)).copy()
    clean = clean.replace({pd.NA: None})
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict("records")


def _analysis_has_stock_pool(analysis: dict[str, Any]) -> bool:
    if not isinstance(analysis, dict):
        return False
    stock_pool = analysis.get("stock_pool")
    if not isinstance(stock_pool, list) or not stock_pool:
        return False
    return any(isinstance(item, dict) and item.get("symbol") for item in stock_pool)


def _merge_stock_selection_pass_into_hybrid_analysis(
    *,
    capital_flow_analysis: dict[str, Any],
    stock_selection_analysis: dict[str, Any],
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Convert stock-selection second pass into the hybrid stock_pool schema."""
    base = dict(capital_flow_analysis) if isinstance(capital_flow_analysis, dict) else {}
    items = stock_selection_analysis.get("candidates", []) if isinstance(stock_selection_analysis, dict) else []
    sector_by_symbol = candidates.set_index("symbol")["sector_level_1"].astype(str).to_dict() if "sector_level_1" in candidates.columns else {}
    pool = []
    for item in items:
        if not isinstance(item, dict) or not item.get("symbol"):
            continue
        symbol = str(item["symbol"])
        score = item.get("agent_score", item.get("llm_stock_score", 50.0))
        conviction = item.get("conviction", item.get("confidence", _score_to_unit(score)))
        pool.append(
            {
                "symbol": symbol,
                "sector_level_1": str(item.get("sector_level_1") or sector_by_symbol.get(symbol, "unknown")),
                "llm_stock_score": _score_to_unit(score),
                "confidence": _score_to_unit(conviction),
                "horizon_bucket": _horizon_from_action_bucket(str(item.get("action_bucket", ""))),
                "key_positive_factors": item.get("key_positive_factors", []),
                "key_risks": item.get("key_risks", []),
                "rationale": str(item.get("rationale", "")),
            }
        )
    by_symbol = {row["symbol"]: row for row in pool}
    for _, row in candidates.iterrows():
        symbol = str(row["symbol"])
        if symbol in by_symbol:
            continue
        pre = float(row.get("pre_llm_score", 0.5) or 0.5)
        by_symbol[symbol] = {
            "symbol": symbol,
            "sector_level_1": str(row.get("sector_level_1", "unknown")),
            "llm_stock_score": pre,
            "confidence": max(0.10, min(0.70, 0.35 + 0.35 * pre)),
            "horizon_bucket": "mid",
            "key_positive_factors": _hybrid_positive_factor_names(row),
            "key_risks": ["old_dealer_risk"] if float(row.get("old_dealer_risk_score", 0.0)) >= 0.55 else [],
            "rationale": "filled from deterministic candidate score because second-pass LLM omitted this symbol",
        }
    ordered_symbols = candidates["symbol"].astype(str).tolist()
    base["stock_pool"] = [by_symbol[symbol] for symbol in ordered_symbols if symbol in by_symbol]
    base["stock_selection_second_pass"] = {
        "used": True,
        "summary": stock_selection_analysis.get("summary", "") if isinstance(stock_selection_analysis, dict) else "",
        "risk_flags": stock_selection_analysis.get("risk_flags", []) if isinstance(stock_selection_analysis, dict) else [],
        "next_research_steps": stock_selection_analysis.get("next_research_steps", []) if isinstance(stock_selection_analysis, dict) else [],
        "chunk_passes": int(stock_selection_analysis.get("chunk_passes", 0)) if isinstance(stock_selection_analysis, dict) else 0,
    }
    return base


def _missing_stock_selection_symbols(stock_selection_analysis: dict[str, Any], candidates: pd.DataFrame) -> list[str]:
    requested = candidates["symbol"].astype(str).tolist() if "symbol" in candidates.columns else []
    items = stock_selection_analysis.get("candidates", []) if isinstance(stock_selection_analysis, dict) else []
    returned = {
        str(item.get("symbol"))
        for item in items
        if isinstance(item, dict) and item.get("symbol")
    }
    return [symbol for symbol in requested if symbol not in returned]


def _horizon_from_action_bucket(action_bucket: str) -> str:
    if "do_t" in action_bucket:
        return "short"
    if "short" in action_bucket:
        return "short"
    return "mid"


def _llm_stock_pool_from_analysis(candidates: pd.DataFrame, analysis: dict[str, Any]) -> pd.DataFrame:
    stock_items = analysis.get("stock_pool", []) if isinstance(analysis, dict) else []
    by_symbol = {}
    for item in stock_items:
        if isinstance(item, dict) and item.get("symbol"):
            by_symbol[str(item["symbol"])] = item
    rows = []
    for _, row in candidates.iterrows():
        item = by_symbol.get(str(row["symbol"]), {})
        score = item.get("llm_stock_score", item.get("agent_score", row.get("pre_llm_score", 0.5)))
        confidence = item.get("confidence", max(0.1, min(0.9, float(row.get("pre_llm_score", 0.5)))))
        rows.append(
            {
                "trade_date": row["trade_date"],
                "symbol": str(row["symbol"]),
                "model_rank": int(row.get("model_rank", len(rows) + 1)),
                "sector_level_1": str(row.get("sector_level_1", "unknown")),
                "llm_stock_score": _score_to_unit(score),
                "llm_confidence": _score_to_unit(confidence),
                "llm_horizon_bucket": str(item.get("horizon_bucket", "mid")),
                "llm_key_positive_factors": item.get("key_positive_factors", []),
                "llm_key_risks": item.get("key_risks", []),
                "llm_rationale": str(item.get("rationale", "")),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["llm_stock_score", "model_rank"], ascending=[False, True]).reset_index(drop=True)
        out["llm_rank"] = out.index + 1
    return out


def _score_to_unit(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.5
    if raw > 1.0:
        raw = raw / 100.0
    return float(max(0.0, min(1.0, raw)))


def _final_hybrid_stock_pool(
    candidates: pd.DataFrame,
    llm_stock_pool: pd.DataFrame,
    *,
    stock_top_n: int,
    capital: float,
    max_base_gross: float,
    max_high_conf_gross: float,
    cash_reserve_min: float,
) -> pd.DataFrame:
    merged = candidates.merge(
        llm_stock_pool[["trade_date", "symbol", "llm_stock_score", "llm_confidence", "llm_rank", "llm_horizon_bucket"]],
        on=["trade_date", "symbol"],
        how="left",
    )
    merged["llm_stock_score"] = pd.to_numeric(merged["llm_stock_score"], errors="coerce").fillna(merged["pre_llm_score"]).clip(0, 1)
    merged["llm_confidence"] = pd.to_numeric(merged["llm_confidence"], errors="coerce").fillna(0.5).clip(0, 1)
    merged["sector_pool_signed_score"] = pd.to_numeric(merged["sector_pool_signed_score"], errors="coerce").fillna(0.0).clip(-1, 1)
    merged["hybrid_score"] = (
        0.34 * merged["factor_rank_score"]
        + 0.25 * merged["llm_stock_score"]
        + 0.18 * merged["sector_pool_score"].clip(0, 1)
        + 0.14 * merged["fundamental_rank_score"]
        + 0.09 * merged["do_t_suitability_score"]
        - 0.20 * merged["old_dealer_risk_score"]
        + 0.05 * merged["sector_pool_signed_score"].clip(-0.5, 0.5)
    ).clip(0, 1)
    merged["action_bucket"] = "core_watch"
    merged.loc[merged["do_t_suitability_score"] >= 0.68, "action_bucket"] = "do_t_watch"
    merged.loc[(merged["hybrid_score"] >= 0.72) & (merged["llm_confidence"] >= 0.65), "action_bucket"] = "core_watch"
    old_block = (
        pd.to_numeric(merged["old_dealer_block"], errors="coerce").fillna(0.0)
        if "old_dealer_block" in merged.columns
        else pd.Series(0.0, index=merged.index)
    )
    merged.loc[(old_block > 0) | (merged["old_dealer_risk_score"] >= 0.70), "action_bucket"] = "avoid"
    out = merged.sort_values(["hybrid_score", "model_rank"], ascending=[False, True]).head(int(stock_top_n)).reset_index(drop=True)
    out["hybrid_rank"] = out.index + 1
    high_conf = float(out["llm_confidence"].head(min(20, len(out))).mean()) if len(out) else 0.0
    gross_ceiling = max_high_conf_gross if high_conf >= 0.72 and float(out["hybrid_score"].head(20).mean()) >= 0.70 else max_base_gross
    gross_ceiling = min(float(gross_ceiling), 1.0 - float(cash_reserve_min))
    tradable_n = int((out["action_bucket"] != "avoid").sum())
    per_name_weight = gross_ceiling / max(1, tradable_n)
    out["research_weight_hint"] = 0.0
    out.loc[out["action_bucket"] != "avoid", "research_weight_hint"] = min(0.05, per_name_weight)
    out["research_amount_hint"] = out["research_weight_hint"] * float(capital) if capital > 0 else 0.0
    out["no_orders_generated"] = True
    return out
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return output_dataset_path


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
    use_silver_panel: Optional[Path] = typer.Option(
        None,
        help="path to a v7 silver/market_panel.parquet — registered as a router source",
    ),
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

    silver_provider = None
    if use_silver_panel is not None:
        from quantagent.data.providers.silver_panel_provider import SilverPanelProvider
        silver_provider = SilverPanelProvider(panel_path=str(use_silver_panel))

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
        # silver panel sits at the head of the priority chain when present
        priority = ("silver_panel", "qlib", "akshare", "baostock", "tushare") if silver_provider is not None else ("qlib", "akshare", "baostock", "tushare")
        router_config = RouterConfig(daily_priority=priority, allow_mock_fallback=allow_mock_fallback)
        router = build_default_router(
            qlib_provider=qlib_provider,
            akshare_provider=akshare_provider,
            baostock_provider=baostock_provider,
            tushare_provider=tushare_provider,
            config=router_config,
        )
        if silver_provider is not None:
            from quantagent.data.router import RoutedProvider
            router.register(RoutedProvider(
                name="silver_panel", provider=silver_provider,
                is_paid=False, quality_baseline=0.92,
            ))

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
