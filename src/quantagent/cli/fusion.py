"""Governed CLI for the factor-fusion search (ATLAS layer L2).

One command, ``search-factor-fusion``, so the web workstation has exactly one
allowlisted entrypoint into the search. It reads a factor panel and a forward
return panel, runs the purged walk-forward protocol over every enumerated blend
scheme, and writes the candidate set, the Pareto frontier and a hashed manifest.

The command deliberately exposes no ``--n-trials`` flag: the trial count is a
property of the enumerated search space, and letting an operator declare it
would make the deflated Sharpe ratio meaningless. Every enumerated candidate is
appended to the persistent trial ledger (family ``nonlinear_fusion``) and the
DSR uses the cumulative family N read from it, which can only add to the
enumerated count.

The PBO/DSR/SPA/PIT/benchmark/holdout checks emitted here are a research-screening
layer, not a production authorization certificate. The CLI does not own the
Stage-4 ExperimentLedger, one-shot FinalHoldoutLedger or executable economic
certification, so even a passing research screen remains production-ineligible.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
from typing import Any, Optional

import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root
from quantagent.research.foundation_gates import (
    evaluate_research_gates,
    fusion_statistical_evidence,
)
from quantagent.research.trial_ledger import (
    FAMILY_NONLINEAR_FUSION,
    TrialLedger,
    TrialRecord,
    config_hash,
    current_git_hash,
    outcome_records,
)

COMMAND_ID = "search-factor-fusion"


# Factor panels often carry labels beside features for convenience. A typo in
# --factor-names must not silently promote those future values into predictors.
# Keep this deliberately narrow: reject semantic label/target names, not every
# feature containing words such as "return" (momentum/lagged return is valid).
_LEAKAGE_FACTOR_NAME = re.compile(
    r"^(?:forward[_-]?return(?:s)?(?:_|$)|future[_-]?return(?:s)?(?:_|$)|label(?:_|$)|target(?:_|$)|y(?:_|$))",
    re.IGNORECASE,
)


def _read_panel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _resolve_forward_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    candidates = [column for column in frame.columns if column.startswith("forward_return")]
    if not candidates:
        raise typer.BadParameter(
            f"forward panel has no '{requested}' column and no forward_return* fallback; "
            f"available columns: {sorted(frame.columns)[:20]}"
        )
    chosen = sorted(candidates)[0]
    typer.echo(f"forward column '{requested}' not found; using '{chosen}'")
    return chosen


def _validate_factor_names_for_leakage(names: tuple[str, ...]) -> None:
    blocked = sorted(name for name in names if _LEAKAGE_FACTOR_NAME.search(name))
    if blocked:
        raise typer.BadParameter(
            "factor_names contains label/forward-looking columns and is blocked fail-closed: "
            f"{blocked}. Use only features observable at the decision timestamp."
        )


def _fusion_trials(
    result: Any, *, run_id: str, run_config_hash: str, git_hash: str
) -> list[TrialRecord]:
    """One ledger entry per enumerated candidate -- controls included, as they count into N."""
    windows = result.fold_windows
    window = {
        "start": min((str(w["trainStart"]) for w in windows), default=""),
        "end": max((str(w["testEnd"]) for w in windows), default=""),
    }
    preferred = result.preferred.candidate_id if result.preferred is not None else None
    frontier = set(result.frontier)
    records: list[TrialRecord] = []
    for candidate in result.candidates:
        if candidate.candidate_id == preferred:
            verdict = "preferred"
        elif candidate.candidate_id in frontier:
            verdict = "frontier"
        else:
            verdict = "dominated"
        records.append(
            TrialRecord(
                family=FAMILY_NONLINEAR_FUSION,
                trial_id=f"{run_id}:{candidate.candidate_id}",
                command=COMMAND_ID,
                git_hash=git_hash,
                config_hash=run_config_hash,
                window=window,
                metrics={
                    **dict(candidate.metrics),
                    "scheme": candidate.scheme,
                    "isControl": bool(candidate.is_control),
                },
                verdict=verdict,
            )
        )
    return records


def _pit_evidence(frame: pd.DataFrame) -> bool | None:
    if "point_in_time_valid" in frame.columns:
        values = frame["point_in_time_valid"].dropna()
        return bool(len(values) and values.astype(bool).all())
    if {"available_at", "trade_date"}.issubset(frame.columns):
        available = pd.to_datetime(frame["available_at"], errors="coerce")
        decision = pd.to_datetime(frame["trade_date"], errors="coerce")
        valid = available.notna() & decision.notna()
        if not bool(valid.any()):
            return None
        return bool((available.loc[valid] <= decision.loc[valid]).all())
    return None


def _fusion_gate_payload(gate: Any, evidence: dict[str, object]) -> dict[str, object]:
    """Serialize a passing/blocked research screen without granting production.

    ``ResearchGateReport.eligible`` is useful evidence, but this CLI cannot prove
    authoritative Stage-4 lineage or consume a one-shot final holdout. Preserve
    the old ``promotionEligible`` field as a backward-compatible *false* value so
    any legacy consumer also fails closed, while exposing the actual statistical
    screen separately as ``researchPromotionEligible``.
    """
    base = dict(gate.as_dict())
    base["promotionEligible"] = False
    base.update(
        {
            "researchPromotionEligible": bool(gate.eligible),
            "productionEligible": False,
            "stage4Governed": False,
            "researchOnly": True,
            "productionBlockers": [
                "factor-fusion CLI has no Stage-4 ExperimentLedger fingerprint lineage (family N comes from the trial ledger)",
                "--holdout-untouched is a research assertion, not a one-shot FinalHoldoutLedger seal",
                "factor-fusion outcome clock is not bound to the Stage-4 executable-label contract",
                "factor-fusion economics are not certified through the strict position-carrying A-share simulator",
            ],
            "statisticalEvidence": evidence,
            "note": (
                "Pareto preference and PBO/DSR/SPA checks are research screening only. "
                "Production eligibility requires the strict Stage-4 ledger/holdout/label/economic chain."
            ),
        }
    )
    return base


@app.command("search-factor-fusion")
def search_factor_fusion(
    factor_panel_path: Path = typer.Option(..., exists=True, dir_okay=False),
    forward_returns_path: Path = typer.Option(..., exists=True, dir_okay=False),
    factor_names: str = typer.Option(..., help="comma-separated factor column names"),
    output_dir: Path = typer.Option(default_reports_root() / "fusion" / "search"),
    forward_column: str = typer.Option("forward_return"),
    horizon_days: int = typer.Option(5, min=1, max=252),
    top_k: int = typer.Option(30, min=1, max=500),
    n_folds: int = typer.Option(4, min=1, max=12),
    embargo_days: int = typer.Option(5, min=0, max=120),
    min_train_days: int = typer.Option(120, min=20),
    min_test_days: int = typer.Option(40, min=5),
    transaction_cost_bps: float = typer.Option(8.0, min=0.0, max=500.0),
    include_genetic: bool = typer.Option(True),
    random_controls: int = typer.Option(8, min=0, max=64),
    single_factor_baselines: int = typer.Option(6, min=0, max=64),
    seed: int = typer.Option(17),
    benchmark_path: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    benchmark_symbol: str = typer.Option("", help="Explicit benchmark, e.g. 000300.SH. Required for a passing research screen."),
    holdout_untouched: bool = typer.Option(
        False,
        help=(
            "Research assertion that an external holdout was not used in selection. "
            "This flag is not a one-shot Stage-4 holdout certificate and cannot grant production eligibility."
        ),
    ),
    enforce_promotion_gates: bool = typer.Option(
        False,
        help="Exit non-zero when the research PBO/DSR/SPA/PIT/benchmark/holdout screen fails; never grants production eligibility.",
    ),
    preference_excess_return: float = typer.Option(0.40, min=0.0, max=1.0),
    preference_annual_return: float = typer.Option(0.20, min=0.0, max=1.0),
    preference_drawdown_control: float = typer.Option(0.25, min=0.0, max=1.0),
    preference_robustness: float = typer.Option(0.15, min=0.0, max=1.0),
):
    """Search factor blends and audit a research-only promotion screen."""
    from quantagent.fusion import (
        FusionSearchConfig,
        ObjectivePreference,
        run_fusion_search,
        save_fusion_artifacts,
    )

    names = tuple(name.strip() for name in factor_names.split(",") if name.strip())
    if not names:
        raise typer.BadParameter("factor_names resolved to an empty list")
    _validate_factor_names_for_leakage(names)

    factor_panel = _read_panel(factor_panel_path)
    forward_panel = _read_panel(forward_returns_path)
    resolved_column = _resolve_forward_column(forward_panel, forward_column)
    forward_panel = forward_panel.rename(columns={resolved_column: "forward_return"})

    missing = [name for name in names if name not in factor_panel.columns]
    if missing:
        raise typer.BadParameter(f"factor columns missing from the panel: {missing}")

    benchmark_returns = None
    if benchmark_path is not None:
        benchmark_frame = _read_panel(benchmark_path)
        if "trade_date" not in benchmark_frame.columns:
            raise typer.BadParameter("benchmark file must contain a trade_date column")
        value_column = next(
            (column for column in ("benchmark_return", "forward_return", "return") if column in benchmark_frame.columns),
            None,
        )
        if value_column is None:
            raise typer.BadParameter("benchmark file must contain benchmark_return, forward_return or return")
        benchmark_returns = (
            benchmark_frame.assign(trade_date=pd.to_datetime(benchmark_frame["trade_date"]))
            .set_index("trade_date")[value_column]
            .astype(float)
            .sort_index()
        )

    config = FusionSearchConfig(
        factor_names=names,
        horizon_days=horizon_days,
        top_k=top_k,
        n_folds=n_folds,
        embargo_days=embargo_days,
        min_train_days=min_train_days,
        min_test_days=min_test_days,
        transaction_cost_bps=transaction_cost_bps,
        include_genetic=include_genetic,
        random_controls=random_controls,
        single_factor_baselines=single_factor_baselines,
        seed=seed,
        benchmark_symbol=benchmark_symbol.strip().upper(),
        preference=ObjectivePreference(
            excess_return=preference_excess_return,
            annual_return=preference_annual_return,
            drawdown_control=preference_drawdown_control,
            robustness=preference_robustness,
        ),
    )

    def _progress(stage: str, completed: int, total: int) -> None:
        typer.echo(f"[{completed} / {total}] {stage}")

    from quantagent.fusion import build_scheme_specs

    run_config_hash = config_hash(
        {
            "command": COMMAND_ID,
            "factorPanel": str(factor_panel_path.resolve()),
            "forwardReturns": str(forward_returns_path.resolve()),
            "forwardColumn": resolved_column,
            "config": dataclasses.asdict(config),
        }
    )
    git_hash = current_git_hash()
    run_id = f"{run_config_hash[:12]}@{pd.Timestamp.now(tz='UTC').isoformat()}"
    # The enumerated search space is fixed before evaluation, so it is reserved here.
    planned = len(
        build_scheme_specs(
            list(names),
            top_k=config.top_k,
            include_genetic=config.include_genetic,
            random_controls=config.random_controls,
            single_factor_baselines=config.single_factor_baselines,
            seed=config.seed,
        )
    )
    dates = pd.to_datetime(factor_panel["trade_date"], errors="coerce")
    # Reserve the run and read prior N under one lock, before compute: crashed
    # runs still count (GOVERNANCE R6); a damaged ledger fails closed.
    ledger = TrialLedger()
    ledger_summary = ledger.begin_run(
        FAMILY_NONLINEAR_FUSION,
        run_id=run_id,
        command=COMMAND_ID,
        git_hash=git_hash,
        config_hash=run_config_hash,
        window={"panelStart": str(dates.min())[:10], "panelEnd": str(dates.max())[:10]},
        planned_trials=planned,
    )
    prior_family_trials = int(ledger_summary["priorFamilyTrials"])
    try:
        result = run_fusion_search(
            factor_panel=factor_panel,
            forward_panel=forward_panel,
            config=config,
            benchmark_returns=benchmark_returns,
            progress=_progress,
            prior_family_trials=prior_family_trials,
        )
    except BaseException as exc:
        try:
            ledger.record_trial(
                FAMILY_NONLINEAR_FUSION, f"{run_id}:failed", COMMAND_ID, git_hash, run_config_hash,
                {}, {"error": f"{type(exc).__name__}: {exc}"}, "failed", trials=0,
            )
        except Exception as ledger_exc:  # keep the original failure primary
            exc.add_note(f"trial ledger failure record not written: {ledger_exc!r}")
        raise
    outcomes = _fusion_trials(result, run_id=run_id, run_config_hash=run_config_hash, git_hash=git_hash)
    ledger.record_trials(outcome_records(outcomes, planned_trials=planned, run_id=run_id))
    run_trials = max(planned, len(outcomes))
    ledger_summary.update(
        {
            "runTrials": run_trials,
            "cumulativeTrials": prior_family_trials + run_trials,
            "dsrTrials": int(result.dsr_trials),
        }
    )
    paths = save_fusion_artifacts(result, output_dir=output_dir)

    preferred = result.preferred
    evidence = fusion_statistical_evidence(
        candidate_navs={candidate.candidate_id: candidate.nav for candidate in result.candidates},
        preferred_id=preferred.candidate_id if preferred is not None else None,
        benchmark_returns=benchmark_returns,
        periods_per_year=config.periods_per_year,
        n_trials=result.dsr_trials,
    )
    gate = evaluate_research_gates(
        pbo=result.pbo,
        dsr_probability=evidence.get("dsrProbability"),
        spa_p_value=evidence.get("spaPValue"),
        benchmark_symbol=config.benchmark_symbol,
        pit_valid=_pit_evidence(factor_panel),
        holdout_untouched=holdout_untouched,
    )
    gate_payload = _fusion_gate_payload(gate, evidence)
    gate_payload["preferredCandidate"] = evidence.get("preferred")
    gate_payload["trialLedger"] = ledger_summary
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "promotion_gate.json"
    gate_path.write_text(json.dumps(gate_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(
        f"n_trials={result.n_trials} family_trials={result.dsr_trials} "
        f"frontier={len(result.frontier)} "
        f"pbo={result.pbo:.4f} benchmark={result.benchmark_mode}"
    )
    if preferred is not None:
        metrics = preferred.metrics
        typer.echo(
            f"preferred={preferred.candidate_id} "
            f"excess={metrics['excessReturn']:+.4f} "
            f"annual={metrics['annualReturn']:+.4f} "
            f"max_dd={metrics['maxDrawdown']:.4f} "
            f"robustness={metrics['robustness']:.3f}"
        )
    else:
        typer.echo("no candidate produced usable out-of-sample observations")
    typer.echo(
        "research_promotion=" + ("PASS" if gate.eligible else "BLOCKED")
        + " production_eligible=NO"
        + f" dsr={evidence.get('dsrProbability')} spa={evidence.get('spaPValue')}"
    )
    for blocker in gate.blockers:
        typer.echo(f"research_promotion_blocker: {blocker}")
    for blocker in gate_payload["productionBlockers"]:
        typer.echo(f"production_blocker: {blocker}")
    typer.echo(f"wrote {len(paths) + 1} research artifacts to {output_dir}")

    if enforce_promotion_gates and not gate.eligible:
        raise typer.Exit(code=2)
    return output_dir
