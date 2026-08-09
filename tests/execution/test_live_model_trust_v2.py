from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.execution.live_model_trust import evaluate_live_model_trust
from quantagent.execution.live_model_trust_v2 import (
    EXECUTION_TIMING_ASSURANCE,
    PROVENANCE_ASSURANCE,
    REQUIRED_ARTIFACT_ROLES,
    REQUIRED_METRIC_SEMANTICS,
    REQUIRED_TRAINER_OBJECTIVE_SEMANTICS,
    sha256_file,
)
from quantagent.execution.live_model_trust_v2_policy import issue_governed_live_model_trust_v2
from quantagent.execution.trusted_backtest_semantics import (
    TRUSTED_EXECUTION_SEMANTICS,
    trusted_cost_model_config,
    trusted_simulation_config,
)
from quantagent.research.selection_governance import evaluate_frozen_candidate
from quantagent.training.semantics import FT_TRANSFORMER_OBJECTIVE_SEMANTICS_VERSION


MODEL_ID = "ft-fresh-2026-v2"
SOURCE_COMMIT = "a" * 40
WINDOW_ID = "fresh-2026-02-02_2026-07-31"
FAMILY = ["linear_control", "ft_transformer"]
SELECTED = "ft_transformer"
ISSUED_AT = datetime(2026, 8, 2, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _common() -> dict[str, object]:
    return {"schema_version": 1, "model_id": MODEL_ID, "source_commit": SOURCE_COMMIT}


def _trials(count: int = 50) -> list[dict[str, str]]:
    return [
        {"trial_id": f"trial-{idx:03d}", "candidate": FAMILY[idx % len(FAMILY)]}
        for idx in range(count)
    ]


def _write_selection_matrix(path: Path):
    dates = pd.bdate_range(end="2026-01-30", periods=160)
    rng = np.random.default_rng(42)
    shared = rng.normal(0.0, 0.002, len(dates))
    frame = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "benchmark": np.zeros(len(dates)),
            "linear_control": 0.00050 + shared,
            "ft_transformer": 0.00055 + shared + rng.normal(0.0, 0.0001, len(dates)),
        }
    )
    frame.to_csv(path, index=False)
    candidates = frame.set_index(pd.DatetimeIndex(dates))[FAMILY]
    benchmark = pd.Series(frame["benchmark"].to_numpy(dtype=float), index=dates)
    report = evaluate_frozen_candidate(
        candidates,
        selected_candidate=SELECTED,
        benchmark_returns=benchmark,
        cumulative_trials=50,
        minimum_observed_days=80,
    )
    assert report.accepted is True, report.rejection_reasons
    return report


def _write_strict_returns(path: Path) -> tuple[float, float]:
    dates = pd.bdate_range(end="2026-01-30", periods=120)
    portfolio = np.full(len(dates), 0.00060)
    benchmark = np.full(len(dates), 0.00020)
    pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "portfolio_return": portfolio,
            "benchmark_return": benchmark,
        }
    ).to_csv(path, index=False)
    return (
        float(np.prod(1.0 + portfolio) - 1.0),
        float(np.prod(1.0 + benchmark) - 1.0),
    )


def _write_fresh_predictions(path: Path) -> tuple[int, str, str]:
    dates = pd.bdate_range("2026-02-02", periods=130)
    rows: list[dict[str, object]] = []
    for idx, day in enumerate(dates):
        rows.extend(
            [
                {"trade_date": day.date().isoformat(), "symbol": "600000.SH", "prediction": 0.01 + idx * 1e-6},
                {"trade_date": day.date().isoformat(), "symbol": "000001.SZ", "prediction": 0.02 + idx * 1e-6},
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(dates), dates[0].date().isoformat(), dates[-1].date().isoformat()


def _build_evidence(tmp_path: Path):
    root = tmp_path / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    files = {role: root / f"{role}.json" for role in REQUIRED_ARTIFACT_ROLES}
    files["model_checkpoint"] = root / "model_checkpoint.pt"
    files["strict_backtest_returns"] = root / "strict_backtest_returns.csv"
    files["statistical_returns"] = root / "statistical_returns.csv"
    files["fresh_oos_predictions"] = root / "fresh_oos_predictions.csv"

    files["model_checkpoint"].write_bytes(b"governed-ft-checkpoint-v2\x00\x01")
    stat_report = _write_selection_matrix(files["statistical_returns"])
    portfolio_total, benchmark_total = _write_strict_returns(files["strict_backtest_returns"])
    fresh_days, fresh_start, fresh_end = _write_fresh_predictions(files["fresh_oos_predictions"])

    checkpoint_sha = sha256_file(files["model_checkpoint"])
    strict_returns_sha = sha256_file(files["strict_backtest_returns"])
    statistical_sha = sha256_file(files["statistical_returns"])
    predictions_sha = sha256_file(files["fresh_oos_predictions"])
    assert strict_returns_sha != statistical_sha

    _write_json(files["trainer_manifest"], _common() | {
        "objective_semantics": REQUIRED_TRAINER_OBJECTIVE_SEMANTICS,
        "checkpoint_sha256": checkpoint_sha,
        "training_cutoff": "2025-12-31",
        "validation_cutoff": "2026-01-30",
    })
    _write_json(files["pre_registration"], _common() | {
        "selection_pre_registered": True,
        "registered_at": "2026-01-15T00:00:00+00:00",
        "candidate_family": FAMILY,
        "max_search_trials": 50,
        "acceptance_window_id": WINDOW_ID,
        "acceptance_start_date": fresh_start,
        "acceptance_end_date": fresh_end,
    })
    _write_json(files["search_ledger"], _common() | {
        "trial_count": 50,
        "candidate_family": FAMILY,
        "selected_candidate": SELECTED,
        "trials": _trials(),
        "completed_at": "2026-01-20T00:00:00+00:00",
        "final_holdout_used_for_selection": False,
        "selection_frozen_before_fresh_oos": True,
    })
    _write_json(files["data_lineage"], _common() | {
        "pit": True,
        "universe_pit": True,
        "training_cutoff": "2025-12-31",
        "validation_cutoff": "2026-01-30",
    })
    _write_json(files["strict_backtest"], _common() | {
        "metric_semantics": REQUIRED_METRIC_SEMANTICS,
        "execution_semantics": TRUSTED_EXECUTION_SEMANTICS,
        "execution_timing_assurance": EXECUTION_TIMING_ASSURANCE,
        "costs_included": True,
        "simulation_config": trusted_simulation_config(),
        "cost_model_config": trusted_cost_model_config(),
        "variant_c_passed": True,
        "checkpoint_sha256": checkpoint_sha,
        "daily_returns_sha256": strict_returns_sha,
        "portfolio_total_return": portfolio_total,
        "benchmark_total_return": benchmark_total,
        "benchmark_excess_positive": portfolio_total > benchmark_total,
    })
    _write_json(files["statistical_gates"], _common() | {
        "selected_candidate": SELECTED,
        "pbo": float(stat_report.pbo),
        "dsr_probability": float(stat_report.dsr_probability),
        "spa_p_value": float(stat_report.spa_pvalue),
        "trial_count": 50,
        "candidate_family": FAMILY,
        "returns_sha256": statistical_sha,
    })
    _write_json(files["fresh_oos"], _common() | {
        "trading_days": fresh_days,
        "final_holdout_reads": 1,
        "contaminated_holdout": False,
        "start_date": fresh_start,
        "end_date": fresh_end,
        "acceptance_window_id": WINDOW_ID,
        "predictions_sha256": predictions_sha,
    })
    _write_json(files["risk_capacity"], _common() | {
        "passed": True,
        "checkpoint_sha256": checkpoint_sha,
        "predictions_sha256": predictions_sha,
        "strict_returns_sha256": strict_returns_sha,
    })

    locations = {role: ("bundle", files[role].relative_to(root).as_posix()) for role in REQUIRED_ARTIFACT_ROLES}
    return root, files, locations, {"bundle": root}


def _issue(tmp_path: Path):
    root, files, locations, roots = _build_evidence(tmp_path)
    manifest = tmp_path / "live_model_trust_v2.json"
    result = issue_governed_live_model_trust_v2(
        manifest,
        model_id=MODEL_ID,
        source_commit=SOURCE_COMMIT,
        artifact_locations=locations,
        artifact_roots=roots,
        issued_at=ISSUED_AT,
    )
    return manifest, root, files, locations, roots, result


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_and_rebind(manifest: Path, role: str, artifact: Path, mutate) -> None:
    doc = _read(artifact)
    mutate(doc)
    _write_json(artifact, doc)
    payload = _read(manifest)
    payload["artifacts"][role]["sha256"] = sha256_file(artifact)
    _write_json(manifest, payload)


def test_valid_bundle_uses_separate_return_evidence(tmp_path: Path) -> None:
    manifest, _, files, _, roots, issued = _issue(tmp_path)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert issued.verification.ok is True and report.ok is True
    assert report.evidence["provenance_assurance"] == PROVENANCE_ASSURANCE
    assert report.evidence["execution_timing_assurance"] == EXECUTION_TIMING_ASSURANCE
    assert report.evidence["fresh_oos_days"] == 130
    assert report.evidence["statistical_observed_days"] == 160
    assert report.evidence["strict_return_days"] == 120
    assert report.evidence["benchmark_excess_positive"] is True
    assert sha256_file(files["strict_backtest_returns"]) != sha256_file(files["statistical_returns"])


def test_selection_and_strict_return_hashes_cannot_be_swapped(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    strict = _read(files["strict_backtest"])
    strict["daily_returns_sha256"] = sha256_file(files["statistical_returns"])
    _write_json(files["strict_backtest"], strict)
    payload = _read(manifest)
    payload["artifacts"]["strict_backtest"]["sha256"] = sha256_file(files["strict_backtest"])
    _write_json(manifest, payload)
    assert "strict_backtest:daily_returns_sha256_mismatch" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons


def test_tamper_deletion_escape_and_symlink_fail_closed(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    with files["strict_backtest_returns"].open("ab") as handle: handle.write(b"x")
    assert any("strict_backtest_returns:sha256_mismatch" in r for r in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons)

    manifest, _, files, _, roots, _ = _issue(tmp_path / "deleted")
    files["fresh_oos_predictions"].unlink()
    assert "fresh_oos_predictions:artifact_missing" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons

    manifest, _, _, _, roots, _ = _issue(tmp_path / "escape")
    outside = tmp_path / "outside.pt"; outside.write_bytes(b"outside")
    payload = _read(manifest)
    payload["artifacts"]["model_checkpoint"] = {"root": "bundle", "path": "../outside.pt", "sha256": sha256_file(outside)}
    _write_json(manifest, payload)
    assert "model_checkpoint:path_invalid" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons

    manifest, _, files, _, roots, _ = _issue(tmp_path / "symlink")
    original = files["model_checkpoint"]; raw = original.read_bytes(); original.unlink()
    external = tmp_path / "external.pt"; external.write_bytes(raw)
    try: original.symlink_to(external)
    except (OSError, NotImplementedError): pytest.skip("symlinks unavailable")
    assert "model_checkpoint:symlink_not_allowed" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons


def test_rehashed_wrong_semantics_and_costs_fail(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    _rewrite_and_rebind(manifest, "trainer_manifest", files["trainer_manifest"], lambda d: d.__setitem__("objective_semantics", "legacy"))
    assert "trainer_manifest:objective_semantics_mismatch" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons

    manifest, _, files, _, roots, _ = _issue(tmp_path / "cost")
    def weaken(doc: dict) -> None:
        doc["simulation_config"]["slippage_bps"] = 0.0
        doc["cost_model_config"]["commission_rate"] = 0.0
    _rewrite_and_rebind(manifest, "strict_backtest", files["strict_backtest"], weaken)
    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons
    assert "strict_backtest:simulation_config_mismatch" in reasons
    assert "strict_backtest:cost_model_config_mismatch" in reasons


def test_strict_summary_must_match_recomputed_returns(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    def lie(doc: dict) -> None:
        doc["portfolio_total_return"] = 0.99
        doc["benchmark_total_return"] = 0.98
    _rewrite_and_rebind(manifest, "strict_backtest", files["strict_backtest"], lie)
    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons
    assert "strict_backtest:portfolio_total_return_mismatch_recomputed" in reasons
    assert "strict_backtest:benchmark_total_return_mismatch_recomputed" in reasons


def test_fresh_coverage_is_recomputed_from_predictions(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    frame = pd.read_csv(files["fresh_oos_predictions"]); last_day = frame["trade_date"].max()
    frame.loc[frame["trade_date"] != last_day].to_csv(files["fresh_oos_predictions"], index=False)
    new_sha = sha256_file(files["fresh_oos_predictions"])
    fresh = _read(files["fresh_oos"]); fresh["predictions_sha256"] = new_sha; _write_json(files["fresh_oos"], fresh)
    risk = _read(files["risk_capacity"]); risk["predictions_sha256"] = new_sha; _write_json(files["risk_capacity"], risk)
    payload = _read(manifest)
    for role in ("fresh_oos_predictions", "fresh_oos", "risk_capacity"):
        payload["artifacts"][role]["sha256"] = sha256_file(files[role])
    _write_json(manifest, payload)
    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons
    assert not any("derived_trading_days_below_120" in r for r in reasons)
    assert any("trading_days_mismatch_predictions" in r for r in reasons)
    assert "fresh_oos:end_date_mismatch_predictions" in reasons


def test_statistical_summary_must_match_recomputation(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    def lie(doc: dict) -> None:
        doc["pbo"] = 0.10; doc["dsr_probability"] = 0.96; doc["spa_p_value"] = 0.01
    _rewrite_and_rebind(manifest, "statistical_gates", files["statistical_gates"], lie)
    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons
    assert any("pbo_mismatch_recomputed" in r for r in reasons)
    assert any("dsr_probability_mismatch_recomputed" in r for r in reasons)


def test_trial_types_ledger_and_budget_are_hard_gates(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    _rewrite_and_rebind(manifest, "search_ledger", files["search_ledger"], lambda d: d.__setitem__("trial_count", 1.7))
    assert "search_ledger:trial_count_invalid" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons

    manifest, _, files, _, roots, _ = _issue(tmp_path / "ledger")
    _rewrite_and_rebind(manifest, "search_ledger", files["search_ledger"], lambda d: d.__setitem__("trials", d["trials"][:-1]))
    assert "search_ledger:trials_length_mismatch_trial_count" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons

    manifest, _, files, _, roots, _ = _issue(tmp_path / "budget")
    _rewrite_and_rebind(manifest, "pre_registration", files["pre_registration"], lambda d: d.__setitem__("max_search_trials", 10))
    assert "search_ledger:trial_count_exceeds_pre_registered_budget" in evaluate_live_model_trust(manifest, artifact_roots=roots).reasons


def test_timing_policy_and_12_role_completeness_are_hard_gates(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    def contaminate(doc: dict) -> None:
        doc["training_cutoff"] = "2026-02-02"; doc["validation_cutoff"] = "2026-02-03"
    _rewrite_and_rebind(manifest, "data_lineage", files["data_lineage"], contaminate)
    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons
    assert any("training_cutoff_not_before_fresh_oos" in r for r in reasons)

    _, _, locations, roots = _build_evidence(tmp_path / "early")
    with pytest.raises(ValueError, match="trust evidence rejected"):
        issue_governed_live_model_trust_v2(
            tmp_path / "early.json", model_id=MODEL_ID, source_commit=SOURCE_COMMIT,
            artifact_locations=locations, artifact_roots=roots,
            issued_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )

    manifest, _, _, _, roots, _ = _issue(tmp_path / "policy")
    payload = _read(manifest); payload["policy"]["max_pbo"] = 0.99; payload["policy"]["execution_timing_assurance"] = "certified"; _write_json(manifest, payload)
    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons
    assert any("v2_policy_mismatch:max_pbo" in r for r in reasons)
    assert any("v2_policy_mismatch:execution_timing_assurance" in r for r in reasons)

    _, _, locations, roots = _build_evidence(tmp_path / "missing")
    assert len(REQUIRED_ARTIFACT_ROLES) == 12
    locations.pop("strict_backtest_returns")
    with pytest.raises(ValueError, match="artifact role mismatch"):
        issue_governed_live_model_trust_v2(
            tmp_path / "missing.json", model_id=MODEL_ID, source_commit=SOURCE_COMMIT,
            artifact_locations=locations, artifact_roots=roots, issued_at=ISSUED_AT,
        )


def test_canonical_ft_semantics_is_locked_to_trainer_source() -> None:
    assert REQUIRED_TRAINER_OBJECTIVE_SEMANTICS == FT_TRANSFORMER_OBJECTIVE_SEMANTICS_VERSION
    source = Path("src/quantagent/training/ft_transformer_trainer.py").read_text(encoding="utf-8")
    assert 'OBJECTIVE_SEMANTICS_VERSION = "ft_transformer_objective_v2_per_date_listwise_validation"' in source
