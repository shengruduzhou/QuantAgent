from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.execution.live_model_trust import evaluate_live_model_trust
from quantagent.execution.live_model_trust_v2 import (
    PROVENANCE_ASSURANCE,
    REQUIRED_ARTIFACT_ROLES,
    REQUIRED_METRIC_SEMANTICS,
    REQUIRED_TRAINER_OBJECTIVE_SEMANTICS,
    sha256_file,
)
from quantagent.execution.live_model_trust_v2_policy import (
    issue_governed_live_model_trust_v2,
)
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
CANDIDATE_FAMILY = ["linear_control", "ft_transformer"]
SELECTED = "ft_transformer"
ISSUED_AT = datetime(2026, 8, 2, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _common() -> dict[str, object]:
    return {"schema_version": 1, "model_id": MODEL_ID, "source_commit": SOURCE_COMMIT}


def _trial_ledger(count: int = 50) -> list[dict[str, str]]:
    return [
        {
            "trial_id": f"trial-{idx:03d}",
            "candidate": CANDIDATE_FAMILY[idx % len(CANDIDATE_FAMILY)],
        }
        for idx in range(count)
    ]


def _write_statistical_returns(path: Path) -> object:
    dates = pd.bdate_range(end="2026-01-30", periods=160)
    rng = np.random.default_rng(42)
    shared = rng.normal(0.0, 0.002, len(dates))
    frame = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "benchmark": np.zeros(len(dates), dtype=float),
            "linear_control": 0.00050 + shared,
            "ft_transformer": 0.00055 + shared + rng.normal(0.0, 0.0001, len(dates)),
        }
    )
    frame.to_csv(path, index=False)
    candidate_returns = frame.set_index(pd.DatetimeIndex(dates))[CANDIDATE_FAMILY]
    benchmark = pd.Series(frame["benchmark"].to_numpy(dtype=float), index=dates)
    report = evaluate_frozen_candidate(
        candidate_returns,
        selected_candidate=SELECTED,
        benchmark_returns=benchmark,
        cumulative_trials=50,
        minimum_observed_days=80,
    )
    assert report.accepted is True, report.rejection_reasons
    return report


def _write_fresh_predictions(path: Path) -> tuple[int, str, str]:
    dates = pd.bdate_range("2026-02-02", periods=130)
    rows: list[dict[str, object]] = []
    for idx, day in enumerate(dates):
        rows.append(
            {"trade_date": day.date().isoformat(), "symbol": "600000.SH", "prediction": 0.01 + idx * 1e-6}
        )
        rows.append(
            {"trade_date": day.date().isoformat(), "symbol": "000001.SZ", "prediction": 0.02 + idx * 1e-6}
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(dates), dates[0].date().isoformat(), dates[-1].date().isoformat()


def _build_evidence(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, tuple[str, str]], dict[str, Path]]:
    root = tmp_path / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    files = {role: root / f"{role}.json" for role in REQUIRED_ARTIFACT_ROLES}
    files["model_checkpoint"] = root / "model_checkpoint.pt"
    files["statistical_returns"] = root / "statistical_returns.csv"
    files["fresh_oos_predictions"] = root / "fresh_oos_predictions.csv"

    files["model_checkpoint"].write_bytes(b"governed-ft-checkpoint-v2\x00\x01")
    stat_report = _write_statistical_returns(files["statistical_returns"])
    fresh_days, fresh_start, fresh_end = _write_fresh_predictions(files["fresh_oos_predictions"])
    checkpoint_sha = sha256_file(files["model_checkpoint"])
    returns_sha = sha256_file(files["statistical_returns"])
    predictions_sha = sha256_file(files["fresh_oos_predictions"])

    _write_json(
        files["trainer_manifest"],
        _common()
        | {
            "objective_semantics": REQUIRED_TRAINER_OBJECTIVE_SEMANTICS,
            "checkpoint_sha256": checkpoint_sha,
            "training_cutoff": "2025-12-31",
            "validation_cutoff": "2026-01-30",
        },
    )
    _write_json(
        files["pre_registration"],
        _common()
        | {
            "selection_pre_registered": True,
            "registered_at": "2026-01-15T00:00:00+00:00",
            "candidate_family": CANDIDATE_FAMILY,
            "max_search_trials": 50,
            "acceptance_window_id": WINDOW_ID,
            "acceptance_start_date": fresh_start,
            "acceptance_end_date": fresh_end,
        },
    )
    _write_json(
        files["search_ledger"],
        _common()
        | {
            "trial_count": 50,
            "candidate_family": CANDIDATE_FAMILY,
            "selected_candidate": SELECTED,
            "trials": _trial_ledger(),
            "completed_at": "2026-01-20T00:00:00+00:00",
            "final_holdout_used_for_selection": False,
            "selection_frozen_before_fresh_oos": True,
        },
    )
    _write_json(
        files["data_lineage"],
        _common()
        | {
            "pit": True,
            "universe_pit": True,
            "training_cutoff": "2025-12-31",
            "validation_cutoff": "2026-01-30",
        },
    )
    _write_json(
        files["strict_backtest"],
        _common()
        | {
            "metric_semantics": REQUIRED_METRIC_SEMANTICS,
            "execution_semantics": TRUSTED_EXECUTION_SEMANTICS,
            "t_plus_one": True,
            "costs_included": True,
            "simulation_config": trusted_simulation_config(),
            "cost_model_config": trusted_cost_model_config(),
            "benchmark_excess_positive": True,
            "variant_c_passed": True,
            "checkpoint_sha256": checkpoint_sha,
            # This hash binds the complete early-OOS candidate return matrix;
            # the governed layer derives the frozen selected column from it.
            "daily_returns_sha256": returns_sha,
        },
    )
    _write_json(
        files["statistical_gates"],
        _common()
        | {
            "selected_candidate": SELECTED,
            "pbo": float(stat_report.pbo),
            "dsr_probability": float(stat_report.dsr_probability),
            "spa_p_value": float(stat_report.spa_pvalue),
            "trial_count": 50,
            "candidate_family": CANDIDATE_FAMILY,
            "returns_sha256": returns_sha,
        },
    )
    _write_json(
        files["fresh_oos"],
        _common()
        | {
            "trading_days": fresh_days,
            "final_holdout_reads": 1,
            "contaminated_holdout": False,
            "start_date": fresh_start,
            "end_date": fresh_end,
            "acceptance_window_id": WINDOW_ID,
            "predictions_sha256": predictions_sha,
        },
    )
    _write_json(
        files["risk_capacity"],
        _common()
        | {
            "passed": True,
            "checkpoint_sha256": checkpoint_sha,
            "predictions_sha256": predictions_sha,
        },
    )

    locations = {
        role: ("bundle", files[role].relative_to(root).as_posix())
        for role in REQUIRED_ARTIFACT_ROLES
    }
    roots = {"bundle": root}
    return root, files, locations, roots


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


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_and_rebind(manifest: Path, role: str, artifact: Path, mutate) -> None:
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    mutate(doc)
    _write_json(artifact, doc)
    payload = _read_manifest(manifest)
    payload["artifacts"][role]["sha256"] = sha256_file(artifact)
    _write_json(manifest, payload)


def test_valid_v2_real_artifact_bundle_passes(tmp_path: Path) -> None:
    manifest, _, _, _, roots, issued = _issue(tmp_path)
    assert issued.verification.ok is True
    payload = _read_manifest(manifest)
    assert "evidence" not in payload
    assert payload["provenance_assurance"] == PROVENANCE_ASSURANCE

    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is True
    assert report.model_id == MODEL_ID
    assert report.evidence["provenance_assurance"] == "hash_bound_unsigned_v1"
    assert report.evidence["fresh_oos_days"] == 130
    assert report.evidence["fresh_prediction_rows"] == 260
    assert report.evidence["fresh_prediction_symbols"] == 2
    assert report.evidence["statistical_observed_days"] == 160
    assert report.evidence["statistical_recomputed"] is True
    assert report.evidence["trial_count"] == 50
    assert report.evidence["pre_registered_max_search_trials"] == 50
    assert report.evidence["execution_semantics"] == TRUSTED_EXECUTION_SEMANTICS


def test_one_byte_artifact_tamper_invalidates_certificate(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    with files["statistical_returns"].open("ab") as handle:
        handle.write(b"x")
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("statistical_returns:sha256_mismatch" in reason for reason in report.reasons)


def test_bound_artifact_deletion_fails_closed(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    files["fresh_oos_predictions"].unlink()
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "fresh_oos_predictions:artifact_missing" in report.reasons


def test_certificate_path_escape_is_rejected_even_with_a_plausible_digest(tmp_path: Path) -> None:
    manifest, root, _, _, roots, _ = _issue(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    payload = _read_manifest(manifest)
    payload["artifacts"]["model_checkpoint"] = {
        "root": "bundle",
        "path": "../outside.pt",
        "sha256": sha256_file(outside),
    }
    _write_json(manifest, payload)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "model_checkpoint:path_invalid" in report.reasons
    assert root not in outside.parents


def test_symlink_substitution_is_rejected(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    original = files["model_checkpoint"]
    bytes_ = original.read_bytes()
    original.unlink()
    external = tmp_path / "external.pt"
    external.write_bytes(bytes_)
    try:
        original.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "model_checkpoint:symlink_not_allowed" in report.reasons


def test_rehashed_but_wrong_trainer_semantics_still_fails(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    _rewrite_and_rebind(
        manifest,
        "trainer_manifest",
        files["trainer_manifest"],
        lambda doc: doc.__setitem__("objective_semantics", "legacy_pointwise_validation"),
    )
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "trainer_manifest:objective_semantics_mismatch" in report.reasons


def test_strict_backtest_execution_and_cost_parameters_are_exactly_bound(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def weaken(doc: dict) -> None:
        doc["simulation_config"]["slippage_bps"] = 0.0
        doc["cost_model_config"]["commission_rate"] = 0.0

    _rewrite_and_rebind(manifest, "strict_backtest", files["strict_backtest"], weaken)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "strict_backtest:simulation_config_mismatch" in report.reasons
    assert "strict_backtest:cost_model_config_mismatch" in report.reasons


def test_fresh_oos_coverage_is_recomputed_from_predictions_not_summary(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    frame = pd.read_csv(files["fresh_oos_predictions"])
    last_day = frame["trade_date"].max()
    frame = frame.loc[frame["trade_date"] != last_day]
    frame.to_csv(files["fresh_oos_predictions"], index=False)
    new_sha = sha256_file(files["fresh_oos_predictions"])

    fresh = json.loads(files["fresh_oos"].read_text(encoding="utf-8"))
    fresh["predictions_sha256"] = new_sha
    _write_json(files["fresh_oos"], fresh)
    risk = json.loads(files["risk_capacity"].read_text(encoding="utf-8"))
    risk["predictions_sha256"] = new_sha
    _write_json(files["risk_capacity"], risk)
    payload = _read_manifest(manifest)
    payload["artifacts"]["fresh_oos_predictions"]["sha256"] = new_sha
    payload["artifacts"]["fresh_oos"]["sha256"] = sha256_file(files["fresh_oos"])
    payload["artifacts"]["risk_capacity"]["sha256"] = sha256_file(files["risk_capacity"])
    _write_json(manifest, payload)

    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("derived_trading_days_below_120" not in reason for reason in report.reasons)
    assert any("trading_days_mismatch_predictions" in reason for reason in report.reasons)
    assert any("end_date_mismatch_predictions" in reason for reason in report.reasons)


def test_fresh_oos_days_and_one_shot_read_are_hard_integer_gates(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def bad(doc: dict) -> None:
        doc["trading_days"] = 119
        doc["final_holdout_reads"] = 2

    _rewrite_and_rebind(manifest, "fresh_oos", files["fresh_oos"], bad)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("trading_days_below_120" in reason for reason in report.reasons)
    assert any("final_holdout_reads_must_equal_1" in reason for reason in report.reasons)


def test_fractional_trial_count_is_invalid_not_truncated(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    _rewrite_and_rebind(
        manifest,
        "search_ledger",
        files["search_ledger"],
        lambda doc: doc.__setitem__("trial_count", 1.7),
    )
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "search_ledger:trial_count_invalid" in report.reasons


def test_probability_fields_reject_bool_negative_and_over_one(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def bad(doc: dict) -> None:
        doc["pbo"] = -0.1
        doc["dsr_probability"] = True
        doc["spa_p_value"] = 1.2

    _rewrite_and_rebind(manifest, "statistical_gates", files["statistical_gates"], bad)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("statistical_gates:pbo" in reason for reason in report.reasons)
    assert any("statistical_gates:dsr" in reason for reason in report.reasons)
    assert any("statistical_gates:spa" in reason for reason in report.reasons)


def test_statistical_summary_must_match_recomputed_pbo_dsr_spa(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def misreport(doc: dict) -> None:
        # All values remain inside the raw pass thresholds; only independent
        # recomputation can detect that the summary was repackaged.
        doc["pbo"] = 0.10
        doc["dsr_probability"] = 0.96
        doc["spa_p_value"] = 0.01

    _rewrite_and_rebind(manifest, "statistical_gates", files["statistical_gates"], misreport)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("pbo_mismatch_recomputed" in reason for reason in report.reasons)
    assert any("dsr_probability_mismatch_recomputed" in reason for reason in report.reasons)


def test_statistical_thresholds_cannot_be_repackaged_as_pass(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def bad(doc: dict) -> None:
        doc["pbo"] = 0.251
        doc["dsr_probability"] = 0.949
        doc["spa_p_value"] = 0.051

    _rewrite_and_rebind(manifest, "statistical_gates", files["statistical_gates"], bad)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("pbo_above_0.25" in reason for reason in report.reasons)
    assert any("dsr_below_0.95" in reason for reason in report.reasons)
    assert any("spa_above_0.05" in reason for reason in report.reasons)


def test_trial_budget_candidate_family_and_full_trial_ledger_are_bound(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def bad_stats(doc: dict) -> None:
        doc["trial_count"] = 2
        doc["candidate_family"] = ["ft_transformer"]

    _rewrite_and_rebind(manifest, "statistical_gates", files["statistical_gates"], bad_stats)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "statistical_gates:trial_count_mismatch_search_ledger" in report.reasons
    assert "statistical_gates:candidate_family_mismatch_search_ledger" in report.reasons

    manifest, _, files, _, roots, _ = _issue(tmp_path / "ledger")
    _rewrite_and_rebind(
        manifest,
        "search_ledger",
        files["search_ledger"],
        lambda doc: doc.__setitem__("trials", doc["trials"][:-1]),
    )
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "search_ledger:trials_length_mismatch_trial_count" in report.reasons


def test_search_cannot_exceed_pre_registered_budget(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    _rewrite_and_rebind(
        manifest,
        "pre_registration",
        files["pre_registration"],
        lambda doc: doc.__setitem__("max_search_trials", 10),
    )
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "search_ledger:trial_count_exceeds_pre_registered_budget" in report.reasons


def test_preregistration_must_name_exact_window_and_precede_it(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def bad(doc: dict) -> None:
        doc["acceptance_window_id"] = "chosen-after-looking"
        doc["registered_at"] = "2026-02-02T00:00:00+00:00"

    _rewrite_and_rebind(manifest, "pre_registration", files["pre_registration"], bad)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "pre_registration:acceptance_window_id_mismatch" in report.reasons
    assert "pre_registration:registered_not_strictly_before_fresh_oos" in report.reasons


def test_search_must_finish_before_fresh_window(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)
    _rewrite_and_rebind(
        manifest,
        "search_ledger",
        files["search_ledger"],
        lambda doc: doc.__setitem__("completed_at", "2026-02-02T00:00:00+00:00"),
    )
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert "search_ledger:completed_not_strictly_before_fresh_oos" in report.reasons


def test_training_or_validation_cutoff_cannot_enter_fresh_window(tmp_path: Path) -> None:
    manifest, _, files, _, roots, _ = _issue(tmp_path)

    def contaminate(doc: dict) -> None:
        doc["training_cutoff"] = "2026-02-02"
        doc["validation_cutoff"] = "2026-02-03"

    _rewrite_and_rebind(manifest, "data_lineage", files["data_lineage"], contaminate)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("data_lineage:training_cutoff_not_before_fresh_oos" in r for r in report.reasons)
    assert any("data_lineage:validation_cutoff_not_before_fresh_oos" in r for r in report.reasons)


def test_issuer_cannot_sign_before_fresh_window_has_finished(tmp_path: Path) -> None:
    _, _, locations, roots = _build_evidence(tmp_path)
    with pytest.raises(ValueError, match="trust evidence rejected"):
        issue_governed_live_model_trust_v2(
            tmp_path / "too-early.json",
            model_id=MODEL_ID,
            source_commit=SOURCE_COMMIT,
            artifact_locations=locations,
            artifact_roots=roots,
            issued_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )


def test_certificate_cannot_weaken_verifier_policy(tmp_path: Path) -> None:
    manifest, _, _, _, roots, _ = _issue(tmp_path)
    payload = _read_manifest(manifest)
    payload["policy"]["max_pbo"] = 0.99
    payload["policy"]["min_fresh_oos_days"] = 1
    payload["policy"]["simulation_config"]["slippage_bps"] = 0.0
    _write_json(manifest, payload)
    report = evaluate_live_model_trust(manifest, artifact_roots=roots)
    assert report.ok is False
    assert any("v2_policy_mismatch:max_pbo" in reason for reason in report.reasons)
    assert any("v2_policy_mismatch:min_fresh_oos_days" in reason for reason in report.reasons)
    assert any("v2_policy_mismatch:simulation_config" in reason for reason in report.reasons)


def test_issuer_refuses_incomplete_artifact_role_set(tmp_path: Path) -> None:
    _, _, locations, roots = _build_evidence(tmp_path)
    locations.pop("risk_capacity")
    with pytest.raises(ValueError, match="artifact role mismatch"):
        issue_governed_live_model_trust_v2(
            tmp_path / "bad.json",
            model_id=MODEL_ID,
            source_commit=SOURCE_COMMIT,
            artifact_locations=locations,
            artifact_roots=roots,
            issued_at=ISSUED_AT,
        )


def test_canonical_ft_semantics_is_locked_to_current_trainer_source() -> None:
    assert REQUIRED_TRAINER_OBJECTIVE_SEMANTICS == FT_TRANSFORMER_OBJECTIVE_SEMANTICS_VERSION
    trainer_source = Path("src/quantagent/training/ft_transformer_trainer.py").read_text(encoding="utf-8")
    assert (
        'OBJECTIVE_SEMANTICS_VERSION = "ft_transformer_objective_v2_per_date_listwise_validation"'
        in trainer_source
    )
