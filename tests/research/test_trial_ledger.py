"""Persistent multiple-testing ledger (ROADMAP P1-4).

The DSR is only as honest as the N it is given. These tests pin down that the
ledger keeps an unforgeable-by-accident cumulative count per family, and that
the governed commands actually feed that cumulative N -- not their own run's
arm/candidate count -- into the deflated Sharpe ratio.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantagent.research.trial_ledger import (
    FAMILY_NONLINEAR_FUSION,
    FAMILY_RDAGENT_FACTOR,
    HISTORICAL_BASELINES,
    TrialLedger,
    TrialLedgerCorruption,
    TrialRecord,
    config_hash,
    default_trial_ledger_path,
)


def _trial(index: int, family: str = FAMILY_NONLINEAR_FUSION) -> TrialRecord:
    return TrialRecord(
        family=family,
        trial_id=f"run@t:arm{index}",
        command="unit-test",
        git_hash="deadbeef",
        config_hash="c" * 64,
        window={"start": "2022-01-03", "end": "2023-06-30"},
        metrics={"rank_ic_mean": 0.01 * index, "nan_metric": float("nan"), "np": np.float64(1.5)},
        verdict="measured",
    )


# ---------------------------------------------------------------------------
# ledger mechanics
# ---------------------------------------------------------------------------


def test_family_count_is_declared_baseline_plus_recorded(tmp_path: Path) -> None:
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    base = HISTORICAL_BASELINES[FAMILY_NONLINEAR_FUSION].count
    assert base == 134  # RESEARCH.md §3 Round 31 estimate, seeded with its source
    assert HISTORICAL_BASELINES[FAMILY_NONLINEAR_FUSION].source
    assert ledger.family_count(FAMILY_NONLINEAR_FUSION) == base

    ledger.record_trial(
        FAMILY_NONLINEAR_FUSION, "t0", "unit-test", "deadbeef", "c" * 64,
        {"start": "2022-01-03", "end": "2022-12-30"}, {"ic": 0.02}, "measured",
    )
    ledger.record_trials([_trial(1), _trial(2), _trial(3, family=FAMILY_RDAGENT_FACTOR)])

    assert ledger.family_count(FAMILY_NONLINEAR_FUSION) == base + 3
    assert ledger.family_count(FAMILY_RDAGENT_FACTOR) == HISTORICAL_BASELINES[FAMILY_RDAGENT_FACTOR].count + 1
    assert ledger.family_count("undeclared_family") == 0
    report = ledger.verify()
    assert report["valid"] and report["records"] == 4
    # Non-finite metrics are stored as null rather than breaking canonical JSON.
    row = ledger.read(FAMILY_NONLINEAR_FUSION)[1]
    assert row["metrics"]["nan_metric"] is None and row["metrics"]["np"] == 1.5


def test_default_path_lives_under_quantagent_home_not_the_repo_runtime() -> None:
    import os

    path = default_trial_ledger_path()
    assert path == Path(os.environ["QUANTAGENT_HOME"]) / "state" / "trial_ledger.jsonl"
    repo_runtime = Path(__file__).resolve().parents[2] / "runtime"
    assert repo_runtime not in path.parents


def test_trial_record_refuses_empty_identity_fields() -> None:
    with pytest.raises(ValueError, match="git_hash"):
        TrialRecord(FAMILY_NONLINEAR_FUSION, "t", "cmd", "", "c", {}, {}, "")


def test_config_hash_ignores_key_order_and_numpy_types() -> None:
    assert config_hash({"a": 1, "b": np.int64(2)}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def _append_worker(path: str, worker: int, batches: int) -> None:
    ledger = TrialLedger(path)
    for batch in range(batches):
        ledger.record_trials(
            [
                TrialRecord(
                    family=FAMILY_NONLINEAR_FUSION,
                    trial_id=f"w{worker}-b{batch}-{k}",
                    command="concurrency-test",
                    git_hash="deadbeef",
                    config_hash="c" * 64,
                    window={},
                    metrics={"k": k},
                    verdict="measured",
                )
                for k in range(3)
            ]
        )


def test_concurrent_writers_never_fork_the_chain(tmp_path: Path) -> None:
    """Round 22 A-04 measured 4 procs x 40 entries => 0 reachable without a lock."""
    path = tmp_path / "ledger.jsonl"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_worker, args=(str(path), w, 10)) for w in range(4)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0
    ledger = TrialLedger(path)
    report = ledger.verify()
    assert report == {"valid": True, "records": 120, "headHash": report["headHash"]}
    rows = ledger.read()
    assert [row["sequence"] for row in rows] == list(range(120))
    # Each batch is contiguous: a batch is one write under one lock.
    for start in range(0, 120, 3):
        ids = [rows[start + k]["trial_id"] for k in range(3)]
        assert len({i.rsplit("-", 1)[0] for i in ids}) == 1, ids


def test_torn_tail_is_detected_and_blocks_further_appends(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = TrialLedger(path)
    ledger.record_trials([_trial(1), _trial(2)])
    with path.open("ab") as handle:  # a writer killed mid-line
        handle.write(b'{"schema":"quantagent.trial_ledger.v1","seq')
    report = ledger.verify()
    assert report["valid"] is False and "torn tail" in report["reason"]
    with pytest.raises(TrialLedgerCorruption):
        ledger.record_trials([_trial(3)])
    with pytest.raises(TrialLedgerCorruption):
        ledger.family_count(FAMILY_NONLINEAR_FUSION)


@pytest.mark.parametrize("mutation", ["edit_metric", "delete_line", "swap_lines", "drop_family"])
def test_tampering_breaks_verification_and_counting(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = TrialLedger(path)
    ledger.record_trials([_trial(i) for i in range(5)])
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutation == "edit_metric":
        row = json.loads(lines[2])
        row["metrics"]["rank_ic_mean"] = 0.99
        lines[2] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    elif mutation == "delete_line":
        del lines[1]
    elif mutation == "swap_lines":
        lines[1], lines[3] = lines[3], lines[1]
    else:  # relabel a trial into another family to shrink this family's N
        row = json.loads(lines[4])
        row["family"] = "elsewhere"
        lines[4] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = ledger.verify()
    assert report["valid"] is False
    assert report["reason"] == "hash chain mismatch"
    with pytest.raises(TrialLedgerCorruption):
        ledger.family_count(FAMILY_NONLINEAR_FUSION)


# ---------------------------------------------------------------------------
# DSR receives the cumulative family N
# ---------------------------------------------------------------------------


def _spy_dsr(monkeypatch, module) -> list[int]:
    captured: list[int] = []
    original = module.deflated_sharpe_ratio

    def spy(*args, **kwargs):
        captured.append(int(kwargs["n_trials"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "deflated_sharpe_ratio", spy)
    return captured


def _comparison_panel(path: Path) -> None:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2022-01-03", periods=170)
    symbols = [f"S{i:03d}" for i in range(40)]
    frames = []
    for date in dates:
        x1, x2 = rng.normal(size=len(symbols)), rng.normal(size=len(symbols))
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": date,
                    "symbol": symbols,
                    "f1": x1,
                    "f2": x2,
                    "forward_return_5d": 0.004 * x1 + 0.003 * x2 + rng.normal(0, 0.02, len(symbols)),
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def _cheap_gbm(monkeypatch) -> None:
    """Test-suite economics: the CLI's default GBM uses every core on a tiny panel."""
    import functools

    import quantagent.cli.nonlinear as nonlinear_cli
    from quantagent.research.model_comparison import ComparisonConfig

    monkeypatch.setattr(
        nonlinear_cli,
        "ComparisonConfig",
        functools.partial(ComparisonConfig, gbm_estimators=30, gbm_n_jobs=2),
    )


def test_audit_nonlinear_factors_feeds_cumulative_family_n_to_dsr(tmp_path, monkeypatch) -> None:
    """Fails on pre-ledger code: DSR used ``n_trials = len(arms)`` (~5), not 134 + arms."""
    import quantagent.research.model_comparison as model_comparison
    import quantagent.research.nonlinear_promotion as nonlinear_promotion
    from quantagent.cli import app

    _cheap_gbm(monkeypatch)

    comparison_n = _spy_dsr(monkeypatch, model_comparison)
    promotion_n = _spy_dsr(monkeypatch, nonlinear_promotion)
    ledger_path = tmp_path / "state" / "trial_ledger.jsonl"
    monkeypatch.setenv("QUANTAGENT_HOME", str(tmp_path))
    panel = tmp_path / "panel.parquet"
    _comparison_panel(panel)
    args = [
        "audit-nonlinear-factors", "--panel-path", str(panel), "--factor-names", "f1,f2",
        "--n-folds", "4", "--holdout-folds", "1", "--valid-size-days", "15",
        "--min-train-days", "60", "--embargo-days", "5",
    ]
    base = HISTORICAL_BASELINES[FAMILY_NONLINEAR_FUSION].count

    first = CliRunner().invoke(app, [*args, "--output-dir", str(tmp_path / "run1")])
    assert first.exit_code == 0, first.stdout
    gate = json.loads((tmp_path / "run1" / "promotion_gate.json").read_text())
    arms = len(gate["comparison"]["arms"])
    assert arms >= 2 and gate["comparison"]["nTrials"] == arms
    assert comparison_n and promotion_n, "the comparison never reached its DSR"
    assert set(comparison_n) == {base + arms}
    assert set(promotion_n) == {base + arms}
    assert gate["comparison"]["dsrTrials"] == base + arms
    assert gate["trialLedger"]["priorFamilyTrials"] == base
    assert gate["trialLedger"]["cumulativeTrials"] == base + arms
    ledger = TrialLedger(ledger_path)
    assert ledger.verify()["valid"]
    assert ledger.family_count(FAMILY_NONLINEAR_FUSION) == base + arms
    assert {row["command"] for row in ledger.read()} == {"audit-nonlinear-factors"}

    # A second identical run is deflated by both runs, not reset to its own arms.
    comparison_n.clear()
    promotion_n.clear()
    second = CliRunner().invoke(app, [*args, "--output-dir", str(tmp_path / "run2")])
    assert second.exit_code == 0, second.stdout
    assert set(comparison_n) == {base + 2 * arms}
    assert set(promotion_n) == {base + 2 * arms}
    assert ledger.family_count(FAMILY_NONLINEAR_FUSION) == base + 2 * arms


def _fusion_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=300)
    symbols = [f"{600000 + index}.SH" for index in range(25)]
    frames = []
    for date in dates:
        signal = rng.normal(size=len(symbols))
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": date,
                    "symbol": symbols,
                    "alpha_a": signal,
                    "alpha_b": rng.normal(size=len(symbols)),
                    "forward_return": 0.02 * signal + 0.01 * rng.normal(size=len(symbols)),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    factors, forwards = tmp_path / "factors.parquet", tmp_path / "forwards.parquet"
    panel[["trade_date", "symbol", "alpha_a", "alpha_b"]].to_parquet(factors, index=False)
    panel[["trade_date", "symbol", "forward_return"]].to_parquet(forwards, index=False)
    return factors, forwards


def test_search_factor_fusion_feeds_cumulative_family_n_to_dsr(tmp_path, monkeypatch) -> None:
    """Fails on pre-ledger code: both fusion DSRs used the enumerated count only."""
    import quantagent.fusion.search_corrected as search_corrected
    import quantagent.research.foundation_gates as foundation_gates
    from quantagent.cli import app

    robustness_n = _spy_dsr(monkeypatch, search_corrected)
    gate_n = _spy_dsr(monkeypatch, foundation_gates)
    monkeypatch.setenv("QUANTAGENT_HOME", str(tmp_path))
    factors, forwards = _fusion_inputs(tmp_path)
    base = HISTORICAL_BASELINES[FAMILY_NONLINEAR_FUSION].count
    # Pre-existing family history from another command must count too.
    TrialLedger(tmp_path / "state" / "trial_ledger.jsonl").record_trials([_trial(i) for i in range(7)])

    result = CliRunner().invoke(
        app,
        [
            "search-factor-fusion", "--factor-panel-path", str(factors),
            "--forward-returns-path", str(forwards), "--factor-names", "alpha_a,alpha_b",
            "--output-dir", str(tmp_path / "fusion"), "--n-folds", "2", "--top-k", "5",
            "--min-train-days", "100", "--min-test-days", "40", "--no-include-genetic",
            "--random-controls", "2", "--single-factor-baselines", "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary = json.loads((tmp_path / "fusion" / "fusion_summary.json").read_text())
    n_trials = summary["nTrials"]
    expected = base + 7 + n_trials
    assert summary["dsrTrials"] == expected
    assert robustness_n and set(robustness_n) == {expected}
    assert gate_n and set(gate_n) == {expected}
    gate = json.loads((tmp_path / "fusion" / "promotion_gate.json").read_text())
    assert gate["trialLedger"]["priorFamilyTrials"] == base + 7
    assert gate["statisticalEvidence"]["dsrTrials"] == expected
    ledger = TrialLedger(tmp_path / "state" / "trial_ledger.jsonl")
    assert ledger.family_count(FAMILY_NONLINEAR_FUSION) == expected
    fusion_rows = [row for row in ledger.read() if row["command"] == "search-factor-fusion"]
    assert len(fusion_rows) == n_trials  # controls are trials too
    assert any(row["metrics"]["isControl"] for row in fusion_rows)


def test_family_history_can_only_raise_fusion_n(tmp_path) -> None:
    """``n_trials`` stays the enumerated space; the ledger adds, never lowers."""
    from quantagent.fusion import FusionSearchConfig, run_fusion_search

    factors, forwards = _fusion_inputs(tmp_path)
    kwargs = dict(
        factor_panel=pd.read_parquet(factors),
        forward_panel=pd.read_parquet(forwards),
        config=FusionSearchConfig(
            factor_names=("alpha_a", "alpha_b"), horizon_days=5, top_k=5, n_folds=2,
            min_train_days=100, min_test_days=40, include_genetic=False,
            random_controls=2, single_factor_baselines=2,
        ),
    )
    plain = run_fusion_search(**kwargs)
    with_history = run_fusion_search(**kwargs, prior_family_trials=134)
    assert plain.n_trials == with_history.n_trials
    assert plain.dsr_trials == plain.n_trials
    assert with_history.dsr_trials == plain.n_trials + 134
    with pytest.raises(ValueError, match="non-negative"):
        run_fusion_search(**kwargs, prior_family_trials=-5)


def test_damaged_ledger_blocks_the_governed_command_before_compute(tmp_path, monkeypatch) -> None:
    from quantagent.cli import app
    import quantagent.cli.nonlinear as nonlinear_cli

    monkeypatch.setenv("QUANTAGENT_HOME", str(tmp_path))
    path = tmp_path / "state" / "trial_ledger.jsonl"
    TrialLedger(path).record_trials([_trial(1)])
    path.write_text(path.read_text().replace('"deadbeef"', '"cafebabe"'), encoding="utf-8")
    monkeypatch.setattr(
        nonlinear_cli, "run_governed_model_comparison",
        lambda *a, **k: pytest.fail("compute must not start on a damaged ledger"),
    )
    panel = tmp_path / "panel.parquet"
    _comparison_panel(panel)
    result = CliRunner().invoke(
        app,
        ["audit-nonlinear-factors", "--panel-path", str(panel), "--factor-names", "f1,f2",
         "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, TrialLedgerCorruption)


def test_rd_agent_loop_records_every_candidate_into_its_family(tmp_path, monkeypatch) -> None:
    """Fails on pre-ledger code: RD-Agent runs left no trial count anywhere."""
    from quantagent.cli import app
    from tests.test_factor_synthesis import _make_panel_with_momentum_signal

    monkeypatch.setenv("QUANTAGENT_HOME", str(tmp_path))
    panel_path = tmp_path / "panel.parquet"
    _make_panel_with_momentum_signal(n_symbols=6, n_days=180).to_parquet(panel_path, index=False)
    out = tmp_path / "synth"
    result = CliRunner().invoke(
        app,
        [
            "synthesize-factors-v7", "--market-panel", str(panel_path), "--output-dir", str(out),
            "--rounds", "2", "--factors-per-round", "3", "--fitness-sample-dates", "0",
            "--fitness-sample-symbols", "0", "--min-validation-tstat", "0",
            "--allow-legacy-unstamped-labels",
        ],
    )
    assert result.exit_code == 0, result.stdout
    feedback = json.loads((out / "rd_agent_task_feedback.json").read_text())
    summary = json.loads((out / "trial_ledger.json").read_text())
    base = HISTORICAL_BASELINES[FAMILY_RDAGENT_FACTOR].count
    assert len(feedback) > 0
    assert summary["runTrials"] == len(feedback)
    assert summary["cumulativeTrials"] == base + len(feedback)
    ledger = TrialLedger(tmp_path / "state" / "trial_ledger.jsonl")
    rows = ledger.read(FAMILY_RDAGENT_FACTOR)
    assert len(rows) == len(feedback)
    # Rejections are trials too.
    assert {row["verdict"] for row in rows} <= {"accepted", "rejected"}
    assert sum(row["verdict"] == "accepted" for row in rows) == sum(bool(r["decision"]) for r in feedback)
    assert ledger.family_count(FAMILY_NONLINEAR_FUSION) == HISTORICAL_BASELINES[FAMILY_NONLINEAR_FUSION].count
