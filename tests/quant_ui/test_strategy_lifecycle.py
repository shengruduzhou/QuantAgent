"""Strategy lifecycle: identity, runs, results and deletion.

These cover the gaps that made saved strategies unmanageable: every save looked
like a new strategy, nothing linked a run to the job or artifacts it produced,
and there was no way to remove anything.
"""

from __future__ import annotations

import json

import pytest

from services.quant_api.app import create_app
from services.quant_api.services.job_diagnostics import diagnose, extract_exception
from services.quant_api.services.run_results import RunResultResolver
from services.quant_api.services.strategies import StrategyService
from tests.quant_ui.test_api import request
from tests.quant_ui.test_strategy_studio import _strategy_payload


def _save(app, **overrides) -> dict:
    payload = {**_strategy_payload(), **overrides}
    saved = request(app, "POST", "/api/strategies", json=payload)
    assert saved.status_code == 200, saved.text
    return saved.json()["data"]


# --- identity ----------------------------------------------------------------
def test_repeated_saves_are_versions_of_one_strategy_not_new_strategies(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)

    first = _save(app, id="alpha-one", name="Alpha One")
    second = _save(app, id="alpha-one", name="Alpha One")
    third = _save(app, id="alpha-one", name="Alpha One")
    assert len({first["version"], second["version"], third["version"]}) == 3

    listing = request(app, "GET", "/api/strategies").json()["data"]

    assert len(listing) == 1
    assert listing[0]["id"] == "alpha-one"
    assert listing[0]["versionCount"] == 3
    assert listing[0]["version"] == third["version"]  # newest is the head


def test_a_strategy_exposes_its_full_version_history(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    _save(app, id="alpha-two", name="Alpha Two")
    head = _save(app, id="alpha-two", name="Alpha Two")

    detail = request(app, "GET", "/api/strategies/alpha-two").json()["data"]

    assert detail["version"] == head["version"]
    assert [item["version"] for item in detail["versions"]] == sorted(
        [item["version"] for item in detail["versions"]], reverse=True
    )
    assert len(detail["versions"]) == 2


def test_an_older_version_can_be_read_back_verbatim(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    first = _save(app, id="alpha-three", name="Alpha Three", topK=25)
    _save(app, id="alpha-three", name="Alpha Three", topK=40)

    detail = request(
        app, "GET", f"/api/strategies/alpha-three?version={first['version']}"
    ).json()["data"]

    assert detail["draft"]["topK"] == 25


# --- deletion ----------------------------------------------------------------
def test_deleting_a_strategy_archives_it_rather_than_destroying_the_record(
    quant_ui_settings,
) -> None:
    app = create_app(quant_ui_settings)
    _save(app, id="alpha-four", name="Alpha Four")

    removed = request(app, "DELETE", "/api/strategies/alpha-four").json()["data"]

    assert request(app, "GET", "/api/strategies").json()["data"] == []
    archive = quant_ui_settings.runtime_root / removed["archivedTo"].replace("runtime/", "", 1)
    assert archive.is_dir()
    assert list(archive.glob("*.json")), "the manifest must survive in the archive"


def test_deleting_one_version_keeps_the_strategy_alive(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    first = _save(app, id="alpha-five", name="Alpha Five")
    _save(app, id="alpha-five", name="Alpha Five")

    request(app, "DELETE", f"/api/strategies/alpha-five?version={first['version']}")

    listing = request(app, "GET", "/api/strategies").json()["data"]
    assert len(listing) == 1
    assert listing[0]["versionCount"] == 1


def test_deleting_an_unknown_strategy_is_a_404_not_a_silent_success(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    assert request(app, "DELETE", "/api/strategies/does-not-exist").status_code == 404


def test_run_outputs_are_only_removed_on_an_explicit_request(quant_ui_settings) -> None:
    service = StrategyService(quant_ui_settings)
    output = quant_ui_settings.runtime_root / "reports" / "lifecycle_fixture"
    output.mkdir(parents=True)
    (output / "evidence.json").write_text("{}", encoding="utf-8")
    (quant_ui_settings.runtime_root / "strategies" / "alpha-six").mkdir(parents=True)
    (quant_ui_settings.runtime_root / "strategies" / "alpha-six" / "v1.json").write_text(
        json.dumps({"id": "alpha-six", "version": "v1", "createdAt": "2026-01-01T00:00:00+00:00",
                    "trustClass": "research_only", "draft": {"name": "Alpha Six"}}),
        encoding="utf-8",
    )
    service.register_run(
        strategy_id="alpha-six", version="v1", job_id="job_x",
        output_dir="runtime/reports/lifecycle_fixture", name="Alpha Six",
    )

    service.delete("alpha-six", delete_outputs=False)
    assert output.exists(), "a plain delete must not destroy research outputs"


def test_output_deletion_refuses_paths_outside_the_runtime_subtree(quant_ui_settings) -> None:
    service = StrategyService(quant_ui_settings)
    assert service._remove_output("/etc/passwd") is not None
    assert service._remove_output("runtime") is not None


# --- runs --------------------------------------------------------------------
def test_a_launched_run_links_strategy_version_job_and_output(quant_ui_settings) -> None:
    service = StrategyService(quant_ui_settings)

    run = service.register_run(
        strategy_id="alpha-seven", version="20260101T000000Z",
        job_id="job_abc", output_dir="runtime/reports/alpha-seven/run_1", name="Alpha Seven",
    )

    stored = service.run(run["runId"])
    assert stored["jobId"] == "job_abc"
    assert stored["strategyVersion"] == "20260101T000000Z"
    assert stored["outputDir"] == "runtime/reports/alpha-seven/run_1"


# --- result resolution -------------------------------------------------------
def _write_run(root, *, acceptance=None, governance=None, verdict=None, pipeline=None):
    root.mkdir(parents=True, exist_ok=True)
    if acceptance is not None:
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "acceptance_report.json").write_text(
            json.dumps(acceptance), encoding="utf-8"
        )
    if governance is not None:
        (root / "portfolio_search").mkdir(parents=True, exist_ok=True)
        (root / "portfolio_search" / "selection_governance.json").write_text(
            json.dumps(governance), encoding="utf-8"
        )
    if verdict is not None:
        (root / "research_verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    if pipeline is not None:
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "full_pipeline_report.json").write_text(
            json.dumps(pipeline), encoding="utf-8"
        )


def test_a_missing_run_directory_reports_absence_not_an_empty_success(quant_ui_settings) -> None:
    resolver = RunResultResolver(quant_ui_settings)

    result = resolver.resolve("runtime/reports/never_ran")

    assert result["status"] == "absent"
    assert result["issues"][0]["code"] == "output_dir_missing"


def test_a_failed_gate_is_reported_as_a_conclusion_with_its_reasons(quant_ui_settings) -> None:
    root = quant_ui_settings.runtime_root / "reports" / "gated_run"
    _write_run(
        root,
        acceptance={
            "failures": ["single_factor_dominance_too_high"],
            "gates": [
                {"name": "rank_ic_mean", "passed": True, "actual": 0.1, "threshold": "> 0.0"},
                {"name": "single_factor_dominance", "passed": False, "actual": 0.71,
                 "threshold": "<= 0.6", "reason": "single_factor_dominance_too_high"},
            ],
        },
        governance={"accepted": True, "pbo": 0.12, "dsr_probability": 0.97,
                    "spa_pvalue": 0.0, "rejection_reasons": []},
        pipeline={"QUANT_ACCEPTANCE_STATUS": "failed"},
    )

    result = RunResultResolver(quant_ui_settings).resolve(str(root))

    assert result["status"] == "complete"
    assert result["conclusion"]["outcome"] == "not_accepted"
    assert result["conclusion"]["promotable"] is False
    assert any("single_factor_dominance" in reason for reason in result["conclusion"]["reasons"])
    assert result["acceptance"]["passedCount"] == 1
    assert result["acceptance"]["totalCount"] == 2


def test_a_clean_run_is_reported_as_accepted_but_still_research_only(quant_ui_settings) -> None:
    root = quant_ui_settings.runtime_root / "reports" / "clean_run"
    _write_run(
        root,
        acceptance={"failures": [], "gates": [
            {"name": "rank_ic_mean", "passed": True, "actual": 0.1, "threshold": "> 0.0"},
        ]},
        governance={"accepted": True, "pbo": 0.1, "dsr_probability": 0.98,
                    "spa_pvalue": 0.01, "rejection_reasons": []},
        pipeline={"QUANT_ACCEPTANCE_STATUS": "passed"},
    )

    conclusion = RunResultResolver(quant_ui_settings).resolve(str(root))["conclusion"]

    assert conclusion["outcome"] == "accepted"
    assert conclusion["promotable"] is True
    assert "research" in conclusion["remediation"] or "人工" in conclusion["remediation"]


def test_a_research_rejection_is_surfaced_with_its_remediation(quant_ui_settings) -> None:
    root = quant_ui_settings.runtime_root / "reports" / "rejected_run"
    _write_run(root, verdict={
        "verdict": "rejected", "code": "overfitting_governance_rejected",
        "title": "候选组合被过拟合治理闸门否决",
        "reasons": ["pbo=0.5429 exceeds 0.2500"],
        "remediation": "减少候选数量或延长 OOS 观测窗口。",
    })

    result = RunResultResolver(quant_ui_settings).resolve(str(root))

    assert result["status"] == "rejected"
    assert result["conclusion"]["outcome"] == "rejected"
    assert result["conclusion"]["reasons"] == ["pbo=0.5429 exceeds 0.2500"]
    assert result["conclusion"]["remediation"]


def test_nav_is_read_from_the_simulator_mapping_shape(quant_ui_settings) -> None:
    root = quant_ui_settings.runtime_root / "reports" / "nav_run"
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "walk_forward_backtest.json").write_text(
        json.dumps({
            "nav": {"2026-01-05": 1_000_000.0, "2026-01-02": 900_000.0, "2026-01-06": 1_100_000.0},
            "orders": [{}, {}], "skipped_orders": [{}], "failed_orders": [],
        }),
        encoding="utf-8",
    )

    backtest = RunResultResolver(quant_ui_settings).resolve(str(root))["backtest"]

    assert backtest["navPoints"] == 3
    assert [point["date"] for point in backtest["nav"]] == ["2026-01-02", "2026-01-05", "2026-01-06"]
    assert backtest["totalReturn"] == pytest.approx(1_100_000 / 900_000 - 1, abs=1e-6)
    assert backtest["orderCount"] == 2


# --- failure diagnosis -------------------------------------------------------
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ValueError: insufficient OOS dates for nested portfolio selection", "insufficient_oos_dates"),
        ("ValueError: benchmark 000300.SH is absent from the market panel", "benchmark_absent"),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory", "out_of_memory"),
        ("ModuleNotFoundError: No module named 'lightgbm'", "dependency_missing"),
        ("FileNotFoundError: No such file or directory: 'labels.parquet'", "input_missing"),
    ],
)
def test_known_failures_are_named_with_a_remediation(line: str, expected: str) -> None:
    failure = diagnose(["some earlier output", line], exit_code=1)

    assert failure.code == expected
    assert failure.remediation, "a named failure must say what to do next"
    assert failure.log_tail, "the evidence must travel with the diagnosis"


def test_an_unknown_failure_says_so_instead_of_guessing() -> None:
    failure = diagnose(["something went sideways"], exit_code=9)

    assert failure.code == "unclassified"
    assert failure.log_tail == ["something went sideways"]


def test_a_signal_death_is_distinguished_from_an_error_exit() -> None:
    failure = diagnose(["no clue"], exit_code=-9)

    assert failure.signal == 9
    assert failure.retryable is True


def test_rich_wrapped_tracebacks_still_yield_the_exception() -> None:
    """Typer renders exceptions inside box drawing; the message must survive."""
    lines = [
        "│ ❱ 1692 │   │   raise ValueError(                                    │",
        "╰──────────────────────────────────────────────────────────────────────╯",
        "ValueError: insufficient OOS dates for nested portfolio selection and final",
        "holdout: dates=60, required=100",
    ]

    assert "insufficient OOS dates" in (extract_exception(lines) or "")


# --- comparison --------------------------------------------------------------
def _register_run(service: StrategyService, strategy_id: str, root, **artifacts) -> str:
    _write_run(root, **artifacts)
    return service.register_run(
        strategy_id=strategy_id, version="v1", job_id=f"job_{strategy_id}",
        output_dir=str(root), name=strategy_id,
    )["runId"]


def test_comparison_aligns_runs_and_marks_the_better_value(quant_ui_settings) -> None:
    service = StrategyService(quant_ui_settings)
    root = quant_ui_settings.runtime_root / "reports"
    left = _register_run(
        service, "left", root / "left",
        governance={"accepted": True, "pbo": 0.30, "dsr_probability": 0.91,
                    "spa_pvalue": 0.02, "rejection_reasons": []},
        acceptance={"failures": [], "gates": [
            {"name": "rank_ic_mean", "passed": True, "actual": 0.05, "threshold": "> 0.0"},
        ]},
        pipeline={"QUANT_ACCEPTANCE_STATUS": "passed"},
    )
    right = _register_run(
        service, "right", root / "right",
        governance={"accepted": True, "pbo": 0.10, "dsr_probability": 0.97,
                    "spa_pvalue": 0.01, "rejection_reasons": []},
        acceptance={"failures": ["x"], "gates": [
            {"name": "rank_ic_mean", "passed": False, "actual": 0.01, "threshold": "> 0.02"},
        ]},
        pipeline={"QUANT_ACCEPTANCE_STATUS": "failed"},
    )

    comparison = service.compare_runs([left, right])

    assert [item["runId"] for item in comparison["runs"]] == [left, right]
    pbo = next(item for item in comparison["metrics"] if item["key"] == "governance.pbo")
    # PBO is better when lower, so the second run wins that column.
    assert pbo["direction"] == "lower"
    assert pbo["bestIndex"] == 1
    dsr = next(item for item in comparison["metrics"] if item["key"] == "governance.dsrProbability")
    assert dsr["bestIndex"] == 1
    gate = next(item for item in comparison["gates"] if item["name"] == "rank_ic_mean")
    assert gate["values"][0]["passed"] is True
    assert gate["values"][1]["passed"] is False


def test_a_run_that_never_produced_a_field_shows_a_gap_not_a_zero(quant_ui_settings) -> None:
    """A missing measurement must never be rendered as a measured value."""
    service = StrategyService(quant_ui_settings)
    root = quant_ui_settings.runtime_root / "reports"
    complete = _register_run(
        service, "complete", root / "complete",
        governance={"accepted": True, "pbo": 0.10, "dsr_probability": 0.97,
                    "spa_pvalue": 0.0, "rejection_reasons": []},
    )
    aborted = _register_run(
        service, "aborted", root / "aborted",
        verdict={"verdict": "rejected", "code": "x", "title": "t", "reasons": []},
    )

    comparison = service.compare_runs([complete, aborted])

    pbo = next(item for item in comparison["metrics"] if item["key"] == "governance.pbo")
    assert pbo["values"][0] == 0.10
    assert pbo["values"][1] is None
    # With only one measured value there is no comparison to win.
    assert pbo["bestIndex"] is None


def test_comparison_is_bounded_and_rejects_unknown_runs(quant_ui_settings) -> None:
    service = StrategyService(quant_ui_settings)

    with pytest.raises(ValueError, match="bounded to 4"):
        service.compare_runs(["a", "b", "c", "d", "e"])
    with pytest.raises(KeyError):
        service.compare_runs(["run_does_not_exist"])
    with pytest.raises(ValueError, match="at least one"):
        service.compare_runs([])


def test_a_missing_acceptance_report_is_never_read_as_a_pass(quant_ui_settings) -> None:
    """The most dangerous failure mode: absent evidence rendered as success."""
    root = quant_ui_settings.runtime_root / "reports" / "no_gates_run"
    _write_run(root, pipeline={"TRAINING_STATUS": "validation_only"})

    result = RunResultResolver(quant_ui_settings).resolve(str(root))

    assert result["conclusion"]["outcome"] == "incomplete"
    assert result["conclusion"]["promotable"] is False
    assert any("acceptance_report.json" in reason for reason in result["conclusion"]["reasons"])


def test_governance_evidence_is_also_required_before_claiming_acceptance(
    quant_ui_settings,
) -> None:
    root = quant_ui_settings.runtime_root / "reports" / "no_governance_run"
    _write_run(
        root,
        acceptance={"failures": [], "gates": [
            {"name": "rank_ic_mean", "passed": True, "actual": 0.1, "threshold": "> 0.0"},
        ]},
        pipeline={"QUANT_ACCEPTANCE_STATUS": "passed"},
    )

    conclusion = RunResultResolver(quant_ui_settings).resolve(str(root))["conclusion"]

    assert conclusion["outcome"] == "incomplete"
    assert any("selection_governance.json" in reason for reason in conclusion["reasons"])


# --- decision council over a completed run -----------------------------------
def test_council_reviews_a_run_and_reports_absent_evidence_as_unknown(
    quant_ui_settings,
) -> None:
    """The council's core discipline: a missing artifact is never a pass."""
    from services.quant_api.services.container import ServiceContainer

    container = ServiceContainer.create(quant_ui_settings)
    root = quant_ui_settings.runtime_root / "reports" / "council_run"
    _write_run(
        root,
        acceptance={"failures": [], "gates": [
            {"name": "no_pit_violations", "passed": True, "actual": 0, "threshold": "0"},
            {"name": "no_mock_or_synthetic", "passed": True, "actual": False, "threshold": "False"},
            {"name": "training_symbols", "passed": True, "actual": 150, "threshold": ">= 2"},
        ]},
        governance={"accepted": True, "pbo": 0.05, "dsr_probability": 0.96,
                    "spa_pvalue": 0.0, "cumulative_trials": 3, "rejection_reasons": []},
        pipeline={"QUANT_ACCEPTANCE_STATUS": "passed"},
    )
    run = container.strategies.register_run(
        strategy_id="council", version="v1", job_id="job_c",
        output_dir=str(root), name="Council Run",
    )

    review = container.council.review_strategy_run(
        run["runId"], container.strategies.results, container.strategies
    )

    verdicts = {item["roleId"]: item["verdict"] for item in review["findings"]}
    assert verdicts["data_quality"] == "pass"
    assert verdicts["fusion_search"] == "pass"
    # No single_factor_dominance gate and no backtest were written, so those
    # roles must abstain rather than clear the run.
    assert verdicts["factor_integrity"] == "unknown"
    assert verdicts["execution_realism"] == "unknown"
    for finding in review["findings"]:
        assert finding["evidence"], "a verdict must name the fields it read"


def test_council_blocks_a_run_whose_gates_failed(quant_ui_settings) -> None:
    from services.quant_api.services.container import ServiceContainer

    container = ServiceContainer.create(quant_ui_settings)
    root = quant_ui_settings.runtime_root / "reports" / "council_blocked"
    _write_run(
        root,
        acceptance={"failures": ["single_factor_dominance_too_high"], "gates": [
            {"name": "no_pit_violations", "passed": True, "actual": 0, "threshold": "0"},
            {"name": "no_mock_or_synthetic", "passed": True, "actual": False, "threshold": "False"},
            {"name": "single_factor_dominance", "passed": False, "actual": 0.72,
             "threshold": "<= 0.6"},
        ]},
        governance={"accepted": True, "pbo": 0.05, "dsr_probability": 0.96,
                    "spa_pvalue": 0.0, "rejection_reasons": []},
        pipeline={"QUANT_ACCEPTANCE_STATUS": "failed"},
    )
    run = container.strategies.register_run(
        strategy_id="council-blocked", version="v1", job_id="job_cb",
        output_dir=str(root), name="Council Blocked",
    )

    review = container.council.review_strategy_run(
        run["runId"], container.strategies.results, container.strategies
    )

    verdicts = {item["roleId"]: item["verdict"] for item in review["findings"]}
    assert verdicts["factor_integrity"] == "blocked"
    assert verdicts["governance"] == "blocked"
    assert review["decision"]["state"] == "BLOCKED"


def test_council_flags_an_unimplementable_book(quant_ui_settings) -> None:
    """Most orders refused by A-share constraints means the book is on paper only."""
    from services.quant_api.services.container import ServiceContainer

    container = ServiceContainer.create(quant_ui_settings)
    root = quant_ui_settings.runtime_root / "reports" / "council_exec"
    _write_run(root, pipeline={"QUANT_ACCEPTANCE_STATUS": "passed"})
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "walk_forward_backtest.json").write_text(
        json.dumps({"nav": {}, "orders": [{}] * 28, "skipped_orders": [{}] * 644,
                    "failed_orders": []}),
        encoding="utf-8",
    )
    run = container.strategies.register_run(
        strategy_id="council-exec", version="v1", job_id="job_ce",
        output_dir=str(root), name="Council Exec",
    )

    review = container.council.review_strategy_run(
        run["runId"], container.strategies.results, container.strategies
    )

    execution = next(
        item for item in review["findings"] if item["roleId"] == "execution_realism"
    )
    assert execution["verdict"] == "warn"
    assert execution["evidence"]["skippedOrderCount"] == 644


def test_council_review_of_an_unknown_run_is_a_key_error(quant_ui_settings) -> None:
    from services.quant_api.services.container import ServiceContainer

    container = ServiceContainer.create(quant_ui_settings)
    with pytest.raises(KeyError):
        container.council.review_strategy_run(
            "run_nope", container.strategies.results, container.strategies
        )
