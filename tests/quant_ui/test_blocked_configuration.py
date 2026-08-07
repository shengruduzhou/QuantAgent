"""A run that cannot answer its own question is *blocked*, not rejected.

`run-full-real-training-v7` used to discover an impossible fold budget in the
portfolio stage — after the whole walk-forward had been paid for — and report it
through the research-rejection path. An operator reading the task centre saw
"研究闸门否决了该候选", which says a hypothesis was tested and refused. Nothing
had been tested: the configuration could never have produced the OOS days its
own selection protocol required.

These cover the three hops that keep the two apart: the verdict object, the CLI
exit status, and the status the job layer files.
"""

from __future__ import annotations

import json

from quantagent.research.verdict import (
    CONFIGURATION_BLOCKED_EXIT_CODE,
    RESEARCH_REJECTED_EXIT_CODE,
    RETURN_DIFFERENCING_DAYS,
    block_infeasible_oos_budget,
    reject_insufficient_oos,
    required_oos_days,
)
from quantagent.training.splitters import WalkForwardSplitConfig, plan_walk_forward
from services.quant_api.services.jobs import TERMINAL_STATUSES, TERMINAL_VERDICTS


def _blocked(**overrides):
    payload = dict(
        achievable_oos_days=80,
        requested_splits=5,
        achievable_splits=4,
        valid_size_days=20,
        min_selection_days=80,
        min_holdout_days=20,
        trading_days_available=300,
        trading_days_required=345,
    )
    payload.update(overrides)
    return block_infeasible_oos_budget(**payload)


def test_blocked_and_rejected_are_different_verdicts_and_exit_codes():
    blocked = _blocked()
    rejected = reject_insufficient_oos(
        observed_days=80, min_selection_days=80, min_holdout_days=20,
    )

    assert blocked.verdict == "blocked"
    assert rejected.verdict == "rejected"
    assert blocked.to_dict()["verdict"] == "blocked"
    assert rejected.to_dict()["verdict"] == "rejected"
    assert CONFIGURATION_BLOCKED_EXIT_CODE != RESEARCH_REJECTED_EXIT_CODE


def test_a_blocked_verdict_states_the_shortfall_and_how_to_clear_it():
    blocked = _blocked()
    payload = blocked.to_dict()

    assert payload["stage"] == "preflight"
    assert payload["metrics"]["requestedSplits"] == 5
    assert payload["metrics"]["achievableSplits"] == 4
    # 80 + 20 + 1 day consumed by NAV differencing.
    assert payload["metrics"]["requiredOosDays"] == 101
    # 101 required / 20 per fold -> the operator needs 6 folds.
    assert payload["metrics"]["minimumSplits"] == 6
    assert payload["remediation"]
    # The panel-span figures have to travel with it: raising nSplits does not
    # help when the data itself is too short.
    assert payload["metrics"]["tradingDaysAvailable"] == 300
    assert payload["metrics"]["tradingDaysRequired"] == 345


def test_remediation_distinguishes_a_small_request_from_a_short_panel():
    """Telling an operator to raise nSplits is wrong when the data is the limit."""
    # The panel is long; the operator simply asked for too few folds.
    config_bound = _blocked(
        requested_splits=2, achievable_splits=2, achievable_oos_days=40,
        trading_days_available=2196, trading_days_required=345,
    )
    assert "nSplits 提高" in config_bound.remediation
    assert "数据本身够用" in config_bound.remediation
    assert "supports only" not in config_bound.reasons[0]

    # The request is right but the panel cannot seat it.
    data_bound = _blocked(
        requested_splits=5, achievable_splits=2, achievable_oos_days=40,
        trading_days_available=300, trading_days_required=345,
    )
    assert "数据跨度不足" in data_bound.remediation
    assert "提高 nSplits 不会有帮助" in data_bound.remediation
    assert "supports only 2 folds" in data_bound.reasons[0]


def test_a_blocked_verdict_is_persisted_where_the_run_writes_its_evidence(tmp_path):
    path = _blocked().persist(tmp_path)

    assert path is not None
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["verdict"] == "blocked"
    assert stored["code"] == "infeasible_oos_budget"


def test_the_job_layer_can_file_a_blocked_run_as_terminal():
    assert "blocked" in TERMINAL_STATUSES
    assert TERMINAL_VERDICTS == {"rejected", "blocked"}


def test_the_budget_reserves_the_day_nav_differencing_consumes():
    """Governance counts daily returns; the splitter counts trading days.

    `nav.pct_change().dropna()` drops the first observation, so a segment of 80
    trading days yields 79 returns. Counting both in "days" made a 100-day OOS
    span look sufficient for an 80+20 protocol; the run then aborted in
    governance with `observed=79, required=80` after the entire portfolio search
    had been evaluated.
    """
    assert RETURN_DIFFERENCING_DAYS == 1
    assert required_oos_days(80, 20) == 101
    assert required_oos_days(5, 5) == 11


def test_the_preflight_plan_agrees_with_the_verdict_it_raises():
    """The blocked message quotes the same fold count the splitter would use."""
    cfg = WalkForwardSplitConfig(
        n_splits=5, valid_size_days=20, min_train_days=120,
        embargo_days=5, purge_days=20, mode="rolling",
    )
    plan = plan_walk_forward(200, cfg)
    assert plan.achievable_splits == 2  # (200 - 145) // 20

    blocked = _blocked(
        achievable_splits=plan.achievable_splits,
        achievable_oos_days=plan.oos_days,
    )
    assert blocked.to_dict()["metrics"]["achievableOosDays"] == 40
