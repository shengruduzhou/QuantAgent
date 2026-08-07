"""A gate must never invent the measurement it is judging.

`excess_return_after_costs` is only defined against a benchmark. When a run had
no benchmark, the gate coerced the missing value to `0.0`, compared `0.0 > 0.0`,
and recorded `excess_return_after_costs_failed` with `actual: 0.0`. An operator
reading that report saw a run that had *measured* zero alpha. Nothing had been
measured at all.

Missing evidence is `unknown`: never a pass, and distinguishable from a measured
failure.
"""

from __future__ import annotations

import pytest

from quantagent.data.v7_quality_gates import (
    GATE_FAIL,
    GATE_PASS,
    GATE_UNKNOWN,
    V7ModelAcceptanceGateConfig,
    evaluate_model_acceptance_gates,
)


def _metrics(**overrides):
    base = {
        "rank_ic_mean": 0.05,
        "rank_ic_stability": 0.5,
        "turnover_adjusted_net_return": 0.08,
        "max_drawdown": -0.03,
        "single_factor_dominance": 0.4,
        "adverse_regime_passed": True,
        "uses_mock_or_synthetic": False,
        "pit_violation_count": 0,
        "selection_pressure_min": 8.0,
        "training_dataset_symbol_count": 400,
        "prediction_symbol_count": 356,
        "effective_universe_min": 120,
        # A complete run has checked for survivorship. Absent, the gate reports
        # unknown — which is the point of the gate, and is asserted separately.
        "survivorship_status": "pass",
        "survivorship_delisted_symbols": 47,
        "survivorship_unknown_sessions": 0,
        # Likewise for label alignment. Every test below that asserts "a complete
        # run passes" was passing *without* this key, i.e. while the gate had no
        # evidence that the labels started after the features were available —
        # the same way the survivorship tests passed before it became a gate.
        "label_alignment_status": "pass",
        "label_alignment_violation_rows": 0,
    }
    base.update(overrides)
    return base


def _config(**overrides):
    base = dict(
        require_paper_report=False,
        require_benchmark=False,
        min_training_symbols=2,
        min_prediction_symbols=1,
        min_effective_universe_by_date=1,
    )
    base.update(overrides)
    return V7ModelAcceptanceGateConfig(**base)


def _gate(report, name):
    return next(gate for gate in report.gates if gate["name"] == name)


def test_missing_benchmark_makes_excess_return_unknown_not_zero():
    report = evaluate_model_acceptance_gates(_metrics(), _config())
    gate = _gate(report, "excess_return_after_costs")

    assert gate["status"] == GATE_UNKNOWN
    # The old behaviour: actual == 0.0, reason == "..._failed".
    assert gate["actual"] is None
    assert gate["reason"] == "excess_return_after_costs_unknown"
    assert gate["detail"]


def test_an_unknown_gate_never_counts_as_passed():
    report = evaluate_model_acceptance_gates(_metrics(), _config())

    assert report.passed is False
    assert report.has_unknowns
    assert "excess_return_after_costs" in [g["name"] for g in report.unknowns]
    assert "excess_return_after_costs_unknown" in report.failures
    # ...and it is not reported as a measured failure.
    assert "excess_return_after_costs_failed" not in report.failures


def test_a_measured_negative_excess_is_a_failure_not_an_unknown():
    report = evaluate_model_acceptance_gates(
        _metrics(excess_return_after_costs=-0.04), _config()
    )
    gate = _gate(report, "excess_return_after_costs")

    assert gate["status"] == GATE_FAIL
    assert gate["actual"] == -0.04
    assert gate["reason"] == "excess_return_after_costs_failed"
    assert not report.has_unknowns


def test_a_measured_positive_excess_passes():
    report = evaluate_model_acceptance_gates(
        _metrics(excess_return_after_costs=0.04), _config()
    )
    gate = _gate(report, "excess_return_after_costs")

    assert gate["status"] == GATE_PASS
    assert gate["actual"] == 0.04
    assert report.passed is True
    assert not report.has_unknowns


def test_report_serialisation_carries_the_unknown_list():
    payload = evaluate_model_acceptance_gates(_metrics(), _config()).to_dict()

    assert payload["unknowns"] == ["excess_return_after_costs"]
    assert payload["passed"] is False
    statuses = {gate["name"]: gate["status"] for gate in payload["gates"]}
    assert statuses["excess_return_after_costs"] == GATE_UNKNOWN
    # Every gate now carries an explicit status, not just a boolean.
    assert all(status in {GATE_PASS, GATE_FAIL, GATE_UNKNOWN} for status in statuses.values())


# --------------------------------------------------------------------------
# DEF-023: no gate may pass on evidence that does not exist
# --------------------------------------------------------------------------
def _permissive_config():
    """Every optional requirement switched off, so only real gates remain."""
    return _config(require_adverse_regime=False)


def test_no_gate_passes_on_an_empty_metrics_dict():
    """The measurement. Four gates used to pass on nothing at all.

    `max_drawdown` (abs(0.0) <= any limit), `single_factor_dominance`
    (0.0 <= any limit), `no_mock_or_synthetic` (absent -> False -> "no synthetic
    data") and `no_pit_violations` (absent -> 0 -> "no look-ahead"). The last two
    are the ones that matter most: a run that was never audited was recorded as
    clean.
    """
    report = evaluate_model_acceptance_gates({}, _permissive_config())

    measured_passes = [
        gate["name"] for gate in report.gates
        if gate["status"] == GATE_PASS and gate.get("required") is not False
    ]
    assert measured_passes == [], (
        f"gates passed with no evidence: {measured_passes}"
    )
    assert report.passed is False
    assert report.has_unknowns


@pytest.mark.parametrize(
    "gate_name",
    [
        "max_drawdown",
        "single_factor_dominance",
        "no_mock_or_synthetic",
        "no_pit_violations",
        "rank_ic_mean",
        "rank_ic_stability",
        "turnover_adjusted_net_return",
        "selection_pressure",
        "training_symbols",
        "prediction_symbols",
        "effective_universe_by_date",
    ],
)
def test_each_measurement_gate_reports_unknown_when_its_metric_is_absent(gate_name):
    report = evaluate_model_acceptance_gates({}, _permissive_config())
    gate = _gate(report, gate_name)

    assert gate["status"] == GATE_UNKNOWN, f"{gate_name} did not report unknown"
    assert gate["actual"] is None, f"{gate_name} reported a fabricated measurement"
    assert gate["passed"] is False
    assert gate["reason"].endswith("_unknown")
    assert gate["detail"], "an unknown gate must say what to do about it"


def test_a_missing_pit_audit_is_not_the_same_as_a_clean_one():
    """The single claim this programme most needs earned rather than defaulted."""
    never_audited = _gate(
        evaluate_model_acceptance_gates({}, _permissive_config()), "no_pit_violations"
    )
    audited_clean = _gate(
        evaluate_model_acceptance_gates(
            _metrics(pit_violation_count=0), _permissive_config()
        ),
        "no_pit_violations",
    )
    audited_dirty = _gate(
        evaluate_model_acceptance_gates(
            _metrics(pit_violation_count=3), _permissive_config()
        ),
        "no_pit_violations",
    )

    assert never_audited["status"] == GATE_UNKNOWN
    assert audited_clean["status"] == GATE_PASS and audited_clean["actual"] == 0
    assert audited_dirty["status"] == GATE_FAIL and audited_dirty["actual"] == 3
    # Three distinct outcomes, three distinct records.
    assert len({g["status"] for g in (never_audited, audited_clean, audited_dirty)}) == 3


def test_a_missing_synthetic_data_audit_is_not_the_same_as_a_clean_one():
    never_audited = _gate(
        evaluate_model_acceptance_gates({}, _permissive_config()), "no_mock_or_synthetic"
    )
    audited_clean = _gate(
        evaluate_model_acceptance_gates(
            _metrics(uses_mock_or_synthetic=False), _permissive_config()
        ),
        "no_mock_or_synthetic",
    )

    assert never_audited["status"] == GATE_UNKNOWN
    assert audited_clean["status"] == GATE_PASS
    assert audited_clean["actual"] is False


def test_a_missing_drawdown_is_not_an_acceptable_drawdown():
    """`abs(0.0) <= 0.1` passed, so an unmeasured run cleared the drawdown limit."""
    never_measured = _gate(
        evaluate_model_acceptance_gates({}, _permissive_config()), "max_drawdown"
    )
    measured_fine = _gate(
        evaluate_model_acceptance_gates(_metrics(max_drawdown=-0.03), _permissive_config()),
        "max_drawdown",
    )
    measured_breach = _gate(
        evaluate_model_acceptance_gates(_metrics(max_drawdown=-0.50), _permissive_config()),
        "max_drawdown",
    )

    assert never_measured["status"] == GATE_UNKNOWN
    assert measured_fine["status"] == GATE_PASS
    assert measured_breach["status"] == GATE_FAIL


def test_a_waived_requirement_is_distinguishable_from_a_measured_pass():
    """"passed" implies a threshold was cleared. A switched-off check cleared nothing."""
    report = evaluate_model_acceptance_gates(_metrics(), _config())

    waived = [g for g in report.gates if g.get("required") is False]
    assert {g["name"] for g in waived} >= {"paper_report", "benchmark"}
    for gate in waived:
        assert gate["reason"] == "waived_by_configuration"
        assert "not required" in str(gate["threshold"])

    measured = _gate(report, "rank_ic_mean")
    assert measured["reason"] == "passed"
    assert measured.get("required") is not False


def test_a_full_set_of_measurements_still_passes():
    """The strictness must not make a genuinely complete run unpassable."""
    report = evaluate_model_acceptance_gates(
        _metrics(
            sharpe=1.5,
            excess_return_after_costs=0.05,
            benchmark_symbol="000300.SH",
            uses_mock_or_synthetic=False,
            pit_violation_count=0,
        ),
        _config(require_adverse_regime=True),
    )

    assert report.has_unknowns is False, [g["name"] for g in report.unknowns]
    assert report.passed is True, report.failures


# --------------------------------------------------------------------------
# M5-02 / DEF-024: survivorship is enforced, and unknown never passes
# --------------------------------------------------------------------------
def test_a_run_that_never_checked_survivorship_reports_unknown():
    """"We did not check for survivorship bias" is not "there is none"."""
    report = evaluate_model_acceptance_gates({}, _permissive_config())
    gate = _gate(report, "survivorship")

    assert gate["status"] == GATE_UNKNOWN
    assert gate["actual"] is None
    assert gate["passed"] is False
    assert "没有检查不等于没有偏差" in gate["detail"]


def test_an_uncheckable_panel_reports_unknown_with_its_counts():
    report = evaluate_model_acceptance_gates(
        _metrics(
            survivorship_status=GATE_UNKNOWN,
            survivorship_unknown_sessions=1_284,
            survivorship_delisted_symbols=0,
        ),
        _permissive_config(),
    )
    gate = _gate(report, "survivorship")

    assert gate["status"] == GATE_UNKNOWN
    assert gate["survivorship_unknown_sessions"] == 1_284
    assert gate["survivorship_delisted_symbols"] == 0
    assert report.passed is False


def test_a_checked_clean_panel_passes_and_a_biased_one_fails():
    clean = _gate(evaluate_model_acceptance_gates(_metrics(), _permissive_config()), "survivorship")
    biased = _gate(
        evaluate_model_acceptance_gates(
            _metrics(survivorship_status=GATE_FAIL), _permissive_config()
        ),
        "survivorship",
    )

    assert clean["status"] == GATE_PASS
    assert clean["survivorship_delisted_symbols"] == 47
    assert biased["status"] == GATE_FAIL
    assert biased["reason"] == "survivorship_bias_present"


def test_the_survivorship_check_can_be_waived_but_says_so():
    report = evaluate_model_acceptance_gates({}, _config(require_survivorship_check=False))
    gate = _gate(report, "survivorship")

    assert gate["status"] == GATE_PASS
    assert gate["required"] is False
    assert gate["reason"] == "waived_by_configuration"


# -- the panel-level judgement -----------------------------------------------
def _masked(rows, master_rows, *, sessions=25, stop_at_delisting=False):
    """Build a masked panel.

    `stop_at_delisting` is what a correctly-built panel does: a security's bars end
    when it stops trading. Left off by default so the fixtures that deliberately
    construct the broken shape stay explicit about it.
    """
    import pandas as pd

    from quantagent.data.ashare.gold_bridge import build_masks

    master = pd.DataFrame(master_rows)
    dates = pd.bdate_range("2026-01-05", periods=sessions)
    delisting = {
        str(row["symbol"]): pd.Timestamp(row["delisting_date"])
        for row in master_rows
        if row.get("delisting_date")
    }
    panel = pd.DataFrame(
        [
            {"symbol": s, "trade_date": d}
            for s in master["symbol"]
            for d in dates
            if not (stop_at_delisting and s in delisting and d >= delisting[s])
        ]
    )
    return build_masks(panel, master=master, st_available=rows)


def test_a_known_delisted_name_with_no_date_makes_survivorship_unknown():
    """DEF-024, measured: it used to contribute as many sessions as a live name."""
    from quantagent.data.ashare.gold_bridge import MASK_UNKNOWN
    from quantagent.data.v7_quality_gates import evaluate_survivorship

    masked = _masked(False, [
        {"symbol": "600001.SH", "listing_date": "2010-01-04",
         "delisting_date": None, "status": "listed"},
        {"symbol": "600003.SH", "listing_date": "2010-01-04",
         "delisting_date": None, "status": "delisted"},
    ])

    dead = masked[masked["symbol"] == "600003.SH"]
    assert (dead["mask_post_delisting"] == MASK_UNKNOWN).all(), (
        "a name known to be delisted with no date was masked as still trading"
    )

    report = evaluate_survivorship(masked)
    assert report.status == GATE_UNKNOWN
    assert report.unknown_sessions == len(dead)
    assert "600003.SH" in report.unknown_symbols
    assert "查不了" in report.detail


def test_a_panel_that_keeps_trading_a_delisted_name_fails():
    """This test used to assert `pass`, and that was the defect (DEF-028).

    The old audit counted a delisted name as "present" only if it had a row with
    `mask_post_delisting == TRUE` — i.e. a bar *after* it stopped existing. So the
    only way to construct a passing panel was to build a broken one, and the
    fixture below does exactly that: 600002.SH is delisted on 2026-01-20 and the
    panel keeps printing bars for it afterwards. That is a real error, and it is now
    reported as one.
    """
    from quantagent.data.v7_quality_gates import evaluate_survivorship

    masked = _masked(True, [
        {"symbol": "600001.SH", "listing_date": "2010-01-04",
         "delisting_date": None, "status": "listed"},
        {"symbol": "600002.SH", "listing_date": "2010-01-04",
         "delisting_date": "2026-01-20", "status": "delisted"},
    ])

    report = evaluate_survivorship(masked)

    assert report.status == GATE_FAIL
    assert report.unknown_sessions == 0
    assert "退市日之后" in report.detail
    # The dead name's post-delisting sessions are still excluded from training —
    # the mask is right, it is the panel that should not contain those rows.
    dead = masked[masked["symbol"] == "600002.SH"]
    assert dead["eligible_for_training"].sum() < len(dead)


def test_a_panel_that_stops_a_delisted_name_on_time_passes():
    """The healthy shape, which the old audit could not express.

    A panel that includes dead names *and* stops each at its delisting date has
    zero `mask_post_delisting == TRUE` rows — by construction. The old audit read
    that as "no delisted names at all, the signature of survivorship bias" and
    returned `unknown`. Measured on the shipped full-universe gold, which has
    exactly this shape: 261 dead names, 424,662 sessions, not one bar past a
    delisting date.
    """
    from quantagent.data.ashare.gold_bridge import MASK_FALSE
    from quantagent.data.v7_quality_gates import evaluate_survivorship

    masked = _masked(True, [
        {"symbol": "600001.SH", "listing_date": "2010-01-04",
         "delisting_date": None, "status": "listed"},
        {"symbol": "600002.SH", "listing_date": "2010-01-04",
         "delisting_date": "2026-01-20", "status": "delisted"},
    ], stop_at_delisting=True)

    assert (masked["mask_post_delisting"] == MASK_FALSE).all()
    report = evaluate_survivorship(masked)
    assert report.status == GATE_PASS
    assert report.delisted_symbols == 1
    assert report.unknown_sessions == 0


def test_a_panel_with_no_delisted_names_at_all_is_reported_unknown():
    """For a full-universe panel that is itself the signature of the bias."""
    from quantagent.data.v7_quality_gates import evaluate_survivorship

    masked = _masked(True, [
        {"symbol": f"60000{i}.SH", "listing_date": "2010-01-04",
         "delisting_date": None, "status": "listed"}
        for i in range(1, 4)
    ])

    report = evaluate_survivorship(masked)

    assert report.status == GATE_UNKNOWN
    assert report.delisted_symbols == 0
    assert "一个退市标的都没有" in report.detail


def test_an_unmasked_panel_cannot_be_judged():
    """Refuses rather than reporting a survivorship verdict it cannot support."""
    import pandas as pd

    from quantagent.data.v7_quality_gates import evaluate_survivorship

    report = evaluate_survivorship(pd.DataFrame({"symbol": ["600001.SH"]}))

    assert report.status == GATE_UNKNOWN
    assert "build_masks" in report.detail


def test_the_report_feeds_the_gate_without_hand_copying():
    from quantagent.data.v7_quality_gates import evaluate_survivorship

    masked = _masked(True, [
        {"symbol": "600001.SH", "listing_date": "2010-01-04",
         "delisting_date": None, "status": "listed"},
        {"symbol": "600002.SH", "listing_date": "2010-01-04",
         "delisting_date": "2026-01-20", "status": "delisted"},
    ], stop_at_delisting=True)
    survivorship = evaluate_survivorship(masked)

    report = evaluate_model_acceptance_gates(
        _metrics(excess_return_after_costs=0.05, **survivorship.as_metrics()),
        _permissive_config(),
    )

    assert _gate(report, "survivorship")["status"] == GATE_PASS
    assert report.has_unknowns is False, [g["name"] for g in report.unknowns]
