"""An audit that cannot run must say so, and a label must not start before its features.

Two defects, one shape.

DEF-025 — the gate hardening was defeated at the producer boundary. Round 15 made
`no_pit_violations` and `no_mock_or_synthetic` report `unknown` when their
measurement was absent. It changed nothing on the real training path, because the
measurement was never absent: `_pit_violations` returned `0` for a frame it could
not read, `_uses_mock_or_synthetic` returned `False` for a frame with no
provenance column, `_aggregate_metrics` wrote `uses_mock_or_synthetic: False` as a
constant, and the CLI injected `pit_violation_count = 0`. The gate dutifully
recorded two *measured* passes on four fabrications.

DEF-026 — nothing had ever compared a row's availability stamp against its own
label window. The v7 builder stamped `available_at = next trading row` while
`v7_label_builder` opened the label window at `close(trade_date)`, so 100% of rows
declared themselves unusable until after they were already being scored. Because
`available_at` is the as-of join key in `merge_pit_features`, that stamp also
admitted third-party features published during the label window: measured at rank
IC +1.0000 for a feature carrying that day's return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.v7_quality_gates import (
    AUDIT_MEASURED,
    AUDIT_NOT_AUDITABLE,
    GATE_FAIL,
    GATE_PASS,
    GATE_UNKNOWN,
    V7ModelAcceptanceGateConfig,
    audit_pit,
    audit_provenance,
    evaluate_data_quality_gates,
    evaluate_label_alignment,
    evaluate_model_acceptance_gates,
)


def _panel(days: int = 30, symbols: tuple[str, ...] = ("000001.SZ", "000002.SZ", "600000.SH")):
    dates = pd.bdate_range("2026-01-05", periods=days)
    rng = np.random.default_rng(11)
    rows = []
    for symbol in symbols:
        price = 10.0
        for date in dates:
            price = max(1.0, price * (1 + float(rng.normal(0, 0.02))))
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1_000_000.0,
                    "amount": price * 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _dataset(market: pd.DataFrame | None = None) -> pd.DataFrame:
    from quantagent.data.v7_dataset_builder import build_market_features
    from quantagent.data.v7_label_builder import build_forward_return_labels

    market = _panel() if market is None else market
    features = build_market_features(market)
    labels = build_forward_return_labels(market, horizons=(1, 5)).frame
    carry = [
        column for column in labels.columns
        if column.startswith("forward_return_") or column.startswith("label_end_")
    ]
    return features.merge(labels[["symbol", "trade_date", *carry]], on=["symbol", "trade_date"], how="inner")


def _gate(report, name: str) -> dict:
    for gate in report.gates:
        if gate["name"] == name:
            return gate
    raise AssertionError(f"no gate named {name}: {[g['name'] for g in report.gates]}")


# ---------------------------------------------------------------------------
# DEF-025: an audit with nothing to read reports nothing, not zero
# ---------------------------------------------------------------------------


def test_pit_audit_declines_when_there_is_nothing_to_compare_against():
    """`available_at` alone answers no question — there must be a reference."""
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "available_at": pd.to_datetime(["2026-01-05", "2026-01-05"]),
        }
    )
    report = audit_pit(frame)
    assert report.status == AUDIT_NOT_AUDITABLE
    # The whole defect in one assertion: this used to be 0.
    assert report.violations is None


def test_pit_audit_declines_when_the_frame_has_no_availability_stamp():
    frame = pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": pd.to_datetime(["2026-01-05"])})
    assert audit_pit(frame).status == AUDIT_NOT_AUDITABLE
    assert audit_pit(frame).violations is None


def test_pit_audit_counts_violations_when_it_can():
    frame = pd.DataFrame(
        {
            "symbol": ["a", "b"],
            "available_at": pd.to_datetime(["2026-01-05", "2026-01-20"]),
            "as_of_date": pd.to_datetime(["2026-01-10", "2026-01-10"]),
        }
    )
    report = audit_pit(frame)
    assert report.status == AUDIT_MEASURED
    assert report.violations == 1
    assert report.reference_column == "as_of_date"


def test_provenance_audit_declines_when_no_column_records_the_source():
    frame = pd.DataFrame({"symbol": ["000001.SZ"], "close": [10.0]})
    report = audit_provenance(frame)
    assert report.status == AUDIT_NOT_AUDITABLE
    # This used to be False, i.e. "audited, and no synthetic data was found".
    assert report.uses_mock_or_synthetic is None


def test_provenance_audit_answers_when_a_source_column_exists():
    clean = pd.DataFrame({"source": ["tickflow", "tickflow"]})
    dirty = pd.DataFrame({"source": ["tickflow", "synthetic_v2"]})
    assert audit_provenance(clean).uses_mock_or_synthetic is False
    assert audit_provenance(dirty).uses_mock_or_synthetic is True


def test_data_quality_report_publishes_unknowns_instead_of_clean_numbers():
    dataset = _dataset()
    report = evaluate_data_quality_gates(dataset)
    assert report.metrics["pit_violation_count"] is None
    assert report.metrics["uses_mock_or_synthetic"] is None
    assert {gate["name"] for gate in report.unknowns} == {"pit_audit", "provenance_audit"}


def test_the_acceptance_gate_reports_unknown_for_an_unaudited_run():
    """End to end: nothing measured, nothing passed.

    Before the fix this exact metrics dict — which is what the training path
    produced — cleared both gates.
    """
    report = evaluate_model_acceptance_gates({}, V7ModelAcceptanceGateConfig())
    assert _gate(report, "no_pit_violations")["status"] == GATE_UNKNOWN
    assert _gate(report, "no_mock_or_synthetic")["status"] == GATE_UNKNOWN
    assert _gate(report, "label_alignment")["status"] == GATE_UNKNOWN
    assert not any(gate["status"] == GATE_PASS for gate in report.gates)


def test_producers_no_longer_supply_the_measurement_they_never_took():
    """The fabrications were two constants in two functions.

    Asserted on the values the functions return rather than on their source, so
    the test tracks what the gate actually receives.
    """
    from quantagent.cli.v7_train import _build_full_pipeline_acceptance_metrics
    from quantagent.training.v7_experiment import _aggregate_metrics

    aggregated = _aggregate_metrics(
        pd.DataFrame({"prediction": [0.1, -0.2], "forward_return_1d": [0.01, -0.02]}),
        [{"rank_ic_mean": 0.02, "net_return": 0.01, "max_drawdown": -0.05, "n_days": 10}],
        coefficients={"1": {"momentum_5d": 0.4, "intercept": 0.0}},
    )
    # Nothing in that call looked at where the rows came from, so it must not
    # claim to know. Previously: `uses_mock_or_synthetic: False`.
    assert "uses_mock_or_synthetic" not in aggregated

    built = _build_full_pipeline_acceptance_metrics(
        training_metrics={"rank_ic_mean": 0.02},
        paper_summary={},
        weight_diagnostics={},
        training_dataset=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"]}),
        predictions=pd.DataFrame({"symbol": ["000001.SZ"]}),
        benchmark_symbol=None,
    )
    # Previously injected as 0, which the gate read as a measured clean PIT audit.
    assert "pit_violation_count" not in built

    report = evaluate_model_acceptance_gates(built, V7ModelAcceptanceGateConfig())
    assert _gate(report, "no_pit_violations")["status"] == GATE_UNKNOWN
    assert _gate(report, "no_mock_or_synthetic")["status"] == GATE_UNKNOWN


# ---------------------------------------------------------------------------
# DEF-026: the label must not start before the features are available
# ---------------------------------------------------------------------------


def test_the_builder_stamps_availability_at_the_row_s_own_close():
    dataset = _dataset()
    assert (pd.to_datetime(dataset["available_at"]) == pd.to_datetime(dataset["trade_date"])).all()


def test_label_alignment_passes_on_the_dataset_the_builder_produces():
    report = evaluate_label_alignment(_dataset())
    assert report.status == GATE_PASS
    assert report.violation_rows == 0
    assert report.by_horizon["forward_return_1d"]["labelled_rows"] > 0


def test_label_alignment_fails_when_the_stamp_lands_inside_the_label_window():
    """The defect, reconstructed: the stamp the builder used to write."""
    dataset = _dataset()
    grouped = dataset.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=False)
    dataset = dataset.sort_values(["symbol", "trade_date"]).copy()
    dataset["available_at"] = grouped["trade_date"].shift(-1).values
    dataset["available_at"] = dataset["available_at"].fillna(
        dataset["trade_date"] + pd.Timedelta(days=1)
    )

    report = evaluate_label_alignment(dataset)
    assert report.status == GATE_FAIL
    assert report.violation_rows > 0
    horizon = report.by_horizon["forward_return_1d"]
    # Not merely late — late *and inside the window the label measures*, which is
    # what makes it a look-ahead rather than a bookkeeping quirk.
    assert horizon["violation_rows_inside_window"] == horizon["violation_rows"]


def test_a_delayed_entry_convention_passes_when_it_declares_itself():
    """`gold_bridge.LABEL_CONVENTION` enters at close(t+1). That is legitimate —
    the audit only requires that the entry instant be published, not assumed."""
    dataset = _dataset()
    dataset["available_at"] = pd.to_datetime(dataset["trade_date"]) + pd.Timedelta(days=1)

    assert evaluate_label_alignment(dataset).status == GATE_FAIL

    dataset["label_entry_at"] = pd.to_datetime(dataset["trade_date"]) + pd.Timedelta(days=1)
    report = evaluate_label_alignment(dataset)
    assert report.status == GATE_PASS
    assert report.entry_basis == "label_entry_at"


@pytest.mark.parametrize(
    "drop",
    ["available_at", "forward_return_"],
)
def test_label_alignment_is_unknown_when_it_cannot_be_checked(drop: str):
    dataset = _dataset()
    dataset = dataset.drop(columns=[c for c in dataset.columns if c.startswith(drop)])
    report = evaluate_label_alignment(dataset)
    assert report.status == GATE_UNKNOWN
    assert report.violation_rows == 0


def test_a_feature_published_during_the_label_window_no_longer_joins_onto_the_row():
    """The measured leak, and the assertion that closes it.

    `merge_pit_features` joins extras as-of the row's `available_at`. With the
    stamp one session late, an extra published on `t+1` — honestly stamped, no
    dishonesty anywhere in the input — landed on the row scored on the `t -> t+1`
    return and reproduced the label exactly: rank IC +1.0000.
    """
    from quantagent.data.v7_dataset_builder import build_market_features, merge_pit_features
    from quantagent.data.v7_label_builder import build_forward_return_labels

    market = _panel()
    features = build_market_features(market)
    labels = build_forward_return_labels(market, horizons=(1,)).frame

    # A feature knowable only at each session's close, carrying that session's return.
    extras = []
    for symbol, frame in market.sort_values(["symbol", "trade_date"]).groupby("symbol"):
        frame = frame.reset_index(drop=True)
        returns = frame["close"].pct_change()
        for index in range(1, len(frame)):
            extras.append(
                {
                    "symbol": symbol,
                    "available_at": frame.loc[index, "trade_date"],
                    "news_score": float(returns.iloc[index]),
                }
            )

    merged = merge_pit_features(features, pd.DataFrame(extras), prefix="")
    merged = merged.merge(
        labels[["symbol", "trade_date", "forward_return_1d"]],
        on=["symbol", "trade_date"],
        how="inner",
    )
    valid = merged.dropna(subset=["news_score", "forward_return_1d"])
    per_date = valid.groupby("trade_date").apply(
        lambda f: f["news_score"].rank().corr(f["forward_return_1d"].rank())
        if len(f) >= 2 else np.nan,
        include_groups=False,
    ).dropna()

    assert len(per_date) > 5
    assert abs(float(per_date.mean())) < 0.9, (
        "an extra published during the label window is reproducing the label"
    )


def test_the_acceptance_gate_blocks_a_run_whose_labels_are_misaligned():
    metrics = {"label_alignment_status": GATE_FAIL, "label_alignment_violation_rows": 12_345}
    report = evaluate_model_acceptance_gates(metrics, V7ModelAcceptanceGateConfig())
    gate = _gate(report, "label_alignment")
    assert gate["status"] == GATE_FAIL
    assert gate["reason"] == "label_window_opens_before_features_are_available"
    assert "label_window_opens_before_features_are_available" in report.failures


def test_a_configuration_may_waive_the_check_but_never_silently():
    report = evaluate_model_acceptance_gates(
        {}, V7ModelAcceptanceGateConfig(require_label_alignment_check=False)
    )
    gate = _gate(report, "label_alignment")
    assert gate["status"] == GATE_PASS
    # Waived is not the same statement as measured-and-clean.
    assert gate["reason"] == "waived_by_configuration"
    assert gate["required"] is False
