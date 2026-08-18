"""A tradability filter that could not run must not report as enabled.

Round 21 / R3b finding.  `_TRADABILITY_CONSTRAINTS` skipped any filter whose
input column was absent from the panel (`if column not in merged.columns:
continue`), while `diagnostics["config"]` kept publishing the REQUESTED value.

That matters because the certified full-universe panel
(`adjusted_market_panel.parquet`) carries no tradability flags at all, and it is
the UI default. So an ST, suspended or limit-locked name entered top-k, the run
reported `rejected=0`, and the diagnostics said `block_st=True`. The report did
not merely omit the gap — it asserted the opposite.

Same defect class as DEF-033 in the execution constraint DSL: an unmeasurable
limit reading as a satisfied one.
"""

from __future__ import annotations

import pandas as pd

from quantagent.portfolio.v7_target_weights import (
    V7TargetWeightsConfig,
    build_v7_target_weights,
)

DATES = pd.to_datetime(["2026-08-17", "2026-08-18"])
SYMBOLS = ["600000.SH", "600001.SH", "600002.SH", "600003.SH",
           "600004.SH", "600005.SH", "600006.SH", "600007.SH"]


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": d, "symbol": s, "alpha_score": float(len(SYMBOLS) - i)}
            for d in DATES
            for i, s in enumerate(SYMBOLS)
        ]
    )


def _market(*, with_flags: bool) -> pd.DataFrame:
    rows = []
    for d in DATES:
        for i, s in enumerate(SYMBOLS):
            row = {
                "trade_date": d, "symbol": s, "close": 10.0,
                "amount": 5e8, "volume": 1e7,
            }
            if with_flags:
                # The top-ranked name is untradable on every count.
                row["is_st"] = i == 0
                row["is_suspended"] = i == 0
                row["is_limit_up"] = i == 0
                row["is_limit_down"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def _config() -> V7TargetWeightsConfig:
    return V7TargetWeightsConfig(
        selection_mode="top_k",
        top_k=2,
        top_k_ratio=None,
        fail_if_top_k_covers_universe=False,
        min_selection_pressure=0.0,
        block_st=True,
    )


def _run(with_flags: bool):
    result = build_v7_target_weights(
        _predictions(), _market(with_flags=with_flags), config=_config()
    )
    return result.target_weights, result.diagnostics


def test_missing_flag_columns_are_reported_as_unenforced() -> None:
    _, diagnostics = _run(with_flags=False)

    unenforced = {u["constraint"] for u in diagnostics["tradability_unenforced"]}
    assert "block_st" in unenforced
    assert all(u["reason"] == "column_absent" for u in diagnostics["tradability_unenforced"])


def test_effective_config_does_not_claim_a_filter_that_never_ran() -> None:
    _, diagnostics = _run(with_flags=False)

    assert diagnostics["config"]["block_st"] is False, (
        "the effective config claimed a filter that had no input column"
    )
    # The ask is preserved separately so the difference is auditable.
    assert diagnostics["config_requested"]["block_st"] is True


def test_present_flag_columns_are_reported_as_enforced() -> None:
    _, diagnostics = _run(with_flags=True)

    assert "block_st" in diagnostics["tradability_enforced"]
    assert diagnostics["tradability_unenforced"] == []
    assert diagnostics["config"]["block_st"] is True


def test_an_enforced_filter_actually_removes_the_name() -> None:
    weights, diagnostics = _run(with_flags=True)

    reasons = {r["reason"] for r in diagnostics["rejected"]}
    assert reasons, "an enforced filter that blocks nothing recorded no rejection"
    if "600000.SH" in weights.columns:
        assert weights["600000.SH"].abs().sum() == 0.0


def test_enforced_and_unenforced_are_never_the_same_fact() -> None:
    """"Nothing was blocked" must not be reachable from "nothing was checked"."""
    _, without = _run(with_flags=False)
    _, with_flags = _run(with_flags=True)

    assert without["tradability_enforced"] != with_flags["tradability_enforced"]
    assert without["tradability_unenforced"] and not with_flags["tradability_unenforced"]
