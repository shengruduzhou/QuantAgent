"""Round-30 ENG_RISK30: the exposure policy inside the strict backtest.

Fail-before: ``run_strict_backtest_v8`` had no ``exposure_policy`` argument
(TypeError) and the simulator had no decision record.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from quantagent.risk.exposure_policy import (
    PRESETS,
    ExposurePolicy,
    account_history_frame,
    decide_exposure,
)

AUDIT = tempfile.mkdtemp(prefix="exposure_policy_audit_")
SYMBOLS = ["600000.SH", "600001.SH", "600002.SH", "000001.SZ", "000002.SZ"]


def _panel(n_sessions: int = 180, *, crash=(60, 85), seed: int = 5, shock_after=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_sessions)
    rows = []
    for k, symbol in enumerate(SYMBOLS):
        price = 10.0 + k
        for i, date in enumerate(dates):
            r = rng.normal(0.0005, 0.012)
            if crash[0] <= i < crash[1]:
                r -= 0.012
            if shock_after is not None and date > shock_after:
                r -= 0.03  # a path change strictly after the shock date
            price = max(0.5, price * (1.0 + r))
            close = round(price, 2)
            rows.append(dict(
                symbol=symbol, trade_date=date, open=close, high=close, low=close, close=close,
                volume=5e8, amount=5e8 * close, is_suspended=False, is_st=False,
                is_limit_up=False, is_limit_down=False,
            ))
    return pd.DataFrame(rows)


def _weekly_targets(panel: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(panel["trade_date"].unique())[:-1:5]
    tw = pd.DataFrame(
        {s: [1.0 / len(SYMBOLS)] * len(dates) for s in SYMBOLS},
        index=pd.DatetimeIndex(dates),
    )
    tw.attrs["target_index_semantics"] = "signal_date"
    return tw


def _cfg() -> AShareExecutionSimulationConfig:
    return AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0, audit_log_dir=AUDIT)


def _run(panel, policy=None):
    kwargs = {} if policy is None else {"exposure_policy": policy}
    return run_strict_backtest_v8(_weekly_targets(panel), panel, config=_cfg(), **kwargs)


def test_flag_off_is_untouched_and_identity_policy_is_inert():
    panel = _panel()
    off = _run(panel)
    identity = _run(panel, PRESETS["identity"])
    pd.testing.assert_series_equal(off.nav, identity.nav, check_exact=True)
    pd.testing.assert_frame_equal(
        off.trades.drop(columns=["client_order_id"]),
        identity.trades.drop(columns=["client_order_id"]),
        check_exact=True,
    )
    assert off.exposure_decisions is None and off.account_path is None
    assert "exposure_policy" not in off.config
    assert identity.config["exposure_policy"]["name"] == "identity"
    assert (identity.exposure_decisions["multiplier"] == 1.0).all()


def test_policy_scales_the_gross_executed_next_session():
    half = ExposurePolicy(name="always_warmup_half", vol_target_annual=0.2, vol_window_sessions=500,
                          vol_min_observations=500, vol_warmup_multiplier=0.5)
    res = _run(_panel(), half)
    decisions = res.exposure_decisions
    assert len(decisions) == len(_weekly_targets(_panel()))
    assert (decisions["multiplier"] == 0.5).all()
    assert (pd.to_datetime(decisions["execution_date"]) > pd.to_datetime(decisions["decision_date"])).all()
    path = res.account_path.iloc[1:]
    frac = path["invested_value"] / path["nav"]
    assert frac.max() < 0.56  # half gross, never the full book
    assert frac.iloc[5:].min() > 0.40


def test_each_backtest_decision_equals_the_policy_on_history_up_to_the_signal_close():
    policy = PRESETS["vt20_ddl"]
    res = _run(_panel(), policy)
    path = res.account_path
    assert len(res.exposure_decisions) > 20
    for row in res.exposure_decisions.to_dict("records"):
        t = pd.Timestamp(row["decision_date"])
        replay = decide_exposure(path, policy, decision_date=t)
        assert replay.multiplier == row["multiplier"]
        assert replay.tier == row["tier"]
        # the simulator had no row after t when it decided
        assert row["observations_after_decision_ignored"] == 0
        assert replay.observations_used == row["observations_used"]


def test_a_path_change_after_t_does_not_change_decisions_up_to_t():
    base_panel = _panel()
    cut = sorted(base_panel["trade_date"].unique())[120]
    shocked_panel = _panel(shock_after=cut)
    policy = PRESETS["vt20_ddl"]
    a = _run(base_panel, policy).exposure_decisions
    b = _run(shocked_panel, policy).exposure_decisions
    a_upto = a[pd.to_datetime(a["decision_date"]) <= cut].reset_index(drop=True)
    b_upto = b[pd.to_datetime(b["decision_date"]) <= cut].reset_index(drop=True)
    assert len(a_upto) > 10
    pd.testing.assert_frame_equal(a_upto, b_upto, check_exact=True)
    # ...and the shock itself is acted on afterwards
    later = b[pd.to_datetime(b["decision_date"]) > cut]
    assert later["multiplier"].min() < a[pd.to_datetime(a["decision_date"]) > cut]["multiplier"].min()


def test_drawdown_ladder_degrosses_the_book_during_a_crash():
    res = _run(_panel(), PRESETS["vt20_ddl"])
    d = res.exposure_decisions
    assert (d["tier"] > 0).any()
    assert d["drawdown_all_time"].max() > 0.08
    assert d.loc[d["tier"] > 0, "multiplier"].max() <= 0.75
    off = _run(_panel())
    assert res.metrics.max_drawdown < off.metrics.max_drawdown


def test_inception_row_is_initial_cash_and_all_time_peak_includes_it():
    res = _run(_panel(), PRESETS["vt20"])
    first = res.account_path.iloc[0]
    assert first["nav"] == 1_000_000.0 and first["invested_value"] == 0.0
    assert pd.Timestamp(first["trade_date"]) == _weekly_targets(_panel()).index[0]


def test_written_bundle_carries_the_decision_record(tmp_path):
    res = _run(_panel(), PRESETS["vt20_ddl"])
    paths = res.write(tmp_path)
    written = pd.read_csv(paths["exposure_decisions"])
    assert list(written["multiplier"]) == list(res.exposure_decisions["multiplier"])
    assert set(written["policy_sha256"]) == {PRESETS["vt20_ddl"].sha256}
    off_paths = _run(_panel()).write(tmp_path / "off")
    assert "exposure_decisions" not in off_paths


def test_account_path_round_trips_into_the_shared_decision_function():
    res = _run(_panel(), PRESETS["vt20"])
    p = res.account_path
    rebuilt = account_history_frame(p["trade_date"], p["nav"], p["invested_value"])
    last = rebuilt["trade_date"].iloc[-1]
    assert decide_exposure(rebuilt, PRESETS["vt20"], decision_date=last).status == "ok"
    assert np.isfinite(rebuilt["nav"]).all()
    assert pytest.approx(float(res.nav.iloc[-1])) == float(p["nav"].iloc[-1])


# ------------------------------------------------------------- canonical evaluator
def _evaluate(tmp_path, monkeypatch, **kwargs):
    import scripts.baseline_protocol as bp

    panel = _panel()
    panel["amount_measured"] = True
    preds = panel[["symbol", "trade_date"]].copy()
    preds["alpha_score"] = np.arange(len(preds)) % 7
    pred_path = tmp_path / "preds.parquet"
    preds.to_parquet(pred_path)
    monkeypatch.setattr(bp, "_load_verified_panel", lambda path, s, e: (panel.copy(), {"verified": True}))
    calls: list[dict] = []
    real = bp.run_strict_backtest_v8

    def spy(*args, **kw):
        calls.append(dict(kw))
        return real(*args, **kw)

    monkeypatch.setattr(bp, "run_strict_backtest_v8", spy)
    out = bp.evaluate(
        str(pred_path), top_k=3, start="2022-01-03", end="2022-09-01", slippage_bps=8.0,
        variants=["C_flags_eligible_delay1"], panel_path="synthetic.parquet",
        save_backtest_dir=str(tmp_path / kwargs.get("tag", "run")),
        **{k: v for k, v in kwargs.items() if k != "tag"},
    )
    return out, calls


def test_canonical_evaluator_without_the_flag_passes_no_policy(tmp_path, monkeypatch):
    out, calls = _evaluate(tmp_path, monkeypatch, tag="off")
    assert "exposure_policy" not in calls[0]
    assert "exposure_policy" not in out
    assert "exposure" not in out["variants"]["C_flags_eligible_delay1"]
    assert not (tmp_path / "off" / "backtest" / "exposure_decisions.csv").exists()
    out_id, _ = _evaluate(tmp_path, monkeypatch, tag="identity", exposure_policy="identity")
    nav_off = pd.read_csv(tmp_path / "off" / "backtest" / "nav.csv")
    nav_id = pd.read_csv(tmp_path / "identity" / "backtest" / "nav.csv")
    pd.testing.assert_frame_equal(nav_off, nav_id, check_exact=True)
    assert out_id["exposure_policy"]["name"] == "identity"


def test_evaluator_flag_stamps_policy_and_writes_decisions(tmp_path, monkeypatch):
    out, calls = _evaluate(tmp_path, monkeypatch, tag="ddl", exposure_policy="vt20_ddl")
    assert calls[0]["exposure_policy"] is PRESETS["vt20_ddl"]
    assert out["exposure_policy"]["policy_sha256"] == PRESETS["vt20_ddl"].sha256
    summary = out["variants"]["C_flags_eligible_delay1"]["exposure"]
    assert summary["signal_decisions"] > 0 and summary["fail_closed_decisions"] == 0
    assert 0.0 < summary["invested_fraction_mean"] <= 1.0
    d = tmp_path / "ddl" / "backtest"
    assert (d / "exposure_decisions.csv").exists() and (d / "account_path.csv").exists()
    import json as _json

    assert _json.loads((d / "metrics.json").read_text())["exposure_policy"]["name"] == "vt20_ddl"


def test_evaluator_rejects_an_unknown_policy_before_reading_data(tmp_path):
    import scripts.baseline_protocol as bp
    from quantagent.risk.exposure_policy import ExposurePolicyError

    with pytest.raises(ExposurePolicyError):
        bp.evaluate(str(tmp_path / "missing.parquet"), top_k=1, start="2022-01-03", end="2022-02-01",
                    slippage_bps=8.0, variants=["C_flags_eligible_delay1"], exposure_policy="nope")
