"""The panel audit tells raw from qfq-as-of-fetch and lots from shares.

Inputs are the recorded Round-29 vendor responses and certified U0 raw rows in
``fixtures/akshare_units_r29``. The two adjustment-step cases apply a known
factor to those real raw closes so the expected basis is known exactly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from quantagent.data.ashare.panel_unit_audit import (
    BASIS_CONSTANT,
    BASIS_QFQ_MULT,
    BASIS_QFQ_SUB,
    BASIS_RAW,
    audit_market_panel,
    classify_adjustment_basis,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "akshare_units_r29"


def _u0() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "u0_daily_bars_raw__2024Q1.csv")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _as_panel(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame[["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]].copy()
    out["source"] = source
    out["point_in_time_valid"] = True
    return out


def test_raw_panel_is_raw_and_its_pit_claim_is_not_refuted():
    u0 = _u0()
    report = audit_market_panel(_as_panel(u0, "u0_copy"), u0, None)
    row = report["by_source"]["u0_copy"]
    assert row["vwap_in_range_rate_x1"] == 1.0
    assert row["close_match_rate_vs_u0"] == 1.0
    assert row["rows_by_adjustment_basis"] == {BASIS_RAW: 290}
    assert report["point_in_time_claim"]["verdict"] == "NOT_REFUTED"


def test_vendor_qfq_prices_with_raw_volume_refute_the_pit_stamp():
    sina_qfq = pd.read_csv(FIXTURES / "ak1_18_60__sina__600519.SH__qfq.csv")
    sina_qfq = sina_qfq.rename(columns={"date": "trade_date"}).assign(symbol="600519.SH")
    sina_qfq["trade_date"] = pd.to_datetime(sina_qfq["trade_date"])
    report = audit_market_panel(_as_panel(sina_qfq, "akshare:sina"), _u0(), None)
    row = report["by_source"]["akshare:sina"]
    assert row["close_match_rate_vs_u0"] == 0.0
    assert row["vwap_in_range_rate_x1"] < 0.05  # raw CNY turnover vs qfq prices
    assert abs(row["median_volume_ratio_vs_u0"] - 1.0) < 1e-4  # volume is raw shares
    assert row["rows_by_adjustment_basis"] == {BASIS_CONSTANT: 58}  # no ex-date in Q1
    claim = report["point_in_time_claim"]
    assert claim["verdict"] == "REFUTED"
    assert claim["rows_stamped_but_non_raw_prices"] == 58


def test_lots_labelled_as_shares_show_up_as_a_0_01_volume_ratio():
    em = pd.read_csv(FIXTURES / "ak1_18_60__east_money__600519.SH__none.csv")
    em = em.rename(columns={"日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
                            "收盘": "close", "成交量": "volume", "成交额": "amount"})
    em["symbol"] = "600519.SH"
    em["trade_date"] = pd.to_datetime(em["trade_date"])
    report = audit_market_panel(_as_panel(em, "akshare:east_money"), _u0(), None)
    row = report["by_source"]["akshare:east_money"]
    assert row["volume_ratio_share_near_0_01"] == 1.0
    assert row["vwap_in_range_rate_x1"] == 0.0
    assert row["vwap_in_range_rate_if_x100"] == 1.0
    assert row["close_match_rate_vs_u0"] == 1.0
    assert row["rows_by_adjustment_basis"] == {BASIS_RAW: 58}


def _with_ex_date(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    u0 = _u0()
    u0 = u0[u0["symbol"] == "000001.SZ"].reset_index(drop=True)
    ex = pd.Timestamp("2024-02-01")
    before = u0["trade_date"] < ex
    panel = u0.copy()
    if kind == "mult":
        panel.loc[before, ["open", "high", "low", "close"]] *= 0.95
        panel.loc[~before, ["open", "high", "low", "close"]] *= 0.99
    else:
        panel.loc[before, ["open", "high", "low", "close"]] -= 0.50
        panel.loc[~before, ["open", "high", "low", "close"]] -= 0.10
    joined = panel.merge(u0[["trade_date", "close"]].rename(columns={"close": "close_ref"}),
                         on="trade_date")
    joined["source"] = "vendor"
    ex_dates = pd.DataFrame({"symbol": ["000001.SZ"], "effective_date": [ex]})
    return joined, ex_dates


def test_ratio_steps_on_ex_dates_are_classified_as_qfq_as_of_fetch():
    joined, ex_dates = _with_ex_date("mult")
    basis = classify_adjustment_basis(joined, ex_dates).iloc[0]
    assert basis["basis"] == BASIS_QFQ_MULT
    assert basis["steps"] == 1 and basis["steps_on_ex_dates"] == 1
    assert bool(basis["qfq_as_of_fetch"]) is True


def test_constant_difference_steps_are_classified_as_subtractive_qfq():
    joined, ex_dates = _with_ex_date("sub")
    basis = classify_adjustment_basis(joined, ex_dates).iloc[0]
    assert basis["basis"] == BASIS_QFQ_SUB
    assert bool(basis["qfq_as_of_fetch"]) is True


def test_steps_off_ex_dates_are_not_called_adjustment():
    joined, _ = _with_ex_date("mult")
    elsewhere = pd.DataFrame({"symbol": ["000001.SZ"], "effective_date": [pd.Timestamp("2024-03-15")]})
    basis = classify_adjustment_basis(joined, elsewhere).iloc[0]
    assert bool(basis["qfq_as_of_fetch"]) is False


def test_script_writes_json_and_markdown_and_exits_3_when_refuted(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "audit_market_panel_units.py"
    spec = importlib.util.spec_from_file_location("_audit_units", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sina_qfq = pd.read_csv(FIXTURES / "ak1_18_60__sina__600519.SH__qfq.csv")
    sina_qfq = sina_qfq.rename(columns={"date": "trade_date"}).assign(symbol="600519.SH")
    sina_qfq["trade_date"] = pd.to_datetime(sina_qfq["trade_date"])
    panel_path = tmp_path / "panel.parquet"
    _as_panel(sina_qfq, "akshare:sina").to_parquet(panel_path, index=False)
    u0_path = tmp_path / "u0.parquet"
    _u0().to_parquet(u0_path, index=False)
    before = panel_path.read_bytes()

    code = module.main([
        "--panel", str(panel_path), "--u0", str(u0_path), "--ex-dates", str(tmp_path / "none"),
        "--start", "2024-01-01", "--end", "2024-03-31", "--out-dir", str(tmp_path / "out"),
    ])
    assert code == 3
    report = json.loads((tmp_path / "out" / "market_panel_unit_audit.json").read_text())
    assert report["point_in_time_claim"]["verdict"] == "REFUTED"
    assert "REFUTED" in (tmp_path / "out" / "market_panel_unit_audit.md").read_text()
    assert panel_path.read_bytes() == before
