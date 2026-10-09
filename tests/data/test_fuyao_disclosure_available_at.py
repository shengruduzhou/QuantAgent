"""Fuyao statements become available the session AFTER their disclosure date (round-29 R3-F07)."""
import pandas as pd

from quantagent.data.providers.fuyao_provider import _normalise_financial_rows
from quantagent.data.v7_dataset_builder import merge_pit_features


def test_after_close_disclosure_not_available_same_session():
    disclosed = pd.Timestamp("2024-04-25 20:00", tz="Asia/Shanghai")   # Thursday evening
    period_end = pd.Timestamp("2024-03-31", tz="Asia/Shanghai")
    rows = [{"thscode": "600000.SH", "report_date_ms": int(disclosed.value // 10**6),
             "period_end_ms": int(period_end.value // 10**6), "net_profit": 123.0}]
    fin = _normalise_financial_rows(rows, statement_type="income", source="fuyao", endpoint="x")
    feats = pd.DataFrame({"symbol": ["600000.SH"] * 2,
                          "trade_date": pd.to_datetime(["2024-04-25", "2024-04-26"]),
                          "available_at": pd.to_datetime(["2024-04-25", "2024-04-26"])})
    merged = merge_pit_features(feats, fin[["symbol", "available_at", "net_profit"]])
    same_day = merged.loc[merged["trade_date"] == "2024-04-25", "net_profit"]
    assert same_day.isna().all(), (
        f"after-close disclosure (available_at={fin['available_at'].iloc[0]}) joined onto the "
        f"same session's row: net_profit={same_day.tolist()}")
    next_day = merged.loc[merged["trade_date"] == "2024-04-26", "net_profit"]
    assert next_day.tolist() == [123.0]


def test_friday_disclosure_is_available_monday():
    disclosed = pd.Timestamp("2024-04-26 18:30", tz="Asia/Shanghai")  # Friday
    period_end = pd.Timestamp("2024-03-31", tz="Asia/Shanghai")
    rows = [{"thscode": "600000.SH", "report_date_ms": int(disclosed.value // 10**6),
             "period_end_ms": int(period_end.value // 10**6), "net_profit": 1.0}]
    fin = _normalise_financial_rows(rows, statement_type="income", source="fuyao", endpoint="x")
    assert fin["available_at"].iloc[0] == pd.Timestamp("2024-04-29")
