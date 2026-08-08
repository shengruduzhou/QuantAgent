from __future__ import annotations

import numpy as np
import pandas as pd

from services.quant_api.services.fund_research import FundResearchService
from services.quant_api.services.market_playbooks_v3 import MarketPlaybookService, PLAYBOOKS


class SyntheticMarket:
    def _provider(self):
        raise AssertionError("provider should not be used by these synthetic strategy tests")


class SyntheticPlaybooks(MarketPlaybookService):
    def _index_bars(self, symbol: str, days: int) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=360)
        drift = 0.0006 if "01" in symbol else 0.0004
        wave = 0.002 * np.sin(np.arange(len(dates)) / 7.0)
        close = 100.0 * np.cumprod(1.0 + drift + wave)
        open_price = close * (1.0 + 0.0003 * np.cos(np.arange(len(dates)) / 5.0))
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.002,
                "low": np.minimum(open_price, close) * 0.998,
                "close": close,
                "volume": 1_000_000.0,
            }
        )

    def _stock_bars(self, symbol: str, days: int) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-02", periods=180)
        close = np.full(len(dates), 100.0)
        close[100:] = 110.0
        volume = np.full(len(dates), 1_000_000.0)
        volume[100] = 2_000_000.0
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": volume,
            }
        )


def test_playbook_catalog_is_exactly_16() -> None:
    assert [item["id"] for item in PLAYBOOKS] == [f"{index:02d}" for index in range(1, 17)]


def test_price_volume_breakout_uses_55_20_60_recipe_and_t_plus_one() -> None:
    payload = SyntheticPlaybooks(SyntheticMarket()).price_volume_breakout(
        "600000.SH", "000300.SH", 8.0
    )
    assert payload["formation"] == {
        "breakoutWindow": 55,
        "exitLowWindow": 20,
        "volumeWindow": 20,
        "volumeMultiple": 1.5,
        "trendWindow": 60,
    }
    assert any(event["action"] == "enter_next_open" for event in payload["events"])
    assert "T+1 open" in payload["assumptions"]["execution"]
    assert "priorHigh55" in payload["rows"][-1]


def test_multi_asset_tsmom_exposes_cash_risk_and_window_sensitivity() -> None:
    payload = SyntheticPlaybooks(SyntheticMarket()).time_series_momentum(
        "I01.TI,I02.TI", "000300.SH", 8.0
    )
    assert payload["assetPool"] == ["I01.TI", "I02.TI"]
    assert set(payload["windowSensitivity"]) == {"60", "120", "180"}
    assert "cashWeight" in payload["rows"][-1]
    assert "activeCount" in payload["rows"][-1]
    assert "T+1 open" in payload["assumptions"]["execution"]
    assert set(payload["latestWeights"]).issubset({"I01.TI", "I02.TI"})


def test_reversal_holding_target_enforces_hold_then_cooldown() -> None:
    index = pd.bdate_range("2026-01-05", periods=16)
    selection = pd.DataFrame({"A": True}, index=index)
    target = MarketPlaybookService._holding_targets(
        selection,
        holding_days=5,
        cooldown_days=5,
    )
    assert target["A"].iloc[:5].tolist() == [1.0] * 5
    assert target["A"].iloc[5:10].tolist() == [0.0] * 5
    assert target["A"].iloc[10] == 1.0


class FakeFundProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_capability(self, path: str, params: dict[str, object]):
        self.calls.append((path, dict(params)))
        if path.endswith("/profile/detail"):
            return {"item": [{"name": "示例ETF"}]}
        if path.endswith("/portfolio/holdings"):
            return {"item": [{"thscode": "600000.SH", "nav_ratio": 5.0, "report_date_ms": 1}]}
        if path.endswith("/performance/nav"):
            return {"item": [{"date_ms": 1, "unit_nav": 1.0}]}
        if path.endswith("/performance/returns"):
            return {"item": [{"period": "1y", "return_rate": 8.0}]}
        if path.endswith("/holders/detail"):
            return {"item": [{"merge_scope": "merged", "report_date_ms": 1}]}
        if path.endswith("/market/snapshot"):
            return {"item": [{"thscode": "510300.SH", "last_price": 4.0}]}
        if path.endswith("/market/historical"):
            return {"item": [{"date_ms": 1, "close_price": 4.0}]}
        raise AssertionError(path)


class FakeFundMarket:
    def __init__(self) -> None:
        self.provider = FakeFundProvider()

    def _provider(self):
        return self.provider


def test_fund_research_uses_server_side_fuyao_contracts() -> None:
    market = FakeFundMarket()
    payload = FundResearchService(market).overview("510300.SH", fund_type="exchange")
    assert payload["fundType"] == "exchange"
    assert payload["panels"]["holdings"]["item"][0]["thscode"] == "600000.SH"
    assert payload["pit"]["market"] == "etf_only_upstream"
    assert payload["issues"] == []
    snapshot_call = next(
        params for path, params in market.provider.calls if path.endswith("/market/snapshot")
    )
    assert snapshot_call == {"thscode": "510300.SH"}
