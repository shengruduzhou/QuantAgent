"""Volume unit truth is decided per RESPONSE, never by a static per-source table.

Fixtures under ``tests/data/fixtures/akshare_units_r29/`` are recorded vendor
responses, not synthetic frames:

* ``ak1_18_60__*`` / ``ak1_19_1__*``: the exact DataFrames akshare 1.18.60 and
  1.19.1 returned for 2024-01-02..2024-03-29 (Round 29 R5 live pull,
  2026-10-02), written to CSV unchanged.
* ``tencent_fqkline__*.json``: the raw ``web.ifzq.gtimg.cn/appstock/app/fqkline``
  ``day`` arrays behind ``TencentSource``.
* ``u0_daily_bars_raw__2024Q1.csv``: the certified U0 raw panel rows (TickFlow,
  shares/CNY) for the same window -- the reference every ratio is taken against.

Defects pinned here (all reproduced against the pre-fix code):

* STAR (688xxx) volume on Tencent is native SHARES; the old normaliser and
  ``TencentSource`` multiplied it by 100 (ratio 100.000032 vs U0).
* akshare >= 1.18.69 skips its own lots->shares x100 for every ``sz000`` symbol,
  so 000001.SZ came out at 0.01x while labelled shares.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from quantagent.data.ashare.contracts import (
    QUALITY_DERIVED,
    QUALITY_OK,
    QUALITY_SUSPECT,
    QUALITY_UNIT_AMBIGUOUS,
)
from quantagent.data.ashare.units import infer_volume_scale, tencent_native_volume_prior
from quantagent.data.providers.akshare_live_provider import (
    AkShareLiveProvider,
    _normalize_akshare_daily,
)
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "akshare_units_r29"


def _recorded(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / f"{name}.csv")


def _u0(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "u0_daily_bars_raw__2024Q1.csv")
    return frame[frame["symbol"] == symbol].reset_index(drop=True)


def _volume_ratio_vs_u0(normalised: pd.DataFrame, symbol: str) -> pd.Series:
    joined = normalised[["trade_date", "volume"]].merge(
        _u0(symbol)[["trade_date", "volume"]], on="trade_date", suffixes=("", "_u0")
    )
    assert len(joined) == 58, "fixture and U0 must overlap on every session"
    return joined["volume"] / joined["volume_u0"]


def _normalise(name: str, symbol: str, source: str, version: str) -> pd.DataFrame:
    return _normalize_akshare_daily(
        _recorded(name), symbol, source=source, adjust="", akshare_version=version
    )


# ---------------------------------------------------------------------------
# recorded responses: one per board / shape named in the IC decision
# ---------------------------------------------------------------------------
class TestRecordedResponses:
    def test_star_tencent_new_shape_is_native_shares_and_verified(self):
        out = _normalise("ak1_19_1__tencent__688981.SH__none", "688981.SH", "tencent", "1.19.1")
        ratio = _volume_ratio_vs_u0(out, "688981.SH")
        assert ratio.between(0.999, 1.001).all(), ratio.describe()
        assert set(out["quality_status"]) == {QUALITY_OK}
        assert set(out["volume_unit"]) == {"shares"}
        assert set(out["raw_volume_unit"]) == {"shares"}
        assert set(out["volume_scale_applied"]) == {1.0}
        assert out["volume_unit_basis"].iloc[0].startswith(
            "tencent_v2_volume_turnover_amount:implied_vwap_in_range:scale=x1"
        )

    def test_star_tencent_legacy_shape_is_not_inflated_and_not_certified(self):
        """1.18.60 shape: no CNY turnover, so the unit cannot be proven."""
        out = _normalise("ak1_18_60__tencent__688981.SH__none", "688981.SH", "tencent", "1.18.60")
        ratio = _volume_ratio_vs_u0(out, "688981.SH")
        assert ratio.between(0.999, 1.001).all(), ratio.describe()  # was 100.000032
        assert set(out["quality_status"]) == {QUALITY_UNIT_AMBIGUOUS}
        assert set(out["volume_unit"]) == {"shares_unverified"}
        assert out["amount"].isna().all()
        assert "tencent_star_native_shares" in out["volume_unit_basis"].iloc[0]

    def test_sz000_stock_on_tencent_new_shape_is_rescaled_by_evidence(self):
        """akshare skips x100 for 'sz000*'; the VWAP check must catch it."""
        out = _normalise("ak1_19_1__tencent__000001.SZ__none", "000001.SZ", "tencent", "1.19.1")
        ratio = _volume_ratio_vs_u0(out, "000001.SZ")
        assert ratio.between(0.999, 1.001).all(), ratio.describe()  # was 0.01
        assert set(out["quality_status"]) == {QUALITY_OK}
        assert set(out["raw_volume_unit"]) == {"lots_100_shares"}
        vwap = out["amount"] / out["volume"]
        assert ((vwap >= out["low"] - 0.01) & (vwap <= out["high"] + 0.01)).all()

    def test_sz000_stock_on_tencent_legacy_shape_uses_lots_prior_unverified(self):
        out = _normalise("ak1_18_60__tencent__000001.SZ__none", "000001.SZ", "tencent", "1.18.60")
        assert _volume_ratio_vs_u0(out, "000001.SZ").between(0.999, 1.001).all()
        assert set(out["quality_status"]) == {QUALITY_UNIT_AMBIGUOUS}

    @pytest.mark.parametrize("symbol", ["600519.SH", "300750.SZ"])
    def test_main_board_and_chinext_tencent_new_shape_controls(self, symbol):
        name = f"ak1_19_1__tencent__{symbol}__none"
        out = _normalise(name, symbol, "tencent", "1.19.1")
        assert _volume_ratio_vs_u0(out, symbol).between(0.999, 1.001).all()
        assert set(out["quality_status"]) == {QUALITY_OK}
        assert set(out["raw_volume_unit"]) == {"shares"}  # akshare already applied x100

    def test_bse_920xxx_on_sina_is_shares(self):
        out = _normalise("ak1_18_60__sina__920000.BJ__none", "920000.BJ", "sina", "1.18.60")
        assert _volume_ratio_vs_u0(out, "920000.BJ").between(0.999, 1.001).all()
        assert set(out["quality_status"]) == {QUALITY_OK}
        assert set(out["raw_volume_unit"]) == {"shares"}

    @pytest.mark.parametrize("symbol", ["600519.SH", "300750.SZ"])
    def test_eastmoney_response_is_proven_lots(self, symbol):
        name = f"ak1_18_60__east_money__{symbol}__none"
        out = _normalise(name, symbol, "east_money", "1.18.60")
        assert _volume_ratio_vs_u0(out, symbol).between(0.999, 1.001).all()
        assert set(out["quality_status"]) == {QUALITY_OK}
        assert set(out["raw_volume_unit"]) == {"lots_100_shares"}
        amount_ratio = out["amount"].to_numpy() / _u0(symbol)["amount"].to_numpy()
        assert pd.Series(amount_ratio).between(0.999, 1.001).all()

    def test_provenance_records_akshare_version_and_endpoint(self):
        out = _normalise("ak1_18_60__sina__688981.SH__none", "688981.SH", "sina", "1.18.60")
        assert set(out["source_endpoint"]) == {"akshare==1.18.60:stock_zh_a_daily"}
        assert set(out["akshare_version"]) == {"1.18.60"}
        assert out["volume_unit_basis"].str.contains("implied_vwap_in_range").all()


class TestAdjustedPricesCannotPassAsRaw:
    """Raw CNY turnover cannot be reconciled with adjusted prices."""

    def test_tencent_qfq_mislabelled_raw_is_ambiguous(self):
        out = _normalise("ak1_19_1__tencent__000001.SZ__qfq", "000001.SZ", "tencent", "1.19.1")
        assert set(out["quality_status"]) == {QUALITY_UNIT_AMBIGUOUS}
        assert set(out["volume_unit"]) == {"shares_unverified"}

    def test_sina_qfq_mislabelled_raw_is_ambiguous(self):
        out = _normalise("ak1_18_60__sina__600519.SH__qfq", "600519.SH", "sina", "1.18.60")
        assert set(out["quality_status"]) == {QUALITY_UNIT_AMBIGUOUS}


class TestVerdictRules:
    def test_mixed_scale_response_is_ambiguous_not_majority_voted(self):
        raw = _recorded("ak1_18_60__east_money__600519.SH__none")
        raw.loc[raw.index[::2], "成交量"] = raw.loc[raw.index[::2], "成交量"] * 100
        out = _normalize_akshare_daily(raw, "600519.SH", source="east_money", akshare_version="t")
        assert set(out["quality_status"]) == {QUALITY_UNIT_AMBIGUOUS}
        assert out["volume_unit_basis"].iloc[0].startswith("ambiguous:0_scales_fit_bulk")

    def test_single_bad_row_under_a_proven_scale_is_suspect_not_ok(self):
        raw = _recorded("ak1_18_60__sina__920000.BJ__none")
        raw.loc[5, "amount"] = raw.loc[5, "amount"] * 10
        out = _normalize_akshare_daily(raw, "920000.BJ", source="sina", akshare_version="t")
        assert out.loc[5, "quality_status"] == QUALITY_SUSPECT
        assert (out.drop(index=5)["quality_status"] == QUALITY_OK).all()
        assert set(out["volume_unit"]) == {"shares"}

    def test_no_turnover_rows_cannot_be_verified(self):
        verdict = infer_volume_scale(
            pd.Series([100.0, 200.0]), pd.Series([0.0, 0.0]),
            pd.Series([10.0, 10.0]), pd.Series([11.0, 11.0]), fallback_scale=100.0,
        )
        assert verdict.status == "unverifiable"
        assert verdict.scale == 100.0
        assert verdict.raw_unit_name == "lots_100_shares_unverified"

    def test_tencent_prior_is_measured_per_board_and_silent_on_bse(self):
        assert tencent_native_volume_prior("688981.SH")[0] == 1.0
        assert tencent_native_volume_prior("689009.SH")[0] == 1.0
        assert tencent_native_volume_prior("000001.SZ")[0] == 100.0
        assert tencent_native_volume_prior("920000.BJ")[0] is None


# ---------------------------------------------------------------------------
# U0 TencentSource (raw fqkline payload, no turnover)
# ---------------------------------------------------------------------------
class _Outcome:
    def __init__(self, payload):
        self.payload, self.ok, self.retry_class = payload, True, "OK"
        self.endpoint = "fixture://fqkline"
        self.retrieved_at = "2026-10-02T00:00:00+00:00"
        self.error, self.status_code, self.latency_s = None, 200, 0.0


class _Client:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self, url, params=None, headers=None):
        return _Outcome(self.payload)


@pytest.mark.parametrize("code,symbol", [("sh688981", "688981.SH"), ("sh600519", "600519.SH")])
def test_u0_tencent_source_scales_by_board_and_refuses_to_certify(code, symbol):
    from quantagent.data.ashare.sources import TencentSource

    payload = json.loads((FIXTURES / f"tencent_fqkline__{code}__2024Q1.json").read_text())
    source = TencentSource(_Client(payload))
    source.WINDOW_YEARS = 100
    result = source.daily_bars(symbol, "2024-01-02", "2024-03-29")
    frame = result.frame.copy()
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    assert _volume_ratio_vs_u0(frame, symbol).between(0.999, 1.001).all()
    assert set(frame["quality_status"]) == {QUALITY_UNIT_AMBIGUOUS}
    assert result.metadata["volume_unit_status"] == "unverifiable"


# ---------------------------------------------------------------------------
# provider: adjust refusal and unit-gated failover
# ---------------------------------------------------------------------------
def _calendar() -> TradingCalendar:
    dates = pd.read_csv(FIXTURES / "u0_daily_bars_raw__2024Q1.csv")["trade_date"].unique()
    return TradingCalendar.from_dates([*sorted(dates), "2024-04-01"])


def _fake_akshare(monkeypatch, *, east=None, sina=None, tencent=None, version="1.18.84"):
    calls: list[str] = []

    def _make(name, payload):
        def api(**kwargs):
            calls.append(name)
            if payload is None:
                raise ConnectionError(f"{name} remote closed")
            return payload.copy()
        return api

    module = types.SimpleNamespace(
        __version__=version,
        stock_zh_a_hist=_make("east_money", east),
        stock_zh_a_daily=_make("sina", sina),
        stock_zh_a_hist_tx=_make("tencent", tencent),
    )
    monkeypatch.setitem(sys.modules, "akshare", module)
    return calls


REQUEST = ProviderRequest("2024-01-02", "2024-03-29", symbols=("688981.SH",))


def test_tencent_without_provable_unit_is_excluded_from_failover(monkeypatch):
    calls = _fake_akshare(
        monkeypatch, tencent=_recorded("ak1_18_60__tencent__688981.SH__none")
    )
    result = AkShareLiveProvider(allow_network=True, trading_calendar=_calendar()).daily_ohlcv(REQUEST)
    assert calls == ["east_money", "sina", "tencent"]
    assert result.frame.empty
    assert result.metadata["failed_symbols"] == ["688981.SH"]
    assert result.metadata["unit_rejections"][0]["source"] == "tencent"
    assert any("tencent_unit_ambiguous" in w for w in result.warnings)
    assert result.point_in_time is False


def test_tencent_with_provable_unit_may_serve_failover(monkeypatch):
    _fake_akshare(monkeypatch, tencent=_recorded("ak1_19_1__tencent__688981.SH__none"))
    result = AkShareLiveProvider(allow_network=True, trading_calendar=_calendar()).daily_ohlcv(REQUEST)
    assert result.metadata["source_by_symbol"] == {"688981.SH": "tencent"}
    assert set(result.frame["quality_status"]) == {QUALITY_OK}
    assert result.metadata["raw_volume_unit_by_source"] == {"tencent": "shares"}
    assert _volume_ratio_vs_u0(result.frame, "688981.SH").between(0.999, 1.001).all()
    assert result.point_in_time is True


def test_ambiguous_primary_falls_through_to_next_source(monkeypatch):
    calls = _fake_akshare(
        monkeypatch,
        east=_recorded("ak1_19_1__tencent__000001.SZ__qfq").rename(
            columns={"date": "日期", "open": "开盘", "close": "收盘", "high": "最高",
                     "low": "最低", "volume": "成交量", "amount": "成交额"}),
        sina=_recorded("ak1_18_60__sina__688981.SH__none"),
    )
    result = AkShareLiveProvider(allow_network=True, trading_calendar=_calendar()).daily_ohlcv(REQUEST)
    assert calls == ["east_money", "sina"]
    assert result.metadata["source_by_symbol"] == {"688981.SH": "sina"}
    assert result.metadata["unit_rejections"][0]["source"] == "east_money"


@pytest.mark.parametrize("adjust", ["qfq", "hfq"])
def test_vendor_adjusted_bars_are_refused_on_the_canonical_path(monkeypatch, adjust):
    calls = _fake_akshare(monkeypatch, sina=_recorded("ak1_18_60__sina__600519.SH__qfq"))
    with pytest.raises(ProviderUnavailable, match="refused on canonical paths"):
        AkShareLiveProvider(allow_network=True, adjust=adjust).daily_ohlcv(REQUEST)
    assert calls == [], "the refusal must happen before any vendor call"


def test_research_only_opt_in_is_named_and_never_pit(monkeypatch):
    _fake_akshare(monkeypatch, sina=_recorded("ak1_18_60__sina__600519.SH__qfq"))
    result = AkShareLiveProvider(
        allow_network=True,
        adjust="qfq",
        source_order=("sina",),
        trading_calendar=_calendar(),
        research_only_allow_vendor_adjusted=True,
    ).daily_ohlcv(ProviderRequest("2024-01-02", "2024-03-29", symbols=("600519.SH",)))
    assert result.point_in_time is False
    assert result.metadata["research_only"] is True
    assert set(result.frame["quality_status"]) == {QUALITY_DERIVED}
    assert not result.frame["point_in_time_valid"].any()
    assert "akshare_vendor_adjusted_research_only" in result.warnings


def test_source_affinity_keeps_a_symbol_on_its_established_source(monkeypatch):
    calls = _fake_akshare(
        monkeypatch,
        east=_recorded("ak1_18_60__east_money__600519.SH__none"),
        sina=_recorded("ak1_18_60__sina__688981.SH__none"),
    )
    result = AkShareLiveProvider(
        allow_network=True, trading_calendar=_calendar(),
        source_affinity={"688981.SH": "sina"},
    ).daily_ohlcv(REQUEST)
    assert calls == ["sina"]
    assert result.metadata["source_switches"] == []


def test_switch_away_from_established_source_is_reported(monkeypatch):
    _fake_akshare(monkeypatch, tencent=_recorded("ak1_19_1__tencent__688981.SH__none"))
    result = AkShareLiveProvider(
        allow_network=True, trading_calendar=_calendar(),
        source_affinity={"688981.SH": "sina"},
    ).daily_ohlcv(REQUEST)
    assert result.metadata["source_switches"] == [{
        "symbol": "688981.SH", "provider_before": "akshare:sina",
        "provider_after": "akshare:tencent", "first_new_trade_date": "2024-01-02",
    }]
    assert any(w.startswith("akshare_source_switch:688981.SH:sina->tencent") for w in result.warnings)
