"""The silver panel never splices vendors or vendor-adjusted prices silently."""

from __future__ import annotations

import json
import sys
import types

import pandas as pd

from quantagent.data.bootstrap.akshare_market_bootstrap import (
    AkShareMarketPanelConfig,
    _market_economic_contract_report,
    _normalise_dtypes,
    build_akshare_market_panel,
)
from quantagent.data.providers.akshare_live_provider import AkShareLiveProvider
from quantagent.data.providers.base import ProviderResult

SESSIONS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def _calendar_akshare(monkeypatch) -> None:
    module = types.SimpleNamespace(
        __version__="1.18.84",
        tool_trade_date_hist_sina=lambda: pd.DataFrame({"trade_date": SESSIONS}),
    )
    monkeypatch.setitem(sys.modules, "akshare", module)


def _row(trade_date: str, available_at: str, source: str, quality: str = "OK") -> dict:
    return {
        "symbol": "600519.SH",
        "trade_date": trade_date,
        "open": 10.0,
        "high": 11.0,
        "low": 9.5,
        "close": 10.5,
        "volume": 1000.0,
        "amount": 10500.0,
        "available_at": available_at,
        "volume_unit": "shares",
        "amount_unit": "CNY",
        "price_adjustment": "raw",
        "point_in_time_valid": True,
        "source": source,
        "source_endpoint": "akshare==1.18.84:stock_zh_a_daily",
        "retrieved_at": "2026-10-02T00:00:00+00:00",
        "quality_status": quality,
        "volume_unit_basis": "implied_vwap_in_range:scale=x1",
    }


def _seed_panel(tmp_path, source: str):
    path = tmp_path / "lake" / "silver" / "market_panel" / "market_panel.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalise_dtypes(pd.DataFrame([_row("2024-01-02", "2024-01-03", source)])).to_parquet(
        path, index=False
    )
    return path


def _config(tmp_path, **kwargs) -> AkShareMarketPanelConfig:
    return AkShareMarketPanelConfig(
        symbols=("600519.SH",),
        start_date="2024-01-03",
        end_date="2024-01-04",
        output_root=str(tmp_path / "lake"),
        allow_network=True,
        **kwargs,
    )


def _serve_from(monkeypatch, source: str, captured: dict) -> None:
    def fake_daily(self, request):
        captured["affinity"] = dict(self.source_affinity or {})
        frame = pd.DataFrame([_row("2024-01-03", "2024-01-04", source)])
        return ProviderResult(frame, source="akshare_live_provider:multi_source", point_in_time=True)

    monkeypatch.setattr(AkShareLiveProvider, "daily_ohlcv", fake_daily)


def test_vendor_adjusted_panel_build_is_refused_before_any_fetch(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(AkShareLiveProvider, "daily_ohlcv", lambda self, r: called.append(r))
    payload = build_akshare_market_panel(_config(tmp_path, adjust="qfq"))
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["akshare_vendor_adjusted_prices_refused"]
    assert called == []


def test_established_source_is_tried_first(monkeypatch, tmp_path):
    _calendar_akshare(monkeypatch)
    _seed_panel(tmp_path, "akshare:sina")
    captured: dict = {}
    _serve_from(monkeypatch, "akshare:sina", captured)
    payload = build_akshare_market_panel(_config(tmp_path))
    assert captured["affinity"] == {"600519.SH": "sina"}
    assert payload["status"] == "passed", payload.get("blockers")


def test_source_switch_without_boundary_record_is_blocked(monkeypatch, tmp_path):
    _calendar_akshare(monkeypatch)
    panel = _seed_panel(tmp_path, "akshare:sina")
    before = panel.read_bytes()
    _serve_from(monkeypatch, "akshare:tencent", {})
    payload = build_akshare_market_panel(_config(tmp_path))
    assert payload["status"] == "blocked"
    assert "akshare_symbol_history_would_mix_sources" in payload["blockers"]
    assert payload["source_boundaries"][0]["provider_before"] == "akshare:sina"
    assert panel.read_bytes() == before, "a blocked build must not touch the panel"


def test_source_switch_with_boundary_record_writes_the_seam(monkeypatch, tmp_path):
    _calendar_akshare(monkeypatch)
    _seed_panel(tmp_path, "akshare:sina")
    _serve_from(monkeypatch, "akshare:tencent", {})
    payload = build_akshare_market_panel(_config(tmp_path, record_source_boundaries=True))
    assert payload["status"] == "passed", payload.get("blockers")
    log = tmp_path / "lake" / "manifests" / "market_panel_source_boundaries.jsonl"
    record = json.loads(log.read_text().splitlines()[0])
    assert record["symbol"] == "600519.SH"
    assert record["boundary_date"] == "2024-01-03"
    assert (record["provider_before"], record["provider_after"]) == ("akshare:sina", "akshare:tencent")
    manifest = json.loads((tmp_path / "lake" / "manifests" / "market_panel.json").read_text())
    assert manifest["extra"]["source_boundaries"][0]["provider_after"] == "akshare:tencent"


def test_economic_contract_rejects_unit_ambiguous_and_suspect_rows():
    frame = pd.DataFrame(
        [
            _row("2024-01-02", "2024-01-03", "akshare:sina"),
            _row("2024-01-03", "2024-01-04", "akshare:tencent", quality="UNIT_AMBIGUOUS"),
            _row("2024-01-04", "2024-01-05", "akshare:sina", quality="SUSPECT"),
        ]
    )
    report = _market_economic_contract_report(_normalise_dtypes(frame))
    assert report["status"] == "failed"
    assert "non_ok_quality_rows:UNIT_AMBIGUOUS=1" in report["violations"]
    assert "non_ok_quality_rows:SUSPECT=1" in report["violations"]
