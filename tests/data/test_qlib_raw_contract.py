"""Qlib normalized bundle restoration is explicit and fail closed."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from quantagent.data.providers.qlib_provider import QlibProvider
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable


def test_normalized_bundle_is_rejected_without_raw_contract():
    with pytest.raises(ProviderUnavailable,match="normalized bars"):
        QlibProvider(provider_uri="unused").daily_ohlcv(
            ProviderRequest('2024-01-02','2024-01-02',symbols=('600000.SH',)))


@pytest.mark.parametrize("factor", [.1, float('nan'), 0., float('inf')])
def test_explicit_raw_restoration_and_invalid_factor(monkeypatch, factor):
    index = pd.MultiIndex.from_tuples([('SH600000',pd.Timestamp('2024-01-02'))],names=['instrument','datetime'])
    frame = pd.DataFrame({'$open':[1.], '$high':[1.1], '$low':[.9], '$close':[1.],
                          '$volume':[1000.], '$factor':[factor], '$raw_amount':[100000.]},index=index)
    monkeypatch.setitem(sys.modules,'qlib',SimpleNamespace(init=lambda **kw: None))
    monkeypatch.setitem(sys.modules,'qlib.data',SimpleNamespace(D=SimpleNamespace(features=lambda *a,**kw: frame)))
    provider = QlibProvider('fixture',raw_amount_field='raw_amount',volume_scale_to_shares=100.)
    request = ProviderRequest('2024-01-02','2024-01-02',symbols=('600000.SH',))
    if factor != .1:
        with pytest.raises(ProviderUnavailable,match='restoration'):
            provider.daily_ohlcv(request)
        return
    result = provider.daily_ohlcv(request)
    assert result.frame.iloc[0]['close'] == pytest.approx(10.)
    assert result.frame.iloc[0]['volume'] == pytest.approx(10000.)
    assert result.frame.iloc[0]['amount'] == pytest.approx(100000.)
    assert result.metadata['adjustment'] == 'raw'


def test_bootstrap_persists_raw_contract_without_caller_certification(monkeypatch, tmp_path):
    from quantagent.data.bootstrap.qlib_bootstrap import QlibBootstrapConfig, build_qlib_market_panel

    index = pd.MultiIndex.from_tuples(
        [("SH600000", pd.Timestamp("2024-01-02"))], names=["instrument", "datetime"],
    )
    frame = pd.DataFrame({
        "$open": [1.], "$high": [1.1], "$low": [.9], "$close": [1.],
        "$volume": [1000.], "$factor": [.1], "$verified_cny": [100000.],
    }, index=index)
    monkeypatch.setitem(sys.modules, "qlib", SimpleNamespace(init=lambda **kwargs: None))
    monkeypatch.setitem(sys.modules, "qlib.data", SimpleNamespace(D=SimpleNamespace(features=lambda *args, **kwargs: frame)))
    bundle = tmp_path / "synthetic_bundle"
    bundle.mkdir()
    result = build_qlib_market_panel(QlibBootstrapConfig(
        provider_uri=str(bundle), start_date="2024-01-02", end_date="2024-01-02",
        symbols=("600000.SH",), output_root=str(tmp_path / "lake"), build_features=False,
        raw_amount_field="verified_cny", volume_scale_to_shares=100.,
        metadata={"production_integrity_certified": True, "volume_unit": "lots", "adjustment": "qfq"},
    ))

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    extra = manifest["extra"]
    assert extra["raw_amount_field"] == "$verified_cny"
    assert extra["volume_scale_to_shares"] == 100.
    assert extra["raw_price_restoration"] == "qlib_ohlc_div_factor"
    assert extra["adjustment"] == "raw"
    assert extra["frequency"] == "1d"
    assert extra["timezone"] == "Asia/Shanghai"
    assert extra["volume_unit"] == "shares"
    assert extra["amount_unit"] == "CNY"
    assert extra["pit_semantics"] == "daily_session_close"
    assert extra["production_integrity_certified"] is False
    market = pd.read_parquet(result["market_path"])
    assert market.iloc[0]["close"] == pytest.approx(10.)
    assert market.iloc[0]["volume"] == pytest.approx(10000.)
    assert market.iloc[0]["amount"] == pytest.approx(100000.)
