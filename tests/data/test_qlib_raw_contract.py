"""Qlib normalized bundle restoration is explicit and fail closed."""
import sys
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
