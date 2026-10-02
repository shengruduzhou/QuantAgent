"""Concurrent list() must not make another request's lookup 404 (round-29 R8-F14)."""

from concurrent.futures import ThreadPoolExecutor

from services.quant_api.adapters.backtests import BacktestAdapter
from services.quant_api.runtime_indexer import RuntimeIndexer


def test_lookups_never_see_a_half_built_index(quant_ui_settings):
    adapter = BacktestAdapter(quant_ui_settings, RuntimeIndexer(quant_ui_settings))
    ids = [item["id"] for item in adapter.list()]
    assert ids

    def work(i: int) -> bool:
        if i % 2:
            adapter.list()
            return True
        return adapter._resolve(ids[i % len(ids)]) is not None

    with ThreadPoolExecutor(16) as pool:
        assert all(pool.map(work, range(400)))
