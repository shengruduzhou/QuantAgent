"""Round 31 F3: a stale runtime index must not freeze requests.

A full rescan of the operator's runtime took 30-47 s under a global lock, and
both TTL expiry and every job completion (invalidate) made the next request
wait for it. With background refresh the stale index is served immediately
and rebuilt once in the background.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import time

from services.quant_api.runtime_indexer.indexer import RuntimeIndexer


def test_stale_index_is_served_while_a_single_refresh_runs(quant_ui_settings, monkeypatch) -> None:
    settings = replace(quant_ui_settings, index_background_refresh=True, index_ttl_seconds=0)
    indexer = RuntimeIndexer(settings)
    first = indexer.scan(force=True)
    assert first

    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    real_walk = indexer._walk

    def slow_walk():
        calls["n"] += 1
        started.set()
        release.wait(5)
        return real_walk()

    monkeypatch.setattr(indexer, "_walk", slow_walk)
    t0 = time.monotonic()
    served = indexer.scan()          # stale (ttl 0): must not block
    indexer.invalidate()             # job completion: must not block either
    served_again = indexer.scan()
    elapsed = time.monotonic() - t0
    assert served == first and served_again == first
    assert elapsed < 1.0
    assert started.wait(2)
    release.set()
    for _ in range(50):
        if not indexer._refreshing:
            break
        time.sleep(0.05)
    assert calls["n"] == 1           # one background rebuild, not one per request


def test_synchronous_mode_still_reflects_invalidation(quant_ui_settings) -> None:
    indexer = RuntimeIndexer(quant_ui_settings)
    before = indexer.scan(force=True)
    marker = quant_ui_settings.runtime_root / "reports" / "round31_marker.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    indexer.invalidate()
    after = indexer.scan()
    assert len(after) == len(before) + 1
