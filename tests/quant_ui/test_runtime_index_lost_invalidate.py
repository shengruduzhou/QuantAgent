"""Round 31 V-02: an invalidate() during an in-flight background rebuild must
not be lost (the walk started before the job wrote its artifact, and used to
mark the index fresh for the whole TTL without it)."""
import dataclasses, threading, time
from pathlib import Path
from services.quant_api.config import ApiSettings
from services.quant_api.runtime_indexer.indexer import RuntimeIndexer


def test_invalidate_during_refresh_is_not_lost(tmp_path):
    runtime = tmp_path / "runtime"; (runtime / "reports").mkdir(parents=True)
    (runtime / "reports" / "old.json").write_text("{}")
    settings = ApiSettings(project_root=tmp_path, runtime_root=runtime,
                           cache_root=tmp_path / "cache", jobs_root=tmp_path / "jobs")
    settings = dataclasses.replace(settings, index_background_refresh=True)
    idx = RuntimeIndexer(settings)
    idx.scan()  # initial synchronous index
    walk = idx._walk
    started = threading.Event()
    def slow_walk():
        out = walk()          # snapshot of the tree taken here
        started.set()
        time.sleep(0.5)       # rest of a 30-47 s production walk
        return out
    idx._walk = slow_walk
    idx._indexed_at = 0.0
    idx.scan()                # stale -> background refresh starts
    assert started.wait(5)
    # a job finishes while the walk is in flight, writes its artifact, invalidates
    (runtime / "reports" / "new_job_result.json").write_text("{}")
    idx.invalidate()
    deadline = time.time() + 10
    while idx._refreshing and time.time() < deadline:   # background refresh(es) complete
        time.sleep(0.05)
    assert not idx._refreshing
    idx._walk = walk
    names = {a["name"] for a in idx.scan()}
    assert "new_job_result.json" in names, (
        f"index fresh for {settings.index_ttl_seconds}s without the job's artifact: {sorted(names)}")
