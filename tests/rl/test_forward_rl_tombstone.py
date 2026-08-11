from __future__ import annotations

import json
from pathlib import Path

from scripts.forward_rl_book import _retire_stale_outputs


def test_retired_forward_exporter_quarantines_all_root_targets(tmp_path: Path) -> None:
    out = tmp_path / "C_rl"
    out.mkdir(parents=True)
    (out / "targets_latest.csv").write_text("symbol,weight\nA,1\n", encoding="utf-8")
    (out / "targets_2026-06-12.csv").write_text("symbol,weight\nA,1\n", encoding="utf-8")
    (out / "last_update.json").write_text('{"target_date":"2026-06-12"}', encoding="utf-8")

    tombstone = _retire_stale_outputs(out)

    assert tombstone["status"] == "RL_FORWARD_BLOCKED"
    assert tombstone["replacement_target_emitted"] is False
    assert tombstone["production_eligible"] is False
    assert not (out / "targets_latest.csv").exists()
    assert not (out / "targets_2026-06-12.csv").exists()
    assert not (out / "last_update.json").exists()

    moved = [Path(path) for path in tombstone["stale_outputs_quarantined"]]
    assert {path.name for path in moved} == {
        "targets_latest.csv",
        "targets_2026-06-12.csv",
        "last_update.json",
    }
    assert all(path.exists() for path in moved)
    blocked = json.loads((out / "BLOCKED.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "RL_FORWARD_BLOCKED"


def test_tombstone_is_idempotent_when_no_stale_targets_exist(tmp_path: Path) -> None:
    out = tmp_path / "C_rl"
    first = _retire_stale_outputs(out)
    second = _retire_stale_outputs(out)

    assert first["stale_outputs_quarantined"] == []
    assert second["stale_outputs_quarantined"] == []
    assert (out / "BLOCKED.json").exists()
