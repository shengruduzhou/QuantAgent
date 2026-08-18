"""Round-22: the governance hash chain must be checkable, and say what broke.

Companion to ``test_audit_chain_concurrency``. That file proves the chain no
longer forks; this one proves a broken chain can be *detected on purpose* --
the A-04 finding was not that corruption went unnoticed but that nothing ever
asked, so 159 orphaned governance records sat in a file that every writer
considered a success.

Covered here: an intact chain, a chain truncated at the front/middle, a chain
truncated at the end (the one shape a hash chain cannot see unaided -- measured
and documented rather than papered over), a tampered middle entry, and the
forked shape A-04 actually produced. Plus the two write-path invariants the fix
rests on: the tail is re-read under the lock, and a failed durable write latches
the log closed instead of resynchronising (DEF-017).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from quantagent.governance import audit as audit_module
from quantagent.governance.audit import (
    GENESIS_HASH,
    AuditChainCorruption,
    AuditEntry,
    AuditLog,
    AuditWriteUnavailable,
)

SUBJECT = "round22 governance chain integrity"


def _fill(log: AuditLog, count: int, *, actor: str = "agent") -> list[AuditEntry]:
    return [
        log.append(kind="ENVELOPE", actor=actor, subject=SUBJECT, payload={"i": i})
        for i in range(count)
    ]


def _lines(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _rewrite(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


class TestIntactChain:
    def test_verify_reports_full_reachability(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        _fill(log, 6)

        result = log.verify()
        assert result["valid"] is True
        assert result["entries_total"] == 6
        assert result["reachable"] == 6
        assert result["orphaned"] == 0
        assert result["head"] != GENESIS_HASH

    def test_require_intact_returns_the_report(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        entries = _fill(log, 4)
        assert log.require_intact()["head"] == entries[-1].entry_hash

    def test_empty_log_is_intact_and_headed_at_genesis(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        assert log.require_intact() == {
            "valid": True, "checked": 0, "reachable": 0,
            "entries_total": 0, "orphaned": 0, "head": GENESIS_HASH,
        }


class TestTruncatedChain:
    def test_dropping_the_first_entry_is_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 5)
        _rewrite(path, _lines(path)[1:])

        result = log.verify()
        assert result["valid"] is False
        assert result["reachable"] == 0
        assert result["entries_total"] == 4
        assert result["orphaned"] == 4
        with pytest.raises(AuditChainCorruption, match="4 are orphaned"):
            log.require_intact()

    def test_dropping_a_middle_entry_is_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 5)
        lines = _lines(path)
        _rewrite(path, lines[:2] + lines[3:])

        result = log.verify()
        assert result["valid"] is False
        assert result["reachable"] == 2
        assert result["orphaned"] == 2
        with pytest.raises(AuditChainCorruption, match="2 of 4 records are reachable"):
            log.require_intact()

    def test_a_torn_final_line_is_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 3)
        lines = _lines(path)
        lines[-1] = lines[-1][: len(lines[-1]) // 2]
        _rewrite(path, lines)

        result = log.verify()
        assert result["valid"] is False
        assert "not a readable audit entry" in result["error"]
        assert result["reachable"] == 2
        with pytest.raises(AuditChainCorruption):
            log.require_intact()

    def test_tail_truncation_needs_an_external_head_anchor(
        self, tmp_path: Path
    ) -> None:
        """Measured limitation, stated rather than hidden.

        Removing entries from the end leaves a chain that is internally perfect:
        nothing inside the file records how long the file should be. Detection
        requires a head remembered from an earlier read.
        """

        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 5)
        head_before = log.verify()["head"]

        _rewrite(path, _lines(path)[:3])

        unaided = log.verify()
        assert unaided["valid"] is True, (
            "tail truncation is genuinely invisible to a bare chain walk; if "
            "this ever starts failing the docstring on require_intact is stale"
        )
        assert unaided["orphaned"] == 0

        with pytest.raises(AuditChainCorruption, match="removed from the end"):
            log.require_intact(expected_head=head_before)


class TestTamperedChain:
    def test_editing_a_middle_entry_is_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 5)

        lines = _lines(path)
        tampered = json.loads(lines[2])
        tampered["payload"] = {"i": 999}
        lines[2] = json.dumps(tampered, ensure_ascii=False)
        _rewrite(path, lines)

        result = log.verify()
        assert result["valid"] is False
        assert "does not match" in result["error"]
        assert result["reachable"] == 2
        assert result["orphaned"] == 3
        with pytest.raises(AuditChainCorruption, match="2 of 5 records are reachable"):
            log.require_intact()

    def test_rehashing_a_tampered_entry_still_breaks_its_successor(
        self, tmp_path: Path
    ) -> None:
        """The forger who repairs one entry's own hash still orphans the rest."""

        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 5)

        lines = _lines(path)
        forged = AuditEntry(**json.loads(lines[2]))
        forged.payload = {"i": 999}
        forged.entry_hash = forged.compute_hash()
        lines[2] = json.dumps(forged.to_dict(), ensure_ascii=False)
        _rewrite(path, lines)

        result = log.verify()
        assert result["valid"] is False
        assert "does not chain to its predecessor" in result["error"]
        assert result["reachable"] == 3
        assert result["orphaned"] == 2

    def test_reordering_two_entries_is_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 4)
        lines = _lines(path)
        lines[1], lines[2] = lines[2], lines[1]
        _rewrite(path, lines)

        assert log.verify()["valid"] is False
        with pytest.raises(AuditChainCorruption):
            log.require_intact()


class TestForkedChain:
    """The exact A-04 shape: two branches claiming the same position."""

    def _forked_log(self, tmp_path: Path) -> tuple[AuditLog, Path]:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        entries = _fill(log, 3, actor="winner")

        # A competing branch chained off entry 0, exactly what two unsynchronised
        # writers used to produce.
        rival_prev = entries[0].entry_hash
        rival_lines: list[str] = []
        for i in range(3):
            rival = AuditEntry(
                sequence=i + 1,
                recorded_at=entries[0].recorded_at,
                kind="ENVELOPE",
                actor="loser",
                subject=SUBJECT,
                payload={"i": i},
                prev_hash=rival_prev,
            )
            rival.entry_hash = rival.compute_hash()
            rival_prev = rival.entry_hash
            rival_lines.append(json.dumps(rival.to_dict(), ensure_ascii=False))

        _rewrite(path, _lines(path) + rival_lines)
        return log, path

    def test_orphans_are_counted_not_merely_flagged(self, tmp_path: Path) -> None:
        log, _ = self._forked_log(tmp_path)

        result = log.verify()
        assert result["valid"] is False
        assert result["entries_total"] == 6
        assert result["reachable"] == 3
        assert result["orphaned"] == 3

    def test_require_intact_names_the_orphan_count(self, tmp_path: Path) -> None:
        log, _ = self._forked_log(tmp_path)
        with pytest.raises(AuditChainCorruption) as excinfo:
            log.require_intact()
        message = str(excinfo.value)
        assert "3 of 6 records are reachable from genesis" in message
        assert "3 are orphaned" in message


class TestTailReading:
    @pytest.mark.parametrize("count", [0, 1, 2, 3, 250])
    def test_seeked_tail_matches_a_full_scan(self, tmp_path: Path, count: int) -> None:
        """`append` reads the tail by seeking; that must equal the honest scan."""

        log = AuditLog(tmp_path / "audit.jsonl")
        _fill(log, count)

        scanned = None
        for entry in log.entries():
            scanned = entry
        assert log.last_entry() == scanned

    def test_tail_survives_non_ascii_payloads_and_blank_lines(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(40):
            log.append(
                kind="DECISION",
                actor="治理",
                subject="全宇宙面板复核",
                payload={"理由": "证据缺失记 unknown" * 20, "i": i},
            )
        expected = log.last_entry()

        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert log.last_entry() == expected

        scanned = None
        for entry in log.entries():
            scanned = entry
        assert scanned == expected

    def test_append_refuses_to_chain_onto_an_edited_tail(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 3)

        lines = _lines(path)
        tampered = json.loads(lines[-1])
        tampered["payload"] = {"i": 999}
        lines[-1] = json.dumps(tampered, ensure_ascii=False)
        _rewrite(path, lines)

        with pytest.raises(AuditChainCorruption, match="does not match its recorded"):
            log.append(kind="DECISION", actor="a", subject=SUBJECT, payload={})
        assert len(_lines(path)) == 3, "a refused append must not write anything"


class TestDurableWriteLatch:
    def test_append_fsyncs_the_log_it_just_wrote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        synced: list[str] = []
        real_fsync = os.fsync

        def _recording_fsync(fd: int) -> None:
            try:
                synced.append(os.readlink(f"/proc/self/fd/{fd}"))
            except OSError:  # pragma: no cover - non-Linux host
                synced.append("<unresolved>")
            real_fsync(fd)

        monkeypatch.setattr(audit_module.os, "fsync", _recording_fsync)
        log.append(kind="DECISION", actor="a", subject=SUBJECT, payload={})
        assert str(path) in synced

    def test_failed_durable_write_latches_the_log_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        _fill(log, 2)

        def _broken_fsync(fd: int) -> None:
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(audit_module.os, "fsync", _broken_fsync)
        with pytest.raises(OSError):
            log.append(kind="DECISION", actor="a", subject=SUBJECT, payload={"n": 3})
        monkeypatch.undo()

        # DEF-017: this instance does not get to guess whether those bytes
        # landed, so it refuses further appends rather than resynchronising.
        with pytest.raises(AuditWriteUnavailable, match="latched closed"):
            log.append(kind="DECISION", actor="a", subject=SUBJECT, payload={"n": 4})

        # A fresh reader re-reads the file honestly and continues from what is
        # actually there -- the documented recovery path, not an in-place resync.
        reopened = AuditLog(path)
        reopened.append(kind="DECISION", actor="a", subject=SUBJECT, payload={"n": 5})
        assert reopened.require_intact()["entries_total"] == len(_lines(path))


class TestInProcessConcurrency:
    def test_threads_sharing_one_log_do_not_collide(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        start = threading.Barrier(6)

        def _worker(tag: int) -> None:
            start.wait(timeout=30)
            for i in range(20):
                log.append(
                    kind="ENVELOPE", actor=f"t{tag}", subject=SUBJECT,
                    payload={"i": i},
                )

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not [t for t in threads if t.is_alive()]

        result = log.require_intact()
        assert result["entries_total"] == 120
        assert result["reachable"] == 120

    def test_separate_instances_on_one_path_do_not_collide(
        self, tmp_path: Path
    ) -> None:
        """Two components each holding their own handle to the same log."""

        path = tmp_path / "audit.jsonl"
        start = threading.Barrier(4)

        def _worker(tag: int) -> None:
            log = AuditLog(path)
            start.wait(timeout=30)
            for i in range(15):
                log.append(
                    kind="ENVELOPE", actor=f"c{tag}", subject=SUBJECT,
                    payload={"i": i},
                )

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not [t for t in threads if t.is_alive()]

        assert AuditLog(path).require_intact()["entries_total"] == 60
