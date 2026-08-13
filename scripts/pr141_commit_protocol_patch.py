from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    # Pending artifacts are staging; execution requires an append-only freeze
    # bound to their exact payload digest.
    replace_once(
        "src/quantagent/paper/pending_signal.py",
        '''PENDING_SIGNAL_SCHEMA_VERSION = "paper_pending_signal_v1"\nPENDING_SIGNAL_STATUS = "pending_next_observed_session"\n''',
        '''PENDING_SIGNAL_SCHEMA_VERSION = "paper_pending_signal_v1"\nPENDING_SIGNAL_STATUS = "pending_next_observed_session"\nPENDING_COMMIT_PROTOCOL = "pending_bound_daily_decision_v1"\n''',
    )
    replace_once(
        "src/quantagent/paper/pending_signal.py",
        '''            return persisted, path\n\n\n__all__ = [\n''',
        '''            return persisted, path\n\n    def discard_staged(\n        self,\n        signal_date: str,\n        *,\n        expected_payload_sha256: str,\n    ) -> bool:\n        """Delete one verified *uncommitted* staging artifact by exact identity.\n\n        Commit-state knowledge deliberately remains outside this store. Callers\n        must hold the account lock, prove the journal has no matching committed\n        daily decision, and pass the exact staged payload hash. This method only\n        makes the per-date deletion itself serialized and durable.\n        """\n        signal_date = pd.Timestamp(signal_date).date().isoformat()\n        path = self.path_for(signal_date)\n        with self._thread_lock, self._exclusive_signal_lock(signal_date):\n            existing = self.read(signal_date)\n            if existing is None:\n                return False\n            if existing.payload_sha256 != str(expected_payload_sha256):\n                raise PendingSignalConflict(\n                    "refusing to discard staged pending signal with mismatched payload identity"\n                )\n            path.unlink()\n            if os.name != "nt":\n                directory_fd = os.open(path.parent, os.O_RDONLY)\n                try:\n                    os.fsync(directory_fd)\n                finally:\n                    os.close(directory_fd)\n            return True\n\n\n__all__ = [\n''',
    )
    replace_once(
        "src/quantagent/paper/pending_signal.py",
        '''    "PENDING_SIGNAL_STATUS",\n''',
        '''    "PENDING_SIGNAL_STATUS",\n    "PENDING_COMMIT_PROTOCOL",\n''',
    )

    # Daily loop imports the protocol marker.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''from quantagent.paper.pending_signal import PendingPaperSignalStore\n''',
        '''from quantagent.paper.pending_signal import (\n    PENDING_COMMIT_PROTOCOL,\n    PendingPaperSignalStore,\n)\n''',
    )

    # Same-date guard: committed marker is authoritative; a current-protocol
    # staging orphan with no marker may be safely recovered under account lock.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    existing = PendingPaperSignalStore(config.pending_signal_dir).read(as_of)\n    if existing is not None:\n        raise PaperAccountStateRefused(\n            "pending paper signal is already frozen for the current signal date; "\n            "refusing to overwrite its hash-bound predictions/target artifacts"\n        )\n    summary_path = Path(config.output_root) / as_of / "daily_loop_summary.json"\n    if summary_path.exists():\n        raise PaperAccountStateRefused(\n            "daily paper decision evidence already exists for the current signal date; "\n            "refusing to overwrite its summary/prediction/target lineage"\n        )\n    journal = PendingExecutionJournal(_execution_journal_path(config))\n    if not journal.verify():\n        raise PaperAccountStateRefused(\n            "pending execution journal verification failed before same-date freeze check"\n        )\n    if journal.daily_decision(as_of) is not None:\n        raise PaperAccountStateRefused(\n            "daily paper decision is already durably frozen for the current signal date"\n        )\n''',
        '''    store = PendingPaperSignalStore(config.pending_signal_dir)\n    journal = PendingExecutionJournal(_execution_journal_path(config))\n    if not journal.verify():\n        raise PaperAccountStateRefused(\n            "pending execution journal verification failed before same-date freeze check"\n        )\n    if journal.daily_decision(as_of) is not None:\n        raise PaperAccountStateRefused(\n            "daily paper decision is already durably frozen for the current signal date"\n        )\n    summary_path = Path(config.output_root) / as_of / "daily_loop_summary.json"\n    if summary_path.exists():\n        raise PaperAccountStateRefused(\n            "daily paper decision evidence already exists for the current signal date; "\n            "refusing to overwrite its summary/prediction/target lineage"\n        )\n    existing = store.read(as_of)\n    if existing is not None:\n        if (\n            existing.source_lineage.get("daily_decision_commit_protocol")\n            == PENDING_COMMIT_PROTOCOL\n        ):\n            # Current-protocol pending without a freeze is staging left by a\n            # crash before commit. The execution consumer rejects it, so under\n            # the same account lock it is safe to remove and recompute.\n            store.discard_staged(\n                as_of, expected_payload_sha256=existing.payload_sha256\n            )\n        else:\n            raise PaperAccountStateRefused(\n                "legacy/ambiguous pending paper signal exists without a committed "\n                "daily-decision marker; operator reconciliation is required"\n            )\n''',
    )

    # Bind target freeze to exact pending and target digests.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''def _freeze_daily_decision(\n    config: DailyPaperLoopConfig,\n    as_of: str,\n    *,\n    decision_kind: str,\n    paper_account_identity_sha256: str,\n    account_evidence: dict[str, object],\n) -> None:\n    """Append an irreversible same-date marker before writing decision artifacts."""\n\n    material = (\n        "quantagent.paper.daily_decision.v1|"\n        f"{as_of}|{paper_account_identity_sha256}"\n    ).encode("utf-8")\n    journal = PendingExecutionJournal(_execution_journal_path(config))\n''',
        '''def _freeze_daily_decision(\n    config: DailyPaperLoopConfig,\n    as_of: str,\n    *,\n    decision_kind: str,\n    paper_account_identity_sha256: str,\n    account_evidence: dict[str, object],\n    pending_payload_sha256: str | None = None,\n    target_weights_sha256: str | None = None,\n) -> None:\n    """Append the irreversible commit marker for one validated daily decision."""\n\n    if decision_kind == "target":\n        if len(str(pending_payload_sha256 or "")) != 64:\n            raise PaperAccountStateRefused(\n                "target daily decision requires exact pending payload binding"\n            )\n        if len(str(target_weights_sha256 or "")) != 64:\n            raise PaperAccountStateRefused(\n                "target daily decision requires exact target-weight binding"\n            )\n        marker_identity = str(pending_payload_sha256)\n    else:\n        material = (\n            "quantagent.paper.daily_decision.v1|"\n            f"{as_of}|{paper_account_identity_sha256}"\n        ).encode("utf-8")\n        marker_identity = sha256(material).hexdigest()\n\n    journal = PendingExecutionJournal(_execution_journal_path(config))\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''        pending_payload_sha256=sha256(material).hexdigest(),\n''',
        '''        pending_payload_sha256=marker_identity,\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''            "assurance": "canonical_account_daily_decision_freeze_v1",\n        },\n''',
        '''            "assurance": "canonical_account_daily_decision_freeze_v1",\n            "commit_protocol": (\n                PENDING_COMMIT_PROTOCOL if decision_kind == "target" else "no_target_v1"\n            ),\n            "target_weights_sha256": (\n                str(target_weights_sha256) if decision_kind == "target" else ""\n            ),\n        },\n''',
    )

    # New target intent explicitly identifies staging protocol and only becomes
    # executable after the bound freeze append succeeds.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''                "canonical_account_nav": str(account_evidence["nav"]),\n            },\n        )\n        _freeze_daily_decision(\n            config,\n            as_of,\n            decision_kind="target",\n            paper_account_identity_sha256=account_identity.payload_sha256,\n            account_evidence=account_evidence,\n        )\n''',
        '''                "canonical_account_nav": str(account_evidence["nav"]),\n                "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,\n            },\n        )\n        try:\n            _freeze_daily_decision(\n                config,\n                as_of,\n                decision_kind="target",\n                paper_account_identity_sha256=account_identity.payload_sha256,\n                account_evidence=account_evidence,\n                pending_payload_sha256=pending.payload_sha256,\n                target_weights_sha256=pending.target_weights_sha256,\n            )\n        except Exception:\n            # If append definitively did not commit, remove the current-protocol\n            # staging artifact. If a marker is visible, the transaction committed\n            # even if the caller did not receive success; preserve it.\n            decision = PendingExecutionJournal(\n                _execution_journal_path(config)\n            ).daily_decision(as_of)\n            if decision is None:\n                PendingPaperSignalStore(config.pending_signal_dir).discard_staged(\n                    as_of, expected_payload_sha256=pending.payload_sha256\n                )\n            raise\n''',
    )

    # Chronological monotonicity: any later durable journal history or pending
    # intent makes a backdated decision unsafe.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    prefix_index = build_canonical_prefix_index(config.canonical_ledger_path)\n    _assert_execution_journal_resolved(\n''',
        '''    cutoff = pd.Timestamp(as_of).date()\n    later_records = []\n    for record in journal.records():\n        try:\n            record_date = pd.Timestamp(record.signal_date).date()\n        except (TypeError, ValueError) as exc:\n            raise PaperAccountStateRefused(\n                f"invalid journal signal_date during chronology check: {record.signal_date!r}"\n            ) from exc\n        if record_date > cutoff:\n            later_records.append(record)\n    if later_records:\n        first = min(later_records, key=lambda row: row.signal_date)\n        raise PaperAccountStateRefused(\n            "paper decision chronology regression refused: later durable journal "\n            f"evidence already exists for signal_date={first.signal_date}"\n        )\n\n    prefix_index = build_canonical_prefix_index(config.canonical_ledger_path)\n    _assert_execution_journal_resolved(\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    cutoff = pd.Timestamp(as_of).date()\n    for path in sorted(root.glob("*.json")):\n''',
        '''    for path in sorted(root.glob("*.json")):\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''        if signal_date >= cutoff:\n            continue\n''',
        '''        if signal_date > cutoff:\n            raise PaperAccountStateRefused(\n                "paper decision chronology regression refused: later pending signal "\n                f"already exists for signal_date={signal_date.isoformat()}"\n            )\n        if signal_date == cutoff:\n            continue\n''',
    )

    # Consumer imports commit protocol and verifies marker-to-pending identity
    # before considering any unexecuted pending economically eligible.
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''from quantagent.paper.execution_journal import (\n    LEGACY_BINDING_STATUS,\n''',
        '''from quantagent.paper.execution_journal import (\n    DAILY_DECISION_STATUS,\n    LEGACY_BINDING_STATUS,\n''',
    )
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''from quantagent.paper.pending_signal import PendingPaperSignal, PendingPaperSignalStore\n''',
        '''from quantagent.paper.pending_signal import (\n    PENDING_COMMIT_PROTOCOL,\n    PendingPaperSignal,\n    PendingPaperSignalStore,\n)\n''',
    )
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''def execute_pending_for_session(\n''',
        '''def _verify_pending_daily_commit(\n    journal: PendingExecutionJournal,\n    pending: PendingPaperSignal,\n    *,\n    expected_paper_account_identity_sha256: str,\n) -> None:\n    if (\n        pending.source_lineage.get("daily_decision_commit_protocol")\n        != PENDING_COMMIT_PROTOCOL\n    ):\n        raise ContinuousPaperExecutionBlocked(\n            "pending signal lacks the staged-intent commit protocol; explicit "\n            "regeneration/reconciliation is required before execution"\n        )\n    decision = journal.daily_decision(pending.signal_date)\n    if decision is None:\n        raise ContinuousPaperExecutionBlocked(\n            "pending signal is staged but not committed by a daily_decision_frozen record"\n        )\n    details = dict(decision.details or {})\n    checks = {\n        "decision_kind": (str(details.get("decision_kind") or ""), "target"),\n        "commit_protocol": (str(details.get("commit_protocol") or ""), PENDING_COMMIT_PROTOCOL),\n        "pending_payload_sha256": (decision.pending_payload_sha256, pending.payload_sha256),\n        "target_weights_sha256": (\n            str(details.get("target_weights_sha256") or ""),\n            pending.target_weights_sha256,\n        ),\n        "paper_account_identity_sha256": (\n            str(details.get("paper_account_identity_sha256") or ""),\n            expected_paper_account_identity_sha256,\n        ),\n        "canonical_account_state_sha256": (\n            str(details.get("canonical_account_state_sha256") or ""),\n            str(pending.source_lineage.get("canonical_account_state_sha256") or ""),\n        ),\n        "canonical_head": (\n            str(details.get("canonical_head") or ""),\n            str(pending.source_lineage.get("canonical_ledger_head_hash") or ""),\n        ),\n        "canonical_records": (\n            str(details.get("canonical_records") or ""),\n            str(pending.source_lineage.get("canonical_ledger_records") or ""),\n        ),\n    }\n    mismatches = [name for name, (left, right) in checks.items() if left != right]\n    if mismatches:\n        raise ContinuousPaperExecutionBlocked(\n            "pending signal commit binding mismatch: " + ",".join(sorted(mismatches))\n        )\n\n\ndef execute_pending_for_session(\n''',
    )
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''        pending_identity_sha = str(\n            pending.source_lineage.get("paper_account_identity_sha256", "")\n        )\n''',
        '''        _verify_pending_daily_commit(\n            journal,\n            pending,\n            expected_paper_account_identity_sha256=account_identity.payload_sha256,\n        )\n\n        pending_identity_sha = str(\n            pending.source_lineage.get("paper_account_identity_sha256", "")\n        )\n''',
    )

    # Persisted tracker symbols are not continuity proof. Explicit account
    # snapshots without a per-episode proof conservatively restart age/horizon.
    replace_once(
        "src/quantagent/portfolio/position_age_tracker.py",
        '''    def begin_session(\n        self,\n        initial_weights: Mapping[str, float] | None,\n        expected_horizons: Mapping[str, int | None] | None = None,\n    ) -> None:\n''',
        '''    def begin_session(\n        self,\n        initial_weights: Mapping[str, float] | None,\n        expected_horizons: Mapping[str, int | None] | None = None,\n        *,\n        continuity_proven_symbols: set[str] | None = None,\n    ) -> None:\n''',
    )
    replace_once(
        "src/quantagent/portfolio/position_age_tracker.py",
        '''        expected_horizons = dict(expected_horizons or {})\n        supplied = initial_weights is not None\n''',
        '''        expected_horizons = dict(expected_horizons or {})\n        supplied = initial_weights is not None\n        continuity_proven = {str(symbol) for symbol in (continuity_proven_symbols or set())}\n''',
    )
    replace_once(
        "src/quantagent/portfolio/position_age_tracker.py",
        '''            if existing is not None:\n                existing.weight = weight\n                if (\n                    symbol in expected_horizons\n                    and expected_horizons[symbol] is not None\n                    and existing.expected_horizon_days\n                    in {None, UNKNOWN_INITIAL_HORIZON_DAYS}\n                ):\n                    existing.expected_horizon_days = int(expected_horizons[symbol])\n                continue\n''',
        '''            if existing is not None:\n                existing.weight = weight\n                if supplied and symbol not in continuity_proven:\n                    # Symbol equality does not prove an uninterrupted lot. The\n                    # account may have sold and reacquired while this tracker was\n                    # offline, so restart conservatively at unknown age.\n                    supplied_horizon = expected_horizons.get(symbol)\n                    existing.entry_date = None\n                    existing.last_seen = None\n                    existing.days_held = 0\n                    existing.expected_horizon_days = (\n                        int(supplied_horizon)\n                        if supplied_horizon is not None\n                        else UNKNOWN_INITIAL_HORIZON_DAYS\n                    )\n                    continue\n                if (\n                    symbol in expected_horizons\n                    and expected_horizons[symbol] is not None\n                    and existing.expected_horizon_days\n                    in {None, UNKNOWN_INITIAL_HORIZON_DAYS}\n                ):\n                    existing.expected_horizon_days = int(expected_horizons[symbol])\n                continue\n''',
    )

    # ----- Regression wiring -------------------------------------------------
    # Existing continuous-execution helper now creates a committed target marker.
    p = Path("tests/paper/test_continuous_pending_execution.py")
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        'from quantagent.paper.execution_journal import PendingExecutionJournal\n',
        'from quantagent.paper.execution_journal import DAILY_DECISION_STATUS, PendingExecutionJournal\n',
        1,
    )
    text = text.replace(
        'from quantagent.paper.pending_signal import PendingPaperSignalStore\n',
        'from quantagent.paper.pending_signal import PENDING_COMMIT_PROTOCOL, PendingPaperSignalStore\n',
        1,
    )
    old = '''def _record(tmp_path, signal_date: str, weight: float):\n    identity = _identity(tmp_path)\n    return PendingPaperSignalStore(tmp_path / "pending").record(\n        signal_date=signal_date,\n        target_weights=_target(signal_date, weight),\n        source_lineage={\n            "model": "test-model",\n            "target_weights_file_sha256": f"sha-{signal_date}-{weight}",\n            "paper_account_identity_sha256": identity.payload_sha256,\n        },\n        created_at=f"{signal_date}T07:00:00+00:00",\n    )[0]\n'''
    new = '''def _record(tmp_path, signal_date: str, weight: float):\n    identity = _identity(tmp_path)\n    pending = PendingPaperSignalStore(tmp_path / "pending").record(\n        signal_date=signal_date,\n        target_weights=_target(signal_date, weight),\n        source_lineage={\n            "model": "test-model",\n            "target_weights_file_sha256": f"sha-{signal_date}-{weight}",\n            "paper_account_identity_sha256": identity.payload_sha256,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_ledger_head_hash": "0" * 64,\n            "canonical_ledger_records": "0",\n            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,\n        },\n        created_at=f"{signal_date}T07:00:00+00:00",\n    )[0]\n    PendingExecutionJournal(tmp_path / "execution.jsonl").append(\n        pending_payload_sha256=pending.payload_sha256,\n        signal_date=signal_date,\n        execution_date=signal_date,\n        status=DAILY_DECISION_STATUS,\n        details={\n            "decision_kind": "target",\n            "paper_account_identity_sha256": identity.payload_sha256,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_records": 0,\n            "canonical_head": "0" * 64,\n            "assurance": "canonical_account_daily_decision_freeze_v1",\n            "commit_protocol": PENDING_COMMIT_PROTOCOL,\n            "target_weights_sha256": pending.target_weights_sha256,\n        },\n    )\n    return pending\n'''
    if old not in text:
        raise RuntimeError("continuous _record helper anchor missing")
    text = text.replace(old, new, 1)
    if "test_uncommitted_staged_pending_is_never_executed" not in text:
        text = text.rstrip() + '''\n\n\ndef test_uncommitted_staged_pending_is_never_executed(tmp_path) -> None:\n    config = _config(tmp_path)\n    identity = _identity(tmp_path)\n    pending = PendingPaperSignalStore(tmp_path / "pending").record(\n        signal_date=FRIDAY,\n        target_weights=_target(FRIDAY, 0.50),\n        source_lineage={\n            "model": "test-model",\n            "paper_account_identity_sha256": identity.payload_sha256,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_ledger_head_hash": "0" * 64,\n            "canonical_ledger_records": "0",\n            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,\n        },\n        created_at=f"{FRIDAY}T07:00:00+00:00",\n    )[0]\n    with pytest.raises(ContinuousPaperExecutionBlocked, match="staged but not committed"):\n        execute_pending_for_session(\n            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS\n        )\n    assert PendingExecutionJournal(config.execution_journal_path).terminal(\n        pending.payload_sha256\n    ) is None\n    assert not (tmp_path / "canonical.jsonl").exists()\n\n\ndef test_mismatched_daily_commit_cannot_execute_pending(tmp_path) -> None:\n    config = _config(tmp_path)\n    identity = _identity(tmp_path)\n    pending = PendingPaperSignalStore(tmp_path / "pending").record(\n        signal_date=FRIDAY,\n        target_weights=_target(FRIDAY, 0.50),\n        source_lineage={\n            "model": "test-model",\n            "paper_account_identity_sha256": identity.payload_sha256,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_ledger_head_hash": "0" * 64,\n            "canonical_ledger_records": "0",\n            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,\n        },\n        created_at=f"{FRIDAY}T07:00:00+00:00",\n    )[0]\n    PendingExecutionJournal(config.execution_journal_path).append(\n        pending_payload_sha256="f" * 64,\n        signal_date=FRIDAY,\n        execution_date=FRIDAY,\n        status=DAILY_DECISION_STATUS,\n        details={\n            "decision_kind": "target",\n            "paper_account_identity_sha256": identity.payload_sha256,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_records": 0,\n            "canonical_head": "0" * 64,\n            "assurance": "canonical_account_daily_decision_freeze_v1",\n            "commit_protocol": PENDING_COMMIT_PROTOCOL,\n            "target_weights_sha256": pending.target_weights_sha256,\n        },\n    )\n    with pytest.raises(ContinuousPaperExecutionBlocked, match="pending_payload_sha256"):\n        execute_pending_for_session(\n            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS\n        )\n'''
    p.write_text(text, encoding="utf-8")

    # Existing foreign-identity test needs a committed marker so identity check is reached.
    p = Path("tests/paper/test_continuous_pending_execution.py")
    text = p.read_text(encoding="utf-8")
    old = '''    pending = PendingPaperSignalStore(tmp_path / "pending").record(\n        signal_date=FRIDAY,\n        target_weights=_target(FRIDAY, 0.50),\n        source_lineage={\n            "model": "test-model",\n            "paper_account_identity_sha256": "0" * 64,\n        },\n        created_at=f"{FRIDAY}T07:00:00+00:00",\n    )[0]\n    with pytest.raises(ContinuousPaperExecutionBlocked, match="mismatched paper-account identity"):\n'''
    new = '''    pending = PendingPaperSignalStore(tmp_path / "pending").record(\n        signal_date=FRIDAY,\n        target_weights=_target(FRIDAY, 0.50),\n        source_lineage={\n            "model": "test-model",\n            "paper_account_identity_sha256": "0" * 64,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_ledger_head_hash": "0" * 64,\n            "canonical_ledger_records": "0",\n            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,\n        },\n        created_at=f"{FRIDAY}T07:00:00+00:00",\n    )[0]\n    PendingExecutionJournal(config.execution_journal_path).append(\n        pending_payload_sha256=pending.payload_sha256,\n        signal_date=FRIDAY,\n        execution_date=FRIDAY,\n        status=DAILY_DECISION_STATUS,\n        details={\n            "decision_kind": "target",\n            "paper_account_identity_sha256": "0" * 64,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_records": 0,\n            "canonical_head": "0" * 64,\n            "assurance": "canonical_account_daily_decision_freeze_v1",\n            "commit_protocol": PENDING_COMMIT_PROTOCOL,\n            "target_weights_sha256": pending.target_weights_sha256,\n        },\n    )\n    with pytest.raises(ContinuousPaperExecutionBlocked, match="paper_account_identity_sha256"):\n'''
    if old in text:
        text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")

    # Daily-loop crash recovery and chronology regressions.
    p = Path("tests/paper/test_daily_loop_pending_execution.py")
    text = p.read_text(encoding="utf-8")
    if "test_freeze_failure_discards_uncommitted_current_protocol_staging" not in text:
        text = text.rstrip() + '''\n\n\ndef test_freeze_failure_discards_uncommitted_current_protocol_staging(tmp_path, monkeypatch) -> None:\n    targets = pd.DataFrame(\n        {"trade_date": [pd.Timestamp("2026-08-07")], "600000.SH": [0.5]}\n    )\n    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=targets)\n    original_append = daily_loop.PendingExecutionJournal.append\n\n    def fail_daily_freeze(self, **kwargs):\n        if kwargs.get("status") == daily_loop.DAILY_DECISION_STATUS:\n            raise OSError("simulated freeze fsync failure")\n        return original_append(self, **kwargs)\n\n    monkeypatch.setattr(daily_loop.PendingExecutionJournal, "append", fail_daily_freeze)\n    with pytest.raises(OSError, match="freeze fsync failure"):\n        daily_loop.run_once(config)\n    assert not (tmp_path / "pending" / f"{as_of}.json").exists()\n    assert daily_loop.PendingExecutionJournal(\n        daily_loop._execution_journal_path(config)\n    ).daily_decision(as_of) is None\n'''
    p.write_text(text, encoding="utf-8")

    p = Path("tests/paper/test_daily_loop_prior_execution_gate.py")
    text = p.read_text(encoding="utf-8")
    if "test_backdated_decision_refuses_later_pending_signal" not in text:
        text = text.rstrip() + '''\n\n\ndef test_backdated_decision_refuses_later_pending_signal(tmp_path) -> None:\n    later = _daily_config(tmp_path, "2026-08-12")\n    _pending(later, "2026-08-12")\n    earlier = _daily_config(tmp_path, "2026-08-11")\n    identity = ensure_paper_account_identity(\n        canonical_ledger_path=earlier.canonical_ledger_path,\n        portfolio_id=earlier.portfolio_id,\n        initial_cash=earlier.initial_cash,\n        identity_path=earlier.account_identity_path,\n    )\n    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later pending"):\n        _assert_prior_pending_signals_resolved(\n            earlier,\n            "2026-08-11",\n            paper_account_identity_sha256=identity.payload_sha256,\n        )\n\n\ndef test_backdated_decision_refuses_later_durable_journal_record(tmp_path) -> None:\n    earlier = _daily_config(tmp_path, "2026-08-11")\n    identity = ensure_paper_account_identity(\n        canonical_ledger_path=earlier.canonical_ledger_path,\n        portfolio_id=earlier.portfolio_id,\n        initial_cash=earlier.initial_cash,\n        identity_path=earlier.account_identity_path,\n    )\n    PendingExecutionJournal(earlier.execution_journal_path).append(\n        pending_payload_sha256="f" * 64,\n        signal_date="2026-08-12",\n        execution_date="2026-08-12",\n        status="execution_started",\n        details={"paper_account_identity_sha256": identity.payload_sha256},\n    )\n    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later durable"):\n        _assert_prior_pending_signals_resolved(\n            earlier,\n            "2026-08-11",\n            paper_account_identity_sha256=identity.payload_sha256,\n        )\n'''
    p.write_text(text, encoding="utf-8")

    # Tracker restart tests: no proof resets; explicit continuity proof preserves.
    p = Path("tests/portfolio/test_position_age_tracker_account_reconciliation.py")
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        '''    # Canonical account recovery is authoritative: SOLD was actually liquidated\n    # before restart, so its persisted lifecycle history must not survive into\n    # the first-date lock calculation. HELD keeps its historical age/horizon.\n    restarted.begin_session({"HELD": 0.30})\n''',
        '''    # Canonical account recovery proves current holdings, but symbol equality\n    # alone does not prove HELD was continuously held while the tracker was offline.\n    restarted.begin_session({"HELD": 0.30})\n''',
    )
    text = text.replace(
        '''    assert int(snapshot.loc["HELD", "expected_horizon_days"]) == 60\n''',
        '''    assert int(snapshot.loc["HELD", "expected_horizon_days"]) == UNKNOWN_INITIAL_HORIZON_DAYS\n    assert restarted.age_for("HELD", pd.Timestamp("2026-08-05")) == 0\n''',
        1,
    )
    if "test_matching_symbol_restart_resets_age_without_continuity_proof" not in text:
        text = text.rstrip() + '''\n\n\ndef test_matching_symbol_restart_resets_age_without_continuity_proof(tmp_path) -> None:\n    state_path = tmp_path / "position_age.parquet"\n    tracker = PositionAgeTracker(state_path=state_path)\n    tracker.record_session(pd.Timestamp("2026-08-01"), {"A": 0.2}, {"A": 5})\n    tracker.record_session(pd.Timestamp("2026-08-02"), {"A": 0.2}, {"A": 5})\n    tracker.persist()\n\n    restarted = PositionAgeTracker.from_state(state_path)\n    assert restarted.age_for("A", pd.Timestamp("2026-08-05")) >= 2\n    restarted.begin_session({"A": 0.2}, {"A": 5})\n    assert restarted.age_for("A", pd.Timestamp("2026-08-05")) == 0\n    assert restarted.is_locked("A", pd.Timestamp("2026-08-05"))\n\n\ndef test_explicit_continuity_proof_preserves_persisted_age(tmp_path) -> None:\n    state_path = tmp_path / "position_age.parquet"\n    tracker = PositionAgeTracker(state_path=state_path)\n    tracker.record_session(pd.Timestamp("2026-08-01"), {"A": 0.2}, {"A": 5})\n    tracker.record_session(pd.Timestamp("2026-08-02"), {"A": 0.2}, {"A": 5})\n    tracker.persist()\n\n    restarted = PositionAgeTracker.from_state(state_path)\n    before = restarted.age_for("A", pd.Timestamp("2026-08-05"))\n    restarted.begin_session(\n        {"A": 0.2},\n        {"A": 5},\n        continuity_proven_symbols={"A"},\n    )\n    assert restarted.age_for("A", pd.Timestamp("2026-08-05")) == before\n'''
    p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
