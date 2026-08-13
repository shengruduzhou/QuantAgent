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
    # --- Daily summary becomes a bound staging artifact ---------------------
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''from hashlib import sha256\nimport json\nfrom pathlib import Path\n\nimport pandas as pd\n''',
        '''from hashlib import sha256\nimport json\nimport os\nfrom pathlib import Path\nfrom uuid import uuid4\n\nimport pandas as pd\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''from quantagent.data.ingestion.daily_evidence_job import DailyEvidenceJob, DailyEvidenceJobConfig\n''',
        '''from quantagent.data.ingestion.daily_evidence_job import DailyEvidenceJob, DailyEvidenceJobConfig\nfrom quantagent.domain.ledger import CanonicalLedger\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''_RESOLVED_PRIOR_TERMINAL_STATUSES = frozenset(\n''',
        '''DAILY_SUMMARY_COMMIT_PROTOCOL = "daily_summary_bound_daily_decision_v1"\n\n\n_RESOLVED_PRIOR_TERMINAL_STATUSES = frozenset(\n''',
    )

    # No-target: build and durably stage the main audit summary before freeze.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''            weights_path = write_v7_target_weights(\n                weights,\n                day_dir / "target_weights.parquet",\n            )\n            _freeze_daily_decision(\n                config,\n                as_of,\n                decision_kind="no_target",\n                paper_account_identity_sha256=account_identity.payload_sha256,\n                account_evidence=account_evidence,\n            )\n        warnings = tuple([*evidence.warnings, "paper_no_target_generated"])\n        summary = {\n''',
        '''            weights_path = write_v7_target_weights(\n                weights,\n                day_dir / "target_weights.parquet",\n            )\n            warnings = tuple([*evidence.warnings, "paper_no_target_generated"])\n            summary = {\n                "daily_decision_commit_protocol": DAILY_SUMMARY_COMMIT_PROTOCOL,\n                "config": asdict(config),\n                "status": "no_target_generated",\n                "evidence_rows": int(len(evidence.frame)),\n                "evidence_warnings": list(evidence.warnings),\n                "account_identity": identity_evidence,\n                "account_state": account_evidence,\n                "blend_diagnostics": blend_result.diagnostics,\n                "target_weight_diagnostics": weights.diagnostics,\n                "execution": {\n                    "status": "no_target_generated",\n                    "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,\n                    "signal_date": as_of,\n                    "pending_signal_path": "",\n                    "executed_fill_count": 0,\n                    "paper_report_written": False,\n                    "paper_book_appended": False,\n                    "paper_account_identity_sha256": account_identity.payload_sha256,\n                    "canonical_account_state_sha256": account_state.account_state_sha256,\n                    "reason": (\n                        "portfolio construction produced no target; absence of a target "\n                        "is not reinterpreted as an all-zero liquidation instruction"\n                    ),\n                },\n                "paper_report": None,\n                "paper_book_path": str(config.paper_book_path),\n            }\n            summary_path = _write_daily_summary(day_dir, summary)\n            summary_sha = _file_sha256(summary_path)\n            try:\n                _freeze_daily_decision(\n                    config,\n                    as_of,\n                    decision_kind="no_target",\n                    paper_account_identity_sha256=account_identity.payload_sha256,\n                    account_evidence=account_evidence,\n                    daily_summary_path=summary_path,\n                    daily_summary_sha256=summary_sha,\n                )\n            except Exception:\n                decision = PendingExecutionJournal(\n                    _execution_journal_path(config)\n                ).daily_decision(as_of)\n                if decision is None:\n                    _discard_staged_daily_summary(\n                        summary_path, expected_sha256=summary_sha\n                    )\n                raise\n        summary = {\n''',
    )
    # Existing post-freeze write must disappear; the in-memory duplicate dict is
    # harmless but is no longer written.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''        _write_daily_summary(day_dir, summary)\n        return DailyPaperLoopResult(\n            status="no_target_generated",\n''',
        '''        return DailyPaperLoopResult(\n            status="no_target_generated",\n''',
    )

    # Target: stage pending + summary, then append one marker binding both.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''        try:\n            _freeze_daily_decision(\n                config,\n                as_of,\n                decision_kind="target",\n                paper_account_identity_sha256=account_identity.payload_sha256,\n                account_evidence=account_evidence,\n                pending_payload_sha256=pending.payload_sha256,\n                target_weights_sha256=pending.target_weights_sha256,\n            )\n        except Exception:\n''',
        '''        warnings = tuple(\n            [*evidence.warnings, "paper_signal_pending_next_observed_session"]\n        )\n        summary = {\n            "daily_decision_commit_protocol": DAILY_SUMMARY_COMMIT_PROTOCOL,\n            "config": asdict(config),\n            "status": "signal_recorded_pending_execution",\n            "evidence_rows": int(len(evidence.frame)),\n            "evidence_warnings": list(evidence.warnings),\n            "account_identity": identity_evidence,\n            "blend_diagnostics": blend_result.diagnostics,\n            "target_weight_diagnostics": weights.diagnostics,\n            "account_state": account_evidence,\n            "execution": {\n                "status": "pending_next_observed_session",\n                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,\n                "signal_date": as_of,\n                "pending_signal_path": str(pending_path),\n                "pending_payload_sha256": pending.payload_sha256,\n                "target_weights_sha256": pending.target_weights_sha256,\n                "predictions_file_sha256": pending.source_lineage["predictions_file_sha256"],\n                "target_weights_file_sha256": pending.source_lineage["target_weights_file_sha256"],\n                "paper_account_identity_sha256": pending.source_lineage["paper_account_identity_sha256"],\n                "canonical_account_state_sha256": pending.source_lineage["canonical_account_state_sha256"],\n                "canonical_ledger_head_hash": pending.source_lineage["canonical_ledger_head_hash"],\n                "executed_fill_count": 0,\n                "paper_report_written": False,\n                "paper_book_appended": False,\n                "reason": (\n                    "T-close target is an intent only; no next observed market session "\n                    "has been executed by this signal-generation call"\n                ),\n            },\n            "paper_report": None,\n            "paper_book_path": str(config.paper_book_path),\n        }\n        summary_path = _write_daily_summary(day_dir, summary)\n        summary_sha = _file_sha256(summary_path)\n        try:\n            _freeze_daily_decision(\n                config,\n                as_of,\n                decision_kind="target",\n                paper_account_identity_sha256=account_identity.payload_sha256,\n                account_evidence=account_evidence,\n                pending_payload_sha256=pending.payload_sha256,\n                target_weights_sha256=pending.target_weights_sha256,\n                daily_summary_path=summary_path,\n                daily_summary_sha256=summary_sha,\n            )\n        except Exception:\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''            if decision is None:\n                PendingPaperSignalStore(config.pending_signal_dir).discard_staged(\n                    as_of, expected_payload_sha256=pending.payload_sha256\n                )\n            raise\n\n    warnings = tuple(\n''',
        '''            if decision is None:\n                PendingPaperSignalStore(config.pending_signal_dir).discard_staged(\n                    as_of, expected_payload_sha256=pending.payload_sha256\n                )\n                _discard_staged_daily_summary(\n                    summary_path, expected_sha256=summary_sha\n                )\n            raise\n\n    warnings = tuple(\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    _write_daily_summary(day_dir, summary)\n    return DailyPaperLoopResult(\n        status="signal_recorded_pending_execution",\n''',
        '''    return DailyPaperLoopResult(\n        status="signal_recorded_pending_execution",\n''',
    )

    # Same-date crash recovery: current-protocol staging summary is removable only
    # if there is no committed marker; legacy/ambiguous summary remains fail-closed.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    summary_path = Path(config.output_root) / as_of / "daily_loop_summary.json"\n    if summary_path.exists():\n        raise PaperAccountStateRefused(\n            "daily paper decision evidence already exists for the current signal date; "\n            "refusing to overwrite its summary/prediction/target lineage"\n        )\n    existing = store.read(as_of)\n''',
        '''    summary_path = Path(config.output_root) / as_of / "daily_loop_summary.json"\n    if summary_path.exists():\n        try:\n            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))\n        except Exception as exc:\n            raise PaperAccountStateRefused(\n                "uncommitted daily summary is unreadable/ambiguous; operator "\n                "reconciliation is required before same-date reuse"\n            ) from exc\n        if (\n            isinstance(summary_payload, dict)\n            and summary_payload.get("daily_decision_commit_protocol")\n            == DAILY_SUMMARY_COMMIT_PROTOCOL\n        ):\n            _discard_staged_daily_summary(\n                summary_path, expected_sha256=_file_sha256(summary_path)\n            )\n        else:\n            raise PaperAccountStateRefused(\n                "legacy/ambiguous daily summary exists without a committed marker; "\n                "refusing same-date overwrite"\n            )\n    existing = store.read(as_of)\n''',
    )

    # Marker requires and binds summary identity for both target and no-target.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    pending_payload_sha256: str | None = None,\n    target_weights_sha256: str | None = None,\n) -> None:\n''',
        '''    pending_payload_sha256: str | None = None,\n    target_weights_sha256: str | None = None,\n    daily_summary_path: Path | None = None,\n    daily_summary_sha256: str | None = None,\n) -> None:\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    if decision_kind == "target":\n''',
        '''    if daily_summary_path is None or not Path(daily_summary_path).is_file():\n        raise PaperAccountStateRefused(\n            "daily decision requires a durable pre-freeze summary artifact"\n        )\n    if len(str(daily_summary_sha256 or "")) != 64:\n        raise PaperAccountStateRefused(\n            "daily decision requires exact summary digest binding"\n        )\n    if _file_sha256(Path(daily_summary_path)) != str(daily_summary_sha256):\n        raise PaperAccountStateRefused(\n            "daily summary changed before decision freeze"\n        )\n\n    if decision_kind == "target":\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''            "target_weights_sha256": (\n                str(target_weights_sha256) if decision_kind == "target" else ""\n            ),\n        },\n''',
        '''            "target_weights_sha256": (\n                str(target_weights_sha256) if decision_kind == "target" else ""\n            ),\n            "daily_summary_path": str(Path(daily_summary_path).resolve(strict=False)),\n            "daily_summary_sha256": str(daily_summary_sha256),\n            "daily_summary_commit_protocol": DAILY_SUMMARY_COMMIT_PROTOCOL,\n        },\n''',
    )

    # Chronology covers both journal execution_date and durable canonical economics.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''        try:\n            record_date = pd.Timestamp(record.signal_date).date()\n        except (TypeError, ValueError) as exc:\n            raise PaperAccountStateRefused(\n                f"invalid journal signal_date during chronology check: {record.signal_date!r}"\n            ) from exc\n        if record_date > cutoff:\n            later_records.append(record)\n    if later_records:\n        first = min(later_records, key=lambda row: row.signal_date)\n        raise PaperAccountStateRefused(\n            "paper decision chronology regression refused: later durable journal "\n            f"evidence already exists for signal_date={first.signal_date}"\n        )\n\n    prefix_index = build_canonical_prefix_index(config.canonical_ledger_path)\n''',
        '''        try:\n            signal_date = pd.Timestamp(record.signal_date).date()\n            execution_date = pd.Timestamp(record.execution_date).date()\n        except (TypeError, ValueError) as exc:\n            raise PaperAccountStateRefused(\n                "invalid journal signal/execution date during chronology check: "\n                f"signal={record.signal_date!r}, execution={record.execution_date!r}"\n            ) from exc\n        if signal_date > cutoff or execution_date > cutoff:\n            later_records.append(record)\n    if later_records:\n        first = min(\n            later_records,\n            key=lambda row: max(\n                pd.Timestamp(row.signal_date).date(),\n                pd.Timestamp(row.execution_date).date(),\n            ),\n        )\n        raise PaperAccountStateRefused(\n            "paper decision chronology regression refused: later durable journal "\n            "evidence already exists for "\n            f"signal_date={first.signal_date}, execution_date={first.execution_date}"\n        )\n\n    canonical = CanonicalLedger(config.canonical_ledger_path)\n    canonical_verification = canonical.verify()\n    if (\n        not bool(canonical_verification.get("valid"))\n        or bool(canonical_verification.get("tornTail"))\n    ):\n        raise PaperAccountStateRefused(\n            "canonical ledger chronology cannot be certified because the chain/tail "\n            "is not fully verifiable"\n        )\n    for record in canonical.read():\n        if record.trade_date is None:\n            continue\n        try:\n            trade_date = pd.Timestamp(record.trade_date).date()\n        except (TypeError, ValueError) as exc:\n            raise PaperAccountStateRefused(\n                f"invalid canonical ledger trade_date during chronology check: {record.trade_date!r}"\n            ) from exc\n        if trade_date > cutoff:\n            raise PaperAccountStateRefused(\n                "paper decision chronology regression refused: canonical economic "\n                f"history already reached trade_date={trade_date.isoformat()}"\n            )\n\n    prefix_index = build_canonical_prefix_index(config.canonical_ledger_path)\n''',
    )

    # Atomic, fsync-backed summary writer + exact staged cleanup.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''def _write_daily_summary(day_dir: Path, summary: dict[str, object]) -> None:\n    day_dir.mkdir(parents=True, exist_ok=True)\n    (day_dir / "daily_loop_summary.json").write_text(\n        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str),\n        encoding="utf-8",\n    )\n''',
        '''def _write_daily_summary(day_dir: Path, summary: dict[str, object]) -> Path:\n    day_dir.mkdir(parents=True, exist_ok=True)\n    target = day_dir / "daily_loop_summary.json"\n    tmp = day_dir / f".{target.name}.{uuid4().hex}.tmp"\n    payload = (\n        json.dumps(\n            summary,\n            ensure_ascii=False,\n            indent=2,\n            sort_keys=True,\n            default=str,\n        )\n        + "\\n"\n    )\n    try:\n        with tmp.open("w", encoding="utf-8") as handle:\n            handle.write(payload)\n            handle.flush()\n            os.fsync(handle.fileno())\n        os.replace(tmp, target)\n        if os.name != "nt":\n            directory_fd = os.open(day_dir, os.O_RDONLY)\n            try:\n                os.fsync(directory_fd)\n            finally:\n                os.close(directory_fd)\n    finally:\n        try:\n            if tmp.exists() or tmp.is_symlink():\n                tmp.unlink()\n        except OSError:\n            pass\n    return target\n\n\ndef _discard_staged_daily_summary(\n    path: Path,\n    *,\n    expected_sha256: str,\n) -> None:\n    if not path.exists():\n        return\n    if _file_sha256(path) != str(expected_sha256):\n        raise PaperAccountStateRefused(\n            "refusing to discard staged daily summary with mismatched digest"\n        )\n    try:\n        payload = json.loads(path.read_text(encoding="utf-8"))\n    except Exception as exc:\n        raise PaperAccountStateRefused(\n            "refusing to discard unreadable staged daily summary"\n        ) from exc\n    if (\n        not isinstance(payload, dict)\n        or payload.get("daily_decision_commit_protocol")\n        != DAILY_SUMMARY_COMMIT_PROTOCOL\n    ):\n        raise PaperAccountStateRefused(\n            "refusing to discard legacy/ambiguous daily summary staging"\n        )\n    path.unlink()\n    if os.name != "nt":\n        directory_fd = os.open(path.parent, os.O_RDONLY)\n        try:\n            os.fsync(directory_fd)\n        finally:\n            os.close(directory_fd)\n''',
    )

    # Execution verifies the bound primary summary before economic side effects.
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''    mismatches = [name for name, (left, right) in checks.items() if left != right]\n    if mismatches:\n        raise ContinuousPaperExecutionBlocked(\n            "pending signal commit binding mismatch: " + ",".join(sorted(mismatches))\n        )\n''',
        '''    mismatches = [name for name, (left, right) in checks.items() if left != right]\n    if mismatches:\n        raise ContinuousPaperExecutionBlocked(\n            "pending signal commit binding mismatch: " + ",".join(sorted(mismatches))\n        )\n    summary_path_text = str(details.get("daily_summary_path") or "").strip()\n    summary_sha = str(details.get("daily_summary_sha256") or "").strip()\n    if len(summary_sha) != 64 or not summary_path_text:\n        raise ContinuousPaperExecutionBlocked(\n            "pending signal commit lacks bound daily summary evidence"\n        )\n    summary_path = Path(summary_path_text)\n    if not summary_path.is_file():\n        raise ContinuousPaperExecutionBlocked(\n            "bound daily summary evidence is missing before pending execution"\n        )\n    if sha256(summary_path.read_bytes()).hexdigest() != summary_sha:\n        raise ContinuousPaperExecutionBlocked(\n            "bound daily summary evidence digest mismatch before pending execution"\n        )\n''',
    )

    # --- Horizon refresh before lock ---------------------------------------
    replace_once(
        "src/quantagent/portfolio/position_age_tracker.py",
        '''    def persist(self) -> Path | None:\n''',
        '''    def update_expected_horizons(\n        self, expected_horizons: Mapping[str, int | None] | None\n    ) -> None:\n        """Refresh known horizons before any holding-period lock decision."""\n        for raw_symbol, raw_horizon in dict(expected_horizons or {}).items():\n            if raw_horizon is None:\n                continue\n            record = self._records.get(str(raw_symbol))\n            if record is not None:\n                record.expected_horizon_days = int(raw_horizon)\n\n    def persist(self) -> Path | None:\n''',
    )
    replace_once(
        "src/quantagent/portfolio/v7_target_weights.py",
        '''    for date, day in preds.groupby("trade_date", sort=True):\n        day_market = market[market["trade_date"] == date]\n''',
        '''    for date, day in preds.groupby("trade_date", sort=True):\n        day_expected_horizons: dict[str, int | None] = {}\n        if age_tracker is not None and theme_frame is not None:\n            today_theme = theme_frame[theme_frame["trade_date"] == date]\n            if not today_theme.empty and "expected_horizon_days" in today_theme.columns:\n                for sym, eh in zip(\n                    today_theme["symbol"], today_theme["expected_horizon_days"]\n                ):\n                    if pd.notna(eh):\n                        day_expected_horizons[str(sym)] = int(eh)\n            age_tracker.update_expected_horizons(day_expected_horizons)\n\n        day_market = market[market["trade_date"] == date]\n''',
    )
    replace_once(
        "src/quantagent/portfolio/v7_target_weights.py",
        '''        if age_tracker is not None:\n            expected_horizons: dict[str, int | None] = {}\n            if theme_frame is not None:\n                today_theme = theme_frame[theme_frame["trade_date"] == date]\n                if not today_theme.empty and "expected_horizon_days" in today_theme.columns:\n                    for sym, eh in zip(today_theme["symbol"], today_theme["expected_horizon_days"]):\n                        if pd.notna(eh):\n                            expected_horizons[str(sym)] = int(eh)\n            age_tracker.record_session(date, weights.to_dict(), expected_horizons)\n''',
        '''        if age_tracker is not None:\n            age_tracker.record_session(\n                date, weights.to_dict(), day_expected_horizons\n            )\n''',
    )

    # --- Tests --------------------------------------------------------------
    # Existing canonical-prefix execution test now creates a committed staged target.
    p = Path("tests/paper/test_canonical_prefix_receipt.py")
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        '''from quantagent.paper.execution_journal import (\n    RECONCILIATION_STATUS,\n    PendingExecutionJournal,\n)\nfrom quantagent.paper.pending_signal import PendingPaperSignalStore\n''',
        '''from quantagent.paper.execution_journal import (\n    DAILY_DECISION_STATUS,\n    RECONCILIATION_STATUS,\n    PendingExecutionJournal,\n)\nfrom quantagent.paper.pending_signal import (\n    PENDING_COMMIT_PROTOCOL,\n    PendingPaperSignalStore,\n)\n''',
        1,
    )
    old = '''    pending = PendingPaperSignalStore(pending_dir).record(\n        signal_date=FRIDAY,\n        target_weights=pd.DataFrame(\n            {"trade_date": [pd.Timestamp(FRIDAY)], SYMBOL: [0.5]}\n        ),\n        source_lineage={"paper_account_identity_sha256": identity.payload_sha256},\n        created_at=f"{FRIDAY}T07:00:00+00:00",\n    )[0]\n    journal_path = tmp_path / "execution.jsonl"\n'''
    new = '''    summary_path = tmp_path / "daily_loop_summary.json"\n    summary_path.write_text('{"test":"committed-summary"}\\n', encoding="utf-8")\n    summary_sha = __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest()\n    pending = PendingPaperSignalStore(pending_dir).record(\n        signal_date=FRIDAY,\n        target_weights=pd.DataFrame(\n            {"trade_date": [pd.Timestamp(FRIDAY)], SYMBOL: [0.5]}\n        ),\n        source_lineage={\n            "paper_account_identity_sha256": identity.payload_sha256,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_ledger_head_hash": "0" * 64,\n            "canonical_ledger_records": "0",\n            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,\n        },\n        created_at=f"{FRIDAY}T07:00:00+00:00",\n    )[0]\n    journal_path = tmp_path / "execution.jsonl"\n    PendingExecutionJournal(journal_path).append(\n        pending_payload_sha256=pending.payload_sha256,\n        signal_date=FRIDAY,\n        execution_date=FRIDAY,\n        status=DAILY_DECISION_STATUS,\n        details={\n            "decision_kind": "target",\n            "paper_account_identity_sha256": identity.payload_sha256,\n            "canonical_account_state_sha256": "1" * 64,\n            "canonical_records": 0,\n            "canonical_head": "0" * 64,\n            "assurance": "canonical_account_daily_decision_freeze_v1",\n            "commit_protocol": PENDING_COMMIT_PROTOCOL,\n            "target_weights_sha256": pending.target_weights_sha256,\n            "daily_summary_path": str(summary_path.resolve()),\n            "daily_summary_sha256": summary_sha,\n            "daily_summary_commit_protocol": "daily_summary_bound_daily_decision_v1",\n        },\n    )\n'''
    if text.count(old) != 1:
        raise RuntimeError(f"canonical prefix pending anchor count={text.count(old)}")
    text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")

    # Old same-date pending is ambiguous, so migrate only the expected wording.
    p = Path("tests/paper/test_daily_loop_account_aware_construction.py")
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        'with pytest.raises(PaperAccountStateRefused, match="already frozen"):',
        'with pytest.raises(PaperAccountStateRefused, match="legacy/ambiguous pending"):',
        1,
    )
    p.write_text(text, encoding="utf-8")

    # Summary transaction + chronology regressions.
    p = Path("tests/paper/test_daily_loop_pending_execution.py")
    text = p.read_text(encoding="utf-8")
    if "test_committed_target_marker_binds_primary_summary_digest" not in text:
        text = text.rstrip() + '''\n\n\ndef test_committed_target_marker_binds_primary_summary_digest(tmp_path, monkeypatch) -> None:\n    targets = pd.DataFrame(\n        {"trade_date": [pd.Timestamp("2026-08-07")], "600000.SH": [0.5]}\n    )\n    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=targets)\n    daily_loop.run_once(config)\n    summary_path = tmp_path / "reports" / as_of / "daily_loop_summary.json"\n    decision = daily_loop.PendingExecutionJournal(\n        daily_loop._execution_journal_path(config)\n    ).daily_decision(as_of)\n    assert decision is not None\n    assert decision.details["daily_summary_sha256"] == sha256(\n        summary_path.read_bytes()\n    ).hexdigest()\n    assert decision.details["daily_summary_commit_protocol"] == daily_loop.DAILY_SUMMARY_COMMIT_PROTOCOL\n\n\ndef test_summary_write_failure_leaves_no_durable_decision(tmp_path, monkeypatch) -> None:\n    targets = pd.DataFrame(\n        {"trade_date": [pd.Timestamp("2026-08-07")], "600000.SH": [0.5]}\n    )\n    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=targets)\n\n    def fail_summary(*_args, **_kwargs):\n        raise OSError("simulated summary fsync failure")\n\n    monkeypatch.setattr(daily_loop, "_write_daily_summary", fail_summary)\n    with pytest.raises(OSError, match="summary fsync failure"):\n        daily_loop.run_once(config)\n    assert daily_loop.PendingExecutionJournal(\n        daily_loop._execution_journal_path(config)\n    ).daily_decision(as_of) is None\n\n\ndef test_uncommitted_current_protocol_summary_is_recoverable(tmp_path) -> None:\n    as_of = "2026-08-07"\n    day_dir = tmp_path / "reports" / as_of\n    summary = {\n        "daily_decision_commit_protocol": daily_loop.DAILY_SUMMARY_COMMIT_PROTOCOL,\n        "status": "staged-before-crash",\n    }\n    summary_path = daily_loop._write_daily_summary(day_dir, summary)\n    config = daily_loop.DailyPaperLoopConfig(\n        as_of_date=as_of,\n        output_root=str(tmp_path / "reports"),\n        pending_signal_dir=str(tmp_path / "pending"),\n        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),\n        execution_journal_path=str(tmp_path / "execution.jsonl"),\n    )\n    daily_loop._assert_current_signal_not_frozen(config, as_of)\n    assert not summary_path.exists()\n\n\ndef test_legacy_uncommitted_summary_remains_fail_closed(tmp_path) -> None:\n    as_of = "2026-08-07"\n    day_dir = tmp_path / "reports" / as_of\n    day_dir.mkdir(parents=True)\n    (day_dir / "daily_loop_summary.json").write_text(\n        '{"status":"legacy"}\\n', encoding="utf-8"\n    )\n    config = daily_loop.DailyPaperLoopConfig(\n        as_of_date=as_of,\n        output_root=str(tmp_path / "reports"),\n        pending_signal_dir=str(tmp_path / "pending"),\n        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),\n        execution_journal_path=str(tmp_path / "execution.jsonl"),\n    )\n    with pytest.raises(daily_loop.PaperAccountStateRefused, match="legacy/ambiguous daily summary"):\n        daily_loop._assert_current_signal_not_frozen(config, as_of)\n'''
    p.write_text(text, encoding="utf-8")

    p = Path("tests/paper/test_daily_loop_prior_execution_gate.py")
    text = p.read_text(encoding="utf-8")
    if "test_backdated_decision_refuses_later_execution_date" not in text:
        text = text.rstrip() + '''\n\n\ndef test_backdated_decision_refuses_later_execution_date(tmp_path) -> None:\n    config = _config(tmp_path, "2026-08-11")\n    PendingExecutionJournal(config.execution_journal_path).append(\n        pending_payload_sha256="e" * 64,\n        signal_date="2026-08-10",\n        execution_date="2026-08-12",\n        status="execution_started",\n        details={"paper_account_identity_sha256": _IDENTITY_SHA},\n    )\n    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*execution_date=2026-08-12"):\n        _assert_prior_pending_signals_resolved(\n            config,\n            "2026-08-11",\n            paper_account_identity_sha256=_IDENTITY_SHA,\n        )\n\n\ndef test_backdated_decision_refuses_later_canonical_trade_date(tmp_path) -> None:\n    config = _config(tmp_path, "2026-08-11")\n    from quantagent.domain.ledger import CanonicalLedger\n    CanonicalLedger(config.canonical_ledger_path).append(None, trade_date="2026-08-12")\n    with pytest.raises(PaperAccountStateRefused, match="canonical economic history.*2026-08-12"):\n        _assert_prior_pending_signals_resolved(\n            config,\n            "2026-08-11",\n            paper_account_identity_sha256=_IDENTITY_SHA,\n        )\n'''
    p.write_text(text, encoding="utf-8")

    # Consumer refuses missing/tampered bound summary.
    p = Path("tests/paper/test_continuous_pending_execution.py")
    text = p.read_text(encoding="utf-8")
    if "test_committed_pending_refuses_missing_bound_summary" not in text:
        text = text.rstrip() + '''\n\n\ndef test_committed_pending_refuses_missing_bound_summary(tmp_path) -> None:\n    config = _config(tmp_path)\n    pending = _record(tmp_path, FRIDAY, 0.50)\n    journal = PendingExecutionJournal(config.execution_journal_path)\n    decision = journal.daily_decision(FRIDAY)\n    assert decision is not None\n    # Retrofit summary binding for the legacy helper's marker, then remove it.\n    # The helper is updated below to bind a real file on all new records.\n    summary_path = Path(decision.details["daily_summary_path"])\n    summary_path.unlink()\n    with pytest.raises(ContinuousPaperExecutionBlocked, match="summary evidence is missing"):\n        execute_pending_for_session(\n            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS\n        )\n    assert journal.terminal(pending.payload_sha256) is None\n'''
    p.write_text(text, encoding="utf-8")

    # Update the helper-created daily marker in continuous pending tests to bind summary.
    p = Path("tests/paper/test_continuous_pending_execution.py")
    text = p.read_text(encoding="utf-8")
    old = '''    PendingExecutionJournal(tmp_path / "execution.jsonl").append(\n        pending_payload_sha256=pending.payload_sha256,\n        signal_date=signal_date,\n        execution_date=signal_date,\n        status=DAILY_DECISION_STATUS,\n        details={\n'''
    new = '''    summary_path = tmp_path / f"summary-{signal_date}.json"\n    summary_path.write_text('{"committed":true}\\n', encoding="utf-8")\n    summary_sha = __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest()\n    PendingExecutionJournal(tmp_path / "execution.jsonl").append(\n        pending_payload_sha256=pending.payload_sha256,\n        signal_date=signal_date,\n        execution_date=signal_date,\n        status=DAILY_DECISION_STATUS,\n        details={\n'''
    if text.count(old) < 1:
        raise RuntimeError("continuous helper marker anchor missing")
    text = text.replace(old, new, 1)
    old_details = '''            "target_weights_sha256": pending.target_weights_sha256,\n        },\n    )\n    return pending\n'''
    new_details = '''            "target_weights_sha256": pending.target_weights_sha256,\n            "daily_summary_path": str(summary_path.resolve()),\n            "daily_summary_sha256": summary_sha,\n            "daily_summary_commit_protocol": "daily_summary_bound_daily_decision_v1",\n        },\n    )\n    return pending\n'''
    if text.count(old_details) < 1:
        raise RuntimeError("continuous helper marker details anchor missing")
    text = text.replace(old_details, new_details, 1)
    p.write_text(text, encoding="utf-8")

    # Horizon updater unit + integration-order proof.
    p = Path("tests/portfolio/test_position_age_tracker_account_reconciliation.py")
    text = p.read_text(encoding="utf-8")
    if "test_late_real_horizon_refreshes_before_next_lock" not in text:
        text = text.rstrip() + '''\n\n\ndef test_late_real_horizon_refreshes_before_next_lock(tmp_path) -> None:\n    tracker = PositionAgeTracker(state_path=tmp_path / "state.parquet")\n    tracker.begin_session({"A": 0.2})\n    assert tracker.is_locked("A", pd.Timestamp("2026-08-05"))\n    tracker.update_expected_horizons({"A": 1})\n    snapshot = tracker.snapshot().set_index("symbol")\n    assert int(snapshot.loc["A", "expected_horizon_days"]) == 1\n'''
    p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
