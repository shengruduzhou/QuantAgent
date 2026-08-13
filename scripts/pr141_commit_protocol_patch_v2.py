from __future__ import annotations

from pathlib import Path

import pr141_commit_protocol_patch as patch


patch.main()

# Preserve integer zero when comparing canonical record counts.
p = Path("src/quantagent/paper/continuous_execution.py")
text = p.read_text(encoding="utf-8")
old = '''        "canonical_records": (\n            str(details.get("canonical_records") or ""),\n            str(pending.source_lineage.get("canonical_ledger_records") or ""),\n        ),\n'''
new = '''        "canonical_records": (\n            (\n                str(details.get("canonical_records"))\n                if details.get("canonical_records") is not None\n                else ""\n            ),\n            str(pending.source_lineage.get("canonical_ledger_records") or ""),\n        ),\n'''
if text.count(old) != 1:
    raise RuntimeError(f"canonical_records normalization anchor count={text.count(old)}")
p.write_text(text.replace(old, new, 1), encoding="utf-8")

# Correct chronology-test helper wiring against this file's actual local API.
p = Path("tests/paper/test_daily_loop_prior_execution_gate.py")
text = p.read_text(encoding="utf-8")
old = '''def test_backdated_decision_refuses_later_pending_signal(tmp_path) -> None:\n    later = _daily_config(tmp_path, "2026-08-12")\n    _pending(later, "2026-08-12")\n    earlier = _daily_config(tmp_path, "2026-08-11")\n    identity = ensure_paper_account_identity(\n        canonical_ledger_path=earlier.canonical_ledger_path,\n        portfolio_id=earlier.portfolio_id,\n        initial_cash=earlier.initial_cash,\n        identity_path=earlier.account_identity_path,\n    )\n    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later pending"):\n        _assert_prior_pending_signals_resolved(\n            earlier,\n            "2026-08-11",\n            paper_account_identity_sha256=identity.payload_sha256,\n        )\n\n\ndef test_backdated_decision_refuses_later_durable_journal_record(tmp_path) -> None:\n    earlier = _daily_config(tmp_path, "2026-08-11")\n    identity = ensure_paper_account_identity(\n        canonical_ledger_path=earlier.canonical_ledger_path,\n        portfolio_id=earlier.portfolio_id,\n        initial_cash=earlier.initial_cash,\n        identity_path=earlier.account_identity_path,\n    )\n    PendingExecutionJournal(earlier.execution_journal_path).append(\n        pending_payload_sha256="f" * 64,\n        signal_date="2026-08-12",\n        execution_date="2026-08-12",\n        status="execution_started",\n        details={"paper_account_identity_sha256": identity.payload_sha256},\n    )\n    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later durable"):\n        _assert_prior_pending_signals_resolved(\n            earlier,\n            "2026-08-11",\n            paper_account_identity_sha256=identity.payload_sha256,\n        )\n'''
new = '''def test_backdated_decision_refuses_later_pending_signal(tmp_path) -> None:\n    later = _config(tmp_path, "2026-08-12")\n    _record(PendingPaperSignalStore(later.pending_signal_dir), "2026-08-12")\n    earlier = _config(tmp_path, "2026-08-11")\n    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later pending"):\n        _assert_prior_pending_signals_resolved(\n            earlier,\n            "2026-08-11",\n            paper_account_identity_sha256=_IDENTITY_SHA,\n        )\n\n\ndef test_backdated_decision_refuses_later_durable_journal_record(tmp_path) -> None:\n    earlier = _config(tmp_path, "2026-08-11")\n    PendingExecutionJournal(earlier.execution_journal_path).append(\n        pending_payload_sha256="f" * 64,\n        signal_date="2026-08-12",\n        execution_date="2026-08-12",\n        status="execution_started",\n        details={"paper_account_identity_sha256": _IDENTITY_SHA},\n    )\n    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later durable"):\n        _assert_prior_pending_signals_resolved(\n            earlier,\n            "2026-08-11",\n            paper_account_identity_sha256=_IDENTITY_SHA,\n        )\n'''
if text.count(old) != 1:
    raise RuntimeError(f"chronology test block anchor count={text.count(old)}")
p.write_text(text.replace(old, new, 1), encoding="utf-8")
