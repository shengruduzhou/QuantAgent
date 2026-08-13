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
    # Do not overclaim canonical/operational historical parity. Legacy migration
    # binds canonical economic truth at lower assurance unless a future protocol
    # proves event-by-event operational correspondence.
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''    operational_economics = _operational_has_reconstructable_economics(\n        operational_state\n    )\n    _assert_recovered_account_consistent(canonical_state, operational_state)\n''',
        '''    _assert_recovered_account_consistent(canonical_state, operational_state)\n''',
    )
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''        "assurance": (\n            "operator_reconciled_legacy_terminal_v2"\n            if operational_economics\n            else "operator_bound_canonical_only_legacy_terminal_v1"\n        ),\n        "operational_economic_reconstruction": (\n            "matched_canonical"\n            if operational_economics\n            else "not_present_canonical_is_record_of_account"\n        ),\n''',
        '''        "assurance": "operator_bound_canonical_only_legacy_terminal_v1",\n        "operational_economic_reconstruction": (\n            "not_claimed_canonical_is_record_of_account"\n        ),\n''',
    )
    replace_once(
        "src/quantagent/paper/legacy_terminal_binding.py",
        '''    assurance = str(details.get("assurance") or "")\n    reconstruction = str(details.get("operational_economic_reconstruction") or "")\n    if assurance == "operator_reconciled_legacy_terminal_v2":\n        if reconstruction != "matched_canonical":\n            raise LegacyTerminalBindingError(\n                "legacy terminal parity assurance lacks matched operational reconstruction"\n            )\n    elif assurance == "operator_bound_canonical_only_legacy_terminal_v1":\n        if reconstruction != "not_present_canonical_is_record_of_account":\n            raise LegacyTerminalBindingError(\n                "legacy terminal canonical-only assurance marker is invalid"\n            )\n    else:\n        # The historical v1 wording was ambiguous: it could claim reconciliation\n        # even when the operational ledger contained no economic reconstruction.\n        # Never silently upgrade that lower-quality evidence.\n        raise LegacyTerminalBindingError("legacy terminal binding assurance is invalid")\n''',
        '''    assurance = str(details.get("assurance") or "")\n    reconstruction = str(details.get("operational_economic_reconstruction") or "")\n    if assurance != "operator_bound_canonical_only_legacy_terminal_v1":\n        # Do not accept a parity-sounding assurance without a separate protocol\n        # that proves corresponding historical economic events, not merely equal\n        # current cash/positions or the presence of arbitrary order records.\n        raise LegacyTerminalBindingError("legacy terminal binding assurance is invalid")\n    if reconstruction != "not_claimed_canonical_is_record_of_account":\n        raise LegacyTerminalBindingError(\n            "legacy terminal canonical-only assurance marker is invalid"\n        )\n''',
    )
    replace_once(
        "src/quantagent/paper/execution_journal.py",
        '''                assurance_contracts = {\n                    "operator_reconciled_legacy_terminal_v2": "matched_canonical",\n                    "operator_bound_canonical_only_legacy_terminal_v1": (\n                        "not_present_canonical_is_record_of_account"\n                    ),\n                }\n''',
        '''                assurance_contracts = {\n                    "operator_bound_canonical_only_legacy_terminal_v1": (\n                        "not_claimed_canonical_is_record_of_account"\n                    ),\n                }\n''',
    )

    # Same filesystem identity must imply same journal identity, including a
    # custom canonical ledger reached through a differently named symlink.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''    canonical = Path(config.canonical_ledger_path)\n    runtime = paper_runtime_paths()\n    if canonical.resolve(strict=False) == runtime.canonical_ledger.resolve(strict=False):\n''',
        '''    canonical = Path(config.canonical_ledger_path).resolve(strict=False)\n    runtime = paper_runtime_paths()\n    if canonical == runtime.canonical_ledger.resolve(strict=False):\n''',
    )

    # Validate no-target source dates before any durable artifacts or freeze.
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''            account_state = fresh_account_state\n            account_evidence = account_state.evidence()\n            # Stage every fallible artifact before the irreversible no-target\n''',
        '''            account_state = fresh_account_state\n            account_evidence = account_state.evidence()\n            _assert_exact_signal_date(\n                predictions,\n                as_of,\n                evidence_name="no-target predictions",\n            )\n            # Stage every fallible artifact before the irreversible no-target\n''',
    )
    replace_once(
        "src/quantagent/paper/daily_loop.py",
        '''def _execution_journal_path(config: DailyPaperLoopConfig) -> str:\n''',
        '''def _assert_exact_signal_date(\n    frame: pd.DataFrame,\n    as_of: str,\n    *,\n    evidence_name: str,\n) -> None:\n    if "trade_date" not in frame.columns or frame.empty:\n        raise PaperAccountStateRefused(\n            f"{evidence_name} lacks a non-empty trade_date column for {as_of}"\n        )\n    parsed = pd.to_datetime(frame["trade_date"], errors="coerce")\n    if parsed.isna().any():\n        raise PaperAccountStateRefused(\n            f"{evidence_name} contains invalid trade_date values"\n        )\n    dates = {pd.Timestamp(value).date().isoformat() for value in parsed}\n    if dates != {str(as_of)}:\n        raise PaperAccountStateRefused(\n            f"{evidence_name} must belong exactly to signal_date={as_of}; "\n            f"observed_dates={sorted(dates)}"\n        )\n\n\ndef _execution_journal_path(config: DailyPaperLoopConfig) -> str:\n''',
    )

    # Persist authoritative tracker reconciliation even when every date rejects.
    replace_once(
        "src/quantagent/portfolio/v7_target_weights.py",
        '''    if not by_date_weights:\n        return V7TargetWeightsResult(\n            pd.DataFrame(),\n            {"status": "all_dates_rejected", "initial_weights": initial_state_diagnostics, **diagnostics},\n        )\n''',
        '''    if not by_date_weights:\n        diagnostics_payload = {\n            "status": "all_dates_rejected",\n            "initial_weights": initial_state_diagnostics,\n            **diagnostics,\n        }\n        if age_tracker is not None:\n            persisted = age_tracker.persist()\n            if persisted is not None:\n                diagnostics_payload["position_state_path"] = str(persisted)\n            diagnostics_payload["position_state_rows"] = int(len(age_tracker.snapshot()))\n        return V7TargetWeightsResult(pd.DataFrame(), diagnostics_payload)\n''',
    )

    # Update legacy-binding assurance expectation and add explicit rejected
    # parity-overclaim regression.
    p = Path("tests/paper/test_daily_decision_and_legacy_binding.py")
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        '== "not_present_canonical_is_record_of_account"',
        '== "not_claimed_canonical_is_record_of_account"',
    )
    if "test_legacy_binding_never_upgrades_to_state_only_operational_parity" not in text:
        text = text.rstrip() + '''\n\n\ndef test_legacy_binding_never_upgrades_to_state_only_operational_parity(tmp_path) -> None:\n    daily = _daily_config(tmp_path, "2026-08-12")\n    prior = _pending(daily, "2026-08-10")\n    journal = PendingExecutionJournal(daily.execution_journal_path)\n    journal.append(\n        pending_payload_sha256=prior.payload_sha256,\n        signal_date=prior.signal_date,\n        execution_date="2026-08-11",\n        status="execution_observed",\n        details={"target_weights_sha256": prior.target_weights_sha256},\n    )\n    binding = bind_legacy_terminal_account(\n        config=_continuous_config(tmp_path),\n        pending_payload_sha256=prior.payload_sha256,\n        as_of_date="2026-08-12",\n        reason="bind canonical record without historical parity overclaim",\n    )\n    assert binding["details"]["assurance"] == "operator_bound_canonical_only_legacy_terminal_v1"\n    assert (\n        binding["details"]["operational_economic_reconstruction"]\n        == "not_claimed_canonical_is_record_of_account"\n    )\n'''
    p.write_text(text, encoding="utf-8")

    # Symlink alias must derive the same custom journal.
    p = Path("tests/paper/test_daily_decision_and_legacy_binding.py")
    text = p.read_text(encoding="utf-8")
    if "test_custom_journal_path_canonicalizes_file_symlink_alias" not in text:
        text = text.rstrip() + '''\n\n\ndef test_custom_journal_path_canonicalizes_file_symlink_alias(tmp_path) -> None:\n    real = tmp_path / "real_custom.jsonl"\n    real.write_text("", encoding="utf-8")\n    alias = tmp_path / "different_alias.jsonl"\n    try:\n        alias.symlink_to(real)\n    except OSError as exc:\n        pytest.skip(f"symlink creation unavailable: {exc}")\n    real_cfg = DailyPaperLoopConfig(\n        as_of_date="2026-08-11", canonical_ledger_path=str(real), execution_journal_path=None\n    )\n    alias_cfg = DailyPaperLoopConfig(\n        as_of_date="2026-08-11", canonical_ledger_path=str(alias), execution_journal_path=None\n    )\n    assert daily_loop._execution_journal_path(real_cfg) == daily_loop._execution_journal_path(alias_cfg)\n'''
    p.write_text(text, encoding="utf-8")

    # Integration regression for stale no-target prediction evidence.
    p = Path("tests/paper/test_daily_loop_pending_execution.py")
    text = p.read_text(encoding="utf-8")
    if "test_no_target_stale_prediction_date_does_not_freeze_day" not in text:
        text = text.rstrip() + '''\n\n\ndef test_no_target_stale_prediction_date_does_not_freeze_day(tmp_path, monkeypatch) -> None:\n    as_of, config, _ = _install_common_mocks(tmp_path, monkeypatch, targets=pd.DataFrame())\n    stale = pd.DataFrame(\n        {\n            "trade_date": [pd.Timestamp("2026-08-06")],\n            "symbol": ["600000.SH"],\n            "prediction": [0.1],\n            "confidence": [0.9],\n        }\n    )\n    monkeypatch.setattr(\n        daily_loop,\n        "predict_v7_alpha",\n        lambda *args, **kwargs: SimpleNamespace(predictions=stale.copy()),\n    )\n    monkeypatch.setattr(\n        daily_loop,\n        "blend_multi_horizon_predictions",\n        lambda *args, **kwargs: SimpleNamespace(\n            blended=stale.copy(), diagnostics={"test": "stale-no-target"}\n        ),\n    )\n    with pytest.raises(PaperAccountStateRefused, match="no-target predictions must belong exactly"):\n        daily_loop.run_once(config)\n    journal = daily_loop.PendingExecutionJournal(daily_loop._execution_journal_path(config))\n    assert journal.daily_decision(as_of) is None\n'''
    p.write_text(text, encoding="utf-8")

    # all_dates_rejected must durably persist the canonical cash-only tracker.
    p = Path("tests/test_v7_target_initial_weights.py")
    text = p.read_text(encoding="utf-8")
    if "test_all_dates_rejected_persists_cash_only_tracker_reconciliation" not in text:
        text = text.rstrip() + '''\n\n\ndef test_all_dates_rejected_persists_cash_only_tracker_reconciliation(tmp_path) -> None:\n    predictions, market, timing = _two_session_inputs()\n    state_path = tmp_path / "position_age.parquet"\n    tracker = v7_target_weights.PositionAgeTracker(state_path=state_path)\n    tracker.begin_session({"STALE": 0.20}, {"STALE": 5})\n    tracker.persist()\n    assert not v7_target_weights.PositionAgeTracker.from_state(state_path).snapshot().empty\n\n    result = build_v7_target_weights(\n        predictions,\n        market,\n        config=_config(\n            holding_period_mode="hard",\n            position_state_path=str(state_path),\n            max_turnover=1.0,\n        ),\n        timing_plan=timing,\n        initial_weights=pd.Series(dtype=float),\n    )\n    assert result.target_weights.empty\n    assert result.diagnostics["status"] == "all_dates_rejected"\n    assert result.diagnostics["position_state_rows"] == 0\n    assert v7_target_weights.PositionAgeTracker.from_state(state_path).snapshot().empty\n'''
    p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
