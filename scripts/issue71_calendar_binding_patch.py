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
    # Low-level certificate binder: make the authoritative calendar a required,
    # digest-bound structured artifact and cross-bind it to prereg/lineage/FRESH.
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''from quantagent.config.paths import quant_paths\n''',
        '''from quantagent.config.paths import quant_paths\nfrom quantagent.execution.acceptance_calendar import load_acceptance_calendar\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''    "statistical_gates",\n    "fresh_oos_predictions",\n''',
        '''    "statistical_gates",\n    "acceptance_calendar",\n    "fresh_oos_predictions",\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''        "statistical_gates",\n        "fresh_oos",\n''',
        '''        "statistical_gates",\n        "acceptance_calendar",\n        "fresh_oos",\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''    fresh_predictions = bindings.get("fresh_oos_predictions")\n\n    trainer = docs.get("trainer_manifest", {})\n''',
        '''    fresh_predictions = bindings.get("fresh_oos_predictions")\n    calendar_binding = bindings.get("acceptance_calendar")\n    acceptance_calendar = None\n    calendar_path = paths.get("acceptance_calendar")\n    if calendar_path is not None:\n        try:\n            acceptance_calendar = load_acceptance_calendar(\n                calendar_path,\n                expected_model_id=model_id,\n                expected_source_commit=source_commit,\n            )\n        except ValueError as exc:\n            reasons.append(str(exc))\n\n    trainer = docs.get("trainer_manifest", {})\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''        if prereg_budget is None:\n            reasons.append("pre_registration:max_search_trials_invalid")\n\n    search = docs.get("search_ledger", {})\n''',
        '''        if prereg_budget is None:\n            reasons.append("pre_registration:max_search_trials_invalid")\n        if acceptance_calendar is not None and calendar_binding is not None:\n            if prereg.get("acceptance_calendar_sha256") != calendar_binding.sha256:\n                reasons.append("pre_registration:acceptance_calendar_sha256_mismatch")\n            if (\n                prereg.get("acceptance_calendar_session_set_sha256")\n                != acceptance_calendar.session_set_sha256\n            ):\n                reasons.append(\n                    "pre_registration:acceptance_calendar_session_set_sha256_mismatch"\n                )\n            if str(prereg.get("acceptance_window_id") or "") != acceptance_calendar.acceptance_window_id:\n                reasons.append("pre_registration:acceptance_calendar_window_mismatch")\n            if str(prereg.get("acceptance_start_date") or "") != acceptance_calendar.window_start_date:\n                reasons.append("pre_registration:acceptance_calendar_start_mismatch")\n            if str(prereg.get("acceptance_end_date") or "") != acceptance_calendar.window_end_date:\n                reasons.append("pre_registration:acceptance_calendar_end_mismatch")\n            calendar_retrieved = _parse_timestamp(acceptance_calendar.source_retrieved_at)\n            calendar_as_of = _parse_timestamp(acceptance_calendar.source_as_of)\n            if prereg_at is not None:\n                if calendar_retrieved is None or calendar_retrieved > prereg_at:\n                    reasons.append("pre_registration:calendar_retrieved_after_registration")\n                if calendar_as_of is None or calendar_as_of > prereg_at:\n                    reasons.append("pre_registration:calendar_as_of_after_registration")\n\n    search = docs.get("search_ledger", {})\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''    if lineage:\n        if lineage.get("pit") is not True:\n            reasons.append("data_lineage:pit_not_true")\n        if lineage.get("universe_pit") is not True:\n            reasons.append("data_lineage:universe_pit_not_true")\n\n    strict = docs.get("strict_backtest", {})\n''',
        '''    if lineage:\n        if lineage.get("pit") is not True:\n            reasons.append("data_lineage:pit_not_true")\n        if lineage.get("universe_pit") is not True:\n            reasons.append("data_lineage:universe_pit_not_true")\n        if acceptance_calendar is not None and calendar_binding is not None:\n            if lineage.get("acceptance_calendar_sha256") != calendar_binding.sha256:\n                reasons.append("data_lineage:acceptance_calendar_sha256_mismatch")\n            if (\n                lineage.get("acceptance_calendar_session_set_sha256")\n                != acceptance_calendar.session_set_sha256\n            ):\n                reasons.append("data_lineage:acceptance_calendar_session_set_sha256_mismatch")\n            if str(lineage.get("acceptance_window_id") or "") != acceptance_calendar.acceptance_window_id:\n                reasons.append("data_lineage:acceptance_calendar_window_mismatch")\n\n    strict = docs.get("strict_backtest", {})\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''        if fresh_predictions and fresh.get("predictions_sha256") != fresh_predictions.sha256:\n            reasons.append("fresh_oos:predictions_sha256_mismatch")\n\n    if prereg and fresh:\n''',
        '''        if fresh_predictions and fresh.get("predictions_sha256") != fresh_predictions.sha256:\n            reasons.append("fresh_oos:predictions_sha256_mismatch")\n        if acceptance_calendar is not None and calendar_binding is not None:\n            if fresh.get("acceptance_calendar_sha256") != calendar_binding.sha256:\n                reasons.append("fresh_oos:acceptance_calendar_sha256_mismatch")\n            if (\n                fresh.get("acceptance_calendar_session_set_sha256")\n                != acceptance_calendar.session_set_sha256\n            ):\n                reasons.append("fresh_oos:acceptance_calendar_session_set_sha256_mismatch")\n            if fresh_window != acceptance_calendar.acceptance_window_id:\n                reasons.append("fresh_oos:acceptance_calendar_window_mismatch")\n            if fresh_start is None or fresh_start.isoformat() != acceptance_calendar.window_start_date:\n                reasons.append("fresh_oos:acceptance_calendar_start_mismatch")\n            if fresh_end is None or fresh_end.isoformat() != acceptance_calendar.window_end_date:\n                reasons.append("fresh_oos:acceptance_calendar_end_mismatch")\n            if fresh_days != acceptance_calendar.trading_days:\n                reasons.append(\n                    f"fresh_oos:trading_days_mismatch_acceptance_calendar:{fresh_days}!="\n                    f"{acceptance_calendar.trading_days}"\n                )\n            if acceptance_calendar.trading_days < min_fresh_oos_days:\n                reasons.append(\n                    f"acceptance_calendar:trading_days_below_{min_fresh_oos_days}:"\n                    f"{acceptance_calendar.trading_days}"\n                )\n\n    if prereg and fresh:\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''    if issued_at and fresh_end and issued_at.date() < fresh_end:\n        reasons.append("v2_issued_before_fresh_oos_end")\n\n    training_cutoff = _parse_date(lineage.get("training_cutoff")) if lineage else None\n''',
        '''    if issued_at and fresh_end and issued_at.date() < fresh_end:\n        reasons.append("v2_issued_before_fresh_oos_end")\n    if acceptance_calendar is not None and issued_at is not None:\n        calendar_retrieved = _parse_timestamp(acceptance_calendar.source_retrieved_at)\n        calendar_as_of = _parse_timestamp(acceptance_calendar.source_as_of)\n        if calendar_retrieved is None or calendar_retrieved > issued_at:\n            reasons.append("acceptance_calendar:retrieved_after_certificate_issue")\n        if calendar_as_of is None or calendar_as_of > issued_at:\n            reasons.append("acceptance_calendar:as_of_after_certificate_issue")\n\n    training_cutoff = _parse_date(lineage.get("training_cutoff")) if lineage else None\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2.py",
        '''        "acceptance_window_id": fresh_window or None,\n        "pbo": pbo,\n''',
        '''        "acceptance_window_id": fresh_window or None,\n        "acceptance_calendar_sha256": (\n            None if calendar_binding is None else calendar_binding.sha256\n        ),\n        "acceptance_calendar_session_set_sha256": (\n            None if acceptance_calendar is None else acceptance_calendar.session_set_sha256\n        ),\n        "acceptance_calendar_source_identity": (\n            None if acceptance_calendar is None else acceptance_calendar.source_identity\n        ),\n        "acceptance_calendar_source_version": (\n            None if acceptance_calendar is None else acceptance_calendar.source_version\n        ),\n        "pbo": pbo,\n''',
    )

    # Governed policy: recompute exact FRESH session tuple and compare to calendar.
    replace_once(
        "src/quantagent/execution/live_model_trust_v2_policy.py",
        '''from quantagent.execution.live_model_evidence import (\n''',
        '''from quantagent.execution.acceptance_calendar import load_acceptance_calendar\nfrom quantagent.execution.live_model_evidence import (\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2_policy.py",
        '''    fresh_path = resolved.get("fresh_oos_predictions")\n    fresh_summary_path = resolved.get("fresh_oos")\n    if fresh_path and fresh_summary_path:\n        try:\n            derived = validate_fresh_predictions(fresh_path)\n            summary = _read_json_object(Path(fresh_summary_path), "fresh_oos")\n''',
        '''    fresh_path = resolved.get("fresh_oos_predictions")\n    fresh_summary_path = resolved.get("fresh_oos")\n    calendar_path = resolved.get("acceptance_calendar")\n    if fresh_path and fresh_summary_path and calendar_path:\n        try:\n            derived = validate_fresh_predictions(fresh_path)\n            summary = _read_json_object(Path(fresh_summary_path), "fresh_oos")\n            calendar = load_acceptance_calendar(\n                calendar_path,\n                expected_model_id=str(payload.get("model_id") or ""),\n                expected_source_commit=str(payload.get("source_commit") or ""),\n            )\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2_policy.py",
        '''            if str(summary.get("end_date") or "") != derived.end_date:\n                reasons.append("fresh_oos:end_date_mismatch_predictions")\n            fresh_start_date = derived.start_date\n            evidence.update(\n                {\n                    "fresh_oos_days": derived.trading_days,\n''',
        '''            if str(summary.get("end_date") or "") != derived.end_date:\n                reasons.append("fresh_oos:end_date_mismatch_predictions")\n            if derived.session_dates != calendar.sessions:\n                missing = sorted(set(calendar.sessions).difference(derived.session_dates))\n                unexpected = sorted(set(derived.session_dates).difference(calendar.sessions))\n                reasons.append(\n                    "fresh_oos_predictions:session_set_mismatch_acceptance_calendar:"\n                    f"missing={missing[:10]}:unexpected={unexpected[:10]}"\n                )\n            fresh_start_date = derived.start_date\n            evidence.update(\n                {\n                    "fresh_oos_days": derived.trading_days,\n''',
    )
    replace_once(
        "src/quantagent/execution/live_model_trust_v2_policy.py",
        '''                    "fresh_prediction_symbols": derived.symbols,\n                }\n            )\n''',
        '''                    "fresh_prediction_symbols": derived.symbols,\n                    "fresh_session_set_exact_match": derived.session_dates == calendar.sessions,\n                    "acceptance_calendar_days": calendar.trading_days,\n                    "acceptance_calendar_session_set_sha256": calendar.session_set_sha256,\n                    "acceptance_calendar_source_identity": calendar.source_identity,\n                    "acceptance_calendar_source_version": calendar.source_version,\n                }\n            )\n''',
    )

    # Test bundle uses an exchange-calendar-shaped 2026 window, not generic bdays.
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''from quantagent.execution.live_model_trust import evaluate_live_model_trust\n''',
        '''from quantagent.execution.acceptance_calendar import (\n    build_acceptance_calendar_payload,\n    canonical_session_set_sha256,\n)\nfrom quantagent.execution.live_model_trust import evaluate_live_model_trust\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''WINDOW_ID = "fresh-2026-02-02_2026-07-31"\nFAMILY = ["linear_control", "ft_transformer"]\nSELECTED = "ft_transformer"\nISSUED_AT = datetime(2026, 8, 2, tzinfo=timezone.utc)\n''',
        '''WINDOW_ID = "fresh-2026-02-24_2026-08-31"\nFAMILY = ["linear_control", "ft_transformer"]\nSELECTED = "ft_transformer"\nISSUED_AT = datetime(2026, 9, 2, tzinfo=timezone.utc)\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''def _write_fresh_predictions(path: Path) -> tuple[int, str, str]:\n    dates = pd.bdate_range("2026-02-02", periods=130)\n    rows: list[dict[str, object]] = []\n    for idx, day in enumerate(dates):\n''',
        '''def _fresh_sessions() -> tuple[str, ...]:\n    # 2026 SSE/SZSE-style test window. Weekend filtering alone is insufficient:\n    # remove weekday exchange closures inside the window explicitly.\n    weekday_closures = {\n        "2026-04-06",  # Qingming\n        "2026-05-01",\n        "2026-05-04",\n        "2026-05-05",  # Labour Day closure\n        "2026-06-19",  # Dragon Boat\n    }\n    sessions = tuple(\n        day.date().isoformat()\n        for day in pd.bdate_range("2026-02-24", "2026-08-31")\n        if day.date().isoformat() not in weekday_closures\n    )\n    assert len(sessions) == 130\n    return sessions\n\n\ndef _write_fresh_predictions(path: Path) -> tuple[int, str, str, tuple[str, ...]]:\n    sessions = _fresh_sessions()\n    rows: list[dict[str, object]] = []\n    for idx, session in enumerate(sessions):\n        day = pd.Timestamp(session)\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''    pd.DataFrame(rows).to_csv(path, index=False)\n    return len(dates), dates[0].date().isoformat(), dates[-1].date().isoformat()\n''',
        '''    pd.DataFrame(rows).to_csv(path, index=False)\n    return len(sessions), sessions[0], sessions[-1], sessions\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''    fresh_days, fresh_start, fresh_end = _write_fresh_predictions(files["fresh_oos_predictions"])\n\n    checkpoint_sha = sha256_file(files["model_checkpoint"])\n''',
        '''    fresh_days, fresh_start, fresh_end, fresh_sessions = _write_fresh_predictions(\n        files["fresh_oos_predictions"]\n    )\n    calendar_payload = build_acceptance_calendar_payload(\n        model_id=MODEL_ID,\n        source_commit=SOURCE_COMMIT,\n        acceptance_window_id=WINDOW_ID,\n        window_start_date=fresh_start,\n        window_end_date=fresh_end,\n        sessions=fresh_sessions,\n        source_provider="SSE_SZSE_OFFICIAL_ARCHIVE",\n        source_identity="sse-szse-2026-session-calendar",\n        source_version="2026.rules+holiday-notices.v1",\n        source_retrieved_at="2025-12-22T08:00:00+08:00",\n        source_as_of="2025-12-22T08:00:00+08:00",\n        source_locators=[\n            "SSE:2026-trading-rules",\n            "SSE:2026-holiday-arrangement",\n            "SZSE:2026-holiday-arrangement",\n        ],\n    )\n    _write_json(files["acceptance_calendar"], calendar_payload)\n\n    checkpoint_sha = sha256_file(files["model_checkpoint"])\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''    predictions_sha = sha256_file(files["fresh_oos_predictions"])\n    assert strict_returns_sha != statistical_sha\n''',
        '''    predictions_sha = sha256_file(files["fresh_oos_predictions"])\n    calendar_sha = sha256_file(files["acceptance_calendar"])\n    calendar_session_sha = calendar_payload["session_set_sha256"]\n    assert strict_returns_sha != statistical_sha\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''        "acceptance_end_date": fresh_end,\n    })\n''',
        '''        "acceptance_end_date": fresh_end,\n        "acceptance_calendar_sha256": calendar_sha,\n        "acceptance_calendar_session_set_sha256": calendar_session_sha,\n    })\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''        "validation_cutoff": "2026-01-30",\n    })\n    _write_json(files["strict_backtest"], _common() | {\n''',
        '''        "validation_cutoff": "2026-01-30",\n        "acceptance_window_id": WINDOW_ID,\n        "acceptance_calendar_sha256": calendar_sha,\n        "acceptance_calendar_session_set_sha256": calendar_session_sha,\n    })\n    _write_json(files["strict_backtest"], _common() | {\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''        "acceptance_window_id": WINDOW_ID,\n        "predictions_sha256": predictions_sha,\n    })\n''',
        '''        "acceptance_window_id": WINDOW_ID,\n        "predictions_sha256": predictions_sha,\n        "acceptance_calendar_sha256": calendar_sha,\n        "acceptance_calendar_session_set_sha256": calendar_session_sha,\n    })\n''',
    )
    replace_once(
        "tests/execution/test_live_model_trust_v2.py",
        '''    assert report.evidence["fresh_oos_days"] == 130\n''',
        '''    assert report.evidence["fresh_oos_days"] == 130\n    assert report.evidence["fresh_session_set_exact_match"] is True\n    assert report.evidence["acceptance_calendar_days"] == 130\n    assert report.evidence["acceptance_calendar_session_set_sha256"] == canonical_session_set_sha256(\n        _fresh_sessions()\n    )\n''',
    )
    # Add exact-set and cross-binding adversarial tests near the end.
    p = Path("tests/execution/test_live_model_trust_v2.py")
    text = p.read_text(encoding="utf-8")
    if "test_same_count_and_boundaries_but_wrong_interior_session_fails_calendar_gate" not in text:
        text = text.rstrip() + '''\n\n\ndef test_same_count_and_boundaries_but_wrong_interior_session_fails_calendar_gate(tmp_path: Path) -> None:\n    manifest, _, files, _, roots, _ = _issue(tmp_path)\n    frame = pd.read_csv(files["fresh_oos_predictions"])\n    # 2026-04-06 is a weekday but an exchange closure in the bound calendar.\n    frame.loc[frame["trade_date"] == "2026-04-07", "trade_date"] = "2026-04-06"\n    frame = frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)\n    frame.to_csv(files["fresh_oos_predictions"], index=False)\n    new_sha = sha256_file(files["fresh_oos_predictions"])\n    fresh = _read(files["fresh_oos"]); fresh["predictions_sha256"] = new_sha; _write_json(files["fresh_oos"], fresh)\n    risk = _read(files["risk_capacity"]); risk["predictions_sha256"] = new_sha; _write_json(files["risk_capacity"], risk)\n    payload = _read(manifest)\n    for role in ("fresh_oos_predictions", "fresh_oos", "risk_capacity"):\n        payload["artifacts"][role]["sha256"] = sha256_file(files[role])\n    _write_json(manifest, payload)\n    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons\n    assert any("fresh_oos_predictions:session_set_mismatch_acceptance_calendar" in r for r in reasons)\n    assert not any("trading_days_mismatch_predictions" in r for r in reasons)\n    assert not any("start_date_mismatch_predictions" in r for r in reasons)\n    assert not any("end_date_mismatch_predictions" in r for r in reasons)\n\n\ndef test_rebinding_calendar_file_alone_cannot_rewrite_pre_registered_window(tmp_path: Path) -> None:\n    manifest, _, files, _, roots, _ = _issue(tmp_path)\n    calendar = _read(files["acceptance_calendar"])\n    calendar["sessions"][30] = "2026-04-06"\n    calendar["sessions"] = sorted(calendar["sessions"])\n    calendar["session_set_sha256"] = canonical_session_set_sha256(calendar["sessions"])\n    _write_json(files["acceptance_calendar"], calendar)\n    payload = _read(manifest)\n    payload["artifacts"]["acceptance_calendar"]["sha256"] = sha256_file(\n        files["acceptance_calendar"]\n    )\n    _write_json(manifest, payload)\n    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons\n    assert "pre_registration:acceptance_calendar_sha256_mismatch" in reasons\n    assert "pre_registration:acceptance_calendar_session_set_sha256_mismatch" in reasons\n    assert "data_lineage:acceptance_calendar_sha256_mismatch" in reasons\n    assert "fresh_oos:acceptance_calendar_sha256_mismatch" in reasons\n\n\ndef test_calendar_source_must_exist_before_pre_registration(tmp_path: Path) -> None:\n    manifest, _, files, _, roots, _ = _issue(tmp_path)\n    calendar = _read(files["acceptance_calendar"])\n    calendar["source"]["retrieved_at"] = "2026-01-16T00:00:00+00:00"\n    calendar["source"]["as_of"] = "2026-01-16T00:00:00+00:00"\n    _write_json(files["acceptance_calendar"], calendar)\n    calendar_sha = sha256_file(files["acceptance_calendar"])\n    for role in ("pre_registration", "data_lineage", "fresh_oos"):\n        doc = _read(files[role]); doc["acceptance_calendar_sha256"] = calendar_sha; _write_json(files[role], doc)\n    payload = _read(manifest)\n    for role in ("acceptance_calendar", "pre_registration", "data_lineage", "fresh_oos"):\n        payload["artifacts"][role]["sha256"] = sha256_file(files[role])\n    _write_json(manifest, payload)\n    reasons = evaluate_live_model_trust(manifest, artifact_roots=roots).reasons\n    assert "pre_registration:calendar_retrieved_after_registration" in reasons\n    assert "pre_registration:calendar_as_of_after_registration" in reasons\n'''
        p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
