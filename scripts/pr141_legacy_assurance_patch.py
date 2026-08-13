from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''def _assert_recovered_account_consistent(\n    canonical_state,\n    operational_state,\n    *,\n    tolerance: float = 1e-6,\n) -> None:\n    operational_positions = _position_quantities(operational_state.portfolio)\n    operational_initial_cash = float(\n        getattr(operational_state.portfolio, "initial_cash", 0.0)\n    )\n    operational_cash = float(operational_state.portfolio.cash)\n    has_operational_economics = bool(\n        operational_state.orders\n        or operational_state.fills\n        or operational_positions\n        or abs(operational_cash - operational_initial_cash) > tolerance\n    )\n\n    canonical_positions = _position_quantities(canonical_state.portfolio)\n    canonical_initial_cash = float(\n        getattr(canonical_state.portfolio, "initial_cash", 0.0)\n    )\n    canonical_cash = float(canonical_state.portfolio.cash)\n    has_canonical_economics = bool(\n        canonical_state.orders\n        or canonical_state.fills\n        or canonical_positions\n        or abs(canonical_cash - canonical_initial_cash) > tolerance\n    )\n    if not has_operational_economics:\n        if has_canonical_economics:\n            raise ContinuousPaperExecutionBlocked(\n                "operational paper ledger has no reconstructable economics while "\n                "the canonical ledger contains account history; refusing to claim "\n                "canonical/operational parity"\n            )\n        return\n\n    if abs(canonical_cash - operational_cash) > tolerance:\n''',
        '''def _operational_has_reconstructable_economics(\n    operational_state,\n    *,\n    tolerance: float = 1e-6,\n) -> bool:\n    operational_positions = _position_quantities(operational_state.portfolio)\n    initial_cash = float(getattr(operational_state.portfolio, "initial_cash", 0.0))\n    cash = float(operational_state.portfolio.cash)\n    return bool(\n        operational_state.orders\n        or operational_state.fills\n        or operational_positions\n        or abs(cash - initial_cash) > tolerance\n    )\n\n\ndef _assert_recovered_account_consistent(\n    canonical_state,\n    operational_state,\n    *,\n    tolerance: float = 1e-6,\n) -> None:\n    # The canonical ledger is the economic record of account. The operational\n    # ledger may legitimately contain only session/control events. If it does\n    # reconstruct economics, however, those economics must agree exactly enough\n    # with canonical state; a conflicting second economic state fails closed.\n    if not _operational_has_reconstructable_economics(\n        operational_state, tolerance=tolerance\n    ):\n        return\n\n    operational_positions = _position_quantities(operational_state.portfolio)\n    operational_cash = float(operational_state.portfolio.cash)\n    canonical_cash = float(canonical_state.portfolio.cash)\n    if abs(canonical_cash - operational_cash) > tolerance:\n''',
    )

    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''    operational_state = recover(\n        EventLedger(config.operational_ledger_path),\n        portfolio_id=identity.portfolio_id,\n        initial_cash=identity.initial_cash,\n    )\n    _assert_recovered_account_consistent(canonical_state, operational_state)\n    if canonical_state.open_orders() or operational_state.open_orders():\n''',
        '''    operational_state = recover(\n        EventLedger(config.operational_ledger_path),\n        portfolio_id=identity.portfolio_id,\n        initial_cash=identity.initial_cash,\n    )\n    operational_economics = _operational_has_reconstructable_economics(\n        operational_state\n    )\n    _assert_recovered_account_consistent(canonical_state, operational_state)\n    if canonical_state.open_orders() or operational_state.open_orders():\n''',
    )

    replace_once(
        "src/quantagent/paper/continuous_execution.py",
        '''        "reason": binding_reason,\n        "assurance": "operator_reconciled_legacy_terminal_v1",\n    }\n''',
        '''        "reason": binding_reason,\n        "assurance": (\n            "operator_reconciled_legacy_terminal_v2"\n            if operational_economics\n            else "operator_bound_canonical_only_legacy_terminal_v1"\n        ),\n        "operational_economic_reconstruction": (\n            "matched_canonical"\n            if operational_economics\n            else "not_present_canonical_is_record_of_account"\n        ),\n    }\n''',
    )

    replace_once(
        "src/quantagent/paper/legacy_terminal_binding.py",
        '''    if str(details.get("assurance") or "") != "operator_reconciled_legacy_terminal_v1":\n        raise LegacyTerminalBindingError("legacy terminal binding assurance is invalid")\n''',
        '''    assurance = str(details.get("assurance") or "")\n    reconstruction = str(details.get("operational_economic_reconstruction") or "")\n    if assurance == "operator_reconciled_legacy_terminal_v2":\n        if reconstruction != "matched_canonical":\n            raise LegacyTerminalBindingError(\n                "legacy terminal parity assurance lacks matched operational reconstruction"\n            )\n    elif assurance == "operator_bound_canonical_only_legacy_terminal_v1":\n        if reconstruction != "not_present_canonical_is_record_of_account":\n            raise LegacyTerminalBindingError(\n                "legacy terminal canonical-only assurance marker is invalid"\n            )\n    else:\n        # The historical v1 wording was ambiguous: it could claim reconciliation\n        # even when the operational ledger contained no economic reconstruction.\n        # Never silently upgrade that lower-quality evidence.\n        raise LegacyTerminalBindingError("legacy terminal binding assurance is invalid")\n''',
    )

    replace_once(
        "src/quantagent/paper/execution_journal.py",
        '''                assurance = str(normalized_details.get("assurance") or "")\n                if assurance != "operator_reconciled_legacy_terminal_v1":\n                    raise ExecutionJournalCorruption(\n                        "legacy_terminal_bound requires explicit migration assurance"\n                    )\n''',
        '''                assurance = str(normalized_details.get("assurance") or "")\n                reconstruction = str(\n                    normalized_details.get("operational_economic_reconstruction") or ""\n                )\n                assurance_contracts = {\n                    "operator_reconciled_legacy_terminal_v2": "matched_canonical",\n                    "operator_bound_canonical_only_legacy_terminal_v1": (\n                        "not_present_canonical_is_record_of_account"\n                    ),\n                }\n                if assurance not in assurance_contracts:\n                    raise ExecutionJournalCorruption(\n                        "legacy_terminal_bound requires explicit unambiguous migration assurance"\n                    )\n                if reconstruction != assurance_contracts[assurance]:\n                    raise ExecutionJournalCorruption(\n                        "legacy_terminal_bound operational reconstruction marker mismatches assurance"\n                    )\n''',
    )

    test = Path("tests/paper/test_daily_decision_and_legacy_binding.py")
    text = test.read_text(encoding="utf-8")
    old = '''def test_recovered_consistency_refuses_empty_operational_history_with_canonical_economics() -> None:\n    held = SimpleNamespace(total=100.0, is_flat=False)\n    canonical = SimpleNamespace(\n        portfolio=SimpleNamespace(\n            positions={"600000.SH": held}, cash=99_000.0, initial_cash=100_000.0\n        ),\n        orders={},\n        fills=[],\n    )\n    operational = SimpleNamespace(\n        portfolio=SimpleNamespace(positions={}, cash=100_000.0, initial_cash=100_000.0),\n        orders={},\n        fills=[],\n    )\n    with pytest.raises(\n        continuous_execution.ContinuousPaperExecutionBlocked,\n        match="operational paper ledger has no reconstructable economics",\n    ):\n        continuous_execution._assert_recovered_account_consistent(canonical, operational)\n'''
    new = '''def test_recovered_consistency_allows_operational_lifecycle_only_state() -> None:\n    held = SimpleNamespace(total=100.0, is_flat=False)\n    canonical = SimpleNamespace(\n        portfolio=SimpleNamespace(\n            positions={"600000.SH": held}, cash=99_000.0, initial_cash=100_000.0\n        ),\n        orders={},\n        fills=[],\n    )\n    operational = SimpleNamespace(\n        portfolio=SimpleNamespace(positions={}, cash=100_000.0, initial_cash=100_000.0),\n        orders={},\n        fills=[],\n    )\n    assert continuous_execution._operational_has_reconstructable_economics(operational) is False\n    continuous_execution._assert_recovered_account_consistent(canonical, operational)\n\n\ndef test_recovered_consistency_rejects_conflicting_operational_economics() -> None:\n    held = SimpleNamespace(total=100.0, is_flat=False)\n    wrong = SimpleNamespace(total=50.0, is_flat=False)\n    canonical = SimpleNamespace(\n        portfolio=SimpleNamespace(\n            positions={"600000.SH": held}, cash=99_000.0, initial_cash=100_000.0\n        ),\n        orders={},\n        fills=[],\n    )\n    operational = SimpleNamespace(\n        portfolio=SimpleNamespace(\n            positions={"600000.SH": wrong}, cash=99_000.0, initial_cash=100_000.0\n        ),\n        orders={},\n        fills=[],\n    )\n    assert continuous_execution._operational_has_reconstructable_economics(operational) is True\n    with pytest.raises(\n        continuous_execution.ContinuousPaperExecutionBlocked,\n        match="position reconciliation failed",\n    ):\n        continuous_execution._assert_recovered_account_consistent(canonical, operational)\n'''
    if old not in text:
        raise RuntimeError("legacy test anchor not found")
    text = text.replace(old, new, 1)
    text = text.replace(
        '''    assert binding["status"] == "legacy_terminal_bound"\n    assert journal.terminal(prior.payload_sha256).record_sha256 == terminal.record_sha256\n''',
        '''    assert binding["status"] == "legacy_terminal_bound"\n    assert binding["details"]["assurance"] == "operator_bound_canonical_only_legacy_terminal_v1"\n    assert (\n        binding["details"]["operational_economic_reconstruction"]\n        == "not_present_canonical_is_record_of_account"\n    )\n    assert journal.terminal(prior.payload_sha256).record_sha256 == terminal.record_sha256\n''',
        1,
    )
    test.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
