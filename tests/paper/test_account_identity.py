from __future__ import annotations

import json

import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.account_identity import (
    ACCOUNT_IDENTITY_SCHEMA,
    PaperAccountIdentityCorruption,
    PaperAccountIdentityError,
    PaperAccountIdentityMigrationRequired,
    PaperAccountIdentityMismatch,
    PaperAccountIdentityStore,
    account_identity_path_for_canonical,
    ensure_paper_account_identity,
)


def test_first_worker_creates_identity_and_exact_reuse_is_stable(tmp_path) -> None:
    path = tmp_path / "account_identity.json"
    store = PaperAccountIdentityStore(path)
    first = store.ensure(
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        created_at="2026-08-11T00:00:00+00:00",
    )
    second = store.ensure(
        portfolio_id="v7-paper",
        initial_cash="1000000.00",
    )
    assert first == second
    assert first.schema_version == ACCOUNT_IDENTITY_SCHEMA
    assert first.initial_cash_cny == "1000000.00"
    assert len(first.payload_sha256) == 64
    assert path.exists()


def test_same_ledger_cannot_be_reinterpreted_with_different_initial_cash(tmp_path) -> None:
    path = tmp_path / "account_identity.json"
    store = PaperAccountIdentityStore(path)
    store.ensure(portfolio_id="v7-paper", initial_cash=1_000_000.0)
    with pytest.raises(PaperAccountIdentityMismatch, match="initial_cash mismatch"):
        store.ensure(portfolio_id="v7-paper", initial_cash=2_000_000.0)


def test_same_ledger_cannot_be_reinterpreted_with_different_portfolio_id(tmp_path) -> None:
    path = tmp_path / "account_identity.json"
    store = PaperAccountIdentityStore(path)
    store.ensure(portfolio_id="v7-paper", initial_cash=1_000_000.0)
    with pytest.raises(PaperAccountIdentityMismatch, match="portfolio_id mismatch"):
        store.ensure(portfolio_id="another-book", initial_cash=1_000_000.0)


def test_digest_tamper_fails_closed(tmp_path) -> None:
    path = tmp_path / "account_identity.json"
    identity = PaperAccountIdentityStore(path).ensure(
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["initial_cash_cny"] = "2000000.00"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PaperAccountIdentityCorruption, match="digest mismatch"):
        PaperAccountIdentityStore(path).read()
    assert identity.initial_cash_cny == "1000000.00"


def test_duplicate_key_and_subcent_cash_fail_closed(tmp_path) -> None:
    path = tmp_path / "account_identity.json"
    path.write_text(
        '{"schema_version":"x","schema_version":"y"}',
        encoding="utf-8",
    )
    with pytest.raises(PaperAccountIdentityCorruption, match="duplicate JSON key"):
        PaperAccountIdentityStore(path).read()
    with pytest.raises(PaperAccountIdentityError, match="CNY cents"):
        PaperAccountIdentityStore(tmp_path / "other.json").ensure(
            portfolio_id="v7-paper",
            initial_cash="100.001",
        )


def test_default_identity_path_is_sibling_of_canonical_ledger(tmp_path) -> None:
    canonical = tmp_path / "paper" / "canonical_ledger.jsonl"
    assert account_identity_path_for_canonical(canonical) == canonical.with_name(
        "account_identity.json"
    )
    identity = ensure_paper_account_identity(
        canonical_ledger_path=canonical,
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
    )
    assert canonical.with_name("account_identity.json").exists()
    assert identity.initial_cash == pytest.approx(100_000.0)


def test_custom_ledgers_in_one_directory_get_distinct_identity_files(tmp_path) -> None:
    first = tmp_path / "paper-a.jsonl"
    second = tmp_path / "paper-b.jsonl"
    assert account_identity_path_for_canonical(first) == tmp_path / "paper-a.account_identity.json"
    assert account_identity_path_for_canonical(second) == tmp_path / "paper-b.account_identity.json"
    assert account_identity_path_for_canonical(first) != account_identity_path_for_canonical(second)


def test_identity_path_canonicalizes_symlink_alias(tmp_path) -> None:
    real = tmp_path / "real.jsonl"
    real.write_text("", encoding="utf-8")
    alias = tmp_path / "alias.jsonl"
    try:
        alias.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    assert account_identity_path_for_canonical(alias) == account_identity_path_for_canonical(real)


def test_nonempty_legacy_canonical_ledger_requires_explicit_identity_migration(tmp_path) -> None:
    canonical = tmp_path / "canonical_ledger.jsonl"
    ledger = CanonicalLedger(canonical)
    # Any valid canonical record means an economic history already exists.  The
    # identity layer may not guess which initial_cash/portfolio_id created it.
    ledger.append(None, trade_date="2026-08-10")
    assert ledger.verify()["valid"] is True
    assert len(ledger) == 1

    with pytest.raises(PaperAccountIdentityMigrationRequired, match="explicit audited migration"):
        ensure_paper_account_identity(
            canonical_ledger_path=canonical,
            portfolio_id="v7-paper",
            initial_cash=100_000.0,
        )
    assert not canonical.with_name("account_identity.json").exists()


def test_empty_existing_canonical_file_can_establish_identity(tmp_path) -> None:
    canonical = tmp_path / "canonical_ledger.jsonl"
    canonical.touch()
    identity = ensure_paper_account_identity(
        canonical_ledger_path=canonical,
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
    )
    assert identity.portfolio_id == "v7-paper"
    assert identity.initial_cash == pytest.approx(100_000.0)
