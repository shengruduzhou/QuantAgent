"""H-032C: strict PIT metadata sourcing, universe reconciliation, entitlement
re-test contract. Guarded skip-if-absent for CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
U0 = REPO / "runtime/data/u0"
PIT = U0 / "pit"

PROVENANCE_COLS = {"symbol", "effective_start", "available_at", "source",
                   "source_timestamp", "source_hash"}


def _pd():
    return pytest.importorskip("pandas")


def test_pit_metadata_manifest_closes_delisting_with_provenance() -> None:
    m = PIT / "pit_metadata_manifest.json"
    if not m.exists():
        pytest.skip("pit metadata not sourced")
    mj = json.loads(m.read_text())
    # delisting/price-limit/ipo are closed; st/suspension/corp-action remain blocked
    assert "delisting_intervals" in mj["closed_fields"]
    assert "price_limit_regimes" in mj["closed_fields"]
    assert set(mj["blocked_fields"]) <= {"st_intervals", "suspension_intervals",
                                         "corporate_action_identity"}
    assert mj["delisting_dates_sourced"] >= 0


def test_pit_interval_tables_carry_full_provenance() -> None:
    pd = _pd()
    for name in ("price_limit_regimes.parquet", "ipo_special_limit_intervals.parquet",
                 "delisting_intervals.parquet"):
        p = PIT / name
        if not p.exists():
            pytest.skip(f"{name} not generated")
        df = pd.read_parquet(p)
        if len(df):
            assert PROVENANCE_COLS <= set(df.columns), name


def test_strict_pit_reflects_delisting_closure() -> None:
    cert = U0 / "u0_strict_pit_certificate.json"
    if not cert.exists():
        pytest.skip("strict pit cert not generated")
    c = json.loads(cert.read_text())
    # delisting must no longer be a blocked PIT field once sourced
    assert "delisting_intervals" not in c["blocked_pit_fields"]
    # training only on full readiness
    assert c["training_permitted"] == (c["decision"] == "FULL_UNIVERSE_DATA_READY")


def test_reconciliation_prevents_bse_dual_identity() -> None:
    r = U0 / "universe_reconciliation.json"
    if not r.exists():
        pytest.skip("reconciliation not generated")
    rj = json.loads(r.read_text())
    guard = rj["dual_identity_guard"]
    # no old 8xxxxx code may exist as a separate BSE security
    assert guard["dual_identity_collisions"] == 0
    assert guard["old_8xxxxx_codes_in_master"] == []


def test_reconciliation_only_adds_authoritative_listings() -> None:
    r = U0 / "universe_reconciliation.json"
    if not r.exists():
        pytest.skip("reconciliation not generated")
    rj = json.loads(r.read_text())
    # additions come from the authoritative akshare BSE list; none rejected-but-added
    assert rj["rejected_identities"] == []
    for sym in rj.get("supplemental_additions_symbols", []):
        assert sym.endswith(".BJ")  # only BSE recent listings this pass


def _backfill_module():
    """Import the backfill script as a module (it is a script, not a package)."""
    import importlib.util
    import sys

    if "bf_mod" in sys.modules:
        return sys.modules["bf_mod"]
    spec = importlib.util.spec_from_file_location(
        "bf_mod", REPO / "scripts/u0_full_universe_backfill.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bf_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frozen_master(pd):
    """A minimal deterministic stand-in for the frozen H-028 master.

    Five securities spanning every board, each carrying the listing/delisting
    metadata the universe depends on. Deliberately tiny and constructed in the
    test: the real artifact is a gitignored runtime file, and depending on it is
    exactly what broke CI.
    """
    return pd.DataFrame([
        {"symbol": "600000.SH", "board": "SH_Main", "name": "浦发银行",
         "listing_date": "1999-11-10", "delisting_date": None, "status": "listed"},
        {"symbol": "000001.SZ", "board": "SZ_Main", "name": "平安银行",
         "listing_date": "1991-04-03", "delisting_date": None, "status": "listed"},
        {"symbol": "300750.SZ", "board": "ChiNext", "name": "宁德时代",
         "listing_date": "2018-06-11", "delisting_date": None, "status": "listed"},
        {"symbol": "688981.SH", "board": "STAR", "name": "中芯国际",
         "listing_date": "2020-07-16", "delisting_date": None, "status": "listed"},
        {"symbol": "600001.SH", "board": "SH_Main", "name": "邯郸钢铁",
         "listing_date": "1998-01-22", "delisting_date": "2009-12-25",
         "status": "delisted"},
    ])


class TestSupplementalUnion:
    """Union semantics for the frozen master and reconciliation additions.

    Hermetic by construction -- every input is built in the test -- while
    asserting more than the previous artifact-dependent version did.
    """

    def test_new_symbols_are_added(self, tmp_path):
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([
            {"symbol": "920002.BJ", "board": "BSE", "name": "北交所标的",
             "listing_date": "2021-11-15", "delisting_date": None, "status": "listed"},
        ])
        out = mod.union_master(master, supplemental)
        assert "920002.BJ" in set(out["symbol"].astype(str))
        assert len(out) == len(master) + 1

    def test_no_duplicate_security_rows(self):
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([
            {"symbol": "600000.SH", "board": "SH_Main", "name": "重复",
             "listing_date": "1999-11-10", "delisting_date": None, "status": "listed"},
            {"symbol": "920002.BJ", "board": "BSE", "name": "新增",
             "listing_date": "2021-11-15", "delisting_date": None, "status": "listed"},
        ])
        out = mod.union_master(master, supplemental)
        assert out["symbol"].astype(str).duplicated().sum() == 0

    def test_supplemental_internal_duplicates_are_collapsed(self):
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([
            {"symbol": "920002.BJ", "board": "BSE", "name": "第一次",
             "listing_date": "2021-11-15", "delisting_date": None, "status": "listed"},
            {"symbol": "920002.BJ", "board": "BSE", "name": "第二次",
             "listing_date": "2021-11-15", "delisting_date": None, "status": "listed"},
        ])
        out = mod.union_master(master, supplemental)
        assert (out["symbol"].astype(str) == "920002.BJ").sum() == 1

    def test_frozen_master_wins_on_conflicting_identity(self):
        """A supplemental row claiming a known symbol must not change anything."""
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([
            {"symbol": "600000.SH", "board": "ChiNext", "name": "冒名顶替",
             "listing_date": "2020-01-01", "delisting_date": "2021-01-01",
             "status": "delisted"},
        ])
        out = mod.union_master(master, supplemental)
        rows = out[out["symbol"] == "600000.SH"]
        # Exactly one row: an appended duplicate is also a precedence failure,
        # and asserting only on .iloc[0] would still see the frozen row.
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["board"] == "SH_Main"
        assert row["name"] == "浦发银行"
        assert row["status"] == "listed"

    def test_listing_and_delisting_metadata_are_not_overwritten(self):
        """The precise unintended-overwrite case: dates must survive intact."""
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([
            {"symbol": "600001.SH", "board": "SH_Main", "name": "邯郸钢铁",
             "listing_date": "2015-01-01", "delisting_date": None, "status": "listed"},
        ])
        out = mod.union_master(master, supplemental)
        rows = out[out["symbol"] == "600001.SH"]
        assert len(rows) == 1, "an appended duplicate also destroys the metadata guarantee"
        row = rows.iloc[0]
        assert row["listing_date"] == "1998-01-22"
        assert row["delisting_date"] == "2009-12-25"
        assert row["status"] == "delisted"

    def test_missing_listing_metadata_in_supplemental_is_preserved_as_missing(self):
        """A new symbol with no dates keeps nulls; nothing is invented."""
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([
            {"symbol": "920003.BJ", "board": "BSE", "name": "缺元数据",
             "listing_date": None, "delisting_date": None, "status": "listed"},
        ])
        out = mod.union_master(master, supplemental)
        row = out[out["symbol"] == "920003.BJ"].iloc[0]
        assert pd.isna(row["listing_date"])
        assert pd.isna(row["delisting_date"])

    def test_supplemental_cannot_widen_the_schema(self):
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([
            {"symbol": "920002.BJ", "board": "BSE", "name": "新增",
             "listing_date": "2021-11-15", "delisting_date": None,
             "status": "listed", "unexpected_column": "should not appear"},
        ])
        out = mod.union_master(master, supplemental)
        assert "unexpected_column" not in out.columns
        assert list(out.columns) == list(master.columns)

    def test_supplemental_without_symbol_column_is_ignored(self):
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        supplemental = pd.DataFrame([{"board": "BSE", "name": "无主键"}])
        out = mod.union_master(master, supplemental)
        assert len(out) == len(master)

    def test_empty_or_absent_supplemental_is_a_no_op(self):
        pd = _pd()
        mod = _backfill_module()
        master = _frozen_master(pd)
        assert len(mod.union_master(master, None)) == len(master)
        assert len(mod.union_master(master, pd.DataFrame())) == len(master)

    def test_load_master_reads_injected_paths(self, tmp_path):
        """The CI fix itself: paths are injectable, so no runtime artifact is needed."""
        pd = _pd()
        mod = _backfill_module()
        master_path = tmp_path / "master.parquet"
        supplemental_path = tmp_path / "supplemental.parquet"
        _frozen_master(pd).to_parquet(master_path, index=False)
        pd.DataFrame([
            {"symbol": "920002.BJ", "board": "BSE", "name": "新增",
             "listing_date": "2021-11-15", "delisting_date": None, "status": "listed"},
        ]).to_parquet(supplemental_path, index=False)

        out = mod.load_master(master_path=master_path,
                              supplemental_path=supplemental_path)
        assert len(out) == 6
        assert "920002.BJ" in set(out["symbol"].astype(str))

    def test_load_master_tolerates_absent_supplemental(self, tmp_path):
        pd = _pd()
        mod = _backfill_module()
        master_path = tmp_path / "master.parquet"
        _frozen_master(pd).to_parquet(master_path, index=False)
        out = mod.load_master(master_path=master_path,
                              supplemental_path=tmp_path / "missing.parquet")
        assert len(out) == 5

    def test_corrupt_supplemental_is_reported_not_swallowed(self, tmp_path, capsys):
        """A corrupt addition previously became a silently smaller universe."""
        pd = _pd()
        mod = _backfill_module()
        master_path = tmp_path / "master.parquet"
        _frozen_master(pd).to_parquet(master_path, index=False)
        corrupt = tmp_path / "corrupt.parquet"
        corrupt.write_bytes(b"this is not parquet")

        out = mod.load_master(master_path=master_path, supplemental_path=corrupt)
        assert len(out) == 5
        assert "WARNING" in capsys.readouterr().out


def test_supplemental_additions_actually_reach_the_backfill_master() -> None:
    """§4: reconciliation additions must be fetched+assembled, not only recorded.

    Artifact-dependent by nature -- it checks the *real* universe -- so it skips
    when either input is absent rather than failing a clean environment.
    """
    pd = _pd()
    add = U0 / "master_supplemental_additions.parquet"
    master = REPO / "runtime/reports/h028/track_a/historical_security_master.parquet"
    if not add.exists() or not master.exists():
        pytest.skip("frozen master or supplemental additions not present on this host")
    extra = set(pd.read_parquet(add)["symbol"].astype(str))
    master_syms = set(_backfill_module().load_master()["symbol"].astype(str))
    assert extra <= master_syms


def test_entitlement_audit_keeps_tickflow_primary_and_no_fabrication() -> None:
    a = REPO / "runtime/reports/h032c/tickflow_entitlement_audit.json"
    if not a.exists():
        pytest.skip("entitlement audit not run yet (may be deferred behind Track-F)")
    aj = json.loads(a.read_text())
    assert "TickFlow" in aj.get("primary_bar_provider", "")
    # if ex_factors is not entitled, corporate actions must be ALTERNATIVE_SOURCE_REQUIRED
    if aj.get("status") != "DEFERRED" and "NOT_ENTITLED" in str(aj.get("ex_factors", "")):
        assert aj["corporate_action_classification"] == "ALTERNATIVE_SOURCE_REQUIRED"
