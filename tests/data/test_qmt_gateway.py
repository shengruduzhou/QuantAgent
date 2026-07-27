"""QMT gateway, entitlement matrix, governed skills and U0 reconciliation.

Every test names a way this layer could produce a false positive: a documented
API read as a granted entitlement, a permission-denied empty read as "no such
data", a truncated window read as full history, a skill dispatched because its
name exists, or a QMT second opinion silently overwriting a verified U0 value.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantagent.data.ashare import source_precedence as sp
from quantagent.data.providers import qmt_entitlement as ent
from quantagent.data.providers import qmt_gateway as gw
from quantagent.data.providers import qmt_skills as sk


# --- entitlement matrix -----------------------------------------------------
class TestEntitlementMatrix:
    def test_serving_requires_rows(self):
        with pytest.raises(ValueError, match="SERVING with 0 rows"):
            ent.EntitlementCell(
                capability="daily_raw", api="x", documented=True, platform="WINDOWS",
                permission_class=ent.BASIC_INCLUDED, probe_status=ent.SERVING,
                rows_returned=0,
            )

    def test_empty_cannot_be_verified_without_entitlement(self):
        """The core rule: an unentitled empty is not an empty dataset."""
        with pytest.raises(ValueError, match="EMPTY_UNVERIFIED"):
            ent.EntitlementCell(
                capability="st_history", api="get_his_st_data", documented=True,
                platform="WINDOWS", permission_class=ent.UNKNOWN_UNTIL_PROBED,
                probe_status=ent.EMPTY_VERIFIED,
            )

    def test_empty_unverified_does_not_prove_absence(self):
        cell = ent.EntitlementCell(
            capability="st_history", api="get_his_st_data", documented=True,
            platform="WINDOWS", permission_class=ent.UNKNOWN_UNTIL_PROBED,
            probe_status=ent.EMPTY_UNVERIFIED,
        )
        assert cell.proves_absence is False
        assert cell.usable is False

    def test_verified_empty_proves_absence(self):
        cell = ent.EntitlementCell(
            capability="st_history", api="get_his_st_data", documented=True,
            platform="WINDOWS", permission_class=ent.VIP_INCLUDED,
            probe_status=ent.EMPTY_VERIFIED,
        )
        assert cell.proves_absence is True

    def test_unknown_permission_class_rejected(self):
        with pytest.raises(ValueError, match="unknown permission_class"):
            ent.EntitlementCell(
                capability="x", api="y", documented=True, platform="WINDOWS",
                permission_class="FREE", probe_status=ent.NOT_PROBED,
            )

    def test_catalogue_covers_every_required_family(self):
        families = {
            ent.CAPABILITY_BY_NAME[c.capability].family for c in ent.CAPABILITY_CATALOGUE
        }
        assert families == {"security", "bars", "pit", "microstructure"}

    def test_catalogue_includes_the_level2_family_and_st(self):
        names = {c.capability for c in ent.CAPABILITY_CATALOGUE}
        for required in ("l2quote", "l2quoteaux", "l2order", "l2transaction",
                         "l2transactioncount", "l2orderqueue", "st_history"):
            assert required in names

    def test_unprobed_matrix_lists_everything_as_blocked(self):
        matrix = ent.unprobed_matrix(platform="Linux", reason="no windows")
        assert len(matrix) == len(ent.CAPABILITY_CATALOGUE)
        assert matrix.serving() == []
        assert all(c.probe_status == ent.PLATFORM_UNAVAILABLE for c in matrix.cells)

    def test_summary_states_the_interpretation_rules(self):
        matrix = ent.unprobed_matrix(platform="Linux", reason="no windows")
        rules = matrix.summary()["interpretation_rules"]
        assert "documented_api_is_not_entitlement" in rules
        assert "empty_is_not_absence" in rules


# --- gateway ----------------------------------------------------------------
class TestGatewayEnvironment:
    def test_non_windows_is_platform_unavailable(self):
        env = gw.probe_environment()
        if env.is_windows:
            pytest.skip("this host is Windows")
        assert env.verdict == ent.PLATFORM_UNAVAILABLE
        assert "Windows" in env.detail

    def test_gateway_raises_rather_than_returning_empty(self):
        env = gw.GatewayEnvironment(
            probed_at="2026-07-28T00:00:00+00:00", os_name="Linux", os_release="x",
            is_windows=False, python_version="3.12", import_error="ModuleNotFoundError",
        )
        gateway = gw.QmtGateway(environment=env)
        with pytest.raises(gw.QmtUnavailable, match="not importable"):
            gateway.fetch_st_history("000004.SZ")

    def test_client_disconnected_is_distinct_from_missing_package(self):
        env = gw.GatewayEnvironment(
            probed_at="t", os_name="Windows", os_release="10", is_windows=True,
            python_version="3.12", xtquant_installed=True, xtdata_importable=True,
            client_connected=False, connect_error="no client",
        )
        gateway = gw.QmtGateway(environment=env)
        with pytest.raises(gw.QmtUnavailable, match="no MiniQMT client answered"):
            gateway.fetch_st_history("000004.SZ")


class TestPermissionAndEmptyClassification:
    @pytest.mark.parametrize("message", [
        "no permission for lv2", "无权限", "未授权", "VIP required", "not allowed",
    ])
    def test_permission_errors_are_recognised(self, message):
        assert gw.looks_like_permission_error(message)

    def test_ordinary_errors_are_not_permission_errors(self):
        assert not gw.looks_like_permission_error("connection reset by peer")
        assert not gw.looks_like_permission_error(None)

    def test_unentitled_empty_is_never_verified(self):
        assert gw.classify_empty(error=None, entitlement_confirmed=False) == ent.EMPTY_UNVERIFIED

    def test_entitled_empty_is_verified(self):
        assert gw.classify_empty(error=None, entitlement_confirmed=True) == ent.EMPTY_VERIFIED

    def test_permission_error_beats_entitlement_flag(self):
        assert gw.classify_empty(error="无权限", entitlement_confirmed=True) == ent.PERMISSION_DENIED


class TestTruncationDetection:
    def test_narrower_start_is_truncation(self):
        assert gw.detect_truncation(
            requested_start="19900101", actual_start="20250101",
            requested_end="20260724", actual_end="20260724",
        )

    def test_narrower_end_is_truncation(self):
        assert gw.detect_truncation(
            requested_start="20250101", actual_start="20250101",
            requested_end="20260724", actual_end="20260101",
        )

    def test_full_window_is_not_truncation(self):
        assert not gw.detect_truncation(
            requested_start="20250101", actual_start="20250101",
            requested_end="20260724", actual_end="20260724",
        )

    def test_wider_actual_is_not_truncation(self):
        assert not gw.detect_truncation(
            requested_start="20250101", actual_start="20200101",
            requested_end="20260724", actual_end="20260724",
        )


class TestOutputPathGuard:
    def test_allowed_root_accepted(self, tmp_path):
        (tmp_path / "runtime/data/qmt_staging").mkdir(parents=True)
        out = gw.assert_allowed_output("runtime/data/qmt_staging/x.parquet", repo_root=tmp_path)
        assert str(out).endswith("x.parquet")

    def test_traversal_refused(self, tmp_path):
        with pytest.raises(gw.QmtWriteRefused):
            gw.assert_allowed_output("runtime/data/qmt_staging/../../../etc/passwd",
                                     repo_root=tmp_path)

    def test_arbitrary_root_refused(self, tmp_path):
        with pytest.raises(gw.QmtWriteRefused):
            gw.assert_allowed_output("src/quantagent/evil.parquet", repo_root=tmp_path)

    def test_module_never_imports_the_trader(self):
        """Read-only separation checked on the parsed import graph, not prose."""
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(gw.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported.add(module)
                imported.update(f"{module}.{a.name}" for a in node.names)
        assert "xtquant.xttrader" not in imported
        assert not any("XtQuantTrader" in name for name in imported)


# --- skills -----------------------------------------------------------------
def _matrix_with(capability: str, status: str) -> ent.EntitlementMatrix:
    matrix = ent.unprobed_matrix(platform="Windows", reason="test")
    for cell in matrix.cells:
        if cell.capability == capability:
            cell.probe_status = status
            cell.permission_class = ent.BASIC_INCLUDED
            if status == ent.SERVING:
                cell.rows_returned = 10
    return matrix


class TestSkillGating:
    def test_unknown_skill_refused(self):
        registry = sk.SkillRegistry(platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_do_whatever")
        assert exc.value.reason == sk.REFUSED_UNKNOWN_SKILL

    def test_windows_skill_refused_on_linux(self):
        registry = sk.SkillRegistry(platform_is_windows=False)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_download_daily",
                               {"symbols": ["600000.SH"], "start": "20250101", "end": "20260101"})
        assert exc.value.reason == sk.REFUSED_PLATFORM
        assert "NOT_RUN_PLATFORM" in str(exc.value)

    def test_skill_refused_when_capability_not_serving(self):
        """A documented API is not a granted entitlement."""
        matrix = _matrix_with("daily_raw", ent.PERMISSION_DENIED)
        registry = sk.SkillRegistry(matrix, platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_download_daily",
                               {"symbols": ["600000.SH"], "start": "20250101", "end": "20260101"})
        assert exc.value.reason == sk.REFUSED_ENTITLEMENT

    def test_skill_allowed_when_capability_serving(self):
        matrix = _matrix_with("daily_raw", ent.SERVING)
        registry = sk.SkillRegistry(matrix, platform_is_windows=True)
        spec = registry.authorize(
            "qmt_download_daily",
            {"symbols": ["600000.SH"], "start": "20250101", "end": "20260101"},
        )
        assert spec.name == "qmt_download_daily"
        assert spec.read_only is True

    def test_trading_parameters_rejected_outright(self):
        matrix = _matrix_with("daily_raw", ent.SERVING)
        registry = sk.SkillRegistry(matrix, platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_download_daily",
                               {"symbols": ["600000.SH"], "start": "20250101",
                                "end": "20260101", "dividend_type": "place order"})
        assert exc.value.reason == sk.REFUSED_TRADING

    def test_level2_probe_is_not_mistaken_for_order_submission(self):
        """'l2order' contains 'order' but is a read-only feed probe."""
        registry = sk.SkillRegistry(_matrix_with("l2order", ent.SERVING),
                                    platform_is_windows=True)
        spec = registry.authorize("qmt_probe_level2", {"symbols": ["600000.SH"]})
        assert spec.name == "qmt_probe_level2"

    def test_invalid_symbol_rejected(self):
        registry = sk.SkillRegistry(_matrix_with("daily_raw", ent.SERVING),
                                    platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_download_daily",
                               {"symbols": ["../../etc/passwd"], "start": "20250101",
                                "end": "20260101"})
        assert exc.value.reason == sk.REFUSED_PARAMETERS

    def test_invalid_date_rejected(self):
        registry = sk.SkillRegistry(_matrix_with("daily_raw", ent.SERVING),
                                    platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_download_daily",
                               {"symbols": ["600000.SH"], "start": "2025-01-01",
                                "end": "20260101"})
        assert exc.value.reason == sk.REFUSED_PARAMETERS

    def test_unknown_parameter_rejected(self):
        registry = sk.SkillRegistry(_matrix_with("daily_raw", ent.SERVING),
                                    platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_download_daily",
                               {"symbols": ["600000.SH"], "start": "20250101",
                                "end": "20260101", "shell": "rm -rf /"})
        assert exc.value.reason == sk.REFUSED_PARAMETERS

    def test_path_traversal_rejected(self):
        registry = sk.SkillRegistry(platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_export_canonical_partitions",
                               {"staging_path": "runtime/data/qmt_staging/../../../etc",
                                "output_path": "runtime/data/qmt_staging/out"})
        assert exc.value.reason == sk.REFUSED_PATH

    def test_absolute_output_path_rejected(self):
        registry = sk.SkillRegistry(platform_is_windows=True)
        with pytest.raises(sk.SkillRefused) as exc:
            registry.authorize("qmt_export_canonical_partitions",
                               {"staging_path": "/etc/passwd",
                                "output_path": "runtime/data/qmt_staging/out"})
        assert exc.value.reason == sk.REFUSED_PATH

    def test_every_refusal_is_audited(self):
        registry = sk.SkillRegistry(platform_is_windows=False)
        for skill in ("qmt_download_daily", "qmt_download_tick"):
            with pytest.raises(sk.SkillRefused):
                registry.authorize(skill, {"symbols": ["600000.SH"],
                                           "start": "20250101", "end": "20260101"})
        records = registry.audit_records()
        assert len(records) == 2
        assert all(r["allowed"] is False for r in records)
        assert all(r["reason"] == sk.REFUSED_PLATFORM for r in records)

    def test_no_skill_grants_trading(self):
        for spec in sk.SKILLS.values():
            assert spec.read_only is True
            assert spec.to_dict()["trading_permitted"] is False


# --- U0 reconciliation ------------------------------------------------------
def _panel(rows):
    return pd.DataFrame(rows)


class TestReconciliation:
    def _row(self, **kw):
        base = {"symbol": "600000.SH", "trade_date": "2026-07-24", "open": 9.08,
                "high": 9.12, "low": 9.02, "close": 9.04, "volume": 50_675_100.0,
                "amount": 459_278_100.0}
        base.update(kw)
        return base

    def test_exact_match(self):
        report = sp.reconcile(_panel([self._row()]), _panel([self._row()]))
        assert report["outcome_counts"].get(sp.MISMATCH_VALUE, 0) == 0
        assert report["outcome_counts"][sp.MATCH] >= 6

    def test_amount_mismatch_detected(self):
        report = sp.reconcile(_panel([self._row()]),
                              _panel([self._row(amount=500_000_000.0)]))
        assert report["outcome_counts"].get(sp.MISMATCH_VALUE, 0) >= 1

    def test_volume_unit_mismatch_is_its_own_outcome(self):
        """手 vs shares is a schema defect, not a data disagreement."""
        report = sp.reconcile(_panel([self._row()]),
                              _panel([self._row(volume=506_751.0)]))
        assert report["outcome_counts"].get(sp.MISMATCH_UNIT, 0) == 1

    def test_adjustment_mismatch_detected_from_constant_ratio(self):
        adjusted = self._row(open=9.08 * 2, high=9.12 * 2, low=9.02 * 2, close=9.04 * 2)
        report = sp.reconcile(_panel([self._row()]), _panel([adjusted]))
        assert report["results"][0]["adjustment_mismatch"] is True
        assert report["outcome_counts"].get(sp.MISMATCH_ADJUSTMENT, 0) == 4

    def test_missing_in_qmt_is_expected_not_a_defect(self):
        report = sp.reconcile(_panel([self._row()]), _panel([]).reindex(
            columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        ))
        assert report["outcome_counts"].get(sp.MISSING_IN_QMT, 0) >= 1

    def test_duplicate_rows_reported(self):
        report = sp.reconcile(_panel([self._row(), self._row()]), _panel([self._row()]))
        assert report["duplicate_rows"]["u0"] == 1


class TestPatchGovernance:
    def _reco(self, outcome, field_name="close", u0=9.04, qmt=9.10):
        return {"results": [{
            "symbol": "600000.SH", "trade_date": "2026-07-24",
            "comparisons": [{"field_name": field_name, "u0_value": u0,
                             "qmt_value": qmt, "outcome": outcome, "detail": ""}],
        }]}

    def test_value_disagreement_does_not_overturn_u0(self):
        patches = sp.propose_patches(self._reco(sp.MISMATCH_VALUE))
        assert patches[0].decision == sp.REJECTED_U0_AUTHORITATIVE
        assert patches[0].applied is False

    def test_gap_is_filled_from_qmt(self):
        patches = sp.propose_patches(self._reco(sp.MISSING_IN_U0, u0=None))
        assert patches[0].decision == sp.APPROVED
        assert patches[0].applied is True

    def test_unit_mismatch_is_never_patched(self):
        patches = sp.propose_patches(self._reco(sp.MISMATCH_UNIT, field_name="volume"))
        assert patches[0].decision == sp.REJECTED_UNIT_MISMATCH

    def test_adjustment_mismatch_is_never_patched(self):
        patches = sp.propose_patches(self._reco(sp.MISMATCH_ADJUSTMENT))
        assert patches[0].decision == sp.REJECTED_U0_AUTHORITATIVE

    def test_every_patch_records_full_provenance(self):
        """All provenance keys present; old_value is legitimately null on a gap fill."""
        patch = sp.propose_patches(self._reco(sp.MISSING_IN_U0, u0=None))[0]
        record = patch.to_dict()
        for key in ("old_provider", "new_provider", "old_value", "new_value",
                    "reason", "validation", "old_hash", "new_hash", "decision"):
            assert key in record, f"{key} missing from the patch record"
        # A gap fill displaced nothing, so old_value must be null -- recording a
        # value there would misrepresent the panel as having held one.
        assert record["old_value"] is None
        assert record["new_value"] is not None
        for key in ("old_provider", "new_provider", "reason", "validation",
                    "old_hash", "new_hash", "decision"):
            assert record[key], f"{key} must be populated on every patch"

    def test_overwrite_candidate_records_both_values(self):
        patch = sp.propose_patches(self._reco(sp.MISMATCH_VALUE, u0=9.04, qmt=9.10))[0]
        record = patch.to_dict()
        assert record["old_value"] == 9.04
        assert record["new_value"] == 9.10
        assert record["old_hash"] != record["new_hash"]

    def test_rejected_patches_stay_in_the_ledger(self):
        """An audit that only lists applied changes hides what was declined."""
        patches = sp.propose_patches(self._reco(sp.MISMATCH_VALUE))
        panel = _panel([{"symbol": "600000.SH", "trade_date": "2026-07-24", "close": 9.04}])
        result, ledger = sp.apply_patches(panel, patches)
        assert ledger["patches_considered"] == 1
        assert ledger["patches_applied"] == 0
        assert ledger["patches_rejected"] == 1
        assert len(ledger["records"]) == 1
        assert result.loc[0, "close"] == 9.04  # untouched

    def test_approved_patch_is_applied_with_provenance_stamp(self):
        patches = sp.propose_patches(self._reco(sp.MISSING_IN_U0, u0=None, qmt=9.10))
        panel = _panel([{"symbol": "600000.SH", "trade_date": "2026-07-24", "close": None}])
        result, ledger = sp.apply_patches(panel, patches)
        assert ledger["patches_applied"] == 1
        assert result.loc[0, "close"] == 9.10
        assert "u0_verified->qmt_xtdata" in str(result.loc[0, "patch_provenance"])

    def test_u0_is_first_in_precedence(self):
        assert sp.PRECEDENCE[0] == sp.PROVIDER_U0
