"""Full-universe Gold: build semantics and artifact integrity.

Split deliberately in two. The hermetic tests build tiny synthetic panels and
assert the *rules* — they run anywhere, including CI, and are the ones that stop
a regression. The artifact tests inspect the real built dataset and skip when it
is absent, because a research host has it and a clean checkout does not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "runtime/data/gold/full_universe"


def _builder():
    """Import the build script as a module (it is a script, not a package)."""
    if "gold_builder" in sys.modules:
        return sys.modules["gold_builder"]
    spec = importlib.util.spec_from_file_location(
        "gold_builder", REPO / "scripts/build_u0_full_universe_gold.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gold_builder"] = module
    spec.loader.exec_module(module)
    return module


def _panel(symbols=("AAA.SZ",), days=90, start="2024-01-02"):
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for symbol in symbols:
        for i, day in enumerate(dates):
            price = 10.0 + i * 0.05
            rows.append({
                "symbol": symbol, "trade_date": day,
                "open": price, "high": price * 1.02, "low": price * 0.98,
                "close": price, "volume": 100_000.0, "amount": price * 100_000.0,
            })
    return pd.DataFrame(rows)


def _master(symbols=("AAA.SZ",), listing="2020-01-02", delisting=None):
    return pd.DataFrame([
        {"symbol": s, "board": "SZ_Main", "listing_date": pd.Timestamp(listing),
         "delisting_date": pd.Timestamp(delisting) if delisting else pd.NaT,
         "status": "delisted" if delisting else "listed"}
        for s in symbols
    ])


class TestPreregisteredRules:
    def test_ipo_seasoning_is_sixty_trading_days(self):
        """A preregistered rule: changing it changes the universe."""
        assert _builder().IPO_SEASONING_TRADING_DAYS == 60

    def test_embargo_is_at_least_the_longest_horizon(self):
        module = _builder()
        assert module.EMBARGO_DAYS >= max(module.HORIZONS)

    def test_horizons_are_the_frozen_set(self):
        assert _builder().HORIZONS == (1, 5, 20)


class TestFeatures:
    def test_features_use_no_future_information(self):
        """Every feature at t must be unchanged by appending data after t."""
        module = _builder()
        panel = _panel(days=120)
        full = module.build_features(panel)
        truncated = module.build_features(panel.iloc[:100].copy())

        for column in module.FEATURE_COLUMNS:
            if column not in full.columns:
                continue
            a = full[column].iloc[:100].to_numpy(dtype=float)
            b = truncated[column].to_numpy(dtype=float)
            assert np.allclose(a, b, equal_nan=True), (
                f"{column} changed when future rows were appended -- it looks ahead"
            )

    def test_feature_set_is_declared(self):
        module = _builder()
        frame = module.build_features(_panel(days=120))
        for column in module.FEATURE_COLUMNS:
            assert column in frame.columns

    def test_amihud_matches_the_direct_definition(self):
        """The transform optimisation must equal the original per-group formula."""
        module = _builder()
        panel = _panel(days=60)
        frame = module.build_features(panel)
        direct = (
            panel.assign(r=panel.groupby("symbol")["close"].pct_change())
            .assign(illiq=lambda d: d["r"].abs() / d["amount"].replace(0, np.nan))
            .groupby("symbol")["illiq"]
            .transform(lambda s: s.rolling(20, min_periods=20).mean())
        )
        assert np.allclose(
            frame["amihud_20d"].to_numpy(dtype=float),
            direct.to_numpy(dtype=float), equal_nan=True)


class TestFolds:
    def test_embargo_gap_separates_train_and_test(self):
        module = _builder()
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=800))
        folds = module.build_folds(dates, n_folds=5, embargo=20)
        assert folds
        for fold in folds:
            assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["test_start"])
            assert fold["embargo_days"] == 20

    def test_folds_are_expanding_not_rolling(self):
        module = _builder()
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=800))
        folds = module.build_folds(dates, n_folds=4, embargo=20)
        starts = {f["train_start"] for f in folds}
        assert len(starts) == 1, "expanding folds all start at the same date"
        ends = [f["train_end"] for f in folds]
        assert ends == sorted(ends)

    def test_test_windows_do_not_overlap_training(self):
        module = _builder()
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=800))
        for fold in module.build_folds(dates, n_folds=4, embargo=20):
            assert pd.Timestamp(fold["embargo_start"]) > pd.Timestamp(fold["train_end"])
            assert pd.Timestamp(fold["test_start"]) > pd.Timestamp(fold["embargo_end"])


class TestQualityChecks:
    def _dataset(self, **overrides):
        frame = _panel(days=30).copy()
        frame["adjustment_method"] = "hfq"
        frame["forward_return_1d"] = 0.01
        frame["mask_is_st"] = "UNKNOWN"
        frame["entry_feasible"] = True
        for key, value in overrides.items():
            frame[key] = value
        return frame

    def test_clean_dataset_passes(self):
        """A full-universe PASS must actually prove dead names are included.

        The old fixture contained only a live security and therefore correctly
        became survivorship=UNKNOWN after the quality gate was hardened. Build a
        tiny but semantically complete universe here: one live name plus one
        name that later delists, with no rows after its delisting date.
        """
        live = self._dataset()
        dead = self._dataset()
        dead["symbol"] = "DEAD.SZ"
        frame = pd.concat([live, dead], ignore_index=True)
        frame["mask_post_delisting"] = "FALSE"
        master = pd.DataFrame(
            [
                {
                    "symbol": "AAA.SZ",
                    "board": "SZ_Main",
                    "listing_date": pd.Timestamp("2020-01-02"),
                    "delisting_date": pd.NaT,
                    "status": "listed",
                },
                {
                    "symbol": "DEAD.SZ",
                    "board": "SZ_Main",
                    "listing_date": pd.Timestamp("2020-01-02"),
                    "delisting_date": pd.Timestamp("2025-12-31"),
                    "status": "delisted",
                },
            ]
        )

        report = _builder().run_quality_checks(frame, master)

        assert report["structurally_valid"], {
            "failed": report["failed_checks"],
            "unknown": report["unknown_checks"],
        }
        assert report["unknown_checks"] == []

    def test_duplicate_security_dates_detected(self):
        frame = pd.concat([self._dataset(), self._dataset()], ignore_index=True)
        report = _builder().run_quality_checks(frame, _master())
        assert "no_duplicate_security_dates" in report["failed_checks"]
        assert report["duplicate_security_dates"] > 0

    def test_pre_listing_rows_detected(self):
        report = _builder().run_quality_checks(
            self._dataset(), _master(listing="2030-01-01"))
        assert "no_pre_listing_rows" in report["failed_checks"]

    def test_post_delisting_rows_detected(self):
        report = _builder().run_quality_checks(
            self._dataset(), _master(delisting="2020-01-01"))
        assert "no_post_delisting_rows" in report["failed_checks"]

    def test_mixed_adjustment_modes_detected(self):
        frame = self._dataset()
        frame.loc[frame.index[:5], "adjustment_method"] = "none"
        report = _builder().run_quality_checks(frame, _master())
        assert "adjustment_mode_declared" in report["failed_checks"]

    def test_negative_volume_detected(self):
        report = _builder().run_quality_checks(self._dataset(volume=-1.0), _master())
        assert "volume_non_negative" in report["failed_checks"]

    def test_non_positive_close_detected(self):
        report = _builder().run_quality_checks(self._dataset(close=0.0), _master())
        assert "close_positive" in report["failed_checks"]

    def test_infeasible_entry_rows_detected(self):
        report = _builder().run_quality_checks(
            self._dataset(entry_feasible=False), _master())
        assert "no_infeasible_entries" in report["failed_checks"]

    def test_counts_are_exposed_at_the_report_root(self):
        """ReadinessEvaluator reads these keys at the root, not nested."""
        report = _builder().run_quality_checks(self._dataset(), _master())
        assert "duplicate_security_dates" in report
        assert "out_of_life_rows" in report


@pytest.mark.skipif(not (GOLD / "manifest.json").exists(),
                    reason="full-universe Gold not built on this host")
class TestBuiltArtifacts:
    """Integrity of the real dataset. Skipped where it has not been built."""

    def _manifest(self):
        return json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))

    def _certificate(self):
        return json.loads((GOLD / "quality_certificate.json").read_text(encoding="utf-8"))

    def test_all_ten_artifacts_exist(self):
        for name in ("manifest.json", "dataset.parquet", "adjusted_market_panel.parquet",
                     "eligibility.parquet", "labels.parquet", "feature_coverage.parquet",
                     "missingness_masks.parquet", "folds.json", "lineage.json",
                     "quality_certificate.json"):
            assert (GOLD / name).exists(), f"missing artifact {name}"

    def test_hashes_agree_across_artifacts(self):
        manifest, certificate = self._manifest(), self._certificate()
        lineage = json.loads((GOLD / "lineage.json").read_text(encoding="utf-8"))
        assert manifest["dataset_hash"] == certificate["dataset_hash"]
        assert manifest["content_hash"] == manifest["dataset_hash"]
        assert lineage["hashes"]["dataset"] == manifest["dataset_hash"]

    def test_row_and_symbol_counts_match_the_parquet(self):
        manifest = self._manifest()
        frame = pd.read_parquet(GOLD / "dataset.parquet", columns=["symbol", "trade_date"])
        assert len(frame) == manifest["rows"]
        assert frame["symbol"].nunique() == manifest["symbols"]
        assert frame.duplicated(subset=["symbol", "trade_date"]).sum() == 0

    def test_universe_is_wider_than_the_retired_frozen_cohort(self):
        """The whole point: 5,790 securities, not the old 3,872."""
        assert self._manifest()["symbols"] > 3872

    def test_certificate_is_granted_and_records_the_st_limitation(self):
        certificate = self._certificate()
        assert certificate["granted"] is True
        assert certificate["certificate"] == "FULL_UNIVERSE_GOLD_READY"
        assert any("not point-in-time complete" in w.lower()
                   for w in certificate["warnings"])

    def test_gold_certificate_does_not_claim_research_readiness(self):
        certificate = self._certificate()
        assert "does NOT permit formal research claims" in certificate["scope_note"]

    def test_all_five_boards_present(self):
        boards = self._manifest()["boards"]
        for board in ("SH_Main", "SZ_Main", "ChiNext", "STAR", "BSE"):
            assert boards.get(board, 0) > 0, f"board {board} absent"

    def test_folds_respect_the_embargo(self):
        folds = json.loads((GOLD / "folds.json").read_text(encoding="utf-8"))
        assert folds["embargo_days"] >= max(folds["horizons"])
        for fold in folds["folds"]:
            assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["test_start"])

    def test_missingness_masks_cover_every_feature(self):
        manifest = self._manifest()
        masks = pd.read_parquet(GOLD / "missingness_masks.parquet")
        for feature in manifest["feature_columns"]:
            assert f"missing_{feature}" in masks.columns

    def test_lineage_records_a_rebuild_command(self):
        lineage = json.loads((GOLD / "lineage.json").read_text(encoding="utf-8"))
        assert "build_u0_full_universe_gold.py" in lineage["rebuild_command"]
        assert lineage["upstream_decisions"]["blocked_pit_fields"] == ["st_intervals"]


@pytest.mark.skipif(not (GOLD / "manifest.json").exists(),
                    reason="full-universe Gold not built on this host")
class TestTrainingDefaultRetired:
    def test_ui_train_template_points_at_full_universe_gold(self):
        source = (REPO / "apps/quant-ui/src/domain/jobTemplates.ts").read_text("utf-8")
        assert "runtime/data/gold/full_universe/dataset.parquet" in source

    def test_ui_train_template_no_longer_defaults_to_the_frozen_cohort(self):
        source = (REPO / "apps/quant-ui/src/domain/jobTemplates.ts").read_text("utf-8")
        train_block = source.split("train:", 1)[1].split("},", 1)[0]
        assert "training_dataset_alpha181" not in train_block
