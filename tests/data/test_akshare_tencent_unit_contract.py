"""Tencent daily bars carry volume in a column named ``amount``.

Measured against Sina (volume in shares, amount in CNY) on 2026-08-14 with
akshare 1.18.60, for 600519 / 000001 / 601398 and ``adjust`` in {"", "qfq"}:

    median(tencent.amount / (sina.volume / 100)) == 1.000000   (exactly)
    tencent.amount / (sina.volume * close)      == 8e-6 .. 1.4e-3

A genuine CNY turnover cannot vary with price level like that. So Tencent's
``amount`` is volume in 100-share lots, it publishes no CNY turnover, and it
publishes no ``volume`` column at all.

Before the fix the normaliser passed that number through as ``amount`` and
stamped ``amount_unit="CNY"``, ``volume_unit="shares"``, ``quality_status="OK"``
-- a turnover figure 1.2e5x too small, self-certified as good, on a source that
is the *failover* and therefore engages precisely when EastMoney is rate-limited.

These tests use synthetic frames shaped exactly like the live payloads so they
run offline; the live shape itself is pinned by the first test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.providers.akshare_live_provider import (
    _CANONICAL_AMOUNT_UNIT,
    _CANONICAL_VOLUME_UNIT,
    _RAW_VOLUME_UNIT_BY_SOURCE,
    _UNIT_UNAVAILABLE,
    _normalize_akshare_daily,
)

#: Exact live Tencent payload for 600519, 2026-07-01..2026-07-08.
#: ``amount`` here is what the endpoint returns; Sina reports 4_247_381 shares
#: for the first row, i.e. 42_473.81 lots.
TENCENT_RAW = pd.DataFrame(
    {
        "date": ["2026-07-01", "2026-07-02", "2026-07-03"],
        "open": [1180.10, 1193.01, 1205.24],
        "close": [1193.01, 1203.00, 1194.45],
        "high": [1196.80, 1215.52, 1210.14],
        "low": [1166.33, 1190.51, 1185.00],
        "amount": [42474.0, 50870.0, 34268.0],
    }
)

#: Sina ground truth for the same rows: volume in shares, amount in CNY.
SINA_SHARES = [4_247_381.0, 5_087_015.0, 3_426_755.0]


def _tencent() -> pd.DataFrame:
    return _normalize_akshare_daily(
        TENCENT_RAW.copy(), "600519.SH", source="tencent", adjust=""
    )


class TestTencentVolumeRecovery:
    def test_the_column_named_amount_becomes_volume_in_shares(self):
        out = _tencent()
        assert "volume" in out.columns
        # lots -> shares, within Tencent's own lot-level rounding
        for got, truth in zip(out["volume"], SINA_SHARES):
            assert got == pytest.approx(truth, rel=2e-5)

    def test_volume_is_not_left_at_lot_scale(self):
        """The 100x that used to be missing entirely."""
        out = _tencent()
        assert out["volume"].iloc[0] == pytest.approx(4_247_400.0)
        assert out["volume"].iloc[0] != pytest.approx(42_474.0)

    def test_raw_unit_provenance_is_detected_per_payload_not_looked_up(self):
        """The unit must come from the shape that arrived, not a static table.

        Tencent has shipped two shapes, so a fixed table entry is guaranteed to
        be wrong for one of them. The lots shape must report lots even though
        the table's default (the legacy shape) says shares.
        """
        assert _RAW_VOLUME_UNIT_BY_SOURCE["tencent"] == "shares"  # legacy default
        out = _tencent()
        assert out["raw_volume_unit"].iloc[0] == "lots_100_shares"
        assert out["volume_unit"].iloc[0] == _CANONICAL_VOLUME_UNIT

    def test_legacy_shape_with_real_volume_and_turnover_is_left_alone(self):
        """Regression on my own over-reach: do not null a genuine CNY amount."""
        legacy = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [10.0],
                "close": [10.5],
                "high": [11.0],
                "low": [9.5],
                "volume": [123_400.0],  # already shares
                "amount": [1_295_000.0],  # already CNY
            }
        )
        out = _normalize_akshare_daily(legacy, "600000.SH", source="tencent", adjust="")
        assert out["volume"].iloc[0] == pytest.approx(123_400.0)
        assert out["amount"].iloc[0] == pytest.approx(1_295_000.0)
        assert out["raw_volume_unit"].iloc[0] == "shares"
        assert out["amount_unit"].iloc[0] == _CANONICAL_AMOUNT_UNIT


class TestTencentAmountIsNotFabricated:
    def test_amount_is_nan_because_tencent_publishes_no_turnover(self):
        out = _tencent()
        assert out["amount"].isna().all()

    def test_amount_unit_is_unavailable_not_cny(self):
        """The label is what downstream trusts, so it must not overstate."""
        out = _tencent()
        assert out["amount_unit"].iloc[0] == _UNIT_UNAVAILABLE
        assert out["amount_unit"].iloc[0] != _CANONICAL_AMOUNT_UNIT

    def test_the_lot_count_never_survives_as_a_turnover_figure(self):
        """Regression on the exact defect: 42474 must never appear as amount.

        Real turnover that session was ~5.07e9 CNY. Passing 42474 through would
        under-report liquidity by ~1.2e5x, which silently inverts any ADV,
        capacity or liquidity-decile filter that reads it.
        """
        out = _tencent()
        assert not (out["amount"] == 42474.0).any()


class TestOtherSourcesUnchanged:
    def test_east_money_still_scales_lots_to_shares(self):
        raw = pd.DataFrame(
            {
                "日期": ["2026-07-01"],
                "开盘": [10.0],
                "最高": [10.5],
                "最低": [9.8],
                "收盘": [10.2],
                "成交量": [1000.0],  # lots
                "成交额": [1_020_000.0],  # CNY
            }
        )
        out = _normalize_akshare_daily(raw, "000001.SZ", source="east_money", adjust="")
        assert out["volume"].iloc[0] == pytest.approx(100_000.0)
        assert out["amount"].iloc[0] == pytest.approx(1_020_000.0)
        assert out["amount_unit"].iloc[0] == _CANONICAL_AMOUNT_UNIT

    def test_sina_volume_is_already_shares_and_is_not_rescaled(self):
        raw = pd.DataFrame(
            {
                "date": ["2026-07-01"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [100_000.0],  # already shares
                "amount": [1_020_000.0],
            }
        )
        out = _normalize_akshare_daily(raw, "000001.SZ", source="sina", adjust="")
        assert out["volume"].iloc[0] == pytest.approx(100_000.0)
        assert out["amount_unit"].iloc[0] == _CANONICAL_AMOUNT_UNIT


class TestUnitLabelsRequireTheColumn:
    def test_absent_volume_is_labelled_unavailable_not_shares(self):
        raw = pd.DataFrame(
            {
                "date": ["2026-07-01"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "amount": [1_020_000.0],
            }
        )
        out = _normalize_akshare_daily(raw, "000001.SZ", source="sina", adjust="")
        assert out["volume_unit"].iloc[0] == _UNIT_UNAVAILABLE

    def test_all_nan_column_does_not_earn_a_canonical_unit(self):
        """An all-NaN column is a missing measurement, not a measured zero."""
        raw = pd.DataFrame(
            {
                "date": ["2026-07-01"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [float("nan")],
                "amount": [float("nan")],
            }
        )
        out = _normalize_akshare_daily(raw, "000001.SZ", source="sina", adjust="")
        assert out["volume_unit"].iloc[0] == _UNIT_UNAVAILABLE
        assert out["amount_unit"].iloc[0] == _UNIT_UNAVAILABLE


class TestSmokeProbeToleratesMissingColumns:
    def test_nonnegative_column_distinguishes_absent_from_negative(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "run_akshare_source_smoke.py"
        spec = importlib.util.spec_from_file_location("_ak_smoke", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        frame = pd.DataFrame({"amount": [1.0, 2.0], "other": [-1.0, 3.0]})
        # absent -> None, so "not published" never reads as "published negative"
        assert module._nonnegative_column(frame, "volume") is None
        assert module._nonnegative_column(frame, "amount") is True
        assert module._nonnegative_column(frame, "other") is False


class TestEmptyVolumeColumnCannotDefeatDetection:
    """A present-but-empty `volume` header is not evidence of the legacy shape.

    Adversarial review found the original fix was structural only: adding an
    all-NaN `volume` column to the lots-shaped payload routed it down the legacy
    branch, passing the lot count through as CNY turnover and stamping
    amount_unit="CNY" -- reproducing the defect the fix existed to remove.
    """

    @staticmethod
    def _lots_payload_with_empty_volume(filler):
        frame = TENCENT_RAW.copy()
        frame["volume"] = filler
        return frame

    @pytest.mark.parametrize("filler", [float("nan"), None])
    def test_all_nan_volume_still_detects_the_lots_shape(self, filler):
        frame = self._lots_payload_with_empty_volume(filler)
        out = _normalize_akshare_daily(frame, "600519.SH", source="tencent", adjust="")
        assert out["amount"].isna().all(), "lot count must not survive as turnover"
        assert out["amount_unit"].iloc[0] == _UNIT_UNAVAILABLE
        assert out["volume"].iloc[0] == pytest.approx(4_247_400.0)

    def test_the_lot_count_never_appears_as_amount_even_with_a_volume_header(self):
        frame = self._lots_payload_with_empty_volume(float("nan"))
        out = _normalize_akshare_daily(frame, "600519.SH", source="tencent", adjust="")
        assert not (out["amount"] == 42474.0).any()

    def test_a_usable_volume_column_still_selects_the_legacy_shape(self):
        """Guard the fix against over-reach: real volume must be believed."""
        frame = TENCENT_RAW.copy()
        frame["volume"] = [4_247_381.0, 5_087_015.0, 3_426_755.0]
        frame["amount"] = [5.07e9, 6.12e9, 4.09e9]
        out = _normalize_akshare_daily(frame, "600519.SH", source="tencent", adjust="")
        assert out["amount"].iloc[0] == pytest.approx(5.07e9)
        assert out["amount_unit"].iloc[0] == _CANONICAL_AMOUNT_UNIT
        assert out["volume"].iloc[0] == pytest.approx(4_247_381.0)


class TestDefaultSourceOrderPrefersCompleteData:
    """Sina must be tried before Tencent, because Tencent has no turnover.

    Measured 2026-08-15 (akshare 1.18.60, 5 symbols x 44 sessions, through the
    provider): sina returns 220/220 volume AND 220/220 amount with
    amount/(volume*close) median 0.9986; tencent returns 220/220 volume and
    0/220 amount, because the source publishes no CNY turnover at all.

    With tencent ahead of sina, any EastMoney outage silently cost the pipeline
    turnover entirely -- which disables every ADV, liquidity and capacity screen
    downstream. EastMoney was in fact unreachable throughout that session.
    """

    def test_sina_is_attempted_before_tencent(self):
        from quantagent.data.providers.akshare_live_provider import _DEFAULT_SOURCE_ORDER

        assert "sina" in _DEFAULT_SOURCE_ORDER, "the only default source with turnover"
        assert _DEFAULT_SOURCE_ORDER.index("sina") < _DEFAULT_SOURCE_ORDER.index("tencent")

    def test_east_money_remains_the_documented_primary(self):
        from quantagent.data.providers.akshare_live_provider import _DEFAULT_SOURCE_ORDER

        assert _DEFAULT_SOURCE_ORDER[0] == "east_money"

    def test_every_default_source_is_routable(self):
        from quantagent.data.providers.akshare_live_provider import (
            _DEFAULT_SOURCE_ORDER,
            _SOURCE_FUNCTIONS,
        )

        assert set(_DEFAULT_SOURCE_ORDER) <= set(_SOURCE_FUNCTIONS)
