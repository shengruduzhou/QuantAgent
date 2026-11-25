"""ROADMAP P1-2: every label artifact declares its clock; consumers refuse guesses and mixes.

delay-0 (``close(t+h)/close(t)``) inflates short-horizon IC by up to 53% against
the delay-1 labels the certified data uses. Before this, ``build-labels-v7``
silently produced delay-0 and the dataset builder merged whatever it was given.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.data import label_contract as lc
from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
from quantagent.data.v7_label_builder import build_forward_return_labels


def _market_panel(days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    rows: list[dict] = []
    rng = np.random.default_rng(5)
    for sidx, symbol in enumerate(("600519.SH", "000858.SZ", "300750.SZ")):
        close = 10.0 + sidx
        for date in dates:
            close *= 1.0 + rng.normal(0.0, 0.02)
            rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + sidx * 100_000,
                    "amount": close * 1_000_000,
                    "available_at": date.strftime("%Y-%m-%d"),
                    "is_suspended": False,
                    "is_st": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                }
            )
    return pd.DataFrame(rows)


def _calendar(tmp_path: Path, days: int = 40) -> Path:
    path = tmp_path / "calendar.parquet"
    pd.DataFrame(
        {"trade_date": pd.date_range("2026-01-02", periods=days, freq="B"), "is_open": True}
    ).to_parquet(path, index=False)
    return path


def _build_labels(tmp_path: Path, choice: str, name: str = "labels.parquet") -> Path:
    market = tmp_path / "market.parquet"
    if not market.exists():
        _market_panel().to_parquet(market, index=False)
    out = tmp_path / name
    extra = ["--market-calendar", str(_calendar(tmp_path))] if choice == "delay1" else []
    result = CliRunner().invoke(
        app,
        ["build-labels-v7", "--market-panel", str(market), "--output", str(out),
         "--horizons", "1,5", "--label-convention", choice, *extra],
    )
    assert result.exit_code == 0, result.output
    return out


def _config(tmp_path: Path, labels: Path, **overrides) -> V7TrainingDatasetConfig:
    base = dict(
        market_panel_path=str(tmp_path / "market.parquet"),
        labels_path=str(labels),
        output_path=str(tmp_path / f"dataset_{labels.stem}.parquet"),
        horizons=(1, 5),
        min_rows=10,
        min_symbols=2,
        min_dates=5,
        factor_library="basic",
    )
    base.update(overrides)
    return V7TrainingDatasetConfig(**base)


# ---------------------------------------------------------------------------
# producers
# ---------------------------------------------------------------------------


def test_build_labels_v7_requires_an_explicit_convention(tmp_path: Path) -> None:
    """Fails on old code, which silently defaulted to delay-0."""
    market = tmp_path / "market.parquet"
    _market_panel().to_parquet(market, index=False)
    result = CliRunner().invoke(
        app, ["build-labels-v7", "--market-panel", str(market), "--output",
              str(tmp_path / "labels.parquet"), "--horizons", "1"],
    )
    assert result.exit_code != 0
    assert "--label-convention" in result.output
    assert not (tmp_path / "labels.parquet").exists()

    bad = CliRunner().invoke(
        app, ["build-labels-v7", "--market-panel", str(market), "--output",
              str(tmp_path / "labels.parquet"), "--horizons", "1", "--label-convention", "t+2"],
    )
    assert bad.exit_code != 0


def test_delay1_and_delay0_are_stamped_and_computed_on_their_own_clock(tmp_path: Path) -> None:
    delay1 = _build_labels(tmp_path, "delay1", "d1.parquet")
    delay0 = _build_labels(tmp_path, "delay0", "d0.parquet")
    assert lc.read_label_convention(delay1) == (lc.EXEC_DELAY1_SESSION, "metadata")
    assert lc.read_label_convention(delay0) == (lc.V7_SAME_CLOSE, "metadata")

    market = _market_panel()
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    one = market[market["symbol"] == "600519.SH"].sort_values("trade_date").reset_index(drop=True)
    close = one["close"]
    expected1 = close.shift(-6) / close.shift(-1) - 1.0  # close(t+1+5)/close(t+1)
    expected0 = close.shift(-5) / close - 1.0
    got1 = pd.read_parquet(delay1)
    got0 = pd.read_parquet(delay0)
    for got, expected in ((got1, expected1), (got0, expected0)):
        sym = got[got["symbol"] == "600519.SH"].sort_values("trade_date")
        merged = one[["trade_date"]].assign(expected=expected).merge(
            sym[["trade_date", "forward_return_5d"]], on="trade_date", how="inner"
        ).dropna()
        assert len(merged) > 20
        np.testing.assert_allclose(merged["forward_return_5d"], merged["expected"], rtol=1e-12)


def test_delay0_library_output_is_unchanged_by_the_new_option() -> None:
    """The library default stays delay-0, bit for bit, so no existing number moves."""
    market = _market_panel()
    default = build_forward_return_labels(market, horizons=(1, 5)).frame
    explicit = build_forward_return_labels(market, horizons=(1, 5), entry_delay_sessions=0).frame
    pd.testing.assert_frame_equal(default, explicit)
    assert default.attrs[lc.LABEL_CONVENTION_ATTR] == lc.V7_SAME_CLOSE
    with pytest.raises(ValueError, match="entry_delay_sessions"):
        build_forward_return_labels(market, horizons=(1,), entry_delay_sessions=2)


def test_csv_labels_get_a_sidecar_stamp_bound_to_their_content(tmp_path: Path) -> None:
    out = _build_labels(tmp_path, "delay1", "labels.csv")
    assert lc.read_label_convention(out) == (lc.EXEC_DELAY1_SESSION, "sidecar")
    with out.open("a", encoding="utf-8") as handle:  # edited after stamping
        handle.write("600519.SH,2099-01-01,0.5\n")
    assert lc.read_label_convention(out) == (None, "unknown")
    with pytest.raises(lc.LabelConventionError):
        lc.require_label_convention(out)


# ---------------------------------------------------------------------------
# consumers
# ---------------------------------------------------------------------------


def test_dataset_builder_refuses_unstamped_labels(tmp_path: Path) -> None:
    """Fails on old code, which merged any labels file it was given."""
    _market_panel().to_parquet(tmp_path / "market.parquet", index=False)
    unstamped = tmp_path / "unstamped.parquet"
    build_forward_return_labels(_market_panel(), horizons=(1, 5)).frame.to_parquet(unstamped, index=False)

    with pytest.raises(lc.LabelConventionError) as excinfo:
        build_v7_training_dataset_artifact(_config(tmp_path, unstamped))
    message = str(excinfo.value)
    assert "no label-convention stamp" in message
    assert "--label-convention delay1" in message and "--allow-legacy-unstamped-labels" in message
    assert not (tmp_path / "dataset_unstamped.parquet").exists()

    legacy = build_v7_training_dataset_artifact(
        _config(tmp_path, unstamped, allow_legacy_unstamped_labels=True)
    )
    assert legacy.feature_schema["label_convention"] == lc.V7_SAME_CLOSE  # inferred from columns
    assert legacy.feature_schema["label_convention_basis"] == "legacy_unstamped_inferred"
    manifest = json.loads(Path(legacy.manifest_path).read_text())
    assert manifest["extra"]["label_convention_basis"] == "legacy_unstamped_inferred"


def test_dataset_builder_records_a_stamped_convention(tmp_path: Path) -> None:
    labels = _build_labels(tmp_path, "delay1")
    result = build_v7_training_dataset_artifact(_config(tmp_path, labels))
    assert result.feature_schema["label_convention"] == lc.EXEC_DELAY1_SESSION
    assert result.feature_schema["label_convention_basis"] == "metadata"
    assert result.summary["label_convention"] == lc.EXEC_DELAY1_SESSION
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["extra"]["label_convention_id"] == lc.EXEC_DELAY1_SESSION


def test_dataset_builder_refuses_mixing_with_a_pinned_schema_of_another_clock(tmp_path: Path) -> None:
    """A walk-forward rebuild pinned to a delay-0 schema must not silently switch to delay-1."""
    delay0 = _build_labels(tmp_path, "delay0", "d0.parquet")
    delay1 = _build_labels(tmp_path, "delay1", "d1.parquet")
    first = build_v7_training_dataset_artifact(_config(tmp_path, delay0))
    with pytest.raises(lc.LabelConventionError, match="refuses to mix label conventions"):
        build_v7_training_dataset_artifact(
            _config(tmp_path, delay1, expected_feature_schema_path=str(first.feature_schema_path))
        )
    # Same clock re-pins fine.
    again = build_v7_training_dataset_artifact(
        _config(tmp_path, delay0, output_path=str(tmp_path / "again.parquet"),
                expected_feature_schema_path=str(first.feature_schema_path))
    )
    assert again.feature_schema["label_convention"] == lc.V7_SAME_CLOSE


def test_dataset_builder_refuses_labels_riding_in_on_the_market_panel(tmp_path: Path) -> None:
    labels = _build_labels(tmp_path, "delay1")
    market = _market_panel()
    delay0 = build_forward_return_labels(market, horizons=(1, 5)).frame
    delay0["trade_date"] = pd.to_datetime(delay0["trade_date"]).dt.strftime("%Y-%m-%d")
    polluted = market.merge(
        delay0[["symbol", "trade_date", "forward_return_5d"]], on=["symbol", "trade_date"], how="left"
    )
    polluted.to_parquet(tmp_path / "market.parquet", index=False)
    with pytest.raises(lc.LabelConventionError, match="refuses to mix label sources"):
        build_v7_training_dataset_artifact(_config(tmp_path, labels))


def test_build_training_dataset_cli_refuses_unstamped_labels_and_names_the_flag(tmp_path: Path) -> None:
    _market_panel().to_parquet(tmp_path / "market.parquet", index=False)
    unstamped = tmp_path / "unstamped.parquet"
    build_forward_return_labels(_market_panel(), horizons=(1, 5)).frame.to_parquet(unstamped, index=False)
    args = [
        "build-training-dataset-v7", "--market-panel", str(tmp_path / "market.parquet"),
        "--labels", str(unstamped), "--output", str(tmp_path / "ds.parquet"), "--horizons", "1,5",
        "--min-rows", "10", "--min-dates", "5", "--factor-library", "basic",
    ]
    refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0
    assert isinstance(refused.exception, lc.LabelConventionError)
    accepted = CliRunner().invoke(app, [*args, "--allow-legacy-unstamped-labels"])
    assert accepted.exit_code == 0, accepted.output


def test_assert_consistent_conventions() -> None:
    assert lc.assert_consistent_conventions({"a": lc.GOLD_DELAY1, "b": None}) == lc.GOLD_DELAY1
    assert lc.assert_consistent_conventions({"a": None}) is None
    with pytest.raises(lc.LabelConventionError):
        lc.assert_consistent_conventions({"a": lc.GOLD_DELAY1, "b": lc.V7_SAME_CLOSE})
    with pytest.raises(lc.LabelConventionError):
        lc.assert_consistent_conventions({"a": lc.GOLD_DELAY1, "b": lc.EXEC_DELAY1_SESSION})


def test_unknown_declared_convention_is_refused(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(pd.DataFrame({"symbol": ["a"], "forward_return_1d": [0.1]}))
    meta = dict(table.schema.metadata or {})
    meta[lc.LABEL_CONVENTION_METADATA_KEY.encode()] = json.dumps({"id": "made_up_v9"}).encode()
    path = tmp_path / "labels.parquet"
    pq.write_table(table.replace_schema_metadata(meta), path)
    with pytest.raises(lc.LabelConventionError, match="unknown label convention"):
        lc.require_label_convention(path)


def test_synthesize_factors_refuses_two_label_sources_and_unstamped_labels(tmp_path: Path) -> None:
    from tests.test_factor_synthesis import _make_panel_with_momentum_signal

    panel = _make_panel_with_momentum_signal(n_symbols=6, n_days=120)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    labels_path = tmp_path / "labels.parquet"
    lc.write_labels_parquet(
        panel[["symbol", "trade_date", "forward_return_5d"]], labels_path, convention=lc.GOLD_DELAY1
    )
    base = ["synthesize-factors-v7", "--market-panel", str(panel_path), "--output-dir",
            str(tmp_path / "out"), "--rounds", "1"]

    both = CliRunner().invoke(app, [*base, "--labels", str(labels_path)])
    assert both.exit_code != 0
    assert isinstance(both.exception, lc.LabelConventionError)
    assert "both" in str(both.exception)

    unstamped_panel = CliRunner().invoke(app, base)  # label rides on an unstamped panel
    assert unstamped_panel.exit_code != 0
    assert isinstance(unstamped_panel.exception, lc.LabelConventionError)


def test_delay1_requires_a_session_calendar(tmp_path: Path) -> None:
    market = tmp_path / "market.parquet"
    _market_panel().to_parquet(market, index=False)
    result = CliRunner().invoke(
        app, ["build-labels-v7", "--market-panel", str(market), "--output",
              str(tmp_path / "labels.parquet"), "--horizons", "1", "--label-convention", "delay1"],
    )
    assert result.exit_code != 0
    assert "--market-calendar" in result.output
    with pytest.raises(ValueError, match="market_sessions"):
        build_forward_return_labels(_market_panel(), horizons=(1,), entry_delay_sessions=1)


def test_delay1_is_on_the_global_session_clock_not_the_next_row() -> None:
    """A symbol suspended for sessions 1-5 must not get an entry 6 sessions later.

    Fails on the row-shift delay-1: its next ROW after session 0 is session 6,
    so the "delay-1" label silently spanned 6+h sessions, past any h-session
    purge/embargo.
    """
    market = _market_panel()
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    suspended = market["symbol"].eq("000858.SZ") & market["trade_date"].isin(sessions[1:6])
    panel = market.loc[~suspended].reset_index(drop=True)

    built = build_forward_return_labels(
        panel, horizons=(1, 5), entry_delay_sessions=1, market_sessions=sessions
    ).frame
    built["trade_date"] = pd.to_datetime(built["trade_date"])
    sym = built[built["symbol"] == "000858.SZ"].set_index("trade_date")
    # Signal at session 0: entry session 1 is suspended => unavailable (the row
    # carries no label at any horizon, so the builder drops it).
    assert sessions[0] not in sym.index or sym.loc[sessions[0], ["forward_return_1d", "forward_return_5d"]].isna().all()
    # The row-shift clock would have produced a value here: next row = session 6.
    row_shift = build_forward_return_labels(panel, horizons=(1,)).frame  # delay-0 reference rows exist
    assert sessions[0] in set(pd.to_datetime(row_shift[row_shift["symbol"] == "000858.SZ"]["trade_date"]))
    # Every label window, available or not, is bounded by 1+h sessions.
    position = {d: i for i, d in enumerate(sessions)}
    for h in (1, 5):
        ends = built[f"label_end_{h}d"].dropna()
        spans = [position[pd.Timestamp(e)] - position[pd.Timestamp(t)]
                 for t, e in zip(built.loc[ends.index, "trade_date"], ends)]
        assert set(spans) == {1 + h}
    # Exit inside the suspension is unavailable too: signal at session 0 with h=1
    # exits at session 2 (suspended); a signal at session 6 is clean again.
    assert not np.isnan(sym.loc[sessions[6], "forward_return_1d"])
    one = market[market["symbol"] == "000858.SZ"].set_index("trade_date")["close"]
    expected = one.loc[sessions[8]] / one.loc[sessions[7]] - 1.0
    assert sym.loc[sessions[6], "forward_return_1d"] == pytest.approx(expected, rel=1e-12)
    # Unsuspended symbols are untouched by the clock check.
    other = built[built["symbol"] == "600519.SH"]["forward_return_5d"]
    assert other.notna().sum() == len(sessions) - 6


def test_delay0_is_byte_identical_with_or_without_a_calendar() -> None:
    market = _market_panel()
    sessions = pd.to_datetime(market["trade_date"]).drop_duplicates().sort_values()
    plain = build_forward_return_labels(market, horizons=(1, 5)).frame
    with_cal = build_forward_return_labels(market, horizons=(1, 5), market_sessions=sessions).frame
    pd.testing.assert_frame_equal(plain, with_cal)


def test_csv_sidecar_without_a_content_hash_is_not_a_stamp(tmp_path: Path) -> None:
    out = _build_labels(tmp_path, "delay0", "labels.csv")
    sidecar = out.with_name(out.name + lc.SIDECAR_SUFFIX)
    payload = json.loads(sidecar.read_text())
    payload.pop("labels_file_sha256")
    sidecar.write_text(json.dumps(payload))
    assert lc.read_label_convention(out)[1] != "sidecar"
    with pytest.raises(lc.LabelConventionError):
        lc.require_label_convention(out)


def test_dataset_builder_never_reads_labels_from_a_sibling_file(tmp_path: Path, monkeypatch) -> None:
    """The stamp is verified on labels.parquet; data must not come from labels.csv."""
    import quantagent.data.dataset_builder.v7_training_dataset as builder

    labels = _build_labels(tmp_path, "delay0")
    other = build_forward_return_labels(_market_panel(), horizons=(1, 5)).frame
    other["forward_return_5d"] = 9.99  # a different, unstamped label source
    other.to_csv(labels.with_suffix(".csv"), index=False)
    real_read = pd.read_parquet

    def flaky(path, *args, **kwargs):
        if Path(path) == labels:
            raise OSError("simulated unreadable parquet")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(builder.pd, "read_parquet", flaky)
    with pytest.raises(RuntimeError, match="refusing a sibling-file fallback"):
        build_v7_training_dataset_artifact(_config(tmp_path, labels))
