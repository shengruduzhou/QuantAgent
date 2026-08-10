from __future__ import annotations

from pathlib import Path

import numpy as np

from quantagent.data.providers.qlib_intraday_reader import read_instrument_minutes


def _write_bin(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([0.0, *values], dtype="<f4").tofile(path)


def _tiny_qlib_root(tmp_path: Path) -> Path:
    root = tmp_path / "qlib"
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "1min.txt").write_text(
        "2024-03-01 09:31:00\n2024-03-01 09:32:00\n",
        encoding="utf-8",
    )
    feature_dir = root / "features" / "sh600000"
    _write_bin(feature_dir / "open.1min.bin", [9.9, 10.9])
    _write_bin(feature_dir / "high.1min.bin", [10.2, 11.2])
    _write_bin(feature_dir / "low.1min.bin", [9.8, 10.8])
    _write_bin(feature_dir / "close.1min.bin", [10.0, 11.0])
    _write_bin(feature_dir / "volume.1min.bin", [100.0, 200.0])
    _write_bin(feature_dir / "factor.1min.bin", [2.0, 2.0])
    return root


def test_adjusted_qlib_prices_are_research_only_and_amount_stays_raw(tmp_path: Path) -> None:
    root = _tiny_qlib_root(tmp_path)

    frame = read_instrument_minutes(root, "600000.SH", adjust=True)

    assert frame["close"].tolist() == [20.0, 22.0]
    assert frame["amount"].tolist() == [1000.0, 2200.0]
    assert (frame["price_adjustment"] == "qfq").all()
    assert not frame["execution_eligible"].any()
    assert (frame["timezone"] == "Asia/Shanghai").all()


def test_raw_qlib_prices_are_execution_eligible(tmp_path: Path) -> None:
    root = _tiny_qlib_root(tmp_path)

    frame = read_instrument_minutes(root, "600000.SH", adjust=False)

    assert frame["close"].tolist() == [10.0, 11.0]
    assert frame["amount"].tolist() == [1000.0, 2200.0]
    assert (frame["price_adjustment"] == "raw").all()
    assert frame["execution_eligible"].all()
