"""Exercise CLI-to-provider wiring with explicitly synthetic Qlib fixtures."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from quantagent.cli import app
import quantagent.cli.v7_train as v7_train
from quantagent.data.providers.base import ProviderRequest, ProviderUnavailable
from quantagent.data.providers.qlib_provider import QlibProvider


@pytest.fixture
def qlib_bundle(monkeypatch, tmp_path):
    bundle = tmp_path / "synthetic_bundle"
    (bundle / "calendars").mkdir(parents=True)
    dates = pd.bdate_range("2024-01-02", periods=8)
    (bundle / "calendars" / "day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")), encoding="utf-8")
    index = pd.MultiIndex.from_product([["SH600000"], dates], names=["instrument", "datetime"])
    frame = pd.DataFrame({
        "$open": 1., "$high": 1.1, "$low": .9, "$close": 1.,
        "$volume": 1000., "$factor": .1, "$turnover_raw_cny": 100000.,
    }, index=index)
    calls = []

    def features(instruments, fields, **kwargs):
        calls.append({"instruments": instruments, "fields": fields, **kwargs})
        return frame[fields]

    monkeypatch.setitem(sys.modules, "qlib", SimpleNamespace(init=lambda **kwargs: None))
    monkeypatch.setitem(sys.modules, "qlib.data", SimpleNamespace(D=SimpleNamespace(features=features)))
    monkeypatch.setattr(v7_train, "default_v7_lake_root", lambda: tmp_path / "lake")
    return bundle, calls


def test_check_qlib_reads_verified_fields(qlib_bundle):
    bundle, calls = qlib_bundle
    result = CliRunner().invoke(app, [
        "check-qlib-v7", "--provider-uri", str(bundle), "--symbols", "600000.SH",
        "--start-date", "2024-01-02", "--end-date", "2024-01-11",
        "--raw-amount-field", "$turnover_raw_cny", "--volume-scale-to-shares", "100",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "passed"
    assert calls[0]["fields"][-1] == "$turnover_raw_cny"


def test_auto_train_bootstrap_restores_raw_values_before_training(qlib_bundle, monkeypatch):
    bundle, calls = qlib_bundle
    training = {}
    monkeypatch.setattr(v7_train, "run_full_real_training_v7", lambda **kwargs: training.update(kwargs))
    result = CliRunner().invoke(app, [
        "auto-train-v7", "--provider-uri", str(bundle), "--symbols", "600000.SH",
        "--horizons", "1", "--raw-amount-field", "turnover_raw_cny",
        "--volume-scale-to-shares", "100",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls[0]["fields"][-1] == "$turnover_raw_cny"
    market = pd.read_parquet(training["market_panel_path"])
    assert market["close"].tolist() == [10.] * 8
    assert market["volume"].tolist() == [10000.] * 8
    assert market["amount"].tolist() == [100000.] * 8
    assert training["labels_path"].exists()
    config = json.loads(result.stdout)["stages"]["market"]["config"]
    assert config["raw_amount_field"] == "turnover_raw_cny"
    assert config["volume_scale_to_shares"] == 100.


@pytest.mark.parametrize("options", [
    [], ["--raw-amount-field", "turnover_raw_cny"],
    ["--raw-amount-field", "close", "--volume-scale-to-shares", "100"],
    ["--raw-amount-field", "turnover_raw_cny", "--volume-scale-to-shares", "nan"],
    ["--raw-amount-field", "turnover_raw_cny", "--volume-scale-to-shares", "inf"],
    ["--raw-amount-field", "turnover_raw_cny", "--volume-scale-to-shares", "0"],
])
@pytest.mark.parametrize("command", ["check-qlib-v7", "build-market-panel-v7", "auto-train-v7"])
def test_cli_invalid_contract_cannot_read_or_train(qlib_bundle, monkeypatch, command, options):
    bundle, calls = qlib_bundle
    training = []
    monkeypatch.setattr(v7_train, "run_full_real_training_v7", lambda **kwargs: training.append(kwargs))
    args = [command, "--provider-uri", str(bundle), "--symbols", "600000.SH", *options]
    if command == "build-market-panel-v7":
        args += ["--start-date", "2024-01-02", "--end-date", "2024-01-11"]
    result = CliRunner().invoke(app, args)
    if command == "check-qlib-v7":
        assert json.loads(result.stdout)["status"] == "unavailable"
    else:
        assert result.exit_code != 0
    assert "raw_amount_field" in result.output or "volume_scale_to_shares" in result.output
    assert calls == []
    assert training == []


@pytest.mark.parametrize("field, scale", [
    (None, 100), (" ", 100), ("$", 100), ("$$amount", 100),
    ("$Close", 100), ("$volume", 100), ("Mean($amount, 5)", 100),
    ("raw_amount", None), ("raw_amount", True), ("raw_amount", -1),
    ("raw_amount", float("nan")), ("raw_amount", float("inf")),
    ("raw_amount", "not-a-number"),
])
def test_invalid_contract_rejected_before_qlib_initialization(monkeypatch, field, scale):
    def unexpected_init(**kwargs):
        pytest.fail("invalid contract must not initialize Qlib")

    monkeypatch.setitem(sys.modules, "qlib", SimpleNamespace(init=unexpected_init))
    provider = QlibProvider("synthetic_bundle", raw_amount_field=field, volume_scale_to_shares=scale)
    with pytest.raises(ProviderUnavailable):
        provider.daily_ohlcv(ProviderRequest("2024-01-02", "2024-01-02", symbols=("600000.SH",)))
