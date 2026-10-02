"""R3-F11: certified labels were overwritten with a same-close file and readiness
only checked that *a* labels.parquet existed."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from quantagent.data import label_contract as lc
from quantagent.data.v7_label_builder import build_forward_return_labels
from quantagent.safety import readiness_tiers as rt


def _panel(days: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    return pd.DataFrame({"symbol": "600000.SH", "trade_date": dates,
                         "close": [10.0 + i for i in range(days)]})


def _gold_labels() -> pd.DataFrame:
    panel = _panel()
    out = panel[["symbol", "trade_date"]].copy()
    out["entry_close_t1"] = panel["close"].shift(-1)
    out["forward_return_1d"] = panel["close"].shift(-2) / panel["close"].shift(-1) - 1
    return out.dropna()


def _certified_dir(tmp_path):
    gold = tmp_path / "data/gold/full_universe"
    gold.mkdir(parents=True)
    sha = lc.write_labels_parquet(_gold_labels(), gold / "labels.parquet",
                                  convention=lc.GOLD_DELAY1)
    manifest = {"label_convention_id": lc.GOLD_DELAY1, "labels_file_sha256": sha,
                "content_hash": "x", "adjustment_method": "hfq"}
    (gold / "manifest.json").write_text(json.dumps(manifest))
    (gold / "quality_certificate.json").write_text(json.dumps(
        {"certificate": "FULL_UNIVERSE_GOLD_READY", "granted": True}))
    return gold, manifest


def test_written_labels_carry_their_convention(tmp_path):
    path = tmp_path / "labels.parquet"
    lc.write_labels_parquet(_gold_labels(), path, convention=lc.GOLD_DELAY1)
    assert lc.read_label_convention(path) == (lc.GOLD_DELAY1, "metadata")


def test_unstamped_files_are_classified_by_their_columns(tmp_path):
    gold = tmp_path / "gold.parquet"
    _gold_labels().to_parquet(gold, index=False)
    v7 = tmp_path / "v7.parquet"
    build_forward_return_labels(_panel(), (1,)).frame.to_parquet(v7, index=False)
    assert lc.read_label_convention(gold) == (lc.GOLD_DELAY1, "inferred")
    assert lc.read_label_convention(v7) == (lc.V7_SAME_CLOSE, "inferred")


def test_the_certified_file_verifies(tmp_path):
    gold, manifest = _certified_dir(tmp_path)
    ok, evidence = lc.verify_labels_against_manifest(gold / "labels.parquet", manifest)
    assert ok is True, evidence


def test_a_same_close_replacement_is_detected(tmp_path):
    """The R3-F11 shape: same path, v7 same-close labels, manifest unchanged."""
    gold, manifest = _certified_dir(tmp_path)
    build_forward_return_labels(_panel(), (1,)).frame.to_parquet(
        gold / "labels.parquet", index=False)
    ok, evidence = lc.verify_labels_against_manifest(gold / "labels.parquet", manifest)
    assert ok is False
    assert evidence["file_convention"] == lc.V7_SAME_CLOSE


def test_same_convention_but_different_content_is_detected(tmp_path):
    gold, manifest = _certified_dir(tmp_path)
    tampered = _gold_labels()
    tampered["forward_return_1d"] *= 2
    lc.write_labels_parquet(tampered, gold / "labels.parquet", convention=lc.GOLD_DELAY1)
    ok, evidence = lc.verify_labels_against_manifest(gold / "labels.parquet", manifest)
    assert ok is False
    assert "content differs" in evidence["reason"]


def test_a_manifest_without_the_contract_is_unknown_not_pass(tmp_path):
    gold, _ = _certified_dir(tmp_path)
    ok, evidence = lc.verify_labels_against_manifest(gold / "labels.parquet",
                                                     {"label_hash": "names-only"})
    assert ok is None
    assert "cannot be shown" in evidence["reason"]


def test_build_labels_v7_refuses_a_certified_directory(tmp_path):
    from quantagent.cli.v7_data import app

    gold, manifest = _certified_dir(tmp_path)
    market = tmp_path / "market.parquet"
    _panel().to_parquet(market, index=False)
    before = lc.file_sha256(gold / "labels.parquet")
    result = CliRunner().invoke(app, ["build-labels-v7", "--market-panel", str(market),
                                      "--output", str(gold / "labels.parquet"),
                                      "--horizons", "1"])
    assert result.exit_code != 0
    assert isinstance(result.exception, lc.CertifiedArtifactError)
    assert lc.file_sha256(gold / "labels.parquet") == before


def test_build_labels_v7_stamps_its_own_convention(tmp_path):
    from quantagent.cli.v7_data import app

    market = tmp_path / "market.parquet"
    _panel().to_parquet(market, index=False)
    out = tmp_path / "v7" / "labels.parquet"
    result = CliRunner().invoke(app, ["build-labels-v7", "--market-panel", str(market),
                                      "--output", str(out), "--horizons", "1"])
    assert result.exit_code == 0, result.output
    assert lc.read_label_convention(out) == (lc.V7_SAME_CLOSE, "metadata")


def _labels_verdict(runtime):
    certificate = rt.ReadinessEvaluator(runtime).full_universe_gold(
        {rt.ENGINEERING_PIPELINE_READY: True})
    payload = certificate.to_dict() if hasattr(certificate, "to_dict") else certificate.__dict__
    requirements = payload.get("requirements") or []
    hit = [r for r in requirements
           if (r.get("name") if isinstance(r, dict) else r.name)
           == "labels_match_certified_convention"]
    assert hit, payload
    req = hit[0]
    return req.get("verdict") if isinstance(req, dict) else req.verdict


def test_readiness_checks_the_label_convention_not_existence(tmp_path):
    gold, manifest = _certified_dir(tmp_path)
    assert _labels_verdict(tmp_path) == rt.PASS
    build_forward_return_labels(_panel(), (1,)).frame.to_parquet(
        gold / "labels.parquet", index=False)
    assert _labels_verdict(tmp_path) == rt.FAIL
