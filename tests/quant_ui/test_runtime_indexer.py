from __future__ import annotations

import json
from hashlib import sha256

from services.quant_api.runtime_indexer import RuntimeIndexer
from services.quant_api.runtime_indexer.parsers import parse_log


def test_runtime_indexer_classifies_and_excludes_internal_cache(quant_ui_settings) -> None:
    indexer = RuntimeIndexer(quant_ui_settings)
    artifacts = indexer.scan(force=True)

    assert artifacts
    assert any(item["kind"] == "backtest" and item["name"] == "metrics.json" for item in artifacts)
    assert any(item["kind"] == "model" and item["name"] == "ft_transformer.pt" for item in artifacts)
    assert any(item["kind"] == "selection" and item["name"] == "hybrid_stock_pool.parquet" for item in artifacts)
    assert all("runtime/cache/quant_ui" not in item["path"] for item in artifacts)
    assert all("/feature_cache/" not in item["path"] for item in artifacts)
    assert all(item["path"].startswith("runtime/") for item in artifacts)

    metrics = next(item for item in artifacts if item["name"] == "metrics.json")
    assert metrics["schemaVersion"] == "quantagent.backtest.metrics.1"
    assert metrics["trustClass"] == "production_ready"
    assert metrics["validationStatus"] == "verified"
    assert metrics["manifestPath"].endswith("metrics.json.manifest.json")
    assert "production_display" in metrics["capabilities"]
    assert "paper_execution" in metrics["capabilities"]
    assert metrics["declaredKind"] == "backtest_metrics"
    assert metrics["kindSource"] == "manifest"
    assert metrics["runIdSource"] == "manifest"
    assert metrics["producer"] == "run-strict-a-share-backtest-v8"
    assert metrics["qualityStatus"] == "passed"
    assert metrics["rows"] == 1
    assert metrics["dateStart"] == "2026-01-02"
    assert metrics["dateEnd"] == "2026-01-05"

    second = indexer.scan()
    assert [item["id"] for item in second] == [item["id"] for item in artifacts]

    filtered = indexer.filter(run_id="fixture_run", horizon="short_5d")
    assert filtered
    assert all(item["runId"] == "fixture_run" for item in filtered)
    assert all(item["horizon"] == "short_5d" for item in filtered)

    catalog = indexer.catalog()
    assert catalog["summary"]["runCount"] >= 1
    assert catalog["summary"]["byCapability"]["production_display"] >= 1
    assert catalog["summary"]["manifestCoverage"] > 0
    assert catalog["runs"][0]["artifactCount"] >= 1

    lineage = indexer.lineage(metrics["id"])
    assert lineage is not None
    assert lineage["status"] == "complete"
    assert lineage["upstream"][0]["artifact"]["name"] == "nav.csv"


def test_empty_runtime_indexer(empty_quant_ui_settings) -> None:
    indexer = RuntimeIndexer(empty_quant_ui_settings)
    assert indexer.scan(force=True) == []
    assert indexer.stats()["artifactCount"] == 0


def test_log_parser_reads_tail_without_returning_entire_file(tmp_path) -> None:
    path = tmp_path / "large.log"
    path.write_text("".join(f"line-{index}\n" for index in range(10_000)), encoding="utf-8")

    result = parse_log(path, limit=3)

    assert result["data"] == ["line-9997", "line-9998", "line-9999"]


def test_unclassified_and_contaminated_artifacts_are_not_production_capable(quant_ui_settings) -> None:
    root = quant_ui_settings.runtime_root / "reports" / "trust_contracts"
    root.mkdir(parents=True)
    unclassified = root / "legacy.json"
    unclassified.write_text('{"value": 1}', encoding="utf-8")

    contaminated = root / "forensics.json"
    contaminated.write_text('{"value": 2}', encoding="utf-8")
    (root / "forensics.json.manifest.json").write_text(
        json.dumps({
            "schema_version": "quantagent.forensics.1",
            "trust_class": "contaminated_holdout_forensics",
            "output_sha256": sha256(contaminated.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )

    artifacts = RuntimeIndexer(quant_ui_settings).scan(force=True)
    legacy_row = next(item for item in artifacts if item["path"].endswith("legacy.json"))
    contaminated_row = next(item for item in artifacts if item["path"].endswith("forensics.json"))

    assert legacy_row["trustClass"] == "unclassified"
    assert legacy_row["validationStatus"] == "unverified"
    assert "production_display" not in legacy_row["capabilities"]
    assert contaminated_row["trustClass"] == "contaminated"
    assert contaminated_row["validationStatus"] == "verified"
    assert "production_display" not in contaminated_row["capabilities"]


def test_hash_mismatch_fails_closed(quant_ui_settings) -> None:
    root = quant_ui_settings.runtime_root / "reports" / "broken_contract"
    root.mkdir(parents=True)
    artifact = root / "result.json"
    artifact.write_text('{"value": 1}', encoding="utf-8")
    (root / "result.json.manifest.json").write_text(
        json.dumps({
            "schema_version": "quantagent.result.1",
            "trust_class": "production_ready",
            "output_sha256": "0" * 64,
        }),
        encoding="utf-8",
    )

    row = next(
        item for item in RuntimeIndexer(quant_ui_settings).scan(force=True)
        if item["path"].endswith("result.json")
    )

    assert row["status"] == "error"
    assert row["validationStatus"] == "invalid"
    assert "production_display" not in row["capabilities"]
    assert row["issues"][0]["code"] == "content_hash_mismatch"
