from __future__ import annotations

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

    second = indexer.scan()
    assert [item["id"] for item in second] == [item["id"] for item in artifacts]

    filtered = indexer.filter(run_id="fixture_run", horizon="short_5d")
    assert filtered
    assert all(item["runId"] == "fixture_run" for item in filtered)
    assert all(item["horizon"] == "short_5d" for item in filtered)


def test_empty_runtime_indexer(empty_quant_ui_settings) -> None:
    indexer = RuntimeIndexer(empty_quant_ui_settings)
    assert indexer.scan(force=True) == []
    assert indexer.stats()["artifactCount"] == 0


def test_log_parser_reads_tail_without_returning_entire_file(tmp_path) -> None:
    path = tmp_path / "large.log"
    path.write_text("".join(f"line-{index}\n" for index in range(10_000)), encoding="utf-8")

    result = parse_log(path, limit=3)

    assert result["data"] == ["line-9997", "line-9998", "line-9999"]
