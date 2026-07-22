from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from services.quant_api.app import create_app
from services.quant_api.services.jobs import JobManager, JobRecord, _progress_from_line


def request(app, method: str, url: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_data_provider_registry_is_explicit_and_never_exposes_credentials(quant_ui_settings, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "do-not-expose-this-token")
    monkeypatch.setattr("services.quant_api.services.data_manager._module_available", lambda _module: True)

    result = request(create_app(quant_ui_settings), "GET", "/api/data/providers")

    assert result.status_code == 200
    payload = result.json()["data"]
    assert payload["supportsCancellation"] is True
    assert payload["providers"][0]["id"] == "tickflow"
    assert payload["coverageEndpoint"] == "/api/data/coverage"
    assert any(provider["id"] == "runtime_catalog" for provider in payload["providers"])
    tushare = next(provider for provider in payload["providers"] if provider["id"] == "tushare_fundamentals")
    assert tushare["configured"] is True
    assert tushare["missingRequirements"] == []
    assert "do-not-expose-this-token" not in result.text
    assert "tokenValue" not in result.text


def test_server_side_coverage_reports_duplicates_and_date_range(quant_ui_settings) -> None:
    source = quant_ui_settings.runtime_root / "import_quarantine" / "bars.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "symbol,trade_date,close\n"
        "000001.SZ,2026-01-02,10\n"
        "000001.SZ,2026-01-02,10\n"
        "000001.SZ,2026-01-05,11\n",
        encoding="utf-8",
    )

    result = request(
        create_app(quant_ui_settings),
        "GET",
        "/api/data/coverage",
        params={"path": "runtime/import_quarantine/bars.csv", "deep": "true"},
    )

    assert result.status_code == 200
    payload = result.json()["data"]
    assert payload["duplicateKeys"] == 1
    assert payload["dateStart"] == "2026-01-02"
    assert payload["dateEnd"] == "2026-01-05"
    assert payload["symbolCount"] == 1


def test_coverage_rejects_files_outside_runtime(quant_ui_settings) -> None:
    outside = quant_ui_settings.project_root / "outside.csv"
    outside.write_text("symbol,trade_date\n000001.SZ,2026-01-02\n", encoding="utf-8")

    result = request(
        create_app(quant_ui_settings),
        "GET",
        "/api/data/coverage",
        params={"path": "outside.csv"},
    )

    assert result.status_code == 422
    assert "inside Runtime" in result.json()["detail"]


def test_quarantine_import_streams_deduplicates_and_writes_manifest(quant_ui_settings) -> None:
    source = quant_ui_settings.runtime_root / "import_quarantine" / "incoming.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "symbol,trade_date,close\n"
        "000001.SZ,2026-01-02,10\n"
        "000001.SZ,2026-01-02,10\n"
        "600519.SH,2026-01-05,1500\n",
        encoding="utf-8",
    )
    output = quant_ui_settings.runtime_root / "data" / "imported" / "validated.parquet"
    env = {**os.environ, "QUANTAGENT_HOME": str(quant_ui_settings.runtime_root)}

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "data_manager_transfer.py"),
            "--operation", "import",
            "--source", str(source),
            "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(__import__("pandas").read_parquet(output)) == 2
    manifest = json.loads(output.with_suffix(".parquet.manifest.json").read_text(encoding="utf-8"))
    assert manifest["duplicates_removed"] == 1
    assert manifest["rows_written"] == 2
    assert len(manifest["sha256"]) == 64


def test_runtime_export_filters_symbols_dates_and_writes_manifest(quant_ui_settings) -> None:
    source = quant_ui_settings.runtime_root / "data" / "source.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "symbol,trade_date,close\n"
        "000001.SZ,2026-01-02,10\n"
        "000001.SZ,2026-02-02,11\n"
        "600519.SH,2026-01-05,1500\n",
        encoding="utf-8",
    )
    output = quant_ui_settings.runtime_root / "exports" / "filtered.csv"
    env = {**os.environ, "QUANTAGENT_HOME": str(quant_ui_settings.runtime_root)}

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "data_manager_transfer.py"),
            "--operation", "export",
            "--source", str(source),
            "--output", str(output),
            "--symbols", "000001.SZ",
            "--start-date", "2026-01-01",
            "--end-date", "2026-01-31",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    exported = __import__("pandas").read_csv(output)
    assert exported[["symbol", "trade_date"]].to_dict("records") == [
        {"symbol": "000001.SZ", "trade_date": "2026-01-02"}
    ]
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    assert manifest["operation"] == "export"
    assert manifest["filters"]["symbols"] == ["000001.SZ"]
    assert manifest["rows_written"] == 1


def test_tickflow_recorder_accepts_server_symbol_file_and_requires_network_confirmation(quant_ui_settings) -> None:
    symbols = quant_ui_settings.runtime_root / "universes" / "held.txt"
    symbols.parent.mkdir(parents=True, exist_ok=True)
    symbols.write_text("000001.SZ\n600519.SH\n", encoding="utf-8")
    manager = JobManager(quant_ui_settings)

    validated = manager.validate(
        "data",
        "record-tickflow-quotes",
        {
            "symbols_file": "runtime/universes/held.txt",
            "loop_seconds": 30,
            "max_iterations": 120,
            "allow_network": True,
        },
    )

    assert validated["valid"] is True
    assert validated["outputPaths"] == ["runtime/data/v7/silver/tick_snapshots"]
    with pytest.raises(ValueError, match="allow_network must be explicitly confirmed"):
        manager.validate(
            "data",
            "record-tickflow-quotes",
            {"symbols": "000001.SZ", "allow_network": False},
        )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('{"progress": 0.42, "rows": 200}', 0.42),
        ('{"batch": 3, "total_batches": 10}', 0.3),
        ('{"iteration": 9, "total_iterations": 12}', 0.75),
        ("[7/20] 600519.SH persisted", 0.35),
    ],
)
def test_provider_progress_is_parsed_from_structured_and_legacy_output(line: str, expected: float) -> None:
    assert _progress_from_line(line) == pytest.approx(expected)


def test_data_job_requires_allowlisted_command_and_explicit_universe(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    arbitrary = request(
        app,
        "POST",
        "/api/jobs/data",
        json={"commandId": "bash", "parameters": {}},
    )
    assert arbitrary.status_code == 422

    no_universe = request(
        app,
        "POST",
        "/api/jobs/data",
        json={
            "commandId": "build-akshare-market-panel-v7",
            "parameters": {
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
                "output": "runtime/data/web/market.parquet",
                "allow_network": True,
            },
        },
    )
    assert no_universe.status_code == 422
    assert "one of" in no_universe.json()["detail"]


def test_queued_job_can_be_cancelled_without_starting_a_process(quant_ui_settings) -> None:
    manager = JobManager(quant_ui_settings)
    manager._jobs["job_cancel"] = JobRecord(
        id="job_cancel",
        type="data",
        status="queued",
        commandId="build-akshare-market-panel-v7",
        createdAt="2026-07-22T00:00:00+00:00",
    )

    cancelled = manager.cancel("job_cancel")

    assert cancelled["status"] == "cancelled"
    assert cancelled["finishedAt"] is not None


def test_terminal_job_cannot_be_cancelled_again(quant_ui_settings) -> None:
    manager = JobManager(quant_ui_settings)
    manager._jobs["job_done"] = JobRecord(
        id="job_done",
        type="data",
        status="succeeded",
        commandId="build-akshare-market-panel-v7",
        createdAt="2026-07-22T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="already finished"):
        manager.cancel("job_done")
