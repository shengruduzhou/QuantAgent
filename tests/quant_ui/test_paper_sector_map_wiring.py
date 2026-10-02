"""The HTTP paper venue's industry map comes from an optional sector-map path.

Without a map the venue refuses a BUY as `industry_unmeasured` (the industry
limit is not silently skipped); a configured map is loaded at startup, and an
unreadable one leaves the map absent and says why instead of taking the API
down.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from services.quant_api.app import create_app
from services.quant_api.config import ApiSettings, default_settings
from services.quant_api.services.paper_orders import PaperOrderService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _settings(tmp_path, **overrides) -> ApiSettings:
    return ApiSettings(
        project_root=PROJECT_ROOT,
        runtime_root=tmp_path / "runtime",
        cache_root=tmp_path / "runtime" / "cache",
        jobs_root=tmp_path / "runtime" / "jobs",
        **overrides,
    ).ensure()


def test_a_configured_sector_map_reaches_the_paper_venue(tmp_path) -> None:
    path = tmp_path / "sector_map.csv"
    pd.DataFrame({"symbol": ["600000.SH", "000001.SZ"], "industry": ["bank", "bank"]}).to_csv(
        path, index=False
    )
    app = create_app(_settings(tmp_path, paper_sector_map=path))
    service = app.state.services.paper_orders
    try:
        assert service.broker.industry_map == {"600000.SH": "bank", "000001.SZ": "bank"}
        assert service.industry_map_source == str(path)
        assert service.industry_map_error is None
    finally:
        service.close()


def test_an_unreadable_sector_map_is_reported_not_fatal(tmp_path) -> None:
    service = PaperOrderService(tmp_path / "p", sector_map_path=tmp_path / "missing.csv")
    try:
        assert service.broker.industry_map == {}
        assert "missing.csv" in service.industry_map_error
    finally:
        service.close()


def test_the_sector_map_path_is_read_from_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTAGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("QUANTAGENT_PAPER_SECTOR_MAP", str(tmp_path / "map.parquet"))
    assert default_settings().paper_sector_map == tmp_path / "map.parquet"
    monkeypatch.delenv("QUANTAGENT_PAPER_SECTOR_MAP")
    assert default_settings().paper_sector_map is None
