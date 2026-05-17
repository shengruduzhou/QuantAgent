"""CLI commands for evidence ingestion (policy / news / disclosure / financial).

Wraps the existing :class:`PolicyIngestor` and :class:`DailyEvidenceJob`
plumbing so an operator can refresh the policy evidence store with a
single command:

    quantagent ingest-policy --since 2025-10-01 --themes auto

The default ``--themes auto`` swaps the legacy keyword map for the
``THEMES_15TH_FIVE_YEAR_PLAN`` pack and turns on ``active_discovery`` so
gov.cn / NDRC / MIIT / 新华社 index pages are walked.
"""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump
from quantagent.config.paths import quant_paths


@app.command("ingest-policy")
def ingest_policy(
    as_of_date: str = typer.Option(..., "--as-of", help="As-of date YYYY-MM-DD."),
    since: str | None = typer.Option(None, "--since", help="Optional lower bound (informational)."),
    themes: str = typer.Option("auto", "--themes", help="auto | legacy — keyword pack to use."),
    allow_network: bool = typer.Option(True, "--allow-network/--no-network"),
    active_discovery: bool = typer.Option(True, "--active-discovery/--no-active-discovery"),
    max_per_source: int = typer.Option(25, "--max-per-source"),
    cache_root: Path | None = typer.Option(None, "--cache-root"),
    store_root: Path | None = typer.Option(None, "--store-root"),
) -> None:
    """Refresh the policy evidence store by walking official index pages."""
    from quantagent.data.ingestion.daily_evidence_job import (
        DailyEvidenceJob,
        DailyEvidenceJobConfig,
    )
    from quantagent.data.ingestion.policy_ingestor import PolicyIngestor
    from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry
    from quantagent.themes.keyword_packs import THEMES_15TH_FIVE_YEAR_PLAN

    paths = quant_paths().ensure()
    cache = Path(cache_root) if cache_root else paths.data_root / "v7" / "evidence"
    store = Path(store_root) if store_root else paths.data_root / "v7" / "evidence" / "store"
    cache.mkdir(parents=True, exist_ok=True)
    store.mkdir(parents=True, exist_ok=True)

    keyword_map = (
        dict(THEMES_15TH_FIVE_YEAR_PLAN) if themes == "auto" else PolicyIngestor.__dataclass_fields__["keyword_to_theme"].default_factory()
    )

    ingestor = PolicyIngestor(
        allow_network=allow_network,
        active_discovery=active_discovery,
        max_articles_per_source=max_per_source,
        keyword_to_theme=keyword_map,
    )
    job = DailyEvidenceJob(
        registry=SourceCredibilityRegistry(),
        ingestors={"policy": ingestor},
    )
    cfg = DailyEvidenceJobConfig(
        as_of_date=as_of_date,
        dry_run=False,
        enabled_sources=("policy",),
        cache_root=str(cache),
        store_root=str(store),
    )
    result = job.run(cfg)
    typer.echo(
        json_dump(
            {
                "as_of_date": as_of_date,
                "since": since,
                "themes": themes,
                "rows": int(len(result.frame)),
                "per_ingestor": result.per_ingestor_rows,
                "warnings": list(result.warnings),
                "metadata": result.metadata,
                "cache_root": str(cache),
                "store_root": str(store),
            }
        )
    )
