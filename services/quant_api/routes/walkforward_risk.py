"""Read-only view of the walk-forward risk-control backtest.

Serves whatever the runner has written so far, so the frontend can poll while a
run is still in progress. The runner writes `status.json` atomically via a
tmp+replace, so a poll never observes a torn file.

This endpoint deliberately performs NO computation and NO defaulting. It reports
`state` exactly as the runner set it and passes metrics through untouched --
including nulls. A missing metric here means "the run has not produced it",
which is a different fact from zero, and the UI is responsible for showing that
difference rather than this layer papering over it.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/walkforward-risk", tags=["walkforward", "risk"])

RESULT_DIR = Path("runtime/walkforward_risk")
STATUS_FILE = RESULT_DIR / "status.json"


def _load() -> dict:
    if not STATUS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=("no walk-forward run found. Start one with "
                    "scripts/walkforward_risk_backtest.py -- this endpoint never "
                    "fabricates a placeholder result."),
        )
    try:
        return json.loads(STATUS_FILE.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=503, detail=f"status file unreadable: {exc}") from exc


@router.get("/status")
async def status() -> dict:
    """Progress only — cheap enough to poll every second."""
    data = _load()
    return {
        "runId": data.get("run_id"),
        "state": data.get("state"),
        "windowsDone": data.get("windows_done", len(data.get("results", []))),
        "windowsTotal": data.get("windows_total"),
        "currentTestYear": data.get("current_test_year"),
        "elapsedSeconds": data.get("elapsed_s"),
        "breadthFloor": data.get("breadth_floor"),
    }


@router.get("/results")
async def results() -> dict:
    """Full per-window results plus the aggregate, as written."""
    data = _load()
    return {
        "runId": data.get("run_id"),
        "state": data.get("state"),
        "windowsTotal": data.get("windows_total"),
        "windowsOkBreadth": data.get("windows_ok_breadth"),
        "breadthFloor": data.get("breadth_floor"),
        "config": data.get("config"),
        "aggregate": data.get("aggregate_over_ok_windows"),
        "note": data.get("note"),
        "results": data.get("results", []),
    }


@router.get("/runs")
async def runs() -> dict:
    """Completed runs on disk, newest first."""
    if not RESULT_DIR.exists():
        return {"runs": []}
    files = sorted(RESULT_DIR.glob("run_*.json"), reverse=True)
    return {"runs": [{"runId": f.stem.removeprefix("run_"),
                      "path": str(f),
                      "sizeBytes": f.stat().st_size} for f in files]}
