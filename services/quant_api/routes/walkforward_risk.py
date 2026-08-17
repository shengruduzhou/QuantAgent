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

#: The strategy search is a separate, isolated run namespace.
SEARCH_DIR = Path("runtime/strategy_search")
SEARCH_STATUS = SEARCH_DIR / "status.json"


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


def _load_search() -> dict:
    if not SEARCH_STATUS.exists():
        raise HTTPException(
            status_code=404,
            detail=("no strategy search found. Start one with "
                    "scripts/strategy_search_walkforward.py."),
        )
    try:
        return json.loads(SEARCH_STATUS.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=503, detail=f"search status unreadable: {exc}") from exc


@router.get("/search")
async def search() -> dict:
    """Best configuration, with the penalty for having searched at all.

    `bestHoldout` is measured on windows the ranking never saw and is the number
    worth believing; `bestSelect` is in-sample for the ranking and is reported
    only so the gap between them is visible. `searchPenalty` carries PBO and the
    expected maximum Sharpe from N noisy draws -- a winner below that threshold
    is not distinguishable from luck, and the UI is expected to say so rather
    than present the winner as a finding.
    """
    d = _load_search()
    return {
        "runId": d.get("run_id"),
        "state": d.get("state"),
        "configsTried": d.get("configs_tried"),
        "configsDone": d.get("configs_done"),
        "configsTotal": d.get("configs_total"),
        "selectUntil": d.get("select_until"),
        "bestConfig": d.get("best_config"),
        "bestSelect": d.get("best_select"),
        "bestHoldout": d.get("best_holdout"),
        "nSelectWindows": d.get("n_select_windows"),
        "nHoldoutWindows": d.get("n_holdout_windows"),
        "searchPenalty": d.get("search_penalty"),
        "leaderboard": d.get("leaderboard", []),
        "note": d.get("note"),
        "elapsedSeconds": d.get("elapsed_s"),
    }
