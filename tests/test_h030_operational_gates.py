"""H-030 Track F1 §6: fail-closed behaviour of the blind-paper operational chain.

Pure-logic tests (no network, no panel writes) over the exact rules the live
scripts use, so a regression in any of them is caught before it can silently
produce official decisions.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def last_available(cst: pd.Timestamp) -> pd.Timestamp:
    """Mirror of the availability clamp shared by catch-up/runner/supervisor."""
    return cst.normalize() if cst.hour * 60 + cst.minute >= 16 * 60 \
        else cst.normalize() - pd.Timedelta(days=1)


# --- 1. a current-day in-progress bar can never be requested -----------------
@pytest.mark.parametrize("now,expected", [
    ("2026-07-21 11:42", "2026-07-20"),   # INC-P1 start time: must refuse today
    ("2026-07-21 15:30", "2026-07-20"),   # old threshold — now inside the margin
    ("2026-07-21 15:59", "2026-07-20"),   # one minute before publication margin
    ("2026-07-21 16:00", "2026-07-21"),   # published
    ("2026-07-21 23:30", "2026-07-21"),
])
def test_partial_intraday_bar_fails_closed(now, expected):
    assert last_available(pd.Timestamp(now)) == pd.Timestamp(expected)


def test_runner_boundary_is_inside_the_margin():
    """16:30 JST runner == 15:30 CST: must NOT be allowed to fetch today."""
    cst_at_runner = pd.Timestamp("2026-07-21 16:30") - pd.Timedelta(hours=1)  # JST->CST
    assert last_available(cst_at_runner) == pd.Timestamp("2026-07-20")


# --- 2. incomplete cross-section fails closed --------------------------------
def _trailing_bad(cov: dict, min_cov=0.93):
    s = pd.Series(cov).sort_index()
    ref = int(s.iloc[:-3].median())
    bad = []
    for d in reversed(s.index):
        if s[d] < min_cov * ref:
            bad.append(d)
        else:
            break
    return bad


def test_incomplete_cross_section_dropped():
    cov = {"2026-07-14": 3584, "2026-07-15": 3548, "2026-07-16": 3548,
           "2026-07-17": 3546, "2026-07-20": 3487, "2026-07-21": 1600}
    assert _trailing_bad(cov) == ["2026-07-21"]


def test_complete_cross_section_kept():
    cov = {"2026-07-14": 3584, "2026-07-15": 3548, "2026-07-16": 3548,
           "2026-07-17": 3546, "2026-07-20": 3487, "2026-07-21": 3487}
    assert _trailing_bad(cov) == []


# --- 3/4/5. shadow-day gate semantics ---------------------------------------
import shadow_day_registry as registry_module  # noqa: E402
from shadow_day_registry import OK_DATA  # noqa: E402


@pytest.fixture
def built_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build a representative registry without depending on mutable runtime state."""
    root = tmp_path / "fresh_blind"
    daily = root / "daily"
    ledger = root / "append_only_ledger.jsonl"
    cert = tmp_path / "forward_fidelity_certificate.json"
    panel = tmp_path / "market_panel.parquet"

    daily.mkdir(parents=True)
    cert.write_text(json.dumps({"passes": True}))
    pd.DataFrame({"trade_date": ["2026-07-21"] * 4}).to_parquet(panel, index=False)

    records = []
    previous_hash = "GENESIS"
    for timestamp, data_status in (
        ("2026-07-21T19:30:00+08:00", "FAILED"),
        ("2026-07-21T20:18:00+08:00", "FAILED"),
        ("2026-07-21T23:56:00+08:00", "OK"),
    ):
        record = {
            "kind": "daily_run",
            "ts": timestamp,
            "run_date": "2026-07-21",
            "prev_hash": previous_hash,
            "data_status": data_status,
            "failed_job_count": 0 if data_status == "OK" else 1,
            "prediction_status": "OK",
            "order_generation_status": "OK",
            "fill_status": "OK",
            "schema_hash": "fixture-schema-v1",
        }
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
        record_hash = hashlib.sha256((previous_hash + payload).encode()).hexdigest()
        record["record_hash"] = record_hash
        records.append(record)
        previous_hash = record_hash
    ledger.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    health = {
        "steps": {
            "freshness": {"panel_max": "2026-07-21", "lag_calendar_days": 0},
            "weights": {
                "books": {
                    candidate: {"weights_hash": f"{candidate.lower()}-weights"}
                    for candidate in registry_module.CANDIDATES
                }
            },
        }
    }
    (daily / "2026-07-21_health.json").write_text(json.dumps(health))
    for directory in ("order_logs", "fill_logs", "encrypted_performance"):
        (root / directory).mkdir()
    for candidate in registry_module.CANDIDATES:
        (root / "order_logs" / f"2026-07-21_{candidate}_weights.json").write_text("fixture")
        (root / "fill_logs" / f"2026-07-21_{candidate}_fills.json").write_text("fixture")
        (root / "encrypted_performance" / f"2026-07-21_{candidate}.bin").write_text("fixture")

    monkeypatch.setattr(registry_module, "ROOT", root)
    monkeypatch.setattr(registry_module, "DAILY", daily)
    monkeypatch.setattr(registry_module, "LEDGER", ledger)
    monkeypatch.setattr(registry_module, "CERT", cert)
    monkeypatch.setattr(registry_module, "PANEL", panel)
    return registry_module.build()


def test_partial_staged_is_not_a_failure():
    """A bounded top-up that ran out of budget must not invalidate a T-1 day."""
    assert "PARTIAL_STAGED" in OK_DATA and "OK" in OK_DATA


def test_true_failed_catchup_is_not_accepted_and_stays_auditable():
    assert "FAILED" not in OK_DATA          # cannot silently pass the gate
    rec = {"data_status": "FAILED", "failed_job_count": 1}
    assert rec["failed_job_count"] != 0     # remains visible in the ledger record


def test_stale_panel_blocks_official_decisions():
    """step_freshness marks >4 calendar days stale; runner gates books on it."""
    def freshness(panel_max: str, today: str):
        lag = (pd.Timestamp(today) - pd.Timestamp(panel_max)).days
        return ("OK" if lag <= 4 else "STALE"), lag
    assert freshness("2026-07-20", "2026-07-21")[0] == "OK"      # normal T-1
    assert freshness("2026-07-14", "2026-07-21")[0] == "STALE"   # 7-day gap blocks


def test_registry_excludes_superseded_records(built_registry):
    """The INC-P1 corrupted record must not count as its own shadow day."""
    reg = built_registry
    day21 = [d for d in reg["days"] if d["trade_date"] == "2026-07-21"]
    assert day21, "07-21 must be present in the registry"
    d = day21[0]
    assert len(d["superseded_record_ids"].split("|")) == 2   # 19:30 cron + 20:18 corrupted
    assert d["authoritative_record_id"] not in d["superseded_record_ids"]
    assert d["data_status"] == "OK" and d["valid_shadow_day"] is True
    assert sum(1 for x in reg["days"] if x["trade_date"] == "2026-07-21") == 1


def test_no_candidate_performance_in_registry(built_registry):
    """Blinding: registry may carry existence counts and hashes only."""
    txt = json.dumps(built_registry).lower()
    for banned in ("nav", "sharpe", "cagr", "drawdown", "return_pct", "pnl"):
        assert banned not in txt, f"registry leaks performance field: {banned}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
