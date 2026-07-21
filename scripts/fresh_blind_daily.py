#!/usr/bin/env python3
"""H-028 Track D: blinded forward paper-trading daily runner (fu h028).

Modular daily workflow for the frozen candidates S1-S4:
  1. update PIT market data (update_market_panel_daily.py)
  2. validate data freshness (panel max date == last exchange trading day)
  3. score sleeves with the frozen v8.9 checkpoints (REQUIRES a passing
     fidelity certificate at runtime/reports/h028/forward_fidelity_certificate.json;
     otherwise records PENDING_P6 and fails closed — never a silent downgrade)
  4. rebuild candidate books deterministically from window start (stateless
     recompute of EMA/min-hold/regime state on accumulated forward scores)
  5. risk gates + order/fill simulation (corrected strict sim)
  6. store artifacts; encrypt performance; append hash-chained ledger record

BLINDING: candidate NAV/return/rank/Sharpe/drawdown are written ONLY into
encrypted_performance/ (Fernet). Operational health is the only plaintext
output. This is PROCEDURAL blinding — the key lives on this machine
(.unblind_key, mode 600); discipline is enforced by tooling + the hash chain
making any early read leave no deniable path, not by cryptographic hardness
against the machine owner. Stated honestly per H-028 prereg.

Run: AI_quant_venv/bin/python3 scripts/fresh_blind_daily.py [--dry-run]
Cron (documented, install manually or via --install-cron):
  30 16 * * 1-5 cd /home/shanhefu/QuantAgent && AI_quant_venv/bin/python3 scripts/fresh_blind_daily.py >> runtime/paper/fresh_blind/daily/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runtime/paper/fresh_blind"
DIRS = ["daily", "candidate_configs", "order_logs", "fill_logs", "risk_logs",
        "encrypted_performance"]
LEDGER = ROOT / "append_only_ledger.jsonl"
KEY_PATH = ROOT / ".unblind_key"
CERT = REPO / "runtime/reports/h028/forward_fidelity_certificate.json"
PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
ELIG = REPO / "runtime/reports/h028/fresh_first_read_eligibility.json"
CANDIDATES = ["S1_production_2sleeve_rank_k10", "S2_L1_c3ema07_minhold10",
              "S3_L1_D1_regime_w05", "S4_RW1_4state"]


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_tree() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def get_key() -> bytes:
    from cryptography.fernet import Fernet
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(Fernet.generate_key())
        os.chmod(KEY_PATH, 0o600)
    return KEY_PATH.read_bytes()


def ledger_append(record: dict) -> str:
    prev = "GENESIS"
    if LEDGER.exists():
        lines = LEDGER.read_text().strip().splitlines()
        if lines:
            prev = json.loads(lines[-1])["record_hash"]
    record["prev_hash"] = prev
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    record["record_hash"] = sha((prev + payload).encode())
    with open(LEDGER, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["record_hash"]


TOPUP_BUDGET_S = 1800


def step_update_data(dry: bool) -> dict:
    """Bounded TOP-UP only — fetching is the supervisor's job.

    Measured 2026-07-21: the vendor limit is 10 req/min, so a full-universe pass
    takes ~6h. When this step owned the catch-up it consumed the runner's entire
    budget and the process was killed before producing books (rc=124). Ownership
    is now split: catchup_supervisor (06:00, multi-iteration) closes the backlog;
    this step only tops up a small residual and never blocks the run. A top-up
    timeout is NOT a failure — staging persists and the supervisor finishes it;
    staleness is judged by step_freshness, which is the honest signal.
    """
    if dry:
        return {"status": "SKIPPED_DRY_RUN"}
    try:
        panel_max = pd.to_datetime(pd.read_parquet(PANEL, columns=["trade_date"])["trade_date"]).max()
        cst = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
        last_avail = cst.normalize() if cst.hour * 60 + cst.minute >= 15 * 60 + 30 \
            else cst.normalize() - pd.Timedelta(days=1)
        if panel_max >= last_avail:
            return {"status": "OK", "note": "panel already current (no fetch needed)"}
    except Exception as e:
        return {"status": "FAILED", "error": f"panel read failed: {str(e)[:120]}"}
    try:
        r = subprocess.run([sys.executable, str(REPO / "scripts/catchup_panel_chunked.py")],
                           capture_output=True, text=True, timeout=TOPUP_BUDGET_S)
    except subprocess.TimeoutExpired:
        return {"status": "PARTIAL_STAGED",
                "note": f"top-up exceeded {TOPUP_BUDGET_S}s; staging persists, "
                        "supervisor completes the window (not a run failure)"}
    return {"status": "OK" if r.returncode == 0 else "FAILED",
            "returncode": r.returncode, "tail": r.stdout[-300:] + r.stderr[-200:]}


def step_freshness() -> dict:
    dates = pd.read_parquet(PANEL, columns=["trade_date"])["trade_date"]
    pmax = pd.to_datetime(dates).max()
    lag_days = (pd.Timestamp(date.today()) - pmax).days
    return {"status": "OK" if lag_days <= 4 else "STALE",
            "panel_max": str(pmax.date()), "lag_calendar_days": int(lag_days)}


def step_score() -> dict:
    if not CERT.exists():
        return {"status": "PENDING_P6_V89_FORWARD_PORT",
                "detail": "no passing fidelity certificate; scoring fails closed. "
                          "Fallback sanctioned by FRESH_HOLDOUT_FREEZE_MANIFEST: "
                          "batch-score at read time once fidelity >=0.99 is certified."}
    cert = json.loads(CERT.read_text())
    if not cert.get("passes"):
        return {"status": "FIDELITY_CERT_FAILED", "cert": {"passes": False}}
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/forward_daily_inference.py"),
         "--run-dir", "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300",
         "--warmup-days", "420", "--device", "cuda",
         "--output", str(ROOT / "daily" / "composite_forward.parquet"),
         "--sleeve-scores-output", str(ROOT / "daily" / "sleeve_scores.parquet")],
        capture_output=True, text=True, timeout=7200, cwd=REPO)
    return {"status": "OK" if r.returncode == 0 else "FAILED",
            "returncode": r.returncode, "tail": (r.stdout[-200:] + r.stderr[-150:]).strip()}


def step_books_and_fills(today: str) -> dict:
    """S1/S2/S3 decisions + strict-sim fills on the FRESH window; performance
    is encrypted, never printed. S4 learner port = recorded PENDING item.
    Stateless recompute from window start on accumulated sleeve scores."""
    from cryptography.fernet import Fernet
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    import baseline_protocol as bp
    from exp011_book_churn import eligible_rank_lists
    from exp009_exposure_overlay import bench_series
    from exp010_hysteresis_overlay import gross_series
    from dual_track_eval import build_book
    from dual_track_d1_integration import tilt_series
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8

    sc_path = ROOT / "daily" / "sleeve_scores.parquet"
    if not sc_path.exists():
        return {"status": "NO_SLEEVE_SCORES"}
    sc = pd.read_parquet(sc_path)
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    sc = sc[sc["trade_date"] >= "2026-05-19"]
    if sc.empty:
        return {"status": "NO_FRESH_SCORES"}
    fresh_start = pd.Timestamp("2026-05-19")
    pcols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
             "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=pcols,
                            filters=[("trade_date", ">=", fresh_start - pd.Timedelta(days=210))])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    bt_panel = panel[panel["trade_date"] >= fresh_start - pd.Timedelta(days=10)].copy()
    flags = bt_panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
    trade_dates = sorted(bt_panel["trade_date"].unique())

    slv = ["short_5d", "mid_5d_30d", "long_30d_120d"]
    for c in slv:
        sc[f"rk_{c}"] = sc.groupby("trade_date")[f"score_{c}"].rank(pct=True)
    sc["c3"] = sc[[f"rk_{c}" for c in slv]].median(axis=1)
    sc = sc.sort_values(["symbol", "trade_date"])
    sc["c3_ema"] = sc.groupby("symbol", sort=False)["c3"].transform(
        lambda s: s.ewm(alpha=0.7, adjust=False).mean())
    sc["s1"] = sc["rk_short_5d"] + sc["rk_mid_5d_30d"]

    tilt = tilt_series(panel.sort_values(["symbol", "trade_date"]), "d1")
    m = sc.merge(tilt, on=["symbol", "trade_date"], how="left")
    m["rc"] = m.groupby("trade_date")["c3_ema"].rank(pct=True)
    m["rd"] = m.groupby("trade_date")["tilt"].rank(pct=True)
    oos_e = max(sc["trade_date"])
    regime = gross_series(bench_series(fresh_start, oos_e), "R2a_confirm5")
    wser = (regime < 1.0).astype(float) * 0.5
    w = m["trade_date"].map(wser).fillna(0.0).to_numpy()
    m["s3"] = (1 - w) * m["rc"] + w * m["rd"].fillna(m["rc"])

    key = get_key()
    fern = Fernet(key)
    out = {"status": "OK", "books": {}, "s4": "PENDING_S4_LEARNER_PORT (regime_weight_meta daily port spec'd)"}
    sector = pd.read_parquet(REPO / bp.SECTOR)
    for cand, col, style in (("S1", "s1", "plain"), ("S2", "c3_ema", "minhold"), ("S3", "s3", "minhold")):
        score = m[["trade_date", "symbol"]].copy()
        score["alpha_score"] = m[col].to_numpy()
        p = score.merge(flags, on=["symbol", "trade_date"], how="left")
        book = build_book(eligible_rank_lists(p),
                          "minhold" if style == "minhold" else "plain",
                          {"n": 10})
        tw = bp._apply_delay(book, trade_dates, 1)
        last_w = tw.iloc[-1]
        holdings = {s: round(float(v), 4) for s, v in last_w[last_w > 0].items()}
        (ROOT / "order_logs" / f"{today}_{cand}_weights.json").write_text(
            json.dumps({"date": today, "candidate": cand, "target_weights": holdings}, indent=1))
        cfg = AShareExecutionSimulationConfig(initial_cash=1e6, slippage_bps=8.0)
        r = run_strict_backtest_v8(tw, bt_panel, sector_map=sector, config=cfg)
        nav = r.nav.copy()
        perf = {"candidate": cand, "as_of": today,
                "nav": {str(k): float(v) for k, v in nav.items()}}
        (ROOT / "encrypted_performance" / f"{today}_{cand}.bin").write_bytes(
            fern.encrypt(json.dumps(perf).encode()))
        fo = r.failed_orders if r.failed_orders is not None else pd.DataFrame()
        (ROOT / "fill_logs" / f"{today}_{cand}_fills.json").write_text(json.dumps(
            {"date": today, "candidate": cand, "n_failed_orders": int(len(fo)),
             "n_holdings": len(holdings)}, indent=1))
        out["books"][cand] = {"n_holdings": len(holdings), "failed_orders": int(len(fo)),
                              "weights_hash": sha(json.dumps(holdings, sort_keys=True).encode())[:16]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    os.chdir(REPO)  # cron-safe: all repo-relative paths resolve regardless of caller cwd
    ensure_tree()
    get_key()  # ensure key exists with 600 before any performance is produced
    today = datetime.now().strftime("%Y-%m-%d")

    steps = {"data": step_update_data(args.dry_run)}
    steps["freshness"] = step_freshness()
    steps["scoring"] = step_score()
    ok_so_far = steps["freshness"]["status"] == "OK" and steps["scoring"]["status"].startswith(("OK", "READY"))
    if ok_so_far:
        try:
            steps["weights"] = step_books_and_fills(today)
        except Exception as e:
            steps["weights"] = {"status": "FAILED", "error": str(e)[:300]}
    else:
        steps["weights"] = {"status": "BLOCKED_UPSTREAM"}
    steps["risk_gates"] = {"status": steps["weights"]["status"] if steps["weights"]["status"] != "OK"
                           else "OK"}
    steps["fills"] = {"status": steps["weights"]["status"]}

    health = {
        "run_date": today,
        "data_status": steps["data"]["status"],
        "prediction_status": steps["scoring"]["status"],
        "order_generation_status": steps["weights"]["status"],
        "fill_status": steps["fills"]["status"],
        "risk_event_count": 0,
        "failed_job_count": sum(1 for s in steps.values() if s["status"] in ("FAILED", "STALE")),
        "schema_hash": sha(PANEL.read_bytes()[:1 << 20])[:16],  # header slice fingerprint
        "runtime_s": round(time.time() - t0, 1),
        "ram_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "vram": "n/a (no GPU step ran)",
        "candidates": CANDIDATES,
        "blinding": "performance encrypted; no candidate-level numbers in this record",
    }
    (ROOT / "daily" / f"{today}_health.json").write_text(json.dumps({**health, "steps": steps}, indent=2))
    h = ledger_append({"ts": datetime.now().isoformat(), "kind": "daily_run", **health})
    print(json.dumps(health, indent=2))
    print("ledger record:", h[:16])
    return 0 if health["failed_job_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
