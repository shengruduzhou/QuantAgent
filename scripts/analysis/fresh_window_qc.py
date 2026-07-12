#!/usr/bin/env python3
"""H-003 fresh-window data-quality certification (NO strategy performance).

Runs every QC gate required before the 2026-05-19+ window may be declared
frozen_future_holdout. Data-level checks only: prices, volumes, flags,
coverage, continuity. It never computes strategy returns, never ranks
candidates, never reads alpha scores.

Exit 0 = all gates pass; exit 1 = at least one gate failed (freeze forbidden,
write FRESH_HOLDOUT_REPAIR_FAILURE.md instead).
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("runtime/data/v7/silver/market_panel/market_panel.parquet")
WIN_START = pd.Timestamp("2026-05-19")
WIN_END = pd.Timestamp("2026-07-02")
SEED_DATE = pd.Timestamp("2026-05-18")
OUT = Path("runtime/reports/fresh_window_qc")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def board_cap(symbol: str, is_st: bool) -> float:
    s = str(symbol).split(".")[0].zfill(6)
    if s.startswith(("30", "68")):
        return 0.20
    if s.startswith(("82", "83", "87", "88", "43", "92")):
        return 0.30
    return 0.05 if is_st else 0.10


def expected_calendar() -> list[str]:
    """Trading calendar for the window from a liquid reference symbol (1 request)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except Exception:
        pass
    import tickflow
    tf = tickflow.TickFlow(api_key=os.environ["TICKFLOW_API_KEY"],
                           base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None)
    k = tf.klines.get("600519.SH", period="1d", count=60, adjust="none", as_dataframe=True)
    d = pd.to_datetime(k["trade_date"])
    return sorted(str(x.date()) for x in d[(d >= WIN_START) & (d <= WIN_END)].unique())


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
            "is_suspended", "is_st", "is_limit_up", "is_limit_down", "source"]
    df = pd.read_parquet(PANEL, columns=cols,
                         filters=[("trade_date", ">=", SEED_DATE - pd.Timedelta(days=45))])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    win = df[(df["trade_date"] >= WIN_START) & (df["trade_date"] <= WIN_END)].copy()
    pre = df[(df["trade_date"] < WIN_START)].copy()

    gates: dict[str, dict] = {}

    def gate(name: str, ok: bool, detail):
        gates[name] = {"pass": bool(ok), "detail": detail}

    # 1) missing trading dates
    cal = expected_calendar()
    have_dates = sorted(str(x.date()) for x in win["trade_date"].unique())
    missing = [d for d in cal if d not in have_dates]
    gate("missing_trading_dates", len(missing) == 0,
         {"expected_days": len(cal), "have_days": len(have_dates), "missing": missing})

    # 2) per-day coverage
    cov = win.groupby("trade_date")["symbol"].nunique()
    seed_n = int(pre[pre["trade_date"] == SEED_DATE]["symbol"].nunique())
    gate("per_day_symbol_count", int(cov.min()) >= int(seed_n * 0.93),
         {"seed_day_symbols": seed_n, "min": int(cov.min()), "max": int(cov.max()),
          "per_day": {str(k.date()): int(v) for k, v in cov.items()}})

    # 3) duplicates
    dup = int(win.duplicated(["symbol", "trade_date"]).sum())
    gate("duplicate_rows", dup == 0, {"count": dup})

    # 4) null OHLCV
    nulls = {c: int(win[c].isna().sum()) for c in ("open", "high", "low", "close", "volume", "amount")}
    gate("null_ohlcv", nulls["close"] == 0 and nulls["open"] == 0, nulls)

    # 5) non-positive anomalies
    bad_px = int((win[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    bad_vol = int((win["volume"] < 0).sum() + (win["amount"] < 0).sum())
    gate("non_positive_anomalies", bad_px == 0 and bad_vol == 0,
         {"non_positive_price_rows": bad_px, "negative_vol_amount_rows": bad_vol})

    # 6) adjusted-price seam check: |ret| across seam within board caps + tolerance
    seam_prev = pre[pre["trade_date"] == SEED_DATE].set_index("symbol")["close"]
    first_day = win[win["trade_date"] == win["trade_date"].min()].set_index("symbol")
    common = first_day.index.intersection(seam_prev.index)
    seam_ret = (first_day.loc[common, "close"] / seam_prev.loc[common] - 1.0)
    caps = pd.Series({s: board_cap(s, bool(first_day.loc[s, "is_st"])) for s in common})
    viol = seam_ret[abs(seam_ret) > caps + 0.02]
    gate("seam_price_continuity", len(viol) <= max(3, int(0.002 * len(common))),
         {"n_checked": int(len(common)), "violations": int(len(viol)),
          "worst": {k: round(float(v), 4) for k, v in viol.abs().nlargest(5).items()}})

    # 7) volume basis: per-symbol median window volume vs pre-seam median (x100 error => ~0.01)
    vs = win.groupby("symbol")["volume"].median()
    vp = pre[pre["trade_date"] >= SEED_DATE - pd.Timedelta(days=30)].groupby("symbol")["volume"].median()
    ratio = (vs / vp).replace([np.inf, -np.inf], np.nan).dropna()
    off = ratio[(ratio < 0.05) | (ratio > 20)]
    gate("volume_basis", len(off) <= max(5, int(0.01 * len(ratio))),
         {"n_checked": int(len(ratio)), "median_ratio": round(float(ratio.median()), 3),
          "outliers": int(len(off))})

    # 8) amount basis: amount ≈ volume x price
    w = win[(win["volume"] > 0) & (win["amount"] > 0)]
    implied = w["amount"] / (w["volume"] * w["close"])
    frac_ok = float(((implied > 0.5) & (implied < 2.0)).mean())
    gate("amount_basis", frac_ok > 0.98, {"fraction_in_band": round(frac_ok, 4),
                                          "median_implied_vwap_over_close": round(float(implied.median()), 3)})

    # 9) prev-close continuity within window.
    # RAW (adjust="none") prices legitimately jump beyond limit bands on
    # ex-dividend / ex-rights days (June = A-share payout season), and the
    # panel's is_st is a current-snapshot broadcast (documented provenance
    # limitation) so recently de-ST'd names can look like 5%-cap violators at
    # +10.x%. Validated 2026-07-04 on the 5 largest violators against the
    # forward-ADJUSTED series: down-jumps −35%/−34%/−33% → adjusted −5.6%/
    # −2.7%/−5.6% (corporate actions), up-jumps +10.9%/+10.7% identical in
    # adjusted series (de-ST cap 10% + low-price rounding). Gate therefore:
    #   hard-fail on physically implausible moves (down <−60%, up >+45%), and
    #   rate-fail if cap+2% violations exceed 0.25% of returns.
    chain = pd.concat([pre[pre["trade_date"] == SEED_DATE], win]).sort_values(["symbol", "trade_date"])
    chain["ret"] = chain.groupby("symbol")["close"].pct_change()
    r = chain.dropna(subset=["ret"])
    capser = r.apply(lambda x: board_cap(x["symbol"], bool(x["is_st"])), axis=1)
    jumps = r[abs(r["ret"]) > capser + 0.02]
    implausible = r[(r["ret"] < -0.60) | (r["ret"] > 0.45)]
    gate("prev_close_continuity",
         len(implausible) == 0 and len(jumps) <= int(0.0025 * len(r)),
         {"n_returns": int(len(r)), "cap_violations": int(len(jumps)),
          "cap_violation_rate": round(float(len(jumps) / max(1, len(r))), 5),
          "implausible_moves": int(len(implausible)),
          "note": "raw-price basis: ex-div/ex-rights + ST-snapshot caps allowed; see manifest QC section"})

    # 10) limit flag sanity
    up_rate = win.groupby("trade_date")["is_limit_up"].mean()
    dn_rate = win.groupby("trade_date")["is_limit_down"].mean()
    flagged = win[win["is_limit_up"] == True]  # noqa: E712
    merged = flagged.merge(chain[["symbol", "trade_date", "ret"]], on=["symbol", "trade_date"], how="left")
    consistent = float((merged["ret"] > 0.04).mean()) if len(merged) else 1.0
    gate("limit_flag_sanity", bool(0.0005 <= up_rate.mean() <= 0.08 and dn_rate.mean() <= 0.05 and consistent > 0.9),
         {"mean_limit_up_rate": round(float(up_rate.mean()), 4),
          "mean_limit_down_rate": round(float(dn_rate.mean()), 4),
          "limit_up_ret_consistency": round(consistent, 3)})

    # 11) ST / suspension sanity
    st_rate = float(win.groupby("trade_date")["is_st"].mean().mean())
    susp_consistent = float((win["is_suspended"] == (win["volume"].fillna(0) <= 0)).mean())
    gate("st_suspension_sanity", bool(0.0 <= st_rate <= 0.10 and susp_consistent > 0.999),
         {"mean_st_rate": round(st_rate, 4), "suspended_eq_zero_volume": round(susp_consistent, 5)})

    # 12) provenance + schema hash + row counts
    src = win["source"].value_counts().to_dict()
    full_md = pd.read_parquet(PANEL, columns=["trade_date"])
    schema_cols = json.dumps(sorted(pd.read_parquet(PANEL, columns=[]).columns.tolist()) if False else
                             sorted(cols), sort_keys=True)  # column-set hash of QC view
    import pyarrow.parquet as pq
    md = pq.ParquetFile(PANEL)
    schema_sig = hashlib.sha256(str(md.schema_arrow).encode()).hexdigest()
    gate("provenance_and_schema", True,
         {"window_sources": {str(k): int(v) for k, v in src.items()},
          "panel_schema_sha256": schema_sig,
          "panel_total_rows": int(md.metadata.num_rows),
          "window_rows": int(len(win))})

    all_pass = all(g["pass"] for g in gates.values())
    report = {
        "window": f"{WIN_START.date()}..{WIN_END.date()}",
        "trading_days_available": len(have_dates),
        "formal_read_threshold_days": 120,
        "below_threshold": len(have_dates) < 120,
        "all_gates_pass": all_pass,
        "gates": gates,
        "peak_rss_gib": round(rss_gib(), 2),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT / "qc_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v["pass"] for k, v in gates.items()}, indent=2))
    print(f"ALL GATES PASS: {all_pass} | days {len(have_dates)} | RSS {report['peak_rss_gib']} GiB")
    print(f"report -> {OUT/'qc_report.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
