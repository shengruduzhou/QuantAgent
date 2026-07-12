#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot stage research; conclusion recorded in stage3_to_6_report.md / memory.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Stage 4 — Cost / turnover / capacity robust daily book.

Goal: make w210_k10 / w111_k5 survive 30-50bps, not just 8bps. Core lever =
TURNOVER REDUCTION (no-trade band / rank buffer / lower frequency) + cost-aware
construction + capacity sizing.

Method: a fast cost-aware NAV proxy (net daily = book_ret - turnover*cost) drives
the broad search across all sections and the *_results.csv; the LEADERBOARD
finalists are then re-confirmed with the trusted strict backtest at 8/15/30/50/100
bps on non-2026 + untouched 2026. No 2026 tuning. No RL. No intraday.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_protocol as bp
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8

ENS = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
OUT = Path("runtime/reports/v89_closed_loop/stage4"); OUT.mkdir(parents=True, exist_ok=True)
ANN = 244
WIN = {"val": ("2024-08-28", "2025-08-31"), "non2026": ("2025-09-01", "2025-12-31"), "y2026": ("2026-01-02", "2026-05-13")}
SLIPS = [8, 15, 30, 50, 100]


def load():
    ens = pd.read_parquet(ENS); ens["trade_date"] = pd.to_datetime(ens["trade_date"])
    pc = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
          "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(bp.PANEL, columns=pc); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp("2024-06-01")]
    sector = pd.read_parquet(bp.SECTOR)
    # fwd1d (close->next close) wide
    px = panel.pivot_table(index="trade_date", columns="symbol", values="close").sort_index()
    fwd1d = (px.shift(-1) / px - 1.0)
    amt = panel.pivot_table(index="trade_date", columns="symbol", values="amount").sort_index()
    adv20 = amt.rolling(20, min_periods=5).mean()
    vol20 = (px.pct_change()).rolling(20, min_periods=5).std()
    # eligibility wide (True = tradable at signal)
    elig = ~(panel.assign(bad=panel["is_st"].fillna(False).astype(bool) | panel["is_suspended"].fillna(False).astype(bool)
                          | panel["is_limit_up"].fillna(False).astype(bool))
             .pivot_table(index="trade_date", columns="symbol", values="bad", aggfunc="max").fillna(False).astype(bool)).reindex_like(px).fillna(False)
    # sleeve rank composites
    R = {}
    for c in ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score"):
        R[c] = ens.pivot_table(index="trade_date", columns="symbol", values=c).rank(axis=1, pct=True)
    comp = {"w210": (2 * R["short_5d_score"] + 1 * R["mid_5d_30d_score"]).reindex_like(px),
            "w111": (R["short_5d_score"] + R["mid_5d_30d_score"] + R["long_30d_120d_score"]).reindex_like(px)}
    return panel, sector, px, fwd1d, adv20, vol20, elig, comp


def build_book(score: pd.DataFrame, elig: pd.DataFrame, *, top_k=10, buffer_rank=None,
               rebalance_every=1, weight="equal", max_w=1.0, vol=None, adv=None) -> pd.DataFrame:
    """Stateful daily target weights with rank-buffer + frequency + weighting."""
    dates = list(score.index)
    buffer_rank = buffer_rank or top_k
    held: list[str] = []
    out = pd.DataFrame(0.0, index=dates, columns=score.columns)
    for di, d in enumerate(dates):
        s = score.loc[d].where(elig.loc[d]).dropna()
        if di % rebalance_every == 0 and len(s):
            ranked = s.sort_values(ascending=False)
            in_buf = set(ranked.index[:buffer_rank])
            keep = [h for h in held if h in in_buf]
            slots = max(0, top_k - len(keep))
            adds = [x for x in ranked.index[:top_k] if x not in keep][:slots]
            held = keep + adds
        if not held:
            continue
        if weight == "equal":
            w = pd.Series(1.0, index=held)
        elif weight == "score":
            w = s.reindex(held).clip(lower=1e-6).fillna(1e-6)
        elif weight == "voladj" and vol is not None:
            iv = (1.0 / vol.loc[d].reindex(held)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            w = iv.clip(lower=1e-6)
        elif weight == "advcap" and adv is not None:
            a = adv.loc[d].reindex(held).fillna(adv.loc[d].median())
            w = a.clip(lower=1e-6)
        else:
            w = pd.Series(1.0, index=held)
        w = w / w.sum()
        w = w.clip(upper=max_w)
        w = w / w.sum()
        out.loc[d, w.index] = w.values
    return out


def proxy_metrics(weights: pd.DataFrame, fwd1d: pd.DataFrame, start, end, slip_bps) -> dict:
    w = weights[(weights.index >= pd.Timestamp(start)) & (weights.index <= pd.Timestamp(end))]
    if w.empty or (w.abs().sum().sum() == 0):
        return {"cagr": -1, "maxDD": 1, "sharpe": 0, "calmar": 0, "turnover": 0, "total": -1}
    f = fwd1d.reindex(index=w.index, columns=w.columns).fillna(0.0)
    book_ret = (w * f).sum(axis=1)
    dturn = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())  # total weight changed
    cost = dturn * (slip_bps + 5.0) / 1e4  # slip + (comm+~half stamp) per unit traded
    net = (book_ret - cost).dropna()
    if len(net) < 5:
        return {"cagr": -1, "maxDD": 1, "sharpe": 0, "calmar": 0, "turnover": 0, "total": -1}
    nav = (1 + net).cumprod()
    n = len(net); total = float(nav.iloc[-1] - 1)
    cagr = float(nav.iloc[-1] ** (ANN / n) - 1)
    dd = float((nav / nav.cummax() - 1).min())
    sharpe = float(net.mean() / net.std() * np.sqrt(ANN)) if net.std() > 0 else 0.0
    return {"cagr": round(cagr, 4), "total": round(total, 4), "maxDD": round(dd, 4),
            "sharpe": round(sharpe, 2), "calmar": round(cagr / abs(dd), 2) if dd < 0 else 0.0,
            "turnover": round(float(dturn.mean() / 2), 4)}


def strict_at(weights: pd.DataFrame, panel, sector, start, end, slip) -> dict:
    di = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    w = weights[(weights.index >= pd.Timestamp(start)) & (weights.index <= pd.Timestamp(end))]
    w = w.loc[:, (w != 0).any(axis=0)]
    if w.empty:
        return {}
    pos = di.searchsorted(w.index) + 1  # t+1
    keep = pos < len(di); tw = w.iloc[keep]; tw.index = di[pos[keep]]
    sim = panel[(panel["trade_date"] >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (panel["trade_date"] <= pd.Timestamp(end) + pd.Timedelta(days=10))]
    res = run_strict_backtest_v8(tw, sim, sector_map=sector, config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=float(slip)))
    m = res.metrics
    return {"cagr": round(m.annualized_return, 4), "total": round(m.total_return, 4), "maxDD": round(m.max_drawdown, 4),
            "sharpe": round(m.sharpe, 3), "calmar": round(m.calmar, 3),
            "turnover": round(float(tw.diff().abs().sum(axis=1).mean() / 2), 4)}


def main() -> int:
    panel, sector, px, fwd1d, adv20, vol20, elig, comp = load()
    print("loaded.", flush=True)
    rows_buf, rows_freq, rows_tw, rows_hyb, rows_net = [], [], [], [], []

    # base books for reference
    base = {"w210": build_book(comp["w210"], elig, top_k=10),
            "w111": build_book(comp["w111"], elig, top_k=5)}

    # 1. rebalance buffer / no-trade band (proxy, val+non2026+2026 at 8/30bps)
    for bk, k in (("w210", 10), ("w111", 5)):
        for br in (k, 15, 20, 30, 50):
            wts = build_book(comp[bk], elig, top_k=k, buffer_rank=br)
            r = {"book": bk, "buffer_rank": br}
            for slip in (8, 30):
                for win in ("non2026", "y2026"):
                    m = proxy_metrics(wts, fwd1d, *WIN[win], slip)
                    r[f"{win}_{slip}bps_cagr"] = m["cagr"]; r[f"{win}_turnover"] = m["turnover"]
            rows_buf.append(r)
    pd.DataFrame(rows_buf).to_csv(OUT / "stage4_rebalance_buffer_results.csv", index=False)
    print("buffer done", flush=True)

    # 2. rebalance frequency
    for bk, k in (("w210", 10), ("w111", 5)):
        for fr in (1, 2, 3, 5):
            wts = build_book(comp[bk], elig, top_k=k, buffer_rank=max(k, 20), rebalance_every=fr)
            r = {"book": bk, "rebalance_every": fr}
            for slip in SLIPS:
                m = proxy_metrics(wts, fwd1d, *WIN["y2026"], slip)
                r[f"y2026_{slip}bps_cagr"] = m["cagr"]
            r["turnover"] = proxy_metrics(wts, fwd1d, *WIN["y2026"], 8)["turnover"]
            rows_freq.append(r)
    pd.DataFrame(rows_freq).to_csv(OUT / "stage4_frequency_results.csv", index=False)
    print("frequency done", flush=True)

    # 3. cost-aware top-k and weighting
    for bk in ("w210", "w111"):
        for k in (5, 7, 10, 15, 20):
            for wsch in ("equal", "score", "voladj", "advcap"):
                wts = build_book(comp[bk], elig, top_k=k, buffer_rank=max(k, 20), weight=wsch, max_w=0.15, vol=vol20, adv=adv20)
                r = {"book": bk, "top_k": k, "weight": wsch}
                for slip in (8, 30, 50):
                    m = proxy_metrics(wts, fwd1d, *WIN["y2026"], slip)
                    r[f"y2026_{slip}bps_cagr"] = m["cagr"]
                r["turnover"] = proxy_metrics(wts, fwd1d, *WIN["y2026"], 8)["turnover"]
                rows_tw.append(r)
    pd.DataFrame(rows_tw).to_csv(OUT / "stage4_topk_weight_results.csv", index=False)
    print("topk/weight done", flush=True)

    # 4. hybrid book (blend the two base books' weights)
    bw210 = build_book(comp["w210"], elig, top_k=10, buffer_rank=20)
    bw111 = build_book(comp["w111"], elig, top_k=5, buffer_rank=20)
    for a in (1.0, 0.75, 0.5, 0.25, 0.0):
        hyb = (a * bw210 + (1 - a) * bw111)
        r = {"alloc_w210": a}
        for slip in SLIPS:
            m = proxy_metrics(hyb, fwd1d, *WIN["y2026"], slip)
            r[f"y2026_{slip}bps_cagr"] = m["cagr"];
        r["turnover"] = proxy_metrics(hyb, fwd1d, *WIN["y2026"], 8)["turnover"]
        rows_hyb.append(r)
    pd.DataFrame(rows_hyb).to_csv(OUT / "stage4_hybrid_results.csv", index=False)
    print("hybrid done", flush=True)

    # 5. net-score: tilt away from illiquid names (liquidity penalty via ADV rank)
    illiq_rank = (-adv20).rank(axis=1, pct=True)  # high = illiquid
    for bk, k in (("w210", 10), ("w111", 5)):
        for pen in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            netscore = comp[bk].rank(axis=1, pct=True) - pen * illiq_rank.reindex_like(comp[bk])
            wts = build_book(netscore, elig, top_k=k, buffer_rank=max(k, 20))
            r = {"book": bk, "liquidity_penalty": pen}
            for slip in (8, 30, 50):
                m = proxy_metrics(wts, fwd1d, *WIN["y2026"], slip)
                r[f"y2026_{slip}bps_cagr"] = m["cagr"]
            r["turnover"] = proxy_metrics(wts, fwd1d, *WIN["y2026"], 8)["turnover"]
            rows_net.append(r)
    pd.DataFrame(rows_net).to_csv(OUT / "stage4_netscore_results.csv", index=False)
    print("netscore done", flush=True)

    # ---- assemble finalists & STRICT-confirm ----
    # finalists chosen by proxy 2026 net CAGR @30bps from buffer + freq + hybrid
    finalists = {
        "w210_base_k10": build_book(comp["w210"], elig, top_k=10),
        "w210_buf20": build_book(comp["w210"], elig, top_k=10, buffer_rank=20),
        "w210_buf30": build_book(comp["w210"], elig, top_k=10, buffer_rank=30),
        "w210_buf20_freq3": build_book(comp["w210"], elig, top_k=10, buffer_rank=20, rebalance_every=3),
        "w111_base_k5": build_book(comp["w111"], elig, top_k=5),
        "w111_buf20": build_book(comp["w111"], elig, top_k=5, buffer_rank=20),
        "w111_buf20_freq3": build_book(comp["w111"], elig, top_k=5, buffer_rank=20, rebalance_every=3),
        "hybrid_50_buf20": 0.5 * build_book(comp["w210"], elig, top_k=10, buffer_rank=20) + 0.5 * build_book(comp["w111"], elig, top_k=5, buffer_rank=20),
    }
    lb = []
    for name, wts in finalists.items():
        row = {"strategy": name}
        for slip in SLIPS:
            m = strict_at(wts, panel, sector, *WIN["y2026"], slip)
            row[f"2026_{slip}bps_cagr"] = m.get("cagr")
        mh = strict_at(wts, panel, sector, *WIN["non2026"], 30)
        row["non2026_30bps_cagr"] = mh.get("cagr")
        m8 = strict_at(wts, panel, sector, *WIN["y2026"], 8)
        row["turnover"] = m8.get("turnover"); row["2026_8bps_maxDD"] = m8.get("maxDD"); row["2026_8bps_sharpe"] = m8.get("sharpe")
        lb.append(row)
        print(f"  {name}: 2026 8bps {row['2026_8bps_cagr']} 30bps {row['2026_30bps_cagr']} 50bps {row['2026_50bps_cagr']} | turnover {row['turnover']}", flush=True)
    lbdf = pd.DataFrame(lb).sort_values("2026_30bps_cagr", ascending=False)
    lbdf.to_csv(OUT / "stage4_cost_robust_leaderboard.csv", index=False)

    # best by 30bps -> save orders/positions
    best = lbdf.iloc[0]["strategy"]
    bw = finalists[best]
    bw.loc[(bw.index >= '2026-01-02')].reset_index().rename(columns={"index": "trade_date"}).melt(
        id_vars="trade_date", var_name="symbol", value_name="weight").query("weight>1e-9").to_parquet(OUT / "stage4_best_config_positions.parquet", index=False)
    rbest = strict_at(bw, panel, sector, *WIN["y2026"], 30)

    # ---- capacity curve (best config) ----
    pos2026 = bw.loc[(bw.index >= '2026-01-02') & (bw.index <= '2026-05-13')]
    adv_win = adv20.reindex(index=pos2026.index, columns=pos2026.columns)
    cap = []
    turn_daily = float(pos2026.diff().abs().sum(axis=1).mean())  # total weight traded/day
    base8 = strict_at(bw, panel, sector, *WIN["y2026"], 8).get("cagr")
    for size in (1e6, 3e6, 5e6, 1e7, 3e7, 5e7, 1e8):
        # per-name daily traded notional = size * (weight change); participation = traded / ADV
        traded_notional = size * turn_daily  # approx daily
        part = (size * pos2026 * (pos2026.diff().abs() > 0)).div(adv_win).replace([np.inf, -np.inf], np.nan)
        avg_part = float(part.stack().mean()) if part.notna().any().any() else np.nan
        max_part = float(part.stack().quantile(0.95)) if part.notna().any().any() else np.nan
        # square-root impact model: impact_bps ~ 10 * sqrt(participation) (per trade)
        impact_bps = 10.0 * np.sqrt(max(avg_part, 0)) * 100 if avg_part == avg_part else np.nan
        eff_slip = 8 + (impact_bps if impact_bps == impact_bps else 0)
        mc = strict_at(bw, panel, sector, *WIN["y2026"], min(eff_slip, 150))
        cap.append({"size_rmb": size, "avg_participation": round(avg_part, 4) if avg_part == avg_part else None,
                    "p95_participation": round(max_part, 4) if max_part == max_part else None,
                    "est_impact_bps": round(impact_bps, 1) if impact_bps == impact_bps else None,
                    "eff_slippage_bps": round(min(eff_slip, 150), 1), "net_cagr": mc.get("cagr"), "maxDD": mc.get("maxDD")})
        print(f"  size {size:.0e}: part {cap[-1]['avg_participation']} impact {cap[-1]['est_impact_bps']}bps -> net CAGR {cap[-1]['net_cagr']}", flush=True)
    pd.DataFrame(cap).to_csv(OUT / "stage4_capacity_curve.csv", index=False)

    # ---- report ----
    lines = ["# Stage 4 — Cost/turnover/capacity robust daily book", "",
             "STRICT-confirmed finalists (2026 net CAGR by slippage; non2026 @30bps):", "",
             "| strategy | 8bps | 15bps | 30bps | 50bps | 100bps | non2026@30 | turnover |", "|---|---|---|---|---|---|---|---|"]
    for _, r in lbdf.iterrows():
        lines.append(f"| {r['strategy']} | {r['2026_8bps_cagr']} | {r['2026_15bps_cagr']} | {r['2026_30bps_cagr']} | "
                     f"{r['2026_50bps_cagr']} | {r['2026_100bps_cagr']} | {r['non2026_30bps_cagr']} | {r['turnover']} |")
    lines += ["", f"Best by 2026@30bps: **{best}** (strict 30bps CAGR {rbest.get('cagr')}, maxDD {rbest.get('maxDD')}, turnover {rbest.get('turnover')})", "",
              "## Capacity curve (best config)", "", "| size RMB | avg part | impact bps | eff slip | net CAGR |", "|---|---|---|---|---|"]
    for c in cap:
        lines.append(f"| {c['size_rmb']:.0e} | {c['avg_participation']} | {c['est_impact_bps']} | {c['eff_slippage_bps']} | {c['net_cagr']} |")
    (OUT / "stage4_cost_robust_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote artifacts to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
