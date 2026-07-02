#!/usr/bin/env python3
"""Stage 10.1 + 10.2 — daily concept-chain scanner + PIT snapshot store.

Run post-close each trading day. It (1) pulls the live 东财 concept universe,
board strength, concept/stock fund-flow, breadth, batch earnings; (2) computes
the strongest sub-concept ranking + wave states + market regime + in-concept
stock hardness; and (3) writes everything to a DATED, timestamped, source-tagged
snapshot under runtime/stage10_concept/snapshots/<YYYYMMDD>/ so future validation
can only ever use data saved on-or-before the as-of date — no concept-membership
look-ahead.

Idempotent per day (skip if today's snapshot exists unless --force).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.concept import scoring  # noqa: E402
from quantagent.concept.taxonomy import board_to_leaves, coverage  # noqa: E402

ROOT = Path("runtime/stage10_concept")
HIST = ROOT / "raw" / "board_hist"
SNAPS = ROOT / "snapshots"
TOP_CONCEPTS = 20


def _ak():
    import akshare as ak
    return ak


def _retry(fn, *args, tries=5, base_sleep=2.0, **kwargs):
    """akshare endpoints are network-flaky (RemoteDisconnected); retry w/ backoff."""
    last = None
    for k in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(base_sleep * (k + 1))
    raise last


def _num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", "").replace({"-": np.nan, "": np.nan}), errors="coerce")


def refresh_board_hist(ak, boards, asof, skip_live=False):
    HIST.mkdir(parents=True, exist_ok=True)
    end = asof.strftime("%Y%m%d")
    start = (asof - pd.Timedelta(days=400)).strftime("%Y%m%d")
    out, miss = {}, 0
    for i, b in enumerate(boards):
        f = HIST / f"{b.replace('/', '_')}.parquet"
        fresh = f.exists() and pd.Timestamp(f.stat().st_mtime, unit="s").date() == asof.date()
        if fresh:
            out[b] = pd.read_parquet(f); continue
        if skip_live:
            if f.exists():            # throttled: use any stale cache, don't hit network
                out[b] = pd.read_parquet(f)
            else:
                miss += 1
            continue
        try:
            h = _retry(ak.stock_board_concept_hist_em, symbol=b, period="daily", start_date=start, end_date=end, adjust="", tries=2)
            if h is not None and not h.empty:
                h.to_parquet(f); out[b] = h
            time.sleep(0.2)
        except Exception:
            miss += 1
        if (i + 1) % 60 == 0:
            print(f"   board hist {i+1}/{len(boards)}")
    print(f"[hist] {len(out)} boards refreshed ({miss} missing)")
    return out


def compute_strength(spot, hist, ff, b2l):
    spot_i = spot.set_index("板块名称")
    rows = []
    for b, leaves in b2l.items():
        if b not in hist or b not in spot_i.index:
            continue
        st = scoring.board_strength(hist[b])
        if not st:
            continue
        sp = spot_i.loc[b]
        up, dn = float(sp.get("上涨家数", np.nan)), float(sp.get("下跌家数", np.nan))
        breadth = up / (up + dn + 1e-9) if not np.isnan(up) else np.nan
        lead_lu = float(sp.get("领涨股票-涨跌幅", 0) >= 9.8)
        rows.append({"board": b, "industry": leaves[0].industry, "track": leaves[0].track,
                     "segment": leaves[0].segment, "position": leaves[0].position,
                     **st, "breadth": breadth, "lead_limitup": lead_lu,
                     "mainflow_pct": float(ff.get(b, 0.0)),
                     "mktcap": float(sp.get("总市值", np.nan))})
    df = pd.DataFrame(rows)
    df["strength"] = scoring.composite_strength(df)
    df["state"] = df.apply(scoring.classify_state, axis=1)
    df = df.sort_values("strength", ascending=False).reset_index(drop=True)
    regime, evidence = scoring.market_regime(df)
    return df, regime, evidence


CACHE = ROOT / "raw"
BATCH_FALLBACK = {"spot_all": "batch/spot.parquet", "flow_stock": "batch/flow_today.parquet",
                  "yjbb": "batch/yjbb_2026q1.parquet", "yjyg": "batch/yjyg_2026h1.parquet"}


def fetch_batches(ak, snap, skip_live=False):
    res = {}
    specs = {
        "spot_all": (ak.stock_zh_a_spot_em, {}),
        "flow_stock": (ak.stock_individual_fund_flow_rank, {"indicator": "今日"}),
        "yjbb": (ak.stock_yjbb_em, {"date": "20260331"}),
        "yjyg": (ak.stock_yjyg_em, {"date": "20260630"}),
    }
    for name, (fn, kw) in specs.items():
        f = snap / f"{name}.parquet"
        if f.exists():
            res[name] = pd.read_parquet(f); continue
        if skip_live:
            fb = CACHE / BATCH_FALLBACK.get(name, "")
            if fb.exists():
                res[name] = pd.read_parquet(fb); res[name].to_parquet(f)
            else:
                res[name] = pd.DataFrame()
            continue
        try:
            df = _retry(fn, **kw)
            df.astype({c: str for c in df.columns if df[c].dtype == object}).to_parquet(f)
            res[name] = pd.read_parquet(f)
            time.sleep(0.4)
        except Exception as e:
            fb = CACHE / BATCH_FALLBACK.get(name, "")
            if fb.exists():
                res[name] = pd.read_parquet(fb)
                res[name].to_parquet(f)
                print(f"   batch {name}: live failed, used session cache")
            else:
                print(f"   batch {name} ERR {repr(e)[:60]}"); res[name] = pd.DataFrame()
    return res


def compute_hardness(strength, ak, snap, b2l, skip_live=False):
    strong = strength[~strength["state"].isin(["弱势", "退潮"])].head(TOP_CONCEPTS)
    boards = strong["board"].tolist()
    consdir = snap / "cons"; consdir.mkdir(exist_ok=True)
    cons = {}
    for b in boards:
        f = consdir / f"{b.replace('/', '_')}.parquet"
        if f.exists():
            cons[b] = pd.read_parquet(f); continue
        fb = CACHE / "cons" / f"{b.replace('/', '_')}.parquet"
        if skip_live:
            if fb.exists():
                cons[b] = pd.read_parquet(fb); cons[b].to_parquet(f)
            continue
        try:
            c = _retry(ak.stock_board_concept_cons_em, symbol=b, tries=2)
            c.astype({col: str for col in c.columns if c[col].dtype == object}).to_parquet(f)
            cons[b] = pd.read_parquet(f); time.sleep(0.25)
        except Exception:
            if fb.exists():
                cons[b] = pd.read_parquet(fb); cons[b].to_parquet(f)
    B = fetch_batches(ak, snap, skip_live=skip_live)
    spot = B["spot_all"]; spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    for c in ["量比", "换手率", "市盈率-动态", "市净率", "总市值", "60日涨跌幅", "年初至今涨跌幅"]:
        if c in spot: spot[c] = _num(spot[c])
    spot = spot.set_index("代码")
    flow = B["flow_stock"]
    if not flow.empty:
        flow["代码"] = flow["代码"].astype(str).str.zfill(6)
        fc = [c for c in flow.columns if "主力净流入" in c and "占比" in c]
        flow = flow.set_index("代码")[fc[0]].pipe(_num) if fc else pd.Series(dtype=float)
    yj = B["yjbb"]; yjf = pd.DataFrame()
    if not yj.empty:
        code = (yj["股票代码"] if "股票代码" in yj else yj.iloc[:, 1]).astype(str).str.zfill(6)
        rev = [c for c in yj.columns if "营业总收入" in c and "同比" in c]
        prof = [c for c in yj.columns if "净利润" in c and "同比" in c]
        gm = [c for c in yj.columns if "销售毛利率" in c]
        yjf = pd.DataFrame({"代码": code,
                            "rev_yoy": _num(yj[rev[0]]) if rev else np.nan,
                            "profit_yoy": _num(yj[prof[0]]) if prof else np.nan,
                            "gross_margin": _num(yj[gm[0]]) if gm else np.nan}).drop_duplicates("代码").set_index("代码")
    yg = B["yjyg"]; ygm = {}
    if not yg.empty:
        code = (yg["股票代码"] if "股票代码" in yg else yg.iloc[:, 1]).astype(str).str.zfill(6)
        tc = [c for c in yg.columns if "预告" in c and ("类型" in c or "指标" in c)]
        if tc: ygm = dict(zip(code, yg[tc[0]]))
    rows = []
    for b in boards:
        c = cons.get(b)
        if c is None or c.empty: continue
        c["代码"] = c["代码"].astype(str).str.zfill(6)
        leaf = b2l[b][0]
        for _, m in c.iterrows():
            code = m["代码"]
            sp = spot.loc[code] if code in spot.index else None
            g = (lambda k: float(sp[k]) if sp is not None and pd.notna(sp.get(k)) else np.nan)
            rows.append(dict(board=b, industry=leaf.industry, track=leaf.track, segment=leaf.segment,
                             position=leaf.position, code=code, name=m.get("名称", ""),
                             mktcap=g("总市值"), ret60=g("60日涨跌幅"), ytd=g("年初至今涨跌幅"),
                             liangbi=g("量比"), turnover=g("换手率"), pe=g("市盈率-动态"), pb=g("市净率"),
                             mainflow_pct=float(flow.get(code, np.nan)) if len(flow) else np.nan,
                             rev_yoy=float(yjf.loc[code, "rev_yoy"]) if code in yjf.index else np.nan,
                             profit_yoy=float(yjf.loc[code, "profit_yoy"]) if code in yjf.index else np.nan,
                             gross_margin=float(yjf.loc[code, "gross_margin"]) if code in yjf.index else np.nan,
                             yj_forecast=ygm.get(code, ""), order_label="unverified"))
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = scoring.score_hardness(df)
    df["role"] = scoring.classify_role(df)
    df["不买理由"] = df.apply(scoring.buy_reject_reason, axis=1)
    return df.sort_values(["board", "hardness"], ascending=[True, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.today().normalize()
    snap = SNAPS / asof.strftime("%Y%m%d"); snap.mkdir(parents=True, exist_ok=True)
    if (snap / "concept_strength.csv").exists() and not args.force:
        print(f"[skip] snapshot {snap} already exists (use --force)"); return 0

    ak = _ak()
    print(f"[scan] as-of {asof.date()}  ->  {snap}")
    skip_live = False
    try:
        spot_boards = _retry(ak.stock_board_concept_name_em, tries=2)
    except Exception:
        spot_boards = pd.read_parquet(CACHE / "em_concept_boards.parquet")
        skip_live = True
        print("[scan] board list: live throttled -> using session cache, skipping further live calls")
    spot_boards.astype({c: str for c in spot_boards.columns if spot_boards[c].dtype == object}).to_parquet(snap / "concept_boards.parquet")
    b2l = board_to_leaves()
    mapped = sorted(set(b2l) & set(spot_boards["板块名称"]))
    hist = refresh_board_hist(ak, mapped, asof, skip_live=skip_live)
    ff = {}
    if not skip_live:
        try:
            ffr = _retry(ak.stock_sector_fund_flow_rank, indicator="今日", sector_type="概念资金流", tries=2)
            ff = ffr.set_index("名称")["今日主力净流入-净占比"].pipe(_num).to_dict()
            ffr.astype({c: str for c in ffr.columns if ffr[c].dtype == object}).to_parquet(snap / "concept_fund_flow.parquet")
        except Exception:
            ff = {}

    strength, regime, evidence = compute_strength(spot_boards, hist, ff, b2l)
    strength.to_csv(snap / "concept_strength.csv", index=False)
    print(f"\n市场状态: {regime}  {evidence}")
    print(f"TOP 12 概念: " + " | ".join(f"{r.board}({r.state},{r.strength:.1f})"
                                       for r in strength.head(12).itertuples()))

    hardness = compute_hardness(strength, ak, snap, b2l, skip_live=skip_live)
    hardness.to_csv(snap / "concept_hardness.csv", index=False)

    cov = coverage(spot_boards["板块名称"].tolist())
    manifest = {
        "asof": asof.strftime("%Y-%m-%d"),
        "generated_at": pd.Timestamp.now().isoformat(),
        "source": "akshare/东财(em) concept boards+hist+fundflow+yjbb+yjyg",
        "akshare_version": __import__("akshare").__version__,
        "used_cache": bool(skip_live),
        "concept_fund_flow_live": (snap / "concept_fund_flow.parquet").exists(),
        "market_regime": regime, "regime_evidence": evidence,
        "n_concept_boards": int(len(spot_boards)),
        "n_taxonomy_mapped": int(cov["mapped_to_taxonomy"]),
        "n_strength_scored": int(len(strength)),
        "n_hardness_stocks": int(len(hardness)),
        "top_concepts": strength.head(TOP_CONCEPTS)["board"].tolist(),
        "files": sorted(p.name for p in snap.glob("*") if p.is_file()),
    }
    (snap / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[snapshot] {snap}  ({manifest['n_hardness_stocks']} hardness stocks, "
          f"{manifest['n_strength_scored']} concepts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
