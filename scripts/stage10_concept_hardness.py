#!/usr/bin/env python3
"""Stage 10 step 2 — 概念硬度 (concept-hardness) in-concept stock scoring (live).

For the currently strongest sub-concepts, pulls constituents and scores every
member on the user's 0-20×4 − 20 hardness rubric using PIT-able batch data:

  概念纯度 0-20   board membership + 市值纯度 proxy (refine w/ 主营 for core picks)
  订单验证 0-20   *flagged 待公告核实* (honest: not faked; verified per pick later)
  业绩兑现 0-20   营收/净利同比 + 毛利率 + 业绩预告 (yjbb/yjyg batch)
  资金量价 0-20   主力净流入 + 60日相对强弱 + 量比 + 换手
  风险扣分 0..-20  高PE/PB + 净利负增长 + 概念不纯

Outputs per stock: hardness score, 短线波段分 / 中线趋势分 / 业绩兑现分, a role
{核心龙头/中军/弹性/补涨/伪概念}, the 不买理由 for low-hardness names, and an
in-concept ranking. Batch endpoints cached so reruns are instant.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.concept.taxonomy import board_to_leaves  # noqa: E402

RAW = Path("runtime/stage10_concept/raw")
BATCH = RAW / "batch"
CONS = RAW / "cons"
OUT = Path("runtime/stage10_concept")
TOP_CONCEPTS = 20


def _ak():
    import akshare as ak
    return ak


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").replace({"-": np.nan, "": np.nan}), errors="coerce")


def fetch_batches():
    ak = _ak()
    BATCH.mkdir(parents=True, exist_ok=True)
    res = {}
    specs = {
        "spot": (ak.stock_zh_a_spot_em, {}),
        "flow_today": (ak.stock_individual_fund_flow_rank, {"indicator": "今日"}),
        "flow_5d": (ak.stock_individual_fund_flow_rank, {"indicator": "5日"}),
        "yjbb_2026q1": (ak.stock_yjbb_em, {"date": "20260331"}),
        "yjyg_2026h1": (ak.stock_yjyg_em, {"date": "20260630"}),
    }
    for name, (fn, kw) in specs.items():
        f = BATCH / f"{name}.parquet"
        if f.exists():
            res[name] = pd.read_parquet(f); continue
        try:
            df = fn(**kw)
            # stringify object cols to dodge arrow '-' errors, then save
            df.astype({c: str for c in df.columns if df[c].dtype == object}).to_parquet(f)
            res[name] = pd.read_parquet(f)
            print(f"  batch {name}: {df.shape}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  batch {name} ERR {repr(e)[:90]}")
            res[name] = pd.DataFrame()
    return res


def fetch_constituents(boards):
    ak = _ak(); CONS.mkdir(parents=True, exist_ok=True)
    out = {}
    for b in boards:
        f = CONS / f"{b.replace('/', '_')}.parquet"
        if f.exists():
            out[b] = pd.read_parquet(f); continue
        try:
            c = ak.stock_board_concept_cons_em(symbol=b)
            c.astype({col: str for col in c.columns if c[col].dtype == object}).to_parquet(f)
            out[b] = pd.read_parquet(f); time.sleep(0.3)
        except Exception as e:
            print(f"  cons {b} ERR {repr(e)[:70]}")
    return out


def main():
    strength = pd.read_csv(OUT / "concept_strength.csv")
    strong = strength[~strength["state"].isin(["弱势", "退潮"])].head(TOP_CONCEPTS)
    boards = strong["board"].tolist()
    print(f"[hardness] scoring inside {len(boards)} strong concepts: {boards[:8]}...")

    cons = fetch_constituents(boards)
    B = fetch_batches()
    spot = B["spot"]; spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    for c in ["最新价", "涨跌幅", "量比", "换手率", "市盈率-动态", "市净率", "总市值", "60日涨跌幅", "年初至今涨跌幅"]:
        if c in spot: spot[c] = _num(spot[c])
    spot = spot.set_index("代码")

    flow = B["flow_today"]
    if not flow.empty:
        flow["代码"] = flow["代码"].astype(str).str.zfill(6)
        fcol = [c for c in flow.columns if "主力净流入" in c and "占比" in c]
        flow["mainflow_pct"] = _num(flow[fcol[0]]) if fcol else np.nan
        flow = flow.set_index("代码")["mainflow_pct"]
    else:
        flow = pd.Series(dtype=float)

    yj = B["yjbb_2026q1"]
    if not yj.empty:
        yj["代码"] = yj["股票代码"].astype(str).str.zfill(6) if "股票代码" in yj else yj.iloc[:, 1].astype(str).str.zfill(6)
        rev = [c for c in yj.columns if "营业总收入" in c and "同比" in c]
        prof = [c for c in yj.columns if "净利润" in c and "同比" in c]
        gm = [c for c in yj.columns if "销售毛利率" in c]
        yj_f = pd.DataFrame({"代码": yj["代码"]})
        yj_f["rev_yoy"] = _num(yj[rev[0]]) if rev else np.nan
        yj_f["profit_yoy"] = _num(yj[prof[0]]) if prof else np.nan
        yj_f["gross_margin"] = _num(yj[gm[0]]) if gm else np.nan
        yj_f = yj_f.drop_duplicates("代码").set_index("代码")
    else:
        yj_f = pd.DataFrame(columns=["rev_yoy", "profit_yoy", "gross_margin"])

    yg = B["yjyg_2026h1"]
    yg_map = {}
    if not yg.empty:
        yg["代码"] = (yg["股票代码"] if "股票代码" in yg else yg.iloc[:, 1]).astype(str).str.zfill(6)
        tcol = [c for c in yg.columns if "预告" in c and ("类型" in c or "指标" in c)]
        if tcol:
            yg_map = yg.drop_duplicates("代码").set_index("代码")[tcol[0]].to_dict()

    b2l = board_to_leaves()
    rows = []
    for b in boards:
        c = cons.get(b)
        if c is None or c.empty:
            continue
        c["代码"] = c["代码"].astype(str).str.zfill(6)
        leaf = b2l[b][0]
        n_members = len(c)
        for _, m in c.iterrows():
            code = m["代码"]; name = m.get("名称", "")
            sp = spot.loc[code] if code in spot.index else None
            cap = float(sp["总市值"]) if sp is not None and pd.notna(sp["总市值"]) else np.nan
            ret60 = float(sp["60日涨跌幅"]) if sp is not None and pd.notna(sp["60日涨跌幅"]) else np.nan
            ytd = float(sp["年初至今涨跌幅"]) if sp is not None and pd.notna(sp["年初至今涨跌幅"]) else np.nan
            lb = float(sp["量比"]) if sp is not None and pd.notna(sp["量比"]) else np.nan
            to = float(sp["换手率"]) if sp is not None and pd.notna(sp["换手率"]) else np.nan
            pe = float(sp["市盈率-动态"]) if sp is not None and pd.notna(sp["市盈率-动态"]) else np.nan
            pb = float(sp["市净率"]) if sp is not None and pd.notna(sp["市净率"]) else np.nan
            mf = float(flow.get(code, np.nan)) if len(flow) else np.nan
            rev = float(yj_f.loc[code, "rev_yoy"]) if code in yj_f.index else np.nan
            pft = float(yj_f.loc[code, "profit_yoy"]) if code in yj_f.index else np.nan
            gm = float(yj_f.loc[code, "gross_margin"]) if code in yj_f.index else np.nan
            yg_t = yg_map.get(code, "")
            rows.append(dict(board=b, industry=leaf.industry, track=leaf.track,
                             segment=leaf.segment, position=leaf.position, n_members=n_members,
                             code=code, name=name, mktcap=cap, ret60=ret60, ytd=ytd, liangbi=lb,
                             turnover=to, pe=pe, pb=pb, mainflow_pct=mf, rev_yoy=rev,
                             profit_yoy=pft, gross_margin=gm, yj_forecast=yg_t))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no constituents scored"); return 1

    # ---------------- 概念硬度 scoring (0-20 x4 − 20) ----------------
    def clip(x, lo, hi): return np.clip(x, lo, hi)
    # 纯度: base 10 for being in the fine sub-concept board; +market-cap purity proxy
    cap_rank = df.groupby("board")["mktcap"].rank(pct=True)           # big = 中军, small = 弹性/纯
    purity = 10.0 + clip((0.5 - (cap_rank - 0.5).abs()) * 8, -2, 6) + 4.0  # mid-small favored, all real members get >=10
    df["score_purity"] = clip(purity, 0, 20)
    # 订单: HONEST placeholder — needs 公告核实; neutral 8, flagged
    df["score_order"] = 8.0
    df["order_status"] = "待公告核实"
    # 业绩: rev_yoy + profit_yoy + gross_margin + 预告
    perf = (clip(df["rev_yoy"].fillna(0) / 5.0, -4, 8) + clip(df["profit_yoy"].fillna(0) / 10.0, -6, 10)
            + clip((df["gross_margin"].fillna(20) - 20) / 5.0, -2, 4))
    perf = perf + df["yj_forecast"].fillna("").apply(lambda t: 3.0 if any(k in str(t) for k in ("预增", "扭亏", "续盈")) else (-3.0 if any(k in str(t) for k in ("预减", "首亏", "续亏")) else 0.0))
    df["score_perf"] = clip(perf + 8.0, 0, 20)
    # 资金量价: mainflow + 60d RS + 量比 + 换手
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    cap_liq = (clip(df["mainflow_pct"].fillna(0) * 2, -6, 8) + clip(df["ret60"].fillna(0) / 8, -4, 8)
               + clip((df["liangbi"].fillna(1) - 1) * 2, -2, 4))
    df["score_capital"] = clip(cap_liq + 8.0, 0, 20)
    # 风险扣分
    risk = (clip((df["pe"].fillna(40) - 60) / 20, 0, 6).fillna(0)          # PE>60 penalized
            + clip((df["pb"].fillna(5) - 8) / 3, 0, 4)
            + (df["profit_yoy"].fillna(0) < -20).astype(float) * 4         # earnings collapse
            + (df["score_purity"] < 10).astype(float) * 3)
    df["score_risk"] = -clip(risk, 0, 20)
    df["hardness"] = df["score_purity"] + df["score_order"] + df["score_perf"] + df["score_capital"] + df["score_risk"]
    # sub-scores
    df["短线波段分"] = df["score_capital"] + clip((df["liangbi"].fillna(1) - 1) * 3, 0, 6)
    df["中线趋势分"] = df["score_perf"] * 0.6 + clip(df["ret60"].fillna(0) / 5, -4, 8)
    df["业绩兑现分"] = df["score_perf"]

    # ---------------- role classification ----------------
    def role(r):
        capr = df[df.board == r["board"]]["mktcap"].rank(pct=True).loc[r.name]
        weak = (r["score_perf"] < 6) and (r["score_capital"] < 6) and (r["score_purity"] < 11)
        if weak:
            return "伪概念"
        if r["score_perf"] >= 11 and r["score_capital"] >= 11 and capr >= 0.6:
            return "核心龙头"
        if capr >= 0.75:
            return "中军"
        if (r["ret60"] is not None) and (r["ret60"] < df[df.board == r["board"]]["ret60"].median()) and r["score_perf"] >= 8:
            return "补涨"
        return "弹性"
    df["role"] = df.apply(role, axis=1)
    df["不买理由"] = df.apply(lambda r: (
        "伪概念:无业绩+无资金+纯度低" if r["role"] == "伪概念" else
        ("业绩不兑现(净利大幅负增)" if (pd.notna(r["profit_yoy"]) and r["profit_yoy"] < -20) else
         ("估值过高(PE/PB极端)" if r["score_risk"] < -6 else ""))), axis=1)

    df = df.sort_values(["board", "hardness"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUT / "concept_hardness.csv", index=False)
    print(f"[write] {OUT/'concept_hardness.csv'}  ({len(df)} stocks across {df.board.nunique()} concepts)")

    # ---- print: per top concept, top hardness stocks ----
    cols = ["name", "role", "ret60", "mainflow_pct", "rev_yoy", "profit_yoy", "pe",
            "score_purity", "score_perf", "score_capital", "score_risk", "hardness", "不买理由"]
    for b in boards[:6]:
        sub = df[df.board == b].head(6)
        if sub.empty:
            continue
        st = strong[strong.board == b].iloc[0]
        print(f"\n=== {b} [{st['industry']}/{st['segment']}] state={st['state']} strength={st['strength']:.1f} ===")
        show = sub[cols].copy()
        for c in ["ret60", "mainflow_pct", "rev_yoy", "profit_yoy"]:
            show[c] = show[c].round(1)
        for c in ["score_purity", "score_perf", "score_capital", "score_risk", "hardness"]:
            show[c] = show[c].round(1)
        with pd.option_context("display.width", 230, "display.max_columns", 30, "display.unicode.east_asian_width", True):
            print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
