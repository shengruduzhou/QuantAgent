#!/usr/bin/env python3
"""Stage 10.2c — mine real 主营业务 vocabulary to grow concept keywords (with provenance).

The purity keyword map fails when 东财 names revenue segments by business-line
(CRO -> 化学业务/测试业务) instead of the concept word. This tool fetches
stock_zygc_em for the top-hardness stocks of each strong concept, extracts the
actual 主营构成 segment names + revenue shares, and aggregates them per concept
so keyword additions are grounded in real data (segment text + example stock +
revenue share) — never hand-guessed.

Output: business_vocab_mined.csv with columns concept, segment, n_stocks,
mean_share, examples. Segments that recur across >=2 pure-play members with
material share are printed as keyword suggestions to add to CONCEPT_KEYWORDS.
Fetches are cached per stock and fail-soft (throttle safe).
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.concept import purity as puritydb  # noqa: E402

ROOT = Path("runtime/stage10_concept")
SNAPS = ROOT / "snapshots"
ZYGC_CACHE = ROOT / "raw" / "zygc"
TOP_PER_CONCEPT = 8
N_CONCEPTS = 25
THROTTLE_AFTER = 4

# generic segment words to ignore as keywords (too broad / non-specific)
_STOP = ("其他", "其它", "境内", "境外", "国内", "国外", "补充", "合计", "主营业务",
         "产品", "服务", "业务", "销售", "制造业", "行业", "地区", "分部")


def _latest_snap():
    ds = sorted(d.name for d in SNAPS.glob("*") if (d / "concept_hardness.csv").exists())
    return ds[-1] if ds else None


def _clean_segment(s: str) -> str:
    s = re.sub(r"\([^)]*\)|（[^）]*）", "", str(s)).strip()   # drop English parens
    return s


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else _latest_snap()
    if not date:
        print("no snapshot"); return 1
    ZYGC_CACHE.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(SNAPS / date / "concept_hardness.csv", dtype={"code": str})
    h["code"] = h["code"].str.zfill(6)
    strength = pd.read_csv(SNAPS / date / "concept_strength.csv")
    concepts = strength.head(N_CONCEPTS)["board"].tolist()

    seg_stats: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    fails = 0; throttled = False
    for board in concepts:
        top = h[h["board"] == board].sort_values("hardness", ascending=False).head(TOP_PER_CONCEPT)
        for _, r in top.iterrows():
            code, name = r["code"], r["name"]
            f = ZYGC_CACHE / f"{code}.parquet"
            if f.exists():
                zygc = pd.read_parquet(f)
            elif throttled:
                continue
            else:
                zygc = puritydb.fetch_zygc(code, allow_network=True)
                if zygc is None or zygc.empty:
                    fails += 1
                    if fails >= THROTTLE_AFTER:
                        throttled = True
                    continue
                fails = 0
                zygc.astype({c: str for c in zygc.columns if zygc[c].dtype == object}).to_parquet(f)
                time.sleep(0.4)
            # pick 按产品分类 (else 按行业分类), latest period
            if "分类类型" in zygc.columns:
                for v in ("按产品分类", "按行业分类"):
                    if (zygc["分类类型"].astype(str) == v).any():
                        zygc = zygc[zygc["分类类型"].astype(str) == v]; break
            dcol = next((c for c in zygc.columns if "日期" in c or "报告期" in c), None)
            if dcol is not None:
                zygc = zygc[zygc[dcol].astype(str) == zygc[dcol].astype(str).max()]
            scol = next((c for c in zygc.columns if "主营构成" in c or "项目" in c), None)
            pcol = next((c for c in zygc.columns if "收入比例" in c), None)
            if scol is None or pcol is None:
                continue
            for _, row in zygc.iterrows():
                seg = _clean_segment(row[scol])
                if not seg or any(w in seg for w in _STOP):
                    continue
                share = pd.to_numeric(str(row[pcol]).replace("%", ""), errors="coerce")
                share = share / 100 if share and share > 1.5 else share
                seg_stats[board][seg].append((name, float(share) if pd.notna(share) else 0.0))

    rows = []
    for board, segs in seg_stats.items():
        for seg, hits in segs.items():
            shares = [s for _, s in hits]
            rows.append({"concept": board, "segment": seg, "n_stocks": len(hits),
                         "mean_share": round(sum(shares) / len(shares), 3),
                         "examples": ", ".join(sorted({n for n, _ in hits})[:4])})
    out = pd.DataFrame(rows).sort_values(["concept", "n_stocks", "mean_share"], ascending=[True, False, False])
    out.to_csv(ROOT / "business_vocab_mined.csv", index=False)
    print(f"[mine] {len(out)} segment rows across {out['concept'].nunique() if len(out) else 0} concepts "
          f"(throttled={throttled}) -> {ROOT/'business_vocab_mined.csv'}")
    # keyword suggestions: segment recurs in >=2 members with material share
    sug = out[(out["n_stocks"] >= 2) & (out["mean_share"] >= 0.15)]
    print("\n=== keyword suggestions (>=2 members, mean_share>=15%) ===")
    for board, grp in sug.groupby("concept"):
        segs = grp["segment"].tolist()
        print(f"  {board}: {segs}  (e.g. {grp.iloc[0]['examples']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
