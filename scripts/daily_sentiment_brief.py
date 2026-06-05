#!/usr/bin/env python3
"""每日舆情短线推断 — daily news-sentiment brief + per-stock overlay signal.

Pipeline (designed to run pre-open via cron):
  1. universe = 人气榜 top-N (or --symbols)
  2. per symbol: fetch 个股新闻, lexicon sentiment, aggregate by 申万 sector
  3. LLM short-term (3-5d) inference over the sector sentiment + top movers
  4. emit markdown brief + a per-symbol sentiment_overlay parquet (sentiment_score,
     attention, short_term_bias) usable as a tactical overlay on the factor pool.

Outputs: runtime/reports/daily/sentiment_brief_<date>.md + sentiment_overlay_<date>.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

# reuse the lexicon from the evidence fetcher
import importlib.util
_spec = importlib.util.spec_from_file_location("_fns", str(Path(__file__).parent / "fetch_news_sentiment.py"))
_fns = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fns)
_sent, _code6 = _fns._sent, _fns._code6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", help="comma list; default = 人气榜 top-N")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--lookback-days", type=int, default=3)
    ap.add_argument("--as-of", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--sector-map", default="runtime/data/v7/silver/sector_map/sector_map.parquet")
    ap.add_argument("--no-llm", action="store_true", help="skip LLM inference (lexicon only)")
    ap.add_argument("--out-dir", type=Path, default=Path("runtime/reports/daily"))
    args = ap.parse_args()

    import akshare as ak
    sector = pd.read_parquet(args.sector_map)[["symbol", "sector_level_1"]].set_index("symbol")["sector_level_1"].to_dict()

    universe = []
    if args.symbols:
        universe = [(s.strip(), None) for s in args.symbols.split(",") if s.strip()]
    else:
        for _attempt in range(3):  # 人气榜 endpoint is intermittently flaky
            try:
                hot = ak.stock_hot_rank_em()
                if hot is not None and not hot.empty and "代码" in hot.columns:
                    universe = [(f"{r['代码']}", int(r['当前排名'])) for _, r in hot.head(args.top_n).iterrows()]
                    break
            except Exception:
                continue
    if not universe:  # fallback to a liquid大盘 watchlist so the job never empties out
        universe = [(s, None) for s in ["600519.SH", "300750.SZ", "601318.SH", "000858.SZ",
                    "600036.SH", "002594.SZ", "601012.SH", "600900.SH", "000333.SZ", "601899.SH"]]

    cutoff = pd.Timestamp(args.as_of) - pd.Timedelta(days=args.lookback_days)
    rows = []
    for sym, rk in universe:
        try:
            news = ak.stock_news_em(symbol=_code6(sym))
        except Exception:
            continue
        if news is None or news.empty:
            continue
        news = news.copy(); news["t"] = pd.to_datetime(news.get("发布时间"), errors="coerce")
        news = news[news["t"].notna() & (news["t"] >= cutoff)]
        if news.empty:
            continue
        s = float(news.apply(lambda r: _sent(str(r.get("新闻标题", "")) + " " + str(r.get("新闻内容", ""))), axis=1).mean())
        sec = sector.get(sym if "." in str(sym) else None) or sector.get(_code6(sym)) or sector.get(f"{_code6(sym)}.SZ") or sector.get(f"{_code6(sym)}.SH")
        rows.append({"symbol": sym, "sector_level_1": sec, "sentiment_score": s,
                     "n_news": int(len(news)), "hot_rank": rk,
                     "attention": (1.0 - rk / 100.0) if rk else 0.0,
                     "short_term_bias": round(0.6 * s + 0.4 * ((1.0 - rk / 100.0) if rk else 0.0), 3)})
    ov = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ov_path = args.out_dir / f"sentiment_overlay_{args.as_of}.parquet"
    ov.to_parquet(ov_path, index=False)

    by_sector = (ov.groupby("sector_level_1")["sentiment_score"].mean().sort_values(ascending=False)
                 if not ov.empty else pd.Series(dtype=float))
    llm_brief = ""
    if not args.no_llm and not ov.empty:
        from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
        c = LLMSkillClient(LLMSkillConfig.from_env())
        payload = {"as_of": args.as_of,
                   "sector_sentiment": {k: round(v, 3) for k, v in by_sector.head(12).items()},
                   "top_positive": ov.nlargest(8, "sentiment_score")[["symbol", "sector_level_1", "sentiment_score", "hot_rank"]].to_dict("records"),
                   "top_negative": ov.nsmallest(6, "sentiment_score")[["symbol", "sector_level_1", "sentiment_score"]].to_dict("records")}
        res = c.invoke("daily_sentiment_inference",
            system_prompt="你是A股短线舆情分析师。基于今日新闻情绪做3-5日短线推断。只输出JSON。",
            user_text=("根据以下今日舆情，给出 JSON: {\"market_tone\":\"risk_on/neutral/risk_off\","
                       "\"hot_sectors\":[板块],\"cooling_sectors\":[板块],\"short_term_view\":\"两三句\","
                       "\"caution\":\"过热/接盘风险提示\"}。数据: " + json.dumps(payload, ensure_ascii=False)),
            fallback={})
        if not res.used_fallback:
            llm_brief = res.output

    md = [f"# 每日舆情短线推断 — {args.as_of}", "",
          f"覆盖 {len(ov)} 只(人气榜top{args.top_n}) · 回看{args.lookback_days}日新闻", ""]
    if isinstance(llm_brief, dict) and llm_brief:
        md += [f"**市场情绪**: {llm_brief.get('market_tone','-')}",
               f"**短线观点**: {llm_brief.get('short_term_view','-')}",
               f"**升温板块**: {llm_brief.get('hot_sectors',[])}  |  **降温板块**: {llm_brief.get('cooling_sectors',[])}",
               f"**风险提示**: {llm_brief.get('caution','-')}", ""]
    md += ["## 板块情绪排序", "", "| 板块 | 平均情绪 |", "|---|---|"]
    for k, v in by_sector.head(12).items():
        md.append(f"| {k} | {v:+.3f} |")
    md += ["", "## 个股短线信号 top10", "", "| symbol | 板块 | 情绪 | 人气 | short_bias |", "|---|---|---|---|---|"]
    for _, r in ov.sort_values("short_term_bias", ascending=False).head(10).iterrows():
        md.append(f"| {r['symbol']} | {r.get('sector_level_1','')} | {r['sentiment_score']:+.2f} | {r.get('hot_rank','-')} | {r['short_term_bias']:+.2f} |")
    md += ["", "> 用法：舆情=个股短线择时/避险overlay，不做主选股因子（避免追高接盘）。"]
    brief_path = args.out_dir / f"sentiment_brief_{args.as_of}.md"
    brief_path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {brief_path} ({len(ov)} stocks) + {ov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
