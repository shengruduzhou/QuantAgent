#!/usr/bin/env python3
"""Fetch 个股新闻舆情 (news sentiment) into canonical evidence, to be COMBINED
with red-header policy so the LLM decides on policy(方向) + sentiment(确认/时机).

Source: akshare ``stock_news_em`` per symbol + ``stock_hot_rank_em`` (人气榜 =
attention). Sentiment is lexicon-scored (fast, deterministic); the final LLM
sees the summarized news and makes the call. Emits a canonical EvidenceRecord
parquet (source_type=news, sentiment_score, entities=[symbol, sector:X]) that
can be concatenated with policy canonical and passed via --canonical-evidence-path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd

POS = ["利好", "增长", "创新高", "大涨", "突破", "中标", "签约", "扩产", "提价", "超预期",
       "盈利", "回购", "增持", "合作", "获批", "订单", "放量", "涨停", "龙头", "景气",
       "补贴", "扶持", "政策支持", "预增", "新高", "拿下", "量产", "放榜", "复苏"]
NEG = ["利空", "下跌", "亏损", "大跌", "跌停", "减持", "质押", "违约", "处罚", "问询",
       "退市", "下滑", "预亏", "商誉减值", "诉讼", "风险", "暴跌", "套牢", "造假",
       "警示", "立案", "停牌", "爆雷", "下修", "裁员", "调查"]


def _sent(text: str) -> float:
    t = str(text or "")
    p = sum(t.count(w) for w in POS)
    n = sum(t.count(w) for w in NEG)
    if p + n == 0:
        return 0.0
    return (p - n) / (p + n)


def _code6(sym: str) -> str:
    return str(sym).split(".")[0].zfill(6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbols")
    g.add_argument("--symbols-from", type=Path)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--max-symbols", type=int, default=60)
    ap.add_argument("--sector-map", default="runtime/data/v7/silver/sector_map/sector_map.parquet")
    ap.add_argument("--as-of", default=dt.date.today().strftime("%Y-%m-%d"))
    ap.add_argument("--output", type=Path, default=Path("runtime/data/v7/raw/news/news_canonical.parquet"))
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        d = pd.read_parquet(args.symbols_from) if args.symbols_from.suffix == ".parquet" else pd.read_csv(args.symbols_from)
        symbols = d["symbol"].astype(str).unique().tolist()
    symbols = symbols[: args.max_symbols]
    sector = pd.read_parquet(args.sector_map)[["symbol", "sector_level_1"]].set_index("symbol")["sector_level_1"].to_dict()

    import akshare as ak
    try:
        hot = ak.stock_hot_rank_em()
        hot_rank = {f"{r['代码']}": int(r['当前排名']) for _, r in hot.iterrows()} if "代码" in hot.columns else {}
    except Exception:
        hot_rank = {}

    cutoff = pd.Timestamp(args.as_of) - pd.Timedelta(days=args.lookback_days)
    rows = []
    for sym in symbols:
        try:
            news = ak.stock_news_em(symbol=_code6(sym))
        except Exception:
            continue
        if news is None or news.empty:
            continue
        news = news.copy()
        news["t"] = pd.to_datetime(news.get("发布时间"), errors="coerce")
        news = news[news["t"].notna() & (news["t"] >= cutoff)]
        if news.empty:
            continue
        scores = news.apply(lambda r: _sent(str(r.get("新闻标题", "")) + " " + str(r.get("新闻内容", ""))), axis=1)
        senti = float(scores.mean())
        n = int(len(news))
        sec = sector.get(sym)
        # attention: top-of-hot-rank -> positive capital-flow tilt
        rk = hot_rank.get(_code6(sym))
        attention = (1.0 - rk / 100.0) if rk else 0.0
        ents = [sym] + ([f"sector:{sec}"] if sec else [])
        title = f"{sym} 近{args.lookback_days}日 {n}条新闻 情绪={senti:+.2f} 热度排名={rk or '-'}"
        rows.append({
            "evidence_id": "news_" + hashlib.sha256(f"{sym}{args.as_of}".encode()).hexdigest()[:16],
            "source_name": "akshare:stock_news_em", "source_type": "news",
            "url_or_file_id": None, "publish_time": pd.Timestamp(args.as_of),
            "crawl_time": pd.Timestamp(args.as_of), "available_at": pd.Timestamp(args.as_of),
            "entity_type": "news_sentiment", "entities": ents, "raw_text_hash": None,
            "extracted_claims": [title], "sentiment_score": senti,
            "policy_direction_score": 0.0, "capital_flow_direction_score": float(max(-1, min(1, attention))),
            "confidence": float(min(1.0, 0.3 + 0.05 * n)), "contradiction_score": 0.0,
            "lag_window_candidates": [1, 3, 5], "audit_trace": {"adapter": "news_sentiment", "n_news": n, "hot_rank": rk},
        })
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    pos = int((out["sentiment_score"] > 0.1).sum()) if len(out) else 0
    neg = int((out["sentiment_score"] < -0.1).sum()) if len(out) else 0
    print(f"wrote {len(out)} news-sentiment records -> {args.output} (pos={pos}, neg={neg})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
