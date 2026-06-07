#!/usr/bin/env python3
"""产业链深度研究 Agent — 舆情/事件全景 → 多产业链(上中下游) → 财报/估值深挖 → 选最优环节.

实现用户的"深挖"工作流（不预设主题；agent 自动从 PIT 新闻里识别当期真正在发酵的多个
政策/产业/事件催化，再逐链深挖）:

  Stage 0 (DATA, PIT): 读取截至 as_of 的真实新闻联播标题(+可选投行研报标题)，作为
                 lookahead-isolated 的舆情/事件来源（信息严格限定 as_of 之前）。
  Stage 1 (LLM): PIT 新闻 → 当期事件全景(分类) + 大盘/风格研判 + 6-8 条产业链，
                 每条含上中下游各 3-5 只代表 A 股(代码+名称+环节角色+概念+逻辑)。
  Stage 2 (DATA): 对【整个】产业链股池打因子分(alpha_score)，并对优先子集拉真实财报
                 (毛利率/归母净利增长率/ROE/营收增长率) + PE 估值（全部 PIT）。
  Stage 3 (LLM): 用真实数字逐链比较利润质量/估值/景气，选出最优环节与重点个股(产业逻辑)。
  Stage 4: 渲染详尽的研报(舆情全景 → 逐链深度 → 风格 → 最终股池 → 风险)。
  Stage 5: UNION(因子池, LLM产业链池) → 可配置混合权重 → 最终股池(较大)。

Output: runtime/reports/monthly/chain_research_<date>.md + chain_pool_<date>.parquet
        + chain_meta_<date>.json (events/chains/picks，供实验与回测复用)

PIT 纪律: Stage0 的新闻是按日期历史回放(news_cctv 支持历史)，Stage1/3 的 system_prompt
严令"绝不能使用 as_of 之后的任何信息或后见之明"。残余风险=LLM 参数记忆里的 hindsight，
由 scripts/chain_oos_validation.py 的 news-ablation 做对照检验。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time as _time
from pathlib import Path

import pandas as pd


def _code6(sym: str) -> str:
    return str(sym).split(".")[0].zfill(6)


def _norm_name(s: str) -> str:
    """Normalize a stock name for matching: drop all whitespace, full-width→half-width A."""
    return "".join(str(s or "").split()).replace("Ａ", "A").replace("Ｂ", "B").strip()


def _load_name2code(ak, cache: str = "runtime/data/v7/silver/code_name_map.parquet") -> dict:
    """Authoritative {normalized_name -> 6-digit code}. LLM hallucinates ~80% of codes, so
    name-authoritative resolution MUST be reliable AND fast.

    CACHE-FIRST: the A-share code↔name map changes rarely (new listings only), while the live
    akshare/eastmoney endpoint THROTTLES and HANGS under heavy use (an unbounded
    ``ak.stock_info_a_code_name()`` once hung the whole run for >200s). So we use a valid
    on-disk cache immediately and only hit the network — timeout-bounded — when the cache is
    missing/too small. Returns {} only if both are unavailable (caller aborts, never trusts LLM codes)."""
    p = Path(cache)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            if len(df) > 4000:
                return {_norm_name(n): _code6(c) for c, n in zip(df["code"].astype(str), df["name"].astype(str))}
        except Exception:
            pass
    # cache miss → rebuild from TICKFLOW instruments (per "all data via tickflow"); timeout-bounded
    try:
        inst = _with_timeout(lambda: _tf_provider()._ensure_all_instruments(), 60.0) or []
        rows = [{"code": _code6(i.get("symbol", "")), "name": str(i.get("name", "")).strip()}
                for i in inst if i.get("symbol") and i.get("name")]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[df["name"].str.len() > 0].drop_duplicates("code")
        if len(df) > 4000:
            try:
                p.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(p, index=False)
            except Exception:
                pass
            return {_norm_name(n): _code6(c) for c, n in zip(df["code"], df["name"])}
    except Exception:
        pass
    if p.exists():  # tickflow flaky → use persisted cache
        df = pd.read_parquet(p)
        return {_norm_name(n): _code6(c) for c, n in zip(df["code"].astype(str), df["name"].astype(str))}
    return {}


def _report_available(period_end: pd.Timestamp, as_of: pd.Timestamp) -> bool:
    """PIT gate: a 报告期 is only knowable at as_of after its regulatory disclosure
    deadline (CSRC). Q1→Apr30, semi→Aug31, Q3→Oct31, annual(12-31)→Apr30 NEXT year.
    This fixes a real lookahead leak: an annual report period (12-31) is NOT public
    until ~Apr of the following year, so Jan/Feb/Mar as-of dates must fall back to Q3."""
    y, m = period_end.year, period_end.month
    if m == 3:
        deadline = pd.Timestamp(y, 4, 30)
    elif m == 6:
        deadline = pd.Timestamp(y, 8, 31)
    elif m == 9:
        deadline = pd.Timestamp(y, 10, 31)
    elif m == 12:
        deadline = pd.Timestamp(y + 1, 4, 30)
    else:
        deadline = period_end + pd.Timedelta(days=60)
    return as_of >= deadline


_FIN_CACHE = Path("runtime/data/v7/silver/fin_snapshot_cache")
_TF_PROVIDER = None


def _tf_provider():
    """Lazy singleton TickflowProvider (paid endpoint). Per user mandate ALL fundamental
    data goes through tickflow — akshare financials were incomplete for key names
    (长电科技600584 / 通富微电002156 先进封装链) and threw throttling hangs."""
    global _TF_PROVIDER
    if _TF_PROVIDER is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(".env", override=False)
        except Exception:
            pass
        from quantagent.data.providers.tickflow_provider import TickflowProvider
        _TF_PROVIDER = TickflowProvider(allow_network=True,
                                        api_endpoint=os.environ.get("TICKFLOW_API_ENDPOINT") or None)
    return _TF_PROVIDER


def _ttm_from_ytd(rows: pd.DataFrame, col: str) -> float | None:
    """TTM of a YTD-cumulative A-share line (eps/net income). For the latest period in year Y
    month M: annual rows (M=12) are already TTM; else TTM = ytd(Y,M) + annual(Y-1) − ytd(Y-1,M)."""
    if rows is None or rows.empty or col not in rows.columns:
        return None
    d = rows.dropna(subset=[col]).copy()
    if d.empty:
        return None
    d["period_end"] = pd.to_datetime(d["period_end"], errors="coerce")
    d = d.dropna(subset=["period_end"]).sort_values("period_end")
    if d.empty:
        return None
    last = d.iloc[-1]
    y, m, val = last["period_end"].year, last["period_end"].month, float(last[col])
    if m == 12:
        return val
    by = {(r["period_end"].year, r["period_end"].month): float(r[col]) for _, r in d.iterrows()}
    pa, py = by.get((y - 1, 12)), by.get((y - 1, m))
    return (val + pa - py) if (pa is not None and py is not None) else None


def _fin_snapshot(symbol: str, as_of: str, close: float | None = None) -> dict:
    """PIT fundamentals via TICKFLOW. The true disclosure gate is tickflow's ``announce_date``
    (no CSRC-deadline guessing). metrics gives 毛利/净利率/ROE/营收增速/净利增速 directly;
    PE = close / TTM-EPS. Disk-cached by (code, as_of) — PIT result is immutable."""
    code = _code6(symbol)
    cf = _FIN_CACHE / f"{code}_{as_of}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            pass
    out = {"gross_margin": None, "net_profit_yoy": None, "roe": None, "rev_yoy": None,
           "net_margin": None, "pe": None, "report_period": None}
    asof_ts = pd.Timestamp(as_of)
    prov = _tf_provider()
    try:
        m = _with_timeout(lambda: prov.financials_metrics(symbol), 20.0)
        if m is not None and not getattr(m, "empty", True) and "announce_date" in m.columns:
            m = m.copy(); m["announce_date"] = pd.to_datetime(m["announce_date"], errors="coerce")
            m = m[m["announce_date"] <= asof_ts].sort_values("period_end")  # PIT
            if not m.empty:
                r = m.iloc[-1]
                def _v(k):
                    v = pd.to_numeric(r.get(k), errors="coerce")
                    return round(float(v), 2) if pd.notna(v) else None
                out["gross_margin"] = _v("gross_margin")
                out["net_margin"] = _v("net_margin")
                out["roe"] = _v("roe")
                out["rev_yoy"] = _v("revenue_yoy")
                out["net_profit_yoy"] = _v("net_income_yoy")
                out["report_period"] = str(pd.to_datetime(r["period_end"]).date())
    except Exception:
        pass
    try:  # PE = PIT close / TTM-EPS (income basic_eps is YTD-cumulative)
        if close is not None and float(close) > 0:
            inc = _with_timeout(lambda: prov.financials_income(symbol), 20.0)
            if inc is not None and not getattr(inc, "empty", True) and "announce_date" in inc.columns:
                inc = inc.copy(); inc["announce_date"] = pd.to_datetime(inc["announce_date"], errors="coerce")
                inc = inc[inc["announce_date"] <= asof_ts]
                eps_col = next((c for c in ("basic_eps", "diluted_eps") if c in inc.columns), None)
                ttm_eps = _ttm_from_ytd(inc, eps_col) if eps_col else None
                if ttm_eps is not None:
                    # loss-makers have negative TTM-EPS → PE is mathematically undefined; show 亏损
                    # (data IS found — the company is losing money — not a missing-data "-").
                    out["pe"] = round(float(close) / ttm_eps, 1) if ttm_eps > 0 else "亏损"
        # fallback: TTM uncomputable but the metrics row already proves a loss → still mark 亏损
        if out["pe"] is None and isinstance(out.get("net_margin"), (int, float)) and out["net_margin"] < 0:
            out["pe"] = "亏损"
    except Exception:
        pass
    try:
        _FIN_CACHE.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return out


def _with_timeout(fn, timeout: float):
    """Run fn() with a hard wall-clock timeout (akshare endpoints can hang/throttle).
    Returns None on timeout/error; the stuck worker thread is abandoned (daemon)."""
    import concurrent.futures as _cf
    ex = _cf.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn).result(timeout=timeout)
    except Exception:
        return None
    finally:
        ex.shutdown(wait=False)


_NEWS_CACHE = Path("runtime/data/v7/silver/news_cache")


def _news_cctv_cached(ak, dstr: str, timeout: float = 20.0):
    """PIT news_cctv for one day, on-disk cached (historical news is immutable) + timeout-bounded."""
    p = _NEWS_CACHE / f"cctv_{dstr}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    nc = _with_timeout(lambda: ak.news_cctv(date=dstr), timeout)
    if nc is not None and not getattr(nc, "empty", True):
        try:
            _NEWS_CACHE.mkdir(parents=True, exist_ok=True)
            nc.to_parquet(p, index=False)
        except Exception:
            pass
    return nc


def _pit_news(ak, as_of: str, lookback_days: int, per_day: int, pit_strict: bool,
              news_mode: str = "real") -> list[str]:
    """Point-in-time舆情/事件来源. news_cctv 支持历史回放(PIT-safe). 在非严格模式下(live)
    额外并入投行研报标题(按 publishDate<=as_of 过滤). 返回去重后的'日期: 标题'行。

    news_mode (供 OOS hindsight 对照):
      real     — 截至 as_of 的真实 PIT 新闻(默认)。
      nonews   — 返回空(LLM 只能靠参数记忆 → 若仍有 edge=hindsight 嫌疑)。
      scrambled— 用 as_of 之前一年的错位窗口新闻(若错的新闻也产生同样 edge → 新闻非驱动)。"""
    if news_mode == "nonews":
        return []
    lines: list[str] = []
    end = pd.Timestamp(as_of)
    if news_mode == "scrambled":
        end = end - pd.Timedelta(days=365)  # 错位一年: 内容与当期事件无关
    for d in pd.bdate_range(end - pd.Timedelta(days=lookback_days), end):
        nc = _news_cctv_cached(ak, d.strftime("%Y%m%d"))
        if nc is not None and not getattr(nc, "empty", True) and "title" in nc.columns:
            lines += [f"{d.date()} [联播] {t}" for t in nc["title"].head(per_day).tolist()]
    # 投行研报标题(行业级): 仅在非PIT-strict(live)用 live 接口，避免历史回放不可得导致的泄漏
    if not pit_strict:
        try:
            br = _with_timeout(lambda: ak.stock_research_report_em(), 15.0)  # 最新研报，仅 live 模式
            if br is None:
                raise RuntimeError("broker reports timeout")
            dcol = next((c for c in br.columns if "日期" in str(c) or "date" in str(c).lower()), None)
            tcol = next((c for c in br.columns if "标题" in str(c) or "title" in str(c).lower()), None)
            if dcol and tcol:
                br = br.copy(); br[dcol] = pd.to_datetime(br[dcol], errors="coerce")
                br = br[br[dcol] <= end].sort_values(dcol).tail(40)
                lines += [f"{r[dcol].date()} [研报] {r[tcol]}" for _, r in br.iterrows()]
        except Exception:
            pass
    # dedup keep order, keep the most recent window
    seen, dedup = set(), []
    for ln in lines:
        key = ln.split("] ", 1)[-1]
        if key not in seen:
            seen.add(key); dedup.append(ln)
    return dedup


def _bond_macro() -> dict:
    """Optional资金面快照 (PIT-light: latest available)."""
    try:
        bf = pd.read_parquet("runtime/data/v7/silver/bond_flows/bond_flows.parquet")
        last = bf.sort_values(bf.columns[0]).iloc[-1]
        return {k: round(float(last[k]), 2) for k in bf.columns if "yield" in k or "spread" in k and pd.notna(last[k])}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default="2026-04-30")
    ap.add_argument("--predictions", default="runtime/tmp/real_preds_20260430.parquet")
    ap.add_argument("--max-stocks-enrich", type=int, default=40, help="拉财报的优先股票数(网络较慢)")
    ap.add_argument("--news-lookback-days", type=int, default=60)
    ap.add_argument("--news-per-day", type=int, default=14)
    ap.add_argument("--n-chains", type=int, default=6, help="LLM 映射的产业链条数")
    ap.add_argument("--stocks-per-segment", type=int, default=4, help="每个上/中/下游环节的代表股数")
    ap.add_argument("--n-factor", type=int, default=20, help="并集中的因子池名额")
    ap.add_argument("--n-chain", type=int, default=15, help="并集中的产业链池名额")
    ap.add_argument("--factor-weight", type=float, default=0.6)
    ap.add_argument("--chain-weight", type=float, default=0.4)
    ap.add_argument("--pit-strict", dest="pit_strict", action="store_true", default=True,
                    help="只用历史可回放的PIT新闻源(默认开, 供OOS)")
    ap.add_argument("--live", dest="pit_strict", action="store_false",
                    help="live模式: 额外并入最新投行研报(仅当 as_of≈今天)")
    ap.add_argument("--news-mode", choices=["real", "nonews", "scrambled"], default="real",
                    help="OOS hindsight 对照: real/nonews(纯参数记忆)/scrambled(错位新闻)")
    ap.add_argument("--out-suffix", default="", help="输出文件名后缀(供 ablation 并存, 如 _nonews)")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=os.environ.get("QUANTAGENT_CHAIN_MODEL", ""),
                    help="Override LLM model. EMPTY (default) = use the configured gemma thinking model "
                         "(strongest reasoning — preferred for quality). gemma's read-timeouts are handled "
                         "by per-chain chunking + backoff, NOT by downgrading. Pass gemini-2.5-flash only "
                         "for fast smoke runs where depth doesn't matter.")
    ap.add_argument("--fast-model", default=os.environ.get("QUANTAGENT_CHAIN_FAST_MODEL", "gemini-2.5-flash"),
                    help="Model for the MECHANICAL chain_stocks enumeration (theme→上中下游A股名单). flash "
                         "is fine here (not reasoning) and avoids gemma's per-minute-quota read-timeouts that "
                         "truncated reports. Reasoning (events/picks) still uses --model (gemma).")
    ap.add_argument("--out-dir", type=Path, default=Path("runtime/reports/monthly"))
    args = ap.parse_args()

    from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
    base_client = LLMSkillClient(LLMSkillConfig.from_env())
    # 分工 (用户定): 真推理(事件分析/个股深度精选=产业逻辑) 用 gemma 保质量; 机械枚举
    # (chain_stocks: 把主题列成上中下游A股名单) 用 flash — gemma 连续大调用被每分钟token配额限流,
    # chain_stocks 反复 360s 读超时削薄研报; flash 几秒返回不超时, 而枚举名单非推理, flash 足够好.
    _is_thinking = any(k in args.model for k in ("gemma", "pro", "thinking")) if args.model else True
    _ov = {"max_input_chars": 28000, "model": args.model} if args.model else {"max_input_chars": 28000}
    # gemma healthy ≈ 43s; a call still silent at 360s = transient overload → fail fast + long backoff.
    _ov["timeout_seconds"] = 360.0 if _is_thinking else 90.0
    client = base_client.with_overrides(**_ov)                                    # 推理: gemma
    fast_client = base_client.with_overrides(model=args.fast_model, timeout_seconds=90, max_input_chars=28000)  # 枚举: flash
    # 枚举多模型兜底: 一个 flash 503/429 就换下一个 — 4 个 flash 变体几乎不可能同时过载 → 链数稳定为 n_chains.
    FAST_MODELS = list(dict.fromkeys([args.fast_model, "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-flash-latest"]))
    print(f"LLM reasoning={args.model or base_client.config.model} (thinking={_is_thinking}); "
          f"enumeration={args.fast_model} (fallbacks={FAST_MODELS[1:]})", flush=True)

    def _invoke_retry(skill, *, system_prompt, user_text, fallback, tries=3, use_client=None, models=None):
        """Retry transient errors (429/503/read-timeout) with long backoff. use_client routes a
        mechanical skill (chain_stocks) to the fast model; models=[...] cycles alternate models per
        attempt (一个过载就换下一个) so chain count stays stable."""
        cl = use_client or client
        n = max(tries, len(models)) if models else tries
        r = None
        for t in range(n):
            this = cl.with_overrides(model=models[t % len(models)]) if models else cl
            r = this.invoke(skill, system_prompt=system_prompt, user_text=user_text, fallback=fallback)
            if not r.used_fallback:
                return r
            mdl = models[t % len(models)] if models else (cl.config.model)
            print(f"  [{skill}] attempt {t+1}/{n} ({mdl}) fallback: {r.fallback_reason}", flush=True)
            if t < n - 1:
                _time.sleep(8.0 * (t + 1) if models else 15.0 * (t + 1))  # shorter backoff when switching models
        return r
    sm = pd.read_parquet("runtime/data/v7/silver/sector_map/sector_map.parquet")[["symbol", "sector_level_1"]]
    universe = set(sm["symbol"])
    code2sym = {_code6(s): s for s in universe}
    code2sector = {_code6(s): sec for s, sec in zip(sm["symbol"], sm["sector_level_1"])}
    preds = pd.read_parquet(args.predictions)
    alpha = {(_code6(s)): v for s, v in zip(preds["symbol"], preds["prediction"])}

    import akshare as ak
    # Authoritative {normalized name -> code}. The LLM hallucinates ~80% of codes, so we
    # resolve by NAME only and DROP anything that doesn't resolve (never trust LLM codes).
    name2code = _load_name2code(ak)
    if len(name2code) < 4000:
        print(f"FATAL: authoritative code-name map too small ({len(name2code)}); refuse to run — "
              "would otherwise trust hallucinated LLM codes. Check network / akshare.")
        return 2
    print(f"name2code: {len(name2code)} authoritative names loaded")

    def _resolve(name: str, code: str = "") -> str | None:
        """NAME-authoritative ONLY. Return the real code, or None (caller drops the stock).
        We deliberately ignore the LLM-emitted code: it is wrong ~80% of the time."""
        nm = _norm_name(name)
        if nm in name2code:
            return name2code[nm]
        # tight fuzzy for suffix diffs (A股/Ⅱ/股份); require ≥3 chars and ≤1 length delta
        if nm and len(nm) >= 3:
            for k, v in name2code.items():
                if (nm in k or k in nm) and abs(len(nm) - len(k)) <= 1:
                    return v
        return None

    # ---- Stage 0: PIT 舆情/事件发现 (信息严格限定 as_of 之前) ----
    news = _pit_news(ak, args.as_of, args.news_lookback_days, args.news_per_day, args.pit_strict, args.news_mode)
    news_blob = "\n".join(news[-400:])
    print(f"Stage0: PIT news lines up to {args.as_of}: {len(news)} (strict={args.pit_strict}, mode={args.news_mode})")

    # ---- Stage 1: split into small calls (thinking-model latency scales with output size,
    #      a single big JSON of events+6 chains×stocks exceeds the socket timeout). ----
    sps = args.stocks_per_segment
    # 1a: events全景 + 大盘研判 + 主题(无个股) — 小输出, 快
    s1a = _invoke_retry("chain_events",
        system_prompt=(f"你是顶级A股产业链研究专家与基金经理。你只知道{args.as_of}及之前的信息，"
                       "绝不能使用之后发生的任何信息、行情或后见之明。只输出一个JSON对象。"),
        user_text=(f"以下是截至{args.as_of}的真实PIT新闻(新闻联播/研报标题)。请：\n"
                   "① 识别此刻【真正在发酵或仍然有效】的重要催化(政策/产业/数据/事件)，分类，尽量全(12-16条)。"
                   "不要只看最近几天：凡是【过去发生但影响仍在持续/尚未兑现】的也要算——长周期政策(十五五/产业扶持)、"
                   "长尾产业趋势(AI算力/先进封装/光模块/商业航天/国产替代)、以及【供应链传导型催化】"
                   "(例：海外大厂或龙头动作→其A股供应链受益，如英伟达Rubin机柜→国产PCB/铜连接/液冷/先进封装(长电/通富)；"
                   "SpaceX/星链IPO进展→卫星材料/特种合金/滤波器(再升/西部材料等))。impact 里写清传导逻辑与受益方向。\n"
                   "② 大盘方向与风格研判(科技涨高后高低切→白酒/影视/农业等防御补涨的轮动节奏；当前资金偏好哪类产业)。\n"
                   f"③ 给出{args.n_chains}条最值得【提前】布局的产业链主题(主题名+催化+景气验证+为何未来1-4周会演绎)。\n"
                   "严禁用as_of之后信息。输出JSON: {"
                   "\"events\":[{\"type\":\"政策/产业/数据/事件\",\"event\":\"具体事件\",\"impact\":\"利好方向/板块\"}],"
                   "\"market_view\":{\"index\":\"大盘研判\",\"style\":\"风格与高低切轮动研判\"},"
                   "\"themes\":[{\"theme\":\"主题\",\"catalyst\":\"对应催化(来自上面新闻或合理传导)\","
                   "\"prosperity\":\"景气验证依据(订单/价格/产量/政策落地)\"}]}。\n\n新闻(PIT):\n" + news_blob),
        fallback={})
    if s1a.used_fallback:
        print("LLM events/themes unavailable:", s1a.fallback_reason); return 1
    events = s1a.output.get("events", []) or []
    market_view = s1a.output.get("market_view", {}) or {}
    themes = (s1a.output.get("themes", []) or [])[:args.n_chains]
    if not themes:
        print("LLM returned no themes"); return 1

    # 1b: 每链单独一调用 映射上中下游个股(只要名称, 代码由 name-authoritative 解析).
    # CH=1: gemma thinking 模型延迟随输出量增长, 单链输出最小 → 几乎不超时(替代降模型的方案).
    chains = []
    CH = 1
    for i in range(0, len(themes), CH):
        batch = themes[i:i + CH]
        s1b = _invoke_retry("chain_stocks",
            system_prompt=(f"你是A股产业链研究专家。只用{args.as_of}及之前信息，绝不用后见之明。只输出一个JSON对象。"),
            user_text=(f"为下列产业链主题逐条映射上中下游代表A股，每个环节{sps}只。"
                       "只给【A股准确简称】(用于代码映射，不要给代码)+环节角色+概念+一句受益逻辑。输出JSON: {"
                       "\"chains\":[{\"theme\":\"(与输入一致)\","
                       "\"upstream\":[{\"name\":\"A股简称\",\"role\":\"环节作用\",\"concept\":\"概念\",\"logic\":\"受益逻辑\"}],"
                       "\"midstream\":[...],\"downstream\":[...]}]}。\n主题: " + json.dumps(batch, ensure_ascii=False)),
            fallback={}, use_client=fast_client, models=FAST_MODELS, tries=4)  # 机械枚举 → flash 多模型兜底
        if s1b.used_fallback:
            print(f"  [chain_stocks batch {i//CH+1}] fallback: {s1b.fallback_reason}"); continue
        tmap = {t.get("theme", ""): t for t in batch}
        for ch in s1b.output.get("chains", []) or []:
            t = tmap.get(ch.get("theme", ""), {})
            ch["catalyst"] = ch.get("catalyst") or t.get("catalyst", "")
            ch["prosperity"] = ch.get("prosperity") or t.get("prosperity", "")
            chains.append(ch)
    if not chains:
        print("LLM returned no chains"); return 1

    # collect + validate codes (whole pool); name-authoritative, drop unresolved/out-of-universe
    rows, dropped = [], 0
    seg_label = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}
    for ch in chains:
        for seg in ("upstream", "midstream", "downstream"):
            for c in ch.get(seg, []) or []:
                code = _resolve(c.get("name", ""))  # NAME-authoritative (ignore LLM code)
                if code and code in code2sym:
                    rows.append({"code": code, "symbol": code2sym[code], "name": c.get("name", ""),
                                 "theme": ch.get("theme", ""), "segment": seg, "seg_label": seg_label[seg],
                                 "role": c.get("role", ""), "concept": c.get("concept", ""),
                                 "logic": c.get("logic", ""), "catalyst": ch.get("catalyst", "")})
                else:
                    dropped += 1
    slots_df = pd.DataFrame(rows).drop_duplicates(["theme", "segment", "code"]).reset_index(drop=True)
    cdf = slots_df.drop_duplicates("code").reset_index(drop=True)  # unique codes for scoring/financials/union
    print(f"Stage1: {len(events)} events, {len(chains)} chains, {len(cdf)} unique valid codes "
          f"({len(slots_df)} chain slots, {dropped} dropped unresolved/out-of-universe)")

    # ---- Stage 2: factor-score WHOLE pool + financials for priority subset ----
    code2rank = dict(zip(preds["symbol"].map(_code6), preds["prediction"].rank(pct=True)))
    cdf["alpha_score"] = cdf["code"].map(alpha)
    cdf["sector_level_1"] = cdf["code"].map(code2sector)
    cdf["factor_rank_pct"] = cdf["code"].map(code2rank).fillna(0.0)  # UNIVERSE percentile (not within-pool)
    # priority for the (slow) financial pull: every resolved chain member is factor-scored,
    # but financial endpoints are slow.  Pull statements for the most repeated /
    # highest-ranked names first instead of trusting LLM output order.
    slot_count = slots_df.groupby("code").size().rename("chain_slot_count")
    cdf = cdf.merge(slot_count, on="code", how="left")
    cdf["chain_slot_count"] = pd.to_numeric(cdf["chain_slot_count"], errors="coerce").fillna(1.0)
    cdf["_enrich_priority"] = (
        0.50 * (cdf["chain_slot_count"].rank(pct=True).fillna(0.0))
        + 0.35 * pd.to_numeric(cdf["factor_rank_pct"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(cdf["alpha_score"], errors="coerce").rank(pct=True).fillna(0.0)
    )
    # PIT close (for PE = close / TTM-EPS) from the tickflow-sourced market panel at as_of
    code2close: dict = {}
    try:
        mp = pd.read_parquet("runtime/data/v7/silver/market_panel/market_panel.parquet",
                             columns=["symbol", "trade_date", "close"])
        mp["trade_date"] = pd.to_datetime(mp["trade_date"], errors="coerce")
        mp = mp[mp["trade_date"] <= pd.Timestamp(args.as_of)].sort_values("trade_date").groupby("symbol").tail(1)
        code2close = {_code6(s): float(c) for s, c in zip(mp["symbol"], mp["close"]) if pd.notna(c)}
    except Exception:
        pass
    fin_cols = ["gross_margin", "net_profit_yoy", "roe", "rev_yoy", "net_margin", "pe", "report_period"]
    # tickflow financials for ALL chain candidates (fast + disk-cached) → COMPLETE tables (用户: 财报必须完整).
    fin_map = {r.code: _fin_snapshot(r.symbol, args.as_of, code2close.get(r.code)) for r in cdf.itertuples()}
    for k in fin_cols:
        cdf[k] = [fin_map.get(c, {}).get(k) for c in cdf["code"]]
    # priority subset narrows only the LLM deep-select payload (limits LLM input), not the data pull.
    enrich = cdf.sort_values("_enrich_priority", ascending=False).head(args.max_stocks_enrich).copy()
    print(f"Stage2: factor-scored {len(cdf)} stocks; tickflow financials for ALL {len(cdf)} candidates; "
          f"LLM deep-select on top {len(enrich)}", flush=True)

    # ---- Stage 3: LLM deep selection with 产业逻辑 (real numbers) ----
    sel = {}
    if not args.no_llm:
        payload = enrich[["code", "name", "theme", "segment", "role", "concept",
                          "gross_margin", "net_margin", "net_profit_yoy", "roe", "rev_yoy", "pe", "alpha_score"]].to_dict("records")
        s3 = _invoke_retry("chain_deep_select",
            system_prompt=(f"你是顶级产业研究员+基金经理。只用{args.as_of}及之前的信息，基于真实财报/估值/因子做"
                           "产业链深度比较选股，给可落地的产业逻辑。只输出一个JSON对象。"),
            user_text=("下面是各产业链环节公司的真实财报(毛利率/净利率/归母净利增长率/ROE/营收增长率)、估值(PE)、量化因子分。"
                       "请逐条产业链比较上中下游，选出最优环节与重点个股(12-18只)，给详细产业逻辑。"
                       "选股标尺(A股偏成长/透支未来，不要只看静态低PE)：优先【成长加速/拐点】(净利增速环比抬升、由负转正、连续高增)、"
                       "【高景气订单兑现】(营收增速与毛利同向、产能紧张)、【供应链卡位】(绑定海外大厂/国产替代龙头)；"
                       "对只稳定不增长的'老登股'(如高PE但零增长的消费白马)降权，除非处在资金避险高低切的补涨节点。"
                       "why 里必须结合具体财报数字与未来1-4周/1季的催化。输出JSON: "
                       "{\"picks\":[{\"code\":\"\",\"name\":\"\",\"chain_role\":\"上/中/下游+作用\","
                       "\"why\":\"产业逻辑(结合财报数字+成长拐点+催化)\",\"conviction\":0到1}],"
                       "\"chain_views\":[{\"theme\":\"\",\"best_segment\":\"\",\"reason\":\"哪环节利润/景气最优\"}]}。"
                       "数据: " + json.dumps(payload, ensure_ascii=False, default=str)),
            fallback={})
        if not s3.used_fallback:
            sel = s3.output

    picks = pd.DataFrame(sel.get("picks", [])) if isinstance(sel, dict) else pd.DataFrame()
    if not picks.empty:
        valid = set(cdf["code"])
        # trust the payload code if it's one we passed in; else re-resolve by name
        picks["code"] = [(_code6(c) if _code6(c) in valid else (_resolve(str(n)) or _code6(c)))
                         for c, n in zip(picks.get("code", ""), picks.get("name", ""))]
        picks = picks[picks["code"].isin(valid)].merge(enrich, on="code", how="left", suffixes=("", "_e"))
        picks["symbol"] = picks["code"].map(code2sym)

    # ---- Stage 5: UNION(因子池, LLM产业链池) — 可配置混合 ----
    fac = preds.copy(); fac["code"] = fac["symbol"].map(_code6)
    fac["factor_rank_pct"] = fac["prediction"].rank(pct=True)
    conv = picks.set_index("code")["conviction"].to_dict() if not picks.empty else {}
    cdf["chain_conviction"] = pd.to_numeric(cdf["code"].map(conv), errors="coerce").fillna(0.6)  # 0.6 baseline
    factor_slots = fac.nlargest(args.n_factor, "factor_rank_pct")[["symbol", "code", "factor_rank_pct"]].copy()
    factor_slots["source"] = "因子"; factor_slots["chain_conviction"] = pd.to_numeric(
        factor_slots["code"].map(conv), errors="coerce").fillna(0.0)
    chain_slots = cdf.sort_values("chain_conviction", ascending=False).head(args.n_chain)[
        ["symbol", "code", "chain_conviction"]].copy()
    chain_slots["source"] = "LLM产业链"
    chain_slots["factor_rank_pct"] = pd.to_numeric(
        chain_slots["code"].map(fac.set_index("code")["factor_rank_pct"]), errors="coerce").fillna(0.0)
    final = pd.concat([factor_slots, chain_slots], ignore_index=True).drop_duplicates("symbol")
    final = final.merge(sm, on="symbol", how="left")
    fw, cw = args.factor_weight, args.chain_weight
    final["mix_score"] = (fw * pd.to_numeric(final["factor_rank_pct"], errors="coerce").fillna(0)
                          + cw * pd.to_numeric(final["chain_conviction"], errors="coerce").fillna(0)).round(3)
    final = final.sort_values("mix_score", ascending=False).reset_index(drop=True)

    # ---- Stage 4: render a detailed research report ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bond = _bond_macro()
    n_chain_only = int((final["source"] == "LLM产业链").sum())
    md = [f"# 产业链深度研究报告 — {args.as_of}", "",
          "*PIT舆情全景 → 多产业链(上中下游)深挖 → 财报/估值 → 选最优环节 → 因子∪产业链 (信息严格限定 as_of 之前)*", "",
          "## 摘要", "",
          f"- 当期识别催化 **{len(events)}** 条；深挖产业链 **{len(chains)}** 条；产业链候选股池 **{len(cdf)}** 只(已全量打因子分)。",
          f"- 大盘研判：{market_view.get('index','-')}",
          f"- 风格研判：{market_view.get('style','-')}",
          f"- 最终股池 **{len(final)}** 只 = 因子池 ∪ LLM产业链池(其中产业链独有 {n_chain_only} 只)，混合权重 因子{fw}/链{cw}。", ""]

    # 一、舆情全景
    md += ["## 一、当期市场舆情全景（来自PIT新闻，agent自动识别）", ""]
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(str(e.get("type", "其他")), []).append(e)
    for typ in ["政策", "产业", "数据", "事件", "其他"]:
        items = [e for t, lst in by_type.items() if typ in t for e in lst]
        if items:
            md += [f"**{typ}面：**"] + [f"- {e.get('event','-')} → {e.get('impact','-')}" for e in items] + [""]
    # any uncategorized
    for typ, lst in by_type.items():
        if not any(k in typ for k in ["政策", "产业", "数据", "事件"]):
            md += [f"**{typ}：**"] + [f"- {e.get('event','-')} → {e.get('impact','-')}" for e in lst] + [""]

    # 二、大盘与资金面
    md += ["## 二、大盘与资金面研判", "", f"- **大盘**：{market_view.get('index','-')}",
           f"- **风格/轮动**：{market_view.get('style','-')}"]
    if bond:
        md.append("- **债市/资金面**：" + " · ".join(f"{k} {v}" for k, v in bond.items()))
    md.append("")

    # 三、逐链深度 (render from the RESOLVED pool so 代码/财报 always agree)
    md += ["## 三、产业链深度（逐链：催化 → 景气 → 上中下游 → 最优环节 → 重点个股）", ""]
    fin_by_code = cdf.set_index("code").to_dict("index")
    def _cell(x):
        return "-" if x is None or (isinstance(x, float) and pd.isna(x)) else (f"{x:.3f}" if isinstance(x, float) and abs(x) < 10 else f"{x}")
    views = {v.get("theme", ""): v for v in (sel.get("chain_views", []) if isinstance(sel, dict) else [])}
    for i, ch in enumerate(chains, 1):
        theme = ch.get("theme", "")
        md += [f"### 链{i}：{theme}", f"- **催化**：{ch.get('catalyst','-')}",
               f"- **景气验证**：{ch.get('prosperity','-')}", ""]
        sub = slots_df[slots_df.theme == theme]
        if sub.empty:
            md.append("*（本链个股名称未能对应到A股代码，已跳过）*\n"); continue
        md += ["| 环节 | 个股(代码) | 角色 | 概念 | 受益逻辑 | 毛利% | 净利率% | 净利增速% | ROE | 营收增速% | PE | 因子分 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for seg in ("upstream", "midstream", "downstream"):
            for r in sub[sub.segment == seg].itertuples():
                f = fin_by_code.get(r.code, {})
                md.append(f"| {r.seg_label} | {r.name}({r.code}) | {r.role} | {r.concept} | {r.logic} | "
                          f"{_cell(f.get('gross_margin'))} | {_cell(f.get('net_margin'))} | {_cell(f.get('net_profit_yoy'))} | "
                          f"{_cell(f.get('roe'))} | {_cell(f.get('rev_yoy'))} | {_cell(f.get('pe'))} | {_cell(f.get('alpha_score'))} |")
        v = views.get(theme)
        if v:
            md += ["", f"> **最优环节**：{v.get('best_segment','-')} — {v.get('reason','-')}"]
        md.append("")

    # 四、重点精选个股(深度)
    if not picks.empty:
        md += ["## 四、重点精选个股（产业逻辑 + 真实财报）", ""]
        for _, p in picks.iterrows():
            md.append(f"### {p.get('name','')}({p.get('code')}) · {p.get('chain_role','')} · 信念{p.get('conviction','-')}")
            md.append(f"{p.get('why','')}")
            e = enrich[enrich.code == p["code"]]
            if not e.empty:
                r = e.iloc[0]
                md.append(f"> 财报({r.get('report_period','-')}): 毛利率{r.gross_margin}% · 净利率{r.get('net_margin','-')}% · "
                          f"归母净利增速{r.net_profit_yoy}% · ROE{r.roe} · 营收增速{r.rev_yoy}% · PE{r.pe} · 因子分{r.alpha_score}")
            md.append("")

    # 五、最终股池
    md += ["## 五、最终股池（因子池 ∪ LLM产业链池，互补混合）", "",
           f"*混合口径：因子模型主导收益排序(已验证 +α)；产业链补充因子覆盖不到的催化驱动名；权重 因子{fw}/链{cw}。共 {len(final)} 只。*", "",
           "| symbol | 板块 | 来源 | 因子分位 | 链信念 | mix |", "|---|---|---|---|---|---|"]
    for _, r in final.iterrows():
        md.append(f"| {r['symbol']} | {r.get('sector_level_1','')} | {r.get('source','')} | "
                  f"{r['factor_rank_pct']:.2f} | {r['chain_conviction']:.2f} | {r['mix_score']:.3f} |")

    # 六、风险
    md += ["", "## 六、风险提示", "",
           "- 产业链由 LLM 推演，个股入池仅供研究参考，非交易指令；财报为 PIT 报告期数据，留意一季报/年报披露节奏。",
           "- 因子与产业链买卖周期不同：因子偏中短(约1-2周再平衡)，产业链偏事件/景气(数周-数月)；建议以因子为主仓、产业链做事件性增配。",
           "- LLM 为2026年模型，历史 as_of 选股存在参数记忆 hindsight 残余风险；以 chain_oos_validation.py 的 news-ablation 做对照。", ""]

    sfx = args.out_suffix or (f"_{args.news_mode}" if args.news_mode != "real" else "")
    out = args.out_dir / f"chain_research_{args.as_of}{sfx}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    keep = ["symbol", "sector_level_1", "source", "factor_rank_pct", "chain_conviction", "mix_score"]
    final[keep].to_parquet(args.out_dir / f"chain_pool_{args.as_of}{sfx}.parquet", index=False)
    # 全量候选池(打了因子分)也落盘，供"对所有股池施行因子"的实验复用
    cdf.to_parquet(args.out_dir / f"chain_candidates_{args.as_of}{sfx}.parquet", index=False)
    meta = {"as_of": args.as_of, "pit_strict": args.pit_strict, "news_mode": args.news_mode, "n_events": len(events),
            "n_chains": len(chains), "n_candidates": int(len(cdf)), "n_final": int(len(final)),
            "events": events, "market_view": market_view,
            "chain_views": sel.get("chain_views", []) if isinstance(sel, dict) else [],
            "factor_weight": fw, "chain_weight": cw,
            "dropped_unresolved_or_out_of_universe": int(dropped),
            "financial_enrichment": {
                "requested": int(args.max_stocks_enrich),
                "completed": int(len(enrich)),
                "coverage": round(float(len(enrich) / max(1, len(cdf))), 4),
                "priority_rule": "0.50 chain_slot_count_rank + 0.35 factor_rank_pct + 0.15 alpha_score_rank",
            },
            "prediction_contract": {
                "as_of": args.as_of,
                "news_mode": args.news_mode,
                "pit_strict": bool(args.pit_strict),
                "no_live_orders": True,
                "requires_oos_ablation": True,
            }}
    (args.out_dir / f"chain_meta_{args.as_of}{sfx}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} | events={len(events)} chains={len(chains)} candidates={len(cdf)} "
          f"picks={len(picks)} final_pool={len(final)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
