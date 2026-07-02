"""Stage 10.3 — order / 公告 / 主营 verification → hard concept labels.

Turns "待公告核实" into one of the user's hard labels by combining three
evidence sources (all PIT, dated):

  confirmed_order     正式公告: 中标 / 重大合同 / 订单 / 供货协议 (with amount)
  confirmed_customer  客户认证 / 定点 / 导入供应链 / 批量供货
  revenue_exposure    主营业务收入 covers the concept's segment (主营构成)
  earnings_verified   定期报告 / 业绩预告 confirms real growth (yjbb/yjyg)
  rumor_only          only 互动易 / 投资者关系 mentions, no 公告
  fake_concept        no 公告 + no 主营 exposure + no earnings — concept-board tag only

The text classifier + label derivation are pure (no network) so they run now
on cached earnings; :func:`fetch_announcements` is the live 公告 hook wired for a
later un-throttled run. Priority high→low: confirmed_order > confirmed_customer
> revenue_exposure > earnings_verified > rumor_only > fake_concept > unverified.
"""
from __future__ import annotations

import re

import pandas as pd

EVIDENCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "order": ("中标", "中标公告", "中标通知", "重大合同", "重大销售合同", "订单", "框架协议",
              "采购协议", "供货协议", "供货合同", "销售合同", "签订", "中标候选人", "成交公告"),
    "customer": ("客户认证", "通过认证", "通过验证", "定点", "定点开发", "供应商", "导入",
                 "进入供应链", "批量供货", "批量交付", "送样", "样品认证", "战略合作协议"),
    "capacity": ("扩产", "产能", "投产", "新建产线", "募投项目", "扩建", "达产", "产能爬坡"),
    "earnings": ("业绩预告", "业绩快报", "定期报告", "年度报告", "半年度报告", "季度报告",
                 "年报", "半年报", "季报", "预增", "扭亏", "净利润"),
    "rumor": ("互动易", "投资者关系", "互动平台", "回复投资者", "投资者问答", "调研"),
}

# amount tokens that turn an order mention into a *material* confirmed order
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(亿元|亿|万元|万)")


def classify_announcement_titles(titles: list[str]) -> dict[str, list[str]]:
    """Group announcement titles by evidence type via keyword match."""
    out: dict[str, list[str]] = {}
    for t in titles:
        t = str(t)
        for ev, kws in EVIDENCE_PATTERNS.items():
            if any(k in t for k in kws):
                out.setdefault(ev, []).append(t)
    return out


def _material_order(titles: list[str]) -> bool:
    """An order/contract title that also names a material amount."""
    for t in titles:
        if any(k in str(t) for k in EVIDENCE_PATTERNS["order"]) and _AMOUNT.search(str(t)):
            return True
    return False


def main_business_covers(main_business: str, concept_keywords: tuple[str, ...]) -> bool:
    if not main_business:
        return False
    return any(k and k in main_business for k in concept_keywords)


def derive_order_label(*, ann_evidence: dict[str, list[str]] | None,
                       profit_yoy: float | None, yj_forecast: str | None,
                       main_business: str | None = None,
                       concept_keywords: tuple[str, ...] = (),
                       purity_score: float | None = None,
                       revenue_exposure_pct: float | None = None) -> str:
    """Combine 公告 + 主营 + earnings evidence into one hard label (priority order).

    performance_mismatch overrides a *positive* order/exposure label when the
    company has the concept/order but its earnings don't deliver (概念强但业绩不兑现).
    """
    ann = ann_evidence or {}
    base = None
    if ann.get("order") and _material_order(ann["order"]):
        base = "confirmed_order"
    elif ann.get("customer"):
        base = "confirmed_customer"
    elif revenue_exposure_pct is not None and revenue_exposure_pct >= 0.10:
        base = "revenue_exposure"
    elif main_business is not None and main_business_covers(main_business, concept_keywords):
        base = "revenue_exposure"
    # concept/order present but earnings collapse -> performance_mismatch
    if base in ("confirmed_order", "confirmed_customer", "revenue_exposure") \
            and profit_yoy is not None and profit_yoy < -20:
        return "performance_mismatch"
    if base:
        return base
    earnings_ok = (profit_yoy is not None and profit_yoy > 0) or (
        yj_forecast and any(k in str(yj_forecast) for k in ("预增", "扭亏", "续盈")))
    if earnings_ok:
        return "earnings_verified"
    if ann.get("order") or ann.get("rumor"):
        return "rumor_only"   # mentioned but no material 公告 / only 互动易
    weak = (profit_yoy is not None and profit_yoy <= 0) or profit_yoy is None
    if weak and (purity_score is None or purity_score < 11):
        return "fake_concept"
    return "unverified"


def label_hardness_offline(hardness: pd.DataFrame) -> pd.Series:
    """No-network labels from cached earnings only (公告/主营 left for live run)."""
    def _lab(r):
        return derive_order_label(ann_evidence=None, profit_yoy=r.get("profit_yoy"),
                                  yj_forecast=r.get("yj_forecast"),
                                  purity_score=r.get("score_purity"))
    return hardness.apply(_lab, axis=1)


def _titles_cninfo(ak, code: str) -> list[str]:
    """巨潮 (cninfo) — primary aggregated source covering SSE/SZSE/BSE."""
    sym = code.split(".")[0]
    df = ak.stock_zh_a_disclosure_report_cninfo(symbol=sym, market="沪深京")
    if df is None or df.empty:
        return []
    col = next((c for c in df.columns if "标题" in c or "公告" in c), df.columns[0])
    return df[col].astype(str).head(80).tolist()


def _titles_em(ak, code: str) -> list[str]:
    """东财 individual notice — auxiliary."""
    df = ak.stock_individual_notice_report(symbol=code.split(".")[0])
    if df is None or df.empty:
        return []
    col = next((c for c in df.columns if "标题" in c or "公告" in c or "名称" in c), df.columns[0])
    return df[col].astype(str).head(80).tolist()


def fetch_announcements(code: str, *, days: int = 120, allow_network: bool = False) -> tuple[list[str], str]:
    """Live 公告 titles + the source used, by priority 巨潮 > 东财(aux). Guarded
    (fail-soft): returns ([], "") on disabled/throttled network so the daily scan
    never breaks. Source priority per user: 巨潮/sse/szse/bse first, 东财/THS aux."""
    if not allow_network:
        return [], ""
    import akshare as ak
    for src, fn in (("cninfo", _titles_cninfo), ("eastmoney", _titles_em)):
        try:
            titles = fn(ak, code)
            if titles:
                return titles, src
        except Exception:
            continue
    return [], ""
