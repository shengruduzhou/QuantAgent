"""盘口微观结构风险识别 — DEFENSIVE ONLY (detect-to-avoid).

合规声明 (COMPLIANCE): this module ONLY DETECTS adverse order-flow patterns so
the strategy can AVOID being the victim (avoid buying into a dump, avoid being
trapped, cut risk on systemic outflow). It does NOT, and must never, generate
orders that manipulate price (无对倒/拉抬/打压/虚假申报). Detection of manipulation
to defend against it is legitimate risk management under Chinese securities law;
executing it is illegal and out of scope.

Detects four flow patterns the user flagged, from EOD intraday-aggregated
features (net_buy_pressure / vwap_deviation / intraday_range_pos /
volume_concentration / close30_volume_share / 北向flow):

1. 一字断魂刀 / 砸盘 (sweep dump): close near low + strongly negative net buy +
   volume climax ⇒ someone swept the bids. Avoid/exit.
2. 游资压盘逼卖 (pressure-sell): heavy late-session selling + below VWAP +
   negative net buy ⇒ pressing retail to sell. Avoid chasing.
3. 对倒虚假量 (wash volume): abnormal volume but no price progress (VWAP≈0,
   weak conviction) ⇒ fake liquidity. Don't trust the breakout.
4. 量化大资金集体出走/避险 (quant exodus, MARKET-wide): cross-sectional net buy
   broadly negative + 北向净流出 ⇒ systemic risk-off. Cut gross exposure.

Pure functions; percentile/cross-sectional based (no fragile hard thresholds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_FEATS = ["net_buy_pressure", "vwap_deviation", "intraday_range_pos",
          "volume_concentration", "close30_volume_share"]


def _pct_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True)


def microstructure_guard(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-stock defensive risk flags. Returns the frame + guard columns.

    Higher *_risk = more adverse. ``guard_action`` ∈ {avoid, caution, ok}.
    """
    out = candidates.copy()
    have = [c for c in _FEATS if c in out.columns]
    for c in have:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    nb = out.get("net_buy_pressure")
    vd = out.get("vwap_deviation")
    irp = out.get("intraday_range_pos")           # 0=收在最低, 1=收在最高
    vc = out.get("volume_concentration")
    c30 = out.get("close30_volume_share")

    n = len(out)
    z = pd.Series(0.5, index=out.index)
    nb_low = (1 - _pct_rank(nb)) if nb is not None else z          # net buy 越低越危险
    irp_low = (1 - _pct_rank(irp)) if irp is not None else z       # 收盘越靠低位越危险
    vc_hi = _pct_rank(vc) if vc is not None else z                 # 量能集中(异常放量)
    vd_low = (1 - _pct_rank(vd)) if vd is not None else z          # 越低于vwap越危险
    c30_hi = _pct_rank(c30) if c30 is not None else z              # 尾盘抛压
    vd_abs_low = (1 - _pct_rank(vd.abs())) if vd is not None else z  # 价格无进展(对倒)

    # 1) 砸盘/断魂刀: 收低 + 净卖 + 放量
    out["sweep_dump_risk"] = (0.4 * irp_low + 0.35 * nb_low + 0.25 * vc_hi).round(3)
    # 2) 压盘逼卖: 尾盘抛压 + 净卖 + 跌破vwap
    out["pressure_sell_risk"] = (0.4 * c30_hi + 0.3 * nb_low + 0.3 * vd_low).round(3)
    # 3) 对倒虚假量: 放量 + 价格无进展
    out["wash_volume_risk"] = (0.6 * vc_hi + 0.4 * vd_abs_low).round(3)
    acute = pd.concat([out["sweep_dump_risk"], out["pressure_sell_risk"]], axis=1).max(axis=1)
    out["guard_action"] = np.where(acute >= 0.80, "avoid",
                            np.where((acute >= 0.65) | (out["wash_volume_risk"] >= 0.80), "caution", "ok"))
    return out


def market_risk_off_level(candidates: pd.DataFrame, north_total: float | None = None) -> dict:
    """市场级'量化集体出走/避险'信号 (pattern 4). Returns level + recommended gross cap."""
    nb = pd.to_numeric(candidates.get("net_buy_pressure"), errors="coerce")
    med_nb = float(nb.median()) if nb is not None and nb.notna().any() else 0.0
    frac_selling = float((nb < 0).mean()) if nb is not None and nb.notna().any() else 0.0
    north_neg = bool(north_total is not None and north_total < 0)
    # broad net-selling + 北向净流出 = systemic risk-off
    score = 0.6 * frac_selling + 0.4 * (1.0 if north_neg else 0.0)
    if frac_selling >= 0.75 and (north_neg or med_nb < 0):
        level, gross_cap = "risk_off", 0.30
    elif frac_selling >= 0.60:
        level, gross_cap = "caution", 0.50
    else:
        level, gross_cap = "normal", 0.80
    return {"level": level, "frac_net_selling": round(frac_selling, 3),
            "median_net_buy": round(med_nb, 4), "north_outflow": north_neg,
            "risk_off_score": round(score, 3), "recommended_gross_cap": gross_cap}
