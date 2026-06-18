"""分时做T 决策层 —— 在因果引擎之上叠加 A股制度约束 + 集合竞价 + 严格 JSON 输出。

把 ``intraday_dot_engine.compute_intraday_state`` 的因子快照，结合持仓
（sellable_qty / today_buy_qty）、市场阶段、集合竞价数据，产出用户规格
第九节定义的严格 JSON 决策。核心红线：

  * T+1：卖出 qty ≤ sellable_qty（昨仓），永不卖 today_buy_qty；
  * 不越涨跌停价、限价单、100股整数倍、现金校验；
  * 正T（低吸→后续卖旧）与反T（卖旧→低位回补）都生成 t_pair_id；
  * confidence ≥ 75 才执行，60-75 小仓/观察，<75 HOLD，制度冲突 REJECT。

研究/回测/信号用途；本层只产出建议，不下真实订单。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantagent.execution.intraday_dot_engine import (
    DotEngineParams,
    IntradayState,
)
from quantagent.execution.intraday_ev_engine import (
    EVDecision,
    EVDecisionConfig,
    IntradayModelSignals,
    decide_ev,
    decide_expected_value,
)


@dataclass(frozen=True)
class Position:
    total_qty: int = 0
    sellable_qty: int = 0          # 昨仓可卖（T+1 合法卖出来源）
    today_buy_qty: int = 0         # 今日买入，今日不可卖
    avg_cost: float = 0.0


@dataclass(frozen=True)
class DecisionConfig:
    conf_execute: float = 75.0
    conf_observe: float = 60.0
    round_lot: int = 100
    base_t_fraction: float = 0.25        # 普通环境单次T占总仓比例
    range_t_fraction: float = 0.40       # 高置信震荡
    weak_t_fraction: float = 0.10
    auction_max_fraction: float = 0.20   # 集合竞价不可撤阶段上限
    min_net_edge_pct: float = 0.004      # 预期净价差门槛（覆盖成本）
    cost_roundtrip_pct: float = 0.0026   # 2×佣金+印花+2×滑点的保守估计


def _round_lot(qty: float, lot: int) -> int:
    return int(max(0, qty) // lot * lot)


def _phase(current_time: str) -> str:
    t = str(current_time)[-8:] if len(str(current_time)) >= 8 else str(current_time)
    if t < "09:15:00":
        return "pre_open"
    if t < "09:20:00":
        return "auction_observe"
    if t < "09:25:00":
        return "auction_no_cancel"
    if t < "09:30:00":
        return "open_prepare"
    if t <= "14:56:59":
        return "continuous"
    if t <= "15:00:00":
        return "close_auction"
    return "closed"


def _empty_decision(symbol: str, time: str, phase: str, reason: str,
                    action: str = "HOLD") -> dict:
    return {
        "symbol": symbol, "time": time, "market_phase": phase,
        "action": action, "side": "NONE", "qty": 0, "price_type": "NONE",
        "limit_price": 0.0, "confidence": 0, "t_pair_id": "", "is_t_trade": False,
        "legal_check": {"t_plus_1_ok": True, "sellable_qty_ok": True,
                        "price_limit_ok": True, "lot_size_ok": True, "cash_ok": True},
        "reason": [reason], "risk_flags": [],
        "failure_control": {"fail_type": "none", "fail_threshold": 0.0, "stop_action": ""},
        "next_watch_levels": {},
    }


def _t_fraction(state: IntradayState, cfg: DecisionConfig, phase: str) -> float:
    if phase == "auction_no_cancel":
        return cfg.auction_max_fraction
    if state.state == "range" and state.reversal_env:
        return cfg.range_t_fraction
    if state.state in ("weak_down", "limit_down"):
        return cfg.weak_t_fraction
    return cfg.base_t_fraction


def decide(
    state: IntradayState | None,
    position: Position,
    *,
    symbol: str,
    current_time: str,
    pre_close: float,
    limit_up: float,
    limit_down: float,
    cash: float = 0.0,
    auction: dict | None = None,
    config: DecisionConfig | None = None,
    pair_id_hint: str | None = None,
) -> dict:
    """生成一条严格 JSON 做T决策（正T/反T/回补/风控卖/观望/拒绝）。"""
    cfg = config or DecisionConfig()
    phase = _phase(current_time)

    if phase in ("pre_open", "closed"):
        return _empty_decision(symbol, current_time, phase, "非交易窗口", "WAIT")
    if phase in ("auction_observe", "open_prepare"):
        # 观察/开盘准备：只预判不下单
        d = _empty_decision(symbol, current_time, phase, "竞价观察阶段，只预判不下单", "WAIT")
        if auction:
            d["reason"].append(f"竞价虚拟价相对昨收 {auction.get('indicative_gap_pct', 0):+.2f}%")
        return d
    if state is None:
        return _empty_decision(symbol, current_time, phase, "无分时数据", "HOLD")
    if state.n_bars < (config or DecisionConfig()).round_lot // 20:  # need a few bars
        return _empty_decision(symbol, current_time, phase, "分时数据不足", "HOLD")

    watch = {
        "intraday_vwap": round(state.vwap, 4),
        "low_buy_line": round(state.low_line, 4),
        "high_sell_line": round(state.high_line, 4),
        "stop_level": round(state.last * (1 - state.band * 0.65), 4),
        "take_profit_level": round(state.last * (1 + state.band * 0.5), 4),
    }
    risk_flags: list[str] = []
    if state.limit_down_risk:
        risk_flags.append("跌停风险")
    if state.limit_up_squeeze:
        risk_flags.append("涨停逼空")
    if state.weak_trend:
        risk_flags.append("弱趋势延续")
    if state.strong_trend:
        risk_flags.append("强趋势延续")

    # ── 失败控制优先 ────────────────────────────────────────────────
    fail_type = "none"
    stop_action = ""
    if state.low_fail_now or state.low_fail_lock:
        fail_type = "low_buy_failure"
        stop_action = "禁止补仓摊低；有昨仓可卖则对冲，否则仅风险提示"
    elif state.high_fail_now or state.high_fail_lock:
        fail_type = "high_sell_failure"
        stop_action = "停止继续卖出；回踩VWAP/短均可回补，逼近涨停则放弃"
    fail_thr = round(state.band * 0.65, 5)
    failure_control = {"fail_type": fail_type, "fail_threshold": fail_thr, "stop_action": stop_action}

    def base(action, side, qty, price, conf, is_t, pair, reasons, legal):
        return {
            "symbol": symbol, "time": current_time, "market_phase": phase,
            "action": action, "side": side, "qty": int(qty),
            "price_type": "LIMIT" if side != "NONE" else "NONE",
            "limit_price": round(float(price), 2) if side != "NONE" else 0.0,
            "confidence": int(round(conf)), "t_pair_id": pair, "is_t_trade": is_t,
            "legal_check": legal, "reason": reasons, "risk_flags": risk_flags,
            "failure_control": failure_control, "next_watch_levels": watch,
        }

    # ── 收盘集合竞价：只允许回补/降风险 ──────────────────────────────
    if phase == "close_auction":
        if fail_type == "high_sell_failure" and cash > 0:
            return base("BUY_BACK", "BUY", 0, state.last, 50, True,
                        pair_id_hint or "", ["收盘竞价：高抛失败的小仓回补窗口（按需）"],
                        {"t_plus_1_ok": True, "sellable_qty_ok": True, "price_limit_ok": True,
                         "lot_size_ok": True, "cash_ok": cash > 0})
        if state.limit_down_risk and position.sellable_qty >= cfg.round_lot:
            qty = _round_lot(position.sellable_qty * 0.5, cfg.round_lot)
            return base("SELL_RISK", "SELL", qty, max(state.last, limit_down), 70, False,
                        "", ["收盘竞价：风险暴露过高，卖出部分昨仓降风险"],
                        {"t_plus_1_ok": True, "sellable_qty_ok": qty <= position.sellable_qty,
                         "price_limit_ok": True, "lot_size_ok": True, "cash_ok": True})
        return _empty_decision(symbol, current_time, phase, "收盘竞价：无需调整", "HOLD")

    # ── 制度硬约束：极端封板 ────────────────────────────────────────
    if state.limit_down_risk and state.last <= limit_down + 0.005:
        if position.sellable_qty >= cfg.round_lot:
            return base("SELL_RISK", "SELL", 0, limit_down, 40, False, "",
                        ["跌停封单：禁止低吸；如需降risk只能挂跌停价排队卖昨仓"],
                        {"t_plus_1_ok": True, "sellable_qty_ok": True, "price_limit_ok": True,
                         "lot_size_ok": True, "cash_ok": True})
        return base("REJECT", "NONE", 0, 0, 0, False, "",
                    ["跌停风险且无可卖昨仓 → 仅风险提示，禁止任何买入"],
                    {"t_plus_1_ok": True, "sellable_qty_ok": False, "price_limit_ok": True,
                     "lot_size_ok": True, "cash_ok": True})

    # ── 预期净价差门槛（覆盖成本）────────────────────────────────────
    if 2 * state.band * 0.5 < cfg.min_net_edge_pct + cfg.cost_roundtrip_pct:
        return base("HOLD", "NONE", 0, 0, 0, False, "",
                    [f"预期净价差不足（带宽{state.band:.3%}），不足覆盖成本 → 不交易"],
                    {"t_plus_1_ok": True, "sellable_qty_ok": True, "price_limit_ok": True,
                     "lot_size_ok": True, "cash_ok": True})

    frac = _t_fraction(state, cfg, phase)
    if phase == "auction_no_cancel":
        frac = min(frac, cfg.auction_max_fraction)

    # ── 反T / 高抛：卖出昨仓（confidence = 高抛可信）──────────────────
    high_conf = state.high_conf
    if state.high_signal and not state.high_fail_lock and not state.strong_trend \
            and not state.limit_up_squeeze:
        if high_conf >= cfg.conf_execute and position.sellable_qty >= cfg.round_lot:
            qty = _round_lot(position.sellable_qty * frac, cfg.round_lot)
            qty = min(qty, position.sellable_qty)
            if qty >= cfg.round_lot:
                price = max(state.high_line, state.last)
                price = min(price, limit_up - 0.01)
                return base("SELL_HIGH", "SELL", qty, price, high_conf, True,
                            pair_id_hint or f"{symbol}-{current_time}-T",
                            [f"反T高抛：{state.high_signal} 高分{state.high_score:.0f}/可信{high_conf:.0f}",
                             f"偏离VWAP+{state.deviation_pct:.2f}%，主动卖比{state.active_sell_ratio:.0f}",
                             "卖出昨仓，后续低位回补降成本（T+1合法）"],
                            {"t_plus_1_ok": True, "sellable_qty_ok": qty <= position.sellable_qty,
                             "price_limit_ok": price < limit_up, "lot_size_ok": qty % cfg.round_lot == 0,
                             "cash_ok": True})
        elif high_conf >= cfg.conf_observe and position.sellable_qty >= cfg.round_lot:
            return base("HOLD", "NONE", 0, 0, high_conf, False, "",
                        [f"高抛信号但可信{high_conf:.0f}<{cfg.conf_execute:.0f} → 只观察"],
                        {"t_plus_1_ok": True, "sellable_qty_ok": True, "price_limit_ok": True,
                         "lot_size_ok": True, "cash_ok": True})

    # ── 正T / 低吸：买入（confidence = 低吸可信）──────────────────────
    low_conf = state.low_conf
    if state.low_signal and not state.low_fail_lock and not state.weak_trend \
            and not state.limit_down_risk:
        if low_conf >= cfg.conf_execute and cash > 0:
            price = min(state.low_line, state.last)
            price = max(price, limit_down + 0.01)
            qty = _round_lot(min(cash / max(price, 1e-6),
                                 position.total_qty * frac if position.total_qty else cash / max(price, 1e-6)),
                             cfg.round_lot)
            action = "BUY_BACK" if (pair_id_hint or fail_type == "high_sell_failure") else "BUY_LOW"
            if qty >= cfg.round_lot and qty * price <= cash + 1e-6:
                return base(action, "BUY", qty, price, low_conf, True,
                            pair_id_hint or f"{symbol}-{current_time}-T",
                            [f"正T低吸：{state.low_signal} 低分{state.low_score:.0f}/可信{low_conf:.0f}",
                             f"偏离VWAP{state.deviation_pct:.2f}%，RSI{state.rsi:.0f}，主动买比{state.active_buy_ratio:.0f}",
                             "买回核心仓T，后续卖出仅用昨仓（T+1合法）" if action == "BUY_LOW"
                             else "高抛失败后的回补腿"],
                            {"t_plus_1_ok": True, "sellable_qty_ok": True,
                             "price_limit_ok": price > limit_down, "lot_size_ok": qty % cfg.round_lot == 0,
                             "cash_ok": qty * price <= cash + 1e-6})
        elif low_conf >= cfg.conf_observe:
            return base("HOLD", "NONE", 0, 0, low_conf, False, "",
                        [f"低吸信号但可信{low_conf:.0f}<{cfg.conf_execute:.0f}或无现金 → 只观察"],
                        {"t_plus_1_ok": True, "sellable_qty_ok": True, "price_limit_ok": True,
                         "lot_size_ok": True, "cash_ok": cash > 0})

    # ── 默认 ────────────────────────────────────────────────────────
    return base("HOLD", "NONE", 0, 0, max(state.low_conf, state.high_conf), False, "",
                [f"状态={state.state}，无达标信号（低{state.low_score:.0f}/高{state.high_score:.0f}）→ 持有等待"],
                {"t_plus_1_ok": True, "sellable_qty_ok": True, "price_limit_ok": True,
                 "lot_size_ok": True, "cash_ok": True})


__all__ = [
    "EVDecision",
    "EVDecisionConfig",
    "IntradayModelSignals",
    "Position",
    "DecisionConfig",
    "decide",
    "decide_ev",
    "decide_expected_value",
]
