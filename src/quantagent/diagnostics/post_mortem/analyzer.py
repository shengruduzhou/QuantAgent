"""Per-trade post-mortem analyser."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config + dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostMortemConfig:
    """Tunable knobs for the analyser.

    The "nearest failing gate" heuristic looks at each gate's detail
    payload and computes how far it was from its threshold (when the
    detail carries enough info — alpha_threshold, liquidity, etc.).
    Gates without a numeric distance are skipped.
    """

    # Margin score function: maps a gate's detail dict to a [0,1] "margin"
    # where 1.0 = passed comfortably and 0.0 = was right on the edge.
    pass    # currently empty; placeholder for future tuning


@dataclass
class PerTradePostMortem:
    trade_id: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbol: str
    entry_price: float
    exit_price: float
    holding_days: int
    realized_pnl_pct: float
    benchmark_return_pct: float
    excess_return_pct: float

    entry_alpha: float | None
    setup_label: str | None
    entry_decision_trace: dict[str, Any]  # serialised DecisionTrace

    nearest_passing_gate: str | None
    nearest_passing_margin: float

    attribution_alpha: float
    attribution_market: float
    attribution_residual: float

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "entry_date": _iso(self.entry_date),
            "exit_date": _iso(self.exit_date),
            "symbol": self.symbol,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "holding_days": int(self.holding_days),
            "realized_pnl_pct": float(self.realized_pnl_pct),
            "benchmark_return_pct": float(self.benchmark_return_pct),
            "excess_return_pct": float(self.excess_return_pct),
            "entry_alpha": _none_if_nan(self.entry_alpha),
            "setup_label": self.setup_label,
            "entry_decision_trace": self.entry_decision_trace,
            "nearest_passing_gate": self.nearest_passing_gate,
            "nearest_passing_margin": float(self.nearest_passing_margin),
            "attribution_alpha": float(self.attribution_alpha),
            "attribution_market": float(self.attribution_market),
            "attribution_residual": float(self.attribution_residual),
            "notes": list(self.notes),
        }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    try:
        return None if np.isnan(value) else float(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# Nearest-failing-gate heuristic
# ---------------------------------------------------------------------------

def _gate_margin(gate_name: str, detail: dict[str, Any]) -> float | None:
    """Return a 0..1 margin score for a passing gate.

    1.0 = far above the threshold; 0.0 = right on the edge. Returns
    ``None`` when the gate doesn't carry enough info to compute a margin.
    """
    if not detail:
        return None
    # Each gate carries a different payload; compute a simple normalised
    # margin from the most natural numeric field.
    if gate_name == "alpha_threshold" and "alpha" in detail:
        a = float(detail["alpha"])
        # higher alpha → larger margin (assume reference ~1.0)
        return float(min(1.0, max(0.0, a)))
    if gate_name == "liquidity" and "amount_cny" in detail:
        # Margin = log-ratio of amount over min_amount.  Capped.
        amt = float(detail["amount_cny"])
        return float(min(1.0, amt / 1_000_000_000.0))  # 10亿 = 1.0
    if gate_name == "price_limit_block" and "daily_return" in detail:
        # Margin = how far from limit-up/down
        r = float(detail["daily_return"])
        # 5% return → margin ~0.5; closer to 9.5% → margin near 0
        return float(min(1.0, max(0.0, 1.0 - abs(r) / 0.095)))
    if gate_name == "fundamental_filter" and "composite_rank" in detail:
        return float(min(1.0, max(0.0, detail["composite_rank"])))
    if gate_name == "policy_aligned" and "signal" in detail:
        # 0 → margin 0.5; +1 → 1.0; -1 → 0.0
        return float(np.clip(0.5 + 0.5 * detail["signal"], 0.0, 1.0))
    if gate_name == "broker_consensus" and "score" in detail:
        return float(np.clip(0.5 + 0.5 * detail["score"], 0.0, 1.0))
    if gate_name == "drawdown_kill" and "dd_20d" in detail:
        # dd=0 → 1.0; dd=-0.20 → 0.0
        return float(np.clip(1.0 + detail["dd_20d"] / 0.20, 0.0, 1.0))
    if gate_name == "concentration_limit" and "proposed" in detail:
        # 0 weight → 1.0; at cap 0.30 → 0.0
        return float(np.clip(1.0 - detail["proposed"] / 0.30, 0.0, 1.0))
    if gate_name == "risk_budget" and "target_weight" in detail:
        return float(np.clip(1.0 - abs(detail["target_weight"]) / 0.03, 0.0, 1.0))
    return None


def _nearest_passing_gate(entry_trace: dict[str, Any]) -> tuple[str | None, float]:
    """Return (gate_name, margin) for the gate closest to failing.

    Only PASSING gates are considered (a FAILED gate would have short-
    circuited the chain — the trade wouldn't exist).  Returns
    ``(None, 1.0)`` when no gate carries a computable margin.
    """
    gates = entry_trace.get("gate_results", []) or []
    candidates: list[tuple[str, float]] = []
    for g in gates:
        if not g.get("passed"):
            continue
        margin = _gate_margin(g["gate_name"], g.get("detail") or {})
        if margin is None:
            continue
        candidates.append((g["gate_name"], float(margin)))
    if not candidates:
        return None, 1.0
    candidates.sort(key=lambda x: x[1])
    return candidates[0]


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def _attribute_return(
    realized: float,
    benchmark: float,
    entry_alpha: float | None,
) -> tuple[float, float, float]:
    """Split realized into (alpha, market, residual).

    Heuristic: market component = min(|realized|, |benchmark|) * sign(benchmark)
    so a stock that's up 5% while market is up 3% gets 3% market + 2% (alpha+residual).
    Alpha component = entry_alpha (the predicted edge); residual = what's left.
    """
    market = float(benchmark)
    if entry_alpha is None or not np.isfinite(entry_alpha):
        alpha_component = 0.0
    else:
        alpha_component = float(entry_alpha)
    residual = float(realized) - market - alpha_component
    return alpha_component, market, residual


# ---------------------------------------------------------------------------
# Single-trade analyzer
# ---------------------------------------------------------------------------

def analyze_trade(
    *,
    trade_id: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    symbol: str,
    entry_price: float,
    exit_price: float,
    benchmark_entry_price: float,
    benchmark_exit_price: float,
    entry_decision_trace: dict[str, Any],
    config: PostMortemConfig | None = None,
) -> PerTradePostMortem:
    cfg = config or PostMortemConfig()
    realized = (float(exit_price) / float(entry_price)) - 1.0
    bench = (float(benchmark_exit_price) / float(benchmark_entry_price)) - 1.0
    excess = realized - bench
    holding_days = int((pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days)

    entry_alpha = None
    setup_label = None
    notes: list[str] = []
    for g in entry_decision_trace.get("gate_results", []):
        if g.get("gate_name") == "alpha_threshold":
            entry_alpha = (g.get("detail") or {}).get("alpha")
        if g.get("gate_name") == "regime_alignment":
            setup_label = (g.get("detail") or {}).get("setup")

    nearest_gate, nearest_margin = _nearest_passing_gate(entry_decision_trace)
    alpha_c, market_c, residual_c = _attribute_return(realized, bench, entry_alpha)

    if realized < bench and entry_alpha is not None and entry_alpha > 0:
        notes.append("alpha_predicted_positive_but_underperformed_bench")
    if nearest_gate is not None and nearest_margin < 0.20:
        notes.append(f"thin_margin_at_entry_on_{nearest_gate}")
    if realized < -0.10:
        notes.append("realized_loss_exceeded_10pct")

    return PerTradePostMortem(
        trade_id=trade_id,
        entry_date=pd.Timestamp(entry_date),
        exit_date=pd.Timestamp(exit_date),
        symbol=symbol,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        holding_days=holding_days,
        realized_pnl_pct=float(realized),
        benchmark_return_pct=float(bench),
        excess_return_pct=float(excess),
        entry_alpha=entry_alpha,
        setup_label=setup_label,
        entry_decision_trace=entry_decision_trace,
        nearest_passing_gate=nearest_gate,
        nearest_passing_margin=nearest_margin,
        attribution_alpha=alpha_c,
        attribution_market=market_c,
        attribution_residual=residual_c,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch (blotter)
# ---------------------------------------------------------------------------

def analyze_blotter(
    blotter: pd.DataFrame,
    *,
    decision_traces: pd.DataFrame | None = None,
    benchmark_prices: pd.Series | None = None,
    config: PostMortemConfig | None = None,
) -> list[PerTradePostMortem]:
    """Analyse a blotter of executed trades.

    Required columns in ``blotter``:
      * ``trade_id``
      * ``symbol``
      * ``entry_date``
      * ``exit_date``
      * ``entry_price``
      * ``exit_price``

    ``decision_traces``: optional long-form audit frame from
    ``traces_to_frame``. We reconstruct a per-(date, symbol)
    serialised trace by grouping back to the per-candidate level.

    ``benchmark_prices``: optional pd.Series indexed by trade_date.
    Missing → use 0 benchmark return.
    """
    if blotter is None or blotter.empty:
        return []
    required = {"trade_id", "symbol", "entry_date", "exit_date", "entry_price", "exit_price"}
    missing = required - set(blotter.columns)
    if missing:
        raise ValueError(f"blotter missing columns: {sorted(missing)}")

    bp = benchmark_prices
    if bp is not None:
        bp = pd.Series(bp).copy()
        bp.index = pd.to_datetime(bp.index)
        bp = bp.sort_index()

    # Reconstruct entry traces by (entry_date, symbol) from the long-form frame
    trace_lookup: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    if decision_traces is not None and not decision_traces.empty:
        dtr = decision_traces.copy()
        dtr["trade_date"] = pd.to_datetime(dtr["trade_date"])
        for (td, sym), grp in dtr.groupby(["trade_date", "symbol"]):
            gate_results = [
                {
                    "gate_name": r["gate_name"],
                    "passed": bool(r["gate_passed"]),
                    "reason": str(r["gate_reason"]),
                    "detail": {},  # long-form doesn't carry the full detail dict
                }
                for _, r in grp.iterrows()
            ]
            final = grp["final_decision"].iloc[-1]
            failed = grp["failed_gate"].iloc[-1]
            trace_lookup[(td, str(sym))] = {
                "candidate_id": f"{_iso(td)}|{sym}",
                "trade_date": _iso(td),
                "symbol": str(sym),
                "final_decision": str(final),
                "failed_gate": None if pd.isna(failed) else str(failed),
                "gate_results": gate_results,
            }

    results: list[PerTradePostMortem] = []
    for _, trade in blotter.iterrows():
        entry_dt = pd.Timestamp(trade["entry_date"])
        exit_dt = pd.Timestamp(trade["exit_date"])
        # Benchmark prices
        if bp is not None and not bp.empty:
            try:
                bench_entry = float(bp.reindex([entry_dt], method="ffill").iloc[0])
                bench_exit = float(bp.reindex([exit_dt], method="ffill").iloc[0])
            except (KeyError, IndexError):
                bench_entry = bench_exit = 1.0  # neutral
        else:
            bench_entry = bench_exit = 1.0

        trace = trace_lookup.get((entry_dt, str(trade["symbol"])), {})
        pm = analyze_trade(
            trade_id=str(trade["trade_id"]),
            entry_date=entry_dt,
            exit_date=exit_dt,
            symbol=str(trade["symbol"]),
            entry_price=float(trade["entry_price"]),
            exit_price=float(trade["exit_price"]),
            benchmark_entry_price=bench_entry,
            benchmark_exit_price=bench_exit,
            entry_decision_trace=trace,
            config=config,
        )
        results.append(pm)
    return results


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_post_mortem_reports(
    post_mortems: Sequence[PerTradePostMortem],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write per-trade JSON files + an aggregate CSV summary.

    Returns paths + a quick summary so callers can act on it.
    """
    out = Path(output_dir)
    (out / "trades").mkdir(parents=True, exist_ok=True)

    # Per-trade JSON
    for pm in post_mortems:
        path = out / "trades" / f"{pm.trade_id}.json"
        path.write_text(json.dumps(pm.to_dict(), indent=2, default=str), encoding="utf-8")

    # Summary CSV
    summary_rows = []
    for pm in post_mortems:
        summary_rows.append(
            {
                "trade_id": pm.trade_id,
                "symbol": pm.symbol,
                "entry_date": _iso(pm.entry_date),
                "exit_date": _iso(pm.exit_date),
                "holding_days": pm.holding_days,
                "realized_pnl_pct": pm.realized_pnl_pct,
                "benchmark_return_pct": pm.benchmark_return_pct,
                "excess_return_pct": pm.excess_return_pct,
                "nearest_passing_gate": pm.nearest_passing_gate,
                "nearest_passing_margin": pm.nearest_passing_margin,
                "attribution_alpha": pm.attribution_alpha,
                "attribution_market": pm.attribution_market,
                "attribution_residual": pm.attribution_residual,
                "n_notes": len(pm.notes),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Aggregate stats
    n = len(post_mortems)
    n_pos = int(sum(1 for p in post_mortems if p.realized_pnl_pct > 0))
    n_excess = int(sum(1 for p in post_mortems if p.excess_return_pct > 0))
    mean_realized = float(summary_df["realized_pnl_pct"].mean()) if n else 0.0
    mean_excess = float(summary_df["excess_return_pct"].mean()) if n else 0.0
    payload = {
        "n_trades": n,
        "win_rate_realized": n_pos / max(1, n),
        "win_rate_excess": n_excess / max(1, n),
        "mean_realized_pnl_pct": mean_realized,
        "mean_excess_pct": mean_excess,
        "trades_dir": str(out / "trades"),
        "summary_csv": str(summary_csv),
    }
    (out / "aggregate_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload
