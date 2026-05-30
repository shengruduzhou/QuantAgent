"""Daily decision report (spec section 12 — explains today's positioning).

Produces a single Markdown document that answers:

* Which sectors were picked today and why.
* Which stocks were picked and why.
* What the total position sizing is and why.
* Which candidates were rejected and which gate stopped them.
* Where the principal risks sit today.

The report is read-only — it does not modify any input. It is the
LLM-friendly view of the deterministic pipeline so an analyst can
audit a single day's decisions without rerunning the model.

LLM safety: this module is the **only** sanctioned place where a
natural-language summary of a decision day is generated, and it
strictly summarises *already-made* decisions. It cannot place
orders. It does not import any execution module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Input bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DailyDecisionInputs:
    """Everything the report needs in one record.

    Every field is optional — the report degrades to "(no data)"
    sections rather than crashing when an upstream input is absent.
    """

    as_of_date: pd.Timestamp
    target_weights: pd.Series | None = None       # symbol → weight (today's)
    prior_weights: pd.Series | None = None        # symbol → weight (yesterday)
    sector_pool: pd.DataFrame | None = None       # cols: sector_level_1, pool_tier
    sector_map: pd.DataFrame | None = None        # cols: symbol, sector_level_1
    fundamental_ranker: pd.DataFrame | None = None  # cols: symbol, composite_rank
    capital_flow_theses: pd.DataFrame | None = None  # thesis frame
    decision_traces: pd.DataFrame | None = None   # traces_to_frame output
    market_regime: str | None = None
    global_conviction: float | None = None
    gross_exposure: float | None = None
    risk_events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DailyDecisionReport:
    as_of_date: pd.Timestamp
    sections: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        order = (
            "summary",
            "sector_picks",
            "stock_picks",
            "position_sizing",
            "rejected_candidates",
            "risk_view",
            "thesis_corroboration",
        )
        parts: list[str] = [f"# Daily Decision Report — {self.as_of_date.date()}\n"]
        for key in order:
            if key in self.sections:
                parts.append(self.sections[key])
        return "\n\n".join(parts).strip() + "\n"

    def write(self, path) -> None:
        from pathlib import Path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _summary(inputs: DailyDecisionInputs) -> str:
    lines: list[str] = ["## Summary\n"]
    regime = inputs.market_regime or "unknown"
    conv = inputs.global_conviction
    gross = inputs.gross_exposure
    lines.append(f"- Market regime: **{regime}**")
    if conv is not None:
        lines.append(f"- Global conviction: **{conv:.3f}**")
    if gross is not None:
        lines.append(f"- Gross exposure: **{gross:.2%}**")
    if inputs.target_weights is not None and not inputs.target_weights.empty:
        n_long = int((inputs.target_weights > 0).sum())
        lines.append(f"- Names with long exposure: **{n_long}**")
    return "\n".join(lines)


def _sector_picks(inputs: DailyDecisionInputs) -> str:
    if inputs.target_weights is None or inputs.target_weights.empty:
        return "## Sector picks\n\n_(no target weights supplied)_"
    if inputs.sector_map is None or inputs.sector_map.empty:
        return "## Sector picks\n\n_(no sector_map supplied — cannot group)_"
    weights = inputs.target_weights.copy()
    weights.index = weights.index.astype(str)
    sm = inputs.sector_map.copy()
    sm["symbol"] = sm["symbol"].astype(str)
    joined = sm.drop_duplicates("symbol", keep="last").set_index("symbol")
    by_sector = (
        weights.to_frame("weight")
        .join(joined["sector_level_1"], how="left")
        .fillna({"sector_level_1": "UNKNOWN"})
        .groupby("sector_level_1")["weight"].sum()
        .sort_values(ascending=False)
    )
    tier_lookup: dict[str, str] = {}
    if inputs.sector_pool is not None and not inputs.sector_pool.empty:
        tier_lookup = dict(
            zip(
                inputs.sector_pool["sector_level_1"].astype(str),
                inputs.sector_pool["pool_tier"].astype(str),
            )
        )
    lines = ["## Sector picks\n", "| sector | weight | pool_tier |", "|---|---|---|"]
    for sector, w in by_sector.items():
        tier = tier_lookup.get(str(sector), "n/a")
        lines.append(f"| {sector} | {float(w):.2%} | {tier} |")
    return "\n".join(lines)


def _stock_picks(inputs: DailyDecisionInputs) -> str:
    if inputs.target_weights is None or inputs.target_weights.empty:
        return "## Stock picks\n\n_(no target weights supplied)_"
    weights = inputs.target_weights.copy()
    weights.index = weights.index.astype(str)
    weights = weights.sort_values(ascending=False)
    fr_lookup: dict[str, float] = {}
    if inputs.fundamental_ranker is not None and not inputs.fundamental_ranker.empty:
        fr = inputs.fundamental_ranker
        if "symbol" in fr.columns and "composite_rank" in fr.columns:
            fr_lookup = dict(
                zip(fr["symbol"].astype(str), fr["composite_rank"].astype(float))
            )
    prior_lookup: dict[str, float] = {}
    if inputs.prior_weights is not None and not inputs.prior_weights.empty:
        prior_lookup = inputs.prior_weights.to_dict()
    lines = [
        "## Stock picks\n",
        "| symbol | weight | Δ vs prior | fundamental rank |",
        "|---|---|---|---|",
    ]
    for sym, w in weights.head(20).items():
        prior = float(prior_lookup.get(str(sym), 0.0))
        delta = float(w) - prior
        fr_rank = fr_lookup.get(str(sym))
        fr_str = f"{fr_rank:.2f}" if fr_rank is not None else "n/a"
        lines.append(f"| {sym} | {float(w):.2%} | {delta:+.2%} | {fr_str} |")
    return "\n".join(lines)


def _position_sizing(inputs: DailyDecisionInputs) -> str:
    lines = ["## Position sizing\n"]
    if inputs.gross_exposure is None:
        lines.append("_(gross exposure not supplied)_")
        return "\n".join(lines)
    gross = inputs.gross_exposure
    lines.append(f"- Gross exposure: **{gross:.2%}**")
    cap = 0.60
    hc_cap = 0.80
    if gross <= cap:
        lines.append(f"- Within default cap ({cap:.0%}).")
    elif gross <= hc_cap:
        lines.append(
            f"- Above default cap ({cap:.0%}) but within high-conviction cap ({hc_cap:.0%})."
        )
        if inputs.global_conviction is not None:
            ok = inputs.global_conviction >= 0.80
            lines.append(
                f"  - global_conviction {inputs.global_conviction:.2f} "
                f"{'meets' if ok else 'does NOT meet'} the 0.80 threshold."
            )
    else:
        lines.append(
            f"- ⚠️ Above high-conviction cap ({hc_cap:.0%}). Decision-chain should have rejected this."
        )
    return "\n".join(lines)


def _rejected_candidates(inputs: DailyDecisionInputs) -> str:
    if inputs.decision_traces is None or inputs.decision_traces.empty:
        return "## Rejected candidates\n\n_(no decision traces supplied)_"
    df = inputs.decision_traces
    rejected = df[df["final_decision"] == "rejected"] if "final_decision" in df.columns else df.iloc[:0]
    if rejected.empty:
        return "## Rejected candidates\n\n_(none today)_"
    by_gate = rejected.groupby("failed_gate").size().sort_values(ascending=False)
    lines = ["## Rejected candidates\n", "| gate | count |", "|---|---|"]
    for gate, count in by_gate.items():
        lines.append(f"| {gate} | {int(count)} |")
    return "\n".join(lines)


def _risk_view(inputs: DailyDecisionInputs) -> str:
    if not inputs.risk_events:
        return "## Risk view\n\n_(no risk events today)_"
    counts: dict[str, int] = {}
    for evt in inputs.risk_events:
        et = str(evt.get("event_type", "unknown"))
        counts[et] = counts.get(et, 0) + 1
    lines = ["## Risk view\n", "| event_type | count |", "|---|---|"]
    for et, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {et} | {count} |")
    return "\n".join(lines)


def _thesis_corroboration(inputs: DailyDecisionInputs) -> str:
    th = inputs.capital_flow_theses
    if th is None or th.empty:
        return "## Thesis corroboration\n\n_(no capital-flow theses supplied)_"
    keep = th[
        [
            c for c in (
                "direction_kind", "direction_value", "thesis_sign",
                "confidence", "contradiction_score", "validation_status",
                "tradability_score",
            )
            if c in th.columns
        ]
    ].copy()
    top = keep.head(10)
    lines = ["## Thesis corroboration\n"]
    cols = list(top.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in top.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_daily_decision_report(inputs: DailyDecisionInputs) -> DailyDecisionReport:
    sections = {
        "summary": _summary(inputs),
        "sector_picks": _sector_picks(inputs),
        "stock_picks": _stock_picks(inputs),
        "position_sizing": _position_sizing(inputs),
        "rejected_candidates": _rejected_candidates(inputs),
        "risk_view": _risk_view(inputs),
        "thesis_corroboration": _thesis_corroboration(inputs),
    }
    return DailyDecisionReport(as_of_date=inputs.as_of_date, sections=sections)


__all__ = [
    "DailyDecisionInputs",
    "DailyDecisionReport",
    "build_daily_decision_report",
]
