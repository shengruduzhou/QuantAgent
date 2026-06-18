"""Dependency-free shared assets for the RD-Agent-style factor loop.

This is a *leaf* module: it imports nothing from QuantAgent and nothing that
touches the network, so the core ``factor_synthesis`` loop can use the
persistent accept/reject memory and the DSL prompt catalogue without pulling
in the LLM client. The actual LLM proposer (``llm_factor_proposer``) and the
standalone ``llm_formula_alpha_candidates`` script both import from here, so
the DSL node catalogue and the memory schema have a single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

# Fast models the proposer cycles to when the primary (often a slow "thinking"
# model under quota throttling) times out or returns nothing usable.
FALLBACK_MODELS: tuple[str, ...] = ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash")

# The exact DSL nodes the LLM is allowed to emit. Anything outside this set
# fails ``parse_expression`` and is rejected before it can touch data. Kept in
# repr form so the model's output is directly parseable.
ALLOWED_NODES: tuple[str, ...] = (
    "Column('open'|'high'|'low'|'close'|'volume'|'amount')",
    "OptionalColumn('pb'|'roe'|'gross_margin'|'debt_to_asset')  # PIT fundamentals; only these resolve to data",
    "Constant(float)",
    "Add(left,right)", "Sub(left,right)", "Mul(left,right)", "Div(numerator,denominator)",
    "Abs(expr)", "Sign(expr)", "Log(expr)", "Rank(expr)", "CsZscore(expr)",
    "Returns(expr, periods)", "Delay(expr, periods)", "Delta(expr, periods)",
    "TsRank(expr, window)", "TsMean(expr, window)", "TsStd(expr, window)",
    "TsSum(expr, window)", "TsMax(expr, window)", "TsMin(expr, window)",
    "TsCorr(left, right, window)", "TsCov(left, right, window)",
    "DecayLinear(expr, window)",
)

# Economic structures known to carry signal on China A-shares — nudges the
# model toward hypotheses with a real mechanism rather than operator soup.
A_SHARE_STRUCTURES: tuple[str, ...] = (
    "short-term reversal", "volume-price divergence", "turnover crowding",
    "liquidity discount", "low volatility", "decayed momentum",
    "intraday positioning (close vs high-low range)", "overnight gap behaviour",
    "volume shock vs trailing distribution", "price vs rolling anchor (52w-high style)",
)

# Escalating research directive, mirroring RD-Agent's RAG escalation: explore
# cheap, attributable factors first; only after the easy space is mapped do we
# ask for richer, higher-IC interaction structures.
RAG_EASY = (
    "Try the easiest, fastest, most clearly-motivated factors first, sweeping different "
    "economic perspectives so that failures are attributable and the SOTA library accumulates "
    "cleanly. Prefer one operator on top of a clear price/volume primitive."
)
RAG_HIGH_IC = (
    "The easy single-mechanism factors have been mapped. Now propose richer factors that can "
    "achieve higher rank-IC: cross-sectional interactions (e.g. price-rank vs volume-rank "
    "correlations), decayed/normalized combinations, and conditional structures — while keeping "
    "every formula economically interpretable and inside the node and complexity budget."
)


def load_memory(path: Path | str | None, max_entries: int = 80) -> list[dict[str, Any]]:
    """Load the most recent ``max_entries`` accept/reject records (JSONL)."""
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-max_entries:]


def append_memory(path: Path | str | None, rows: Sequence[dict[str, Any]]) -> None:
    """Append accept/reject records to the persistent JSONL memory."""
    if path is None or not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# Keyword map used to bucket a free-text hypothesis/description into one of the
# economic structures above. First match wins; anything unmatched is "other".
STRUCTURE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("short-term reversal", ("reversal", "revert", "mean-revert", "mean revert", "contrarian")),
    ("volume-price divergence", ("divergence", "volume-price", "price-volume", "volume price", "price volume")),
    ("turnover crowding", ("turnover", "crowd")),
    ("liquidity discount", ("liquidity", "illiquid", "amihud")),
    ("low volatility", ("low vol", "low-vol", "volatility", "idiosyncratic vol", "variance")),
    ("decayed momentum", ("momentum", "decay", "trend")),
    ("intraday positioning (close vs high-low range)", ("intraday", "high-low", "high low", "close location", "range position", "clv")),
    ("overnight gap behaviour", ("overnight", "gap", "open vs", "open-to")),
    ("volume shock vs trailing distribution", ("volume shock", "volume spike", "abnormal volume", "volume surge")),
    ("price vs rolling anchor (52w-high style)", ("anchor", "52", "rolling high", "near high", "high-water", "distance to high")),
)

_HORIZONS: tuple[str, ...] = ("short_5d", "mid_5d_30d", "long_30d_120d")


def classify_structure(text: str) -> str:
    """Bucket a hypothesis/description string into a known economic structure."""
    blob = str(text or "").lower()
    for structure, keywords in STRUCTURE_KEYWORDS:
        if any(kw in blob for kw in keywords):
            return structure
    return "other"


def classify_horizon(hint: str | int | None) -> str:
    """Normalise a free-text/numeric horizon hint to a canonical bucket."""
    if isinstance(hint, (int, float)):
        w = int(hint)
        if w <= 5:
            return "short_5d"
        if w <= 30:
            return "mid_5d_30d"
        return "long_30d_120d"
    blob = str(hint or "").lower()
    if blob in _HORIZONS:
        return blob
    if "long" in blob or "120" in blob:
        return "long_30d_120d"
    if "mid" in blob or "medium" in blob or "30" in blob:
        return "mid_5d_30d"
    if "short" in blob or "5d" in blob:
        return "short_5d"
    return "unspecified"


def coverage_map(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate accept/reject knowledge by (structure, horizon).

    This is the lightweight port of RD-Agent's knowledge management: instead of
    a flat recent-list, the proposer is shown which economic structures × horizons
    have been mined (with the best out-of-sample tradable IC achieved) and which
    have been attempted repeatedly without ever surviving the gates — so each new
    round can steer at the orthogonal whitespace rather than re-mining dead ends.
    """
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entries:
        structure = str(e.get("structure") or "other")
        horizon = str(e.get("horizon") or "unspecified")
        cell = cells.setdefault(
            (structure, horizon),
            {"structure": structure, "horizon": horizon, "attempted": 0,
             "accepted": 0, "best_validation_rank_ic": 0.0},
        )
        cell["attempted"] += 1
        if e.get("status") == "selected":
            cell["accepted"] += 1
            ic = e.get("validation_rank_ic")
            try:
                ic_f = float(ic)
            except (TypeError, ValueError):
                ic_f = None
            if ic_f is not None and ic_f > cell["best_validation_rank_ic"]:
                cell["best_validation_rank_ic"] = ic_f
    covered = sorted(
        cells.values(),
        key=lambda c: (-c["accepted"], -c["best_validation_rank_ic"], -c["attempted"]),
    )
    crowded = [c for c in covered if c["attempted"] >= 3 and c["accepted"] == 0]
    return {"cells": covered, "crowded_but_failing": crowded}


def uncovered_directions(
    entries: Sequence[dict[str, Any]],
    structures: Sequence[str] = A_SHARE_STRUCTURES,
) -> list[str]:
    """Economic structures with no accepted factor yet — the orthogonal whitespace."""
    accepted_structs = {str(e.get("structure")) for e in entries if e.get("status") == "selected"}
    return [s for s in structures if s not in accepted_structs]


def memory_digest(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compress past outcomes into prompt-sized feedback for the next round."""
    rejected = [e for e in entries if e.get("status") != "selected"]
    selected = [e for e in entries if e.get("status") == "selected"]
    reason_counts: dict[str, int] = {}
    for e in rejected:
        key = str(e.get("status") or "rejected")
        reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "past_rejected_count_by_reason": reason_counts,
        "recent_rejected_examples": [
            {
                "expression": e.get("raw_expression") or e.get("expression"),
                "reason": e.get("status"),
                "validation_rank_ic": e.get("validation_rank_ic"),
            }
            for e in rejected[-12:]
        ],
        "recent_accepted_examples": [
            {
                "expression": e.get("expression"),
                "validation_rank_ic": e.get("validation_rank_ic"),
            }
            for e in selected[-8:]
        ],
    }


__all__ = [
    "FALLBACK_MODELS",
    "ALLOWED_NODES",
    "A_SHARE_STRUCTURES",
    "STRUCTURE_KEYWORDS",
    "RAG_EASY",
    "RAG_HIGH_IC",
    "load_memory",
    "append_memory",
    "memory_digest",
    "classify_structure",
    "classify_horizon",
    "coverage_map",
    "uncovered_directions",
]
