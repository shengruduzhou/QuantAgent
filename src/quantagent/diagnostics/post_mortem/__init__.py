"""Stage 5.4 — per-trade post-mortem.

For each executed trade we reconstruct:
* the 14-gate decision trace at entry,
* the realized outcome (P&L, holding period, vs benchmark),
* counterfactuals (which gate was closest to blocking the trade?),
* attribution (alpha component vs market beta vs residual).
"""

from quantagent.diagnostics.post_mortem.analyzer import (
    PerTradePostMortem,
    PostMortemConfig,
    analyze_blotter,
    analyze_trade,
    write_post_mortem_reports,
)

__all__ = [
    "PerTradePostMortem",
    "PostMortemConfig",
    "analyze_blotter",
    "analyze_trade",
    "write_post_mortem_reports",
]
