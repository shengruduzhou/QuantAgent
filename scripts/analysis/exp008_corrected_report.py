#!/usr/bin/env python3
"""INC-E1 corrected EXP-008 report + 15bps sensitivity regen.

Reads the corrected wf_h008 outputs (produced by exp008_walkforward_eval.py
under the promoted fix) and the preserved pre_inc_e1/ copies, emits a
before/after markdown table, and regenerates the report-only 15bps cost
sensitivity for C2/C3_ema0.7 under the corrected simulator (the old
cost_sensitivity_15bps.json was an ad-hoc pre-INC-E1 artifact with no
generator).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import baseline_protocol as bp  # noqa: E402
from exp008_walkforward_eval import (  # noqa: E402
    FOLDS, TOP_K, ANN, build_candidates, sleeve_frame, cagr,
)
from quantagent.backtest.ashare_execution_simulator import (  # noqa: E402
    AShareExecutionSimulationConfig,
)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008"
QUAR = pd.Timestamp("2025-09-01")


def regen_15bps() -> dict:
    """Report-only 15bps sensitivity for the two headline candidates, corrected."""
    sector = pd.read_parquet(REPO / bp.SECTOR)
    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
                  "amount", "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    out: dict[str, dict[str, float]] = {"C2_prod_rank110": {}, "C3_ema0.7": {}}
    for fold, spec in FOLDS.items():
        oos_s, oos_e = map(pd.Timestamp, spec["oos"])
        frame = sleeve_frame(fold)
        cands = build_candidates(frame[frame["trade_date"] <= oos_e], oos_s)
        panel = pd.read_parquet(REPO / bp.PANEL, columns=panel_cols,
                                filters=[("trade_date", ">=", oos_s - pd.Timedelta(days=10)),
                                         ("trade_date", "<=", oos_e)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
        trade_dates = sorted(panel["trade_date"].unique())
        cfg = AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=15.0)
        for name in ("C2_prod_rank110", "C3_ema0.7"):
            p = cands[name].merge(flags, on=["symbol", "trade_date"], how="left")
            tw = bp._target_weights(p, "alpha_score", TOP_K, eligible_only=True,
                                    delay_days=1, trade_dates=trade_dates)
            res = run_strict_backtest_v8(tw, panel, sector_map=sector, config=cfg)
            nav = res.nav.copy(); nav.index = pd.to_datetime(nav.index)
            r = nav.pct_change().dropna().to_numpy()
            out[name][fold] = round(cagr(r), 4)
    return out


def md_table(new: pd.DataFrame, old: pd.DataFrame) -> str:
    rows = ["| Fold | Candidate | CAGR pre→post | MaxDD pre→post | Turnover pre→post |",
            "|------|-----------|---------------|----------------|-------------------|"]
    m = old.merge(new, on=["fold", "candidate"], suffixes=("_o", "_n"))
    for _, r in m.iterrows():
        rows.append(
            f"| {r['fold']} | {r['candidate']} | "
            f"{r['cagr_o']*100:+.1f}% → **{r['cagr_n']*100:+.1f}%** | "
            f"{r['maxdd_o']*100:.1f}% → {r['maxdd_n']*100:.1f}% | "
            f"{r['turnover_o']:.3f} → **{r['turnover_n']:.3f}** |")
    return "\n".join(rows)


def main() -> int:
    new = pd.read_csv(OUT / "candidate_fold_metrics.csv")
    old = pd.read_csv(OUT / "pre_inc_e1" / "candidate_fold_metrics.csv")
    snew = json.loads((OUT / "wf_summary.json").read_text())
    sold = json.loads((OUT / "pre_inc_e1" / "wf_summary.json").read_text())

    print("regenerating corrected 15bps sensitivity ...")
    s15 = regen_15bps()
    (OUT / "cost_sensitivity_15bps_corrected.json").write_text(json.dumps(s15, indent=2))

    agg_new = snew["aggregates"]
    agg_old = sold["aggregates"]
    lines = ["| Candidate | median CAGR pre→post | worst fold pre→post | max turnover pre→post | DSR pre→post |",
             "|-----------|----------------------|---------------------|-----------------------|--------------|"]
    for c in agg_new:
        an, ao = agg_new[c], agg_old[c]
        dn = snew["dsr_stitched"][c]; do = sold["dsr_stitched"][c]
        lines.append(
            f"| {c} | {ao['median_cagr']*100:+.1f}% → **{an['median_cagr']*100:+.1f}%** | "
            f"{ao['min_cagr']*100:+.1f}% → {an['min_cagr']*100:+.1f}% | "
            f"{ao['max_turnover']:.3f} → **{an['max_turnover']:.3f}** | "
            f"{do:.3f} → **{dn:.3f}** |")

    report = f"""# EXP-008 CORRECTED under INC-E1 fix (2026-07-06)

**Trusted-evaluator order-dedup fix promoted (commit 7f09453). All 24 variant-C
fold evaluations re-run under the corrected simulator; pre-INC-E1 copies
preserved in `wf_h008/pre_inc_e1/`.** Runtime {snew.get('runtime_sec','?')}s,
peak RSS {snew.get('peak_rss_gib','?')} GiB. CPU-only, zero retraining, zero
fresh-holdout contact (all folds OOS < 2025-09-01, quarantine guard armed).

## Headline: the pre-INC-E1 "low turnover" was an order-drop artifact

Every candidate's true turnover is 3–13× higher than recorded. The EMA books
that appeared to "solve" the 0.10/day turnover gate (0.05–0.19) actually churn
0.57–1.04/day. **EXP-011's core claim — "turnover gate is SOLVED at the book
layer" — is refuted: it was dropped incremental orders, not low churn.**

## Aggregate before → after (pre-INC-E1 → corrected)

{chr(10).join(lines)}

- **fold-block PBO:** {sold['fold_block_pbo']} → **{snew['fold_block_pbo']}**
- **N (cumulative trials):** {snew.get('cumulative_trials_N')}

## Per-fold before → after

{md_table(new, old)}

## Corrected 15bps cost sensitivity (report-only, regenerated)

C2_prod_rank110: {json.dumps(s15['C2_prod_rank110'])}
C3_ema0.7:       {json.dumps(s15['C3_ema0.7'])}

## Interpretation

1. **DSR ≈ 0 for every blend** (max 0.0485, C3_ema0.3; was 0.55–0.74). After
   multiple-testing correction at N={snew.get('cumulative_trials_N')}, **no blend
   has a statistically significant turnover-adjusted Sharpe.** The family's
   apparent edge was substantially an execution artifact.
2. **Median fold CAGR collapsed** across the board; the incumbent-style
   C2_prod_rank110 is now the worst of the set on median ({agg_new['C2_prod_rank110']['median_cagr']*100:+.1f}%)
   and every candidate's median excess vs benchmark is negative
   ({min(agg_new[c]['median_excess'] for c in agg_new)*100:.0f}%..{max(agg_new[c]['median_excess'] for c in agg_new)*100:.0f}%).
3. **F2 crash is worse, not better** (−53.7%..−70.9% vs bench −33.1%);
   C3_ema0.7 remains least-bad at −56.7%. Crash exposure is signal-level.
4. **Direction preserved, magnitude destroyed:** EMA smoothing still dominates
   the fast daily-reselection books (C1/C2/median) on median CAGR and DD — the
   qualitative H-008 conclusion (smoothing helps; C2 incumbent is not a strong
   anchor) survives, but the economics are far poorer and the turnover gate is
   universally, badly violated.
5. **PBO fell to {snew['fold_block_pbo']}** only because everything is now
   consistently mediocre — with DSR ≈ 0 this is not a positive signal.

## Consequences for the mission

- **Track A (book-churn control) is re-opened, not closed.** Real churn is
  0.57–1.35/day, so genuine holding-period / partial-adjust / throttle rules may
  now *actually* help (their EXP-011 "success" was artifact). EXP-011's other
  conclusion ("every slow book deepens the F2 crash") must be re-tested under
  corrected fills before it is trusted.
- **Trust anchors (+17.3% / +17.25%) still pending re-run** — they pass through
  the same simulator and are expected to fall materially.
- **Production config unchanged** (red line); the incumbent looks weaker than
  believed but no auto-replacement. FRESH window (~2026-11) remains the arbiter.
"""
    (REPO / "EXP008_CORRECTED_INC_E1.md").write_text(report, encoding="utf-8")
    print("wrote EXP008_CORRECTED_INC_E1.md")
    print("wrote cost_sensitivity_15bps_corrected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
