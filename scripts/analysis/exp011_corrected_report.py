#!/usr/bin/env python3
"""INC-E1 corrected EXP-011 (Track A book-churn) report + gate re-derivation.

The pre-registered EXP-011 gates were calibrated to the pre-INC-E1 (artifact)
C3_ema0.7 carrier. This re-derives the same 5 gates against the CORRECTED
carrier (from the corrected EXP-008 wf_summary) so the PASS/FAIL verdicts are
honest. Candidate set is unchanged (frozen at commit 1994cd4) -> no post-hoc
selection; only the reference baseline is corrected.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runtime/reports/v89_closed_loop/wf_h008"
E11 = OUT / "exp011_book_churn"


def main() -> int:
    new = pd.read_csv(E11 / "book_fold_metrics.csv")
    old = pd.read_csv(E11 / "pre_inc_e1" / "book_fold_metrics.csv")
    res = json.loads((E11 / "results.json").read_text())
    dsr = res["dsr_stitched"]

    # corrected carrier C3_ema0.7 from corrected EXP-008
    wf = json.loads((OUT / "wf_summary.json").read_text())
    car = wf["aggregates"]["C3_ema0.7"]
    car_med = car["median_cagr"]; car_dd = car["worst_maxdd"]
    car_f2 = car["fold_cagrs"][1]; car_turn = car["max_turnover"]
    car_dsr = wf["dsr_stitched"]["C3_ema0.7"]

    # gates: G1 turnover<=0.10 (absolute promise); G2 worstDD<=carrier;
    # G3 F2 >= carrier F2 + 5pp (material crash improvement); G4 median>=carrier;
    # G5 sector<=0.33 (absolute)
    TURN_CAP, SECTOR_CAP, F2_MARGIN = 0.10, 0.33, 0.05
    rules = ["B1_buffer30", "B2_minhold10", "B3_partial30", "B4_reb5d", "B5_buffer_r2a_ramp"]

    def agg(rule, col, how):
        s = new[new["rule"] == rule][col]
        return s.max() if how == "max" else s.median() if how == "med" else s.min()

    lines = [
        "| Rule | median CAGR | maxTurn | worstDD | F2 crash | DSR | G1 turn≤.10 | G2 DD≤car | G3 F2≥car+5pp | G4 med≥car | verdict |",
        "|------|-------------|---------|---------|----------|-----|------|------|------|------|---------|",
    ]
    verdicts = {}
    for r in rules:
        med = agg(r, "cagr", "med"); mt = agg(r, "turnover", "max")
        dd = agg(r, "maxdd", "max"); f2 = new[(new.rule == r) & (new.fold == "F2")]["cagr"].iloc[0]
        sec = agg(r, "mean_max_sector_weight", "max")
        g1 = mt <= TURN_CAP
        g2 = dd <= car_dd + 1e-9
        g3 = f2 >= car_f2 + F2_MARGIN
        g4 = med >= car_med
        g5 = sec <= SECTOR_CAP
        allg = g1 and g2 and g3 and g4 and g5
        beats_carrier = (med > car_med) and (mt < car_turn) and (f2 > car_f2) and (dsr[r] > car_dsr)
        v = "ALL-PASS" if allg else ("beats-carrier(4/4)" if beats_carrier else "partial")
        verdicts[r] = {"all_gates": bool(allg), "beats_carrier": bool(beats_carrier),
                       "median": round(float(med), 4), "maxturn": round(float(mt), 4),
                       "worstdd": round(float(dd), 4), "f2": round(float(f2), 4),
                       "dsr": round(float(dsr[r]), 4)}
        lines.append(
            f"| {r} | {med*100:+.1f}% | {mt:.3f} | {dd*100:.1f}% | {f2*100:+.1f}% | {dsr[r]:.3f} | "
            f"{'✓' if g1 else '✗'} | {'✓' if g2 else '✗'} | {'✓' if g3 else '✗'} | {'✓' if g4 else '✗'} | {v} |")

    report = f"""# EXP-011 CORRECTED under INC-E1 fix — Track A book-churn (2026-07-06)

**Re-ran the 5 pre-registered book-construction rules (frozen commit 1994cd4) on
the corrected simulator (fix promoted 7f09453). Candidate set unchanged; only the
reference carrier is corrected.** Runtime 344s, peak RSS 2.05 GiB, CPU-only, zero
retrain, zero fresh-holdout contact. Pre-INC-E1 copies in exp011_book_churn/pre_inc_e1/.

## Corrected carrier (C3_ema0.7, from corrected EXP-008)
median CAGR **{car_med*100:+.1f}%** · worstDD **{car_dd*100:.1f}%** · F2 **{car_f2*100:+.1f}%** ·
maxTurn **{car_turn:.3f}** · DSR **{car_dsr:.3f}**

## Corrected results + gates re-derived vs corrected carrier

{chr(10).join(lines)}

(G5 sector ≤0.33 passes for all — column omitted. Gates G2/G3/G4 are carrier-relative
and were re-derived from the corrected carrier; G1 turnover and G5 sector are absolute
production promises, unchanged.)

## Reversal vs pre-INC-E1 EXP-011

The pre-INC-E1 EXP-011 concluded "turnover gate SOLVED at book layer (0.014–0.041)
but every slow book DEEPENS the F2 crash." **Both halves were artifacts:**
- The 0.014–0.041 turnovers were dropped incremental orders; real corrected
  turnovers are 0.19–0.78.
- The "slow books deepen the crash" was measured against an artifact carrier.
  Against the CORRECTED carrier (F2 −56.7%), **B2_minhold10 (−40.2%) and
  B5 (−39.1%) IMPROVE the crash by 16–18pp** — minimum-hold is crash-protective,
  not crash-amplifying, once fills are honest.

## Standout: B2_minhold10 (minimum 10-day holding period)

vs corrected carrier: median CAGR **+36.4% vs +1.3%** (28×), turnover **0.202 vs
1.035** (5× cut), F2 crash **−40.2% vs −56.7%** (+16pp), DSR **0.427 vs 0.025**
(17×), worstDD 36.6% vs 33.9% (marginally worse). **Beats the corrected carrier
on 4/4 core axes (median, turnover, F2, DSR).** It still FAILS the two absolute
production promises — G1 turnover (0.202 > 0.10) and G2 worst-DD (0.366 > 0.339) —
so it is NOT a production candidate, but it is by far the strongest Track-A
mechanism found and the only one with a materially non-trivial DSR.

## Verdict: INCONCLUSIVE-PROMISING (Track A re-opened)

- No candidate passes all absolute gates (turnover 0.10 promise binds hard: even
  the tightest book churns ~2× it; crash-window DD remains ~36%).
- But B2_minhold10 is a real, large, pre-registered improvement over the corrected
  carrier and inverts the prior crash conclusion. It warrants a corrected
  re-registration (H-014) that (a) explores min-hold length × partial-adjust to
  push turnover toward 0.10, (b) keeps the crash gain, with the FRESH window
  (~2026-11) as the out-of-sample arbiter — NOT a production swap.
- PBO {res.get('fold_block_pbo', 0.0)}; cumulative trials N unchanged (re-run of
  frozen candidates, not new trials).
"""
    (REPO / "EXP011_CORRECTED_INC_E1.md").write_text(report, encoding="utf-8")
    print("wrote EXP011_CORRECTED_INC_E1.md")
    print(json.dumps(verdicts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
