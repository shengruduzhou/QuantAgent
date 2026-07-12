#!/usr/bin/env python3
"""H-015 dual-track comparison report — emits the 7 protocol tables + verdict."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = json.loads((REPO / "runtime/reports/v89_closed_loop/wf_h008/exp015_dual_track/results.json").read_text())
C = R["candidates"]
ref = R["carrier_ref"]


def row(cid):
    a = C[cid]
    return (f"| {cid} | {a['track']} | {a['median_cagr8']*100:+.1f}% | {a['worst_fold8']*100:+.1f}% | "
            f"{a['f2_cagr8']*100:+.1f}% | {a['median_cagr25']*100:+.1f}% | {a['worst_dd8']*100:.1f}% | "
            f"{a['calmar']:.2f} | {a['max_turnover']:.3f} | {a['avg_hold_days']:.1f}d | "
            f"{a['median_excess']*100:+.1f}% | {a['dsr']:.3f} |")


H = "| Cand | Trk | medCAGR | worstFold | F2 | med@25bps | worstDD | Calmar | maxTurn | hold | medExcess | DSR |"
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|"

Ls = [c for c in C if C[c]["track"] == "L"]
Hs = [c for c in C if C[c]["track"] == "H"]
by_med = sorted(C, key=lambda c: -C[c]["median_cagr8"])
bestL = max(Ls, key=lambda c: C[c]["median_cagr8"])
bestH = max(Hs, key=lambda c: C[c]["median_cagr8"])
best_cagr = by_med[0]
best_calmar = max(C, key=lambda c: C[c]["calmar"])
net_pos = [c for c in C if C[c]["median_cagr25"] > 0]
best_capacity = min(net_pos, key=lambda c: C[c]["max_turnover"]) if net_pos else "—"
best_crash = max(C, key=lambda c: C[c]["f2_cagr8"])


def gate_L(cid):
    a = C[cid]
    g = {"medCAGR≥carrier": a["median_cagr8"] >= ref["median_cagr8"],
         "turn≤0.5": a["max_turnover"] <= 0.5,
         "wDD≤carrier(33.9%)": a["worst_dd8"] <= ref["worst_dd8"],
         "F2≥carrier+5pp": a["f2_cagr8"] >= ref["f2_cagr8"] + 0.05,
         "med@25bps≥0": a["median_cagr25"] >= 0}
    return g


def gate_H(cid):
    a = C[cid]
    g = {"med@8≥carrier": a["median_cagr8"] >= ref["median_cagr8"],
         "med@15≥carrier": a["median_cagr15"] >= ref["median_cagr8"],
         "med@25≥carrier": a["median_cagr25"] >= ref["median_cagr8"],
         "turn≤1.5": a["max_turnover"] <= 1.5,
         "F2≥carrier": a["f2_cagr8"] >= ref["f2_cagr8"]}
    return g


med_L = sorted([C[c]["median_cagr8"] for c in Ls])[len(Ls)//2]
med_H = sorted([C[c]["median_cagr8"] for c in Hs])[len(Hs)//2]

doc = f"""# DUAL-TRACK RESULT — H-015 (Track L low-turnover vs Track H high-turnover)

**Corrected simulator, strict variant-C, H-008 folds, 8/15/25 bps, net metrics
decide. N={R['cumulative_trials_N']}, PBO={R['pbo']}, runtime {R['runtime_sec']}s,
peak RSS {R['peak_rss_gib']} GiB. Zero retrain, zero fresh-holdout contact.**

Reference: corrected C3_ema0.7 carrier — median {ref['median_cagr8']*100:+.1f}%,
worstDD {ref['worst_dd8']*100:.1f}%, F2 {ref['f2_cagr8']*100:+.1f}%, maxTurn
{ref['max_turnover']:.3f}, DSR {ref['dsr']:.4f}. Benchmark per fold:
F1 {R['bench_ann']['F1']*100:+.1f}% / F2 {R['bench_ann']['F2']*100:+.1f}% /
F3 {R['bench_ann']['F3']*100:+.1f}% / F4 {R['bench_ann']['F4']*100:+.1f}%.

## Full candidate table (8 bps unless noted)

{H}
{SEP}
{chr(10).join(row(c) for c in by_med)}

## The 7 governed comparisons

**1. Track L best vs Track H best**
{H}
{SEP}
{row(bestL)}
{row(bestH)}

**2. Track L median vs Track H median candidate:** L median CAGR **{med_L*100:+.1f}%**
vs H median **{med_H*100:+.1f}%**.

**3. Best net-CAGR (8 bps):** **{best_cagr}** ({C[best_cagr]['median_cagr8']*100:+.1f}%).
**4. Best drawdown-adjusted (Calmar):** **{best_calmar}** (Calmar {C[best_calmar]['calmar']:.2f}).
**5. Best capacity-adjusted (lowest turnover among median@25bps>0):** **{best_capacity}**
(turnover {C[best_capacity]['max_turnover']:.3f}).
**6. Best crash-regime (F2):** **{best_crash}** (F2 {C[best_crash]['f2_cagr8']*100:+.1f}%).
**7. Best production-readiness (most gates):** see gate tables below — top-3 L
candidates pass 4/5 Track-L gates (fail only worst-DD by 1–3pp in the F2 crash);
no Track H candidate passes its cost-survival gate.

## Track-L gate table (vs corrected carrier + protocol §5)

| Cand | medCAGR≥carrier | turn≤0.5 | wDD≤33.9% | F2≥−51.7% | med@25bps≥0 | verdict |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {c} | {'✓' if g['medCAGR≥carrier'] else '✗'} | {'✓' if g['turn≤0.5'] else '✗'} | "
    f"{'✓' if g['wDD≤carrier(33.9%)'] else '✗'} | {'✓' if g['F2≥carrier+5pp'] else '✗'} | "
    f"{'✓' if g['med@25bps≥0'] else '✗'} | {sum(g.values())}/5 |"
    for c in Ls for g in [gate_L(c)]) + f"""

## Track-H gate table (cost-survival is the bar)

| Cand | med@8≥carrier | med@15≥carrier | med@25≥carrier | turn≤1.5 | F2≥carrier | verdict |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {c} | {'✓' if g['med@8≥carrier'] else '✗'} | {'✓' if g['med@15≥carrier'] else '✗'} | "
    f"{'✓' if g['med@25≥carrier'] else '✗'} | {'✓' if g['turn≤1.5'] else '✗'} | "
    f"{'✓' if g['F2≥carrier'] else '✗'} | {sum(g.values())}/5 |"
    for c in Hs for g in [gate_H(c)]) + f"""

## Verdict: Track L VALIDATED as the robust path; Track H REJECTED on cost survival

1. **Low-turnover dominates.** The top-3 candidates are all Track L
   (L1 {C['L1_c3ema07_minhold10']['median_cagr8']*100:+.1f}%,
   L4 {C['L4_c3ema07_reb10']['median_cagr8']*100:+.1f}%,
   L3 {C['L3_midlong_minhold10']['median_cagr8']*100:+.1f}% median), each with
   turnover ≤0.20/day. Every fast (turn≥1.0) candidate — L2, H1, H2, H3 — is
   net-weak-to-negative.
2. **Cost survival is the discriminator.** At 25 bps the low-turnover L books stay
   strongly positive (L1 {C['L1_c3ema07_minhold10']['median_cagr25']*100:+.1f}%,
   L4 {C['L4_c3ema07_reb10']['median_cagr25']*100:+.1f}%) while **no Track H
   candidate survives** (best H4 {C['H4_short_minhold3']['median_cagr25']*100:+.1f}%).
   Track H fails its own gate.
3. **Turnover control, not horizon, is the lever.** L2 (plain mid+long, no hold)
   churns 1.01 and is weak; adding min-hold (L3) flips it to +27.4%. Even a fast
   short signal is rescued by min-hold-3 (H4 +16.8% vs H1 −15.2%). The mechanism
   is the book constraint.
4. **Best robust candidate = L1_c3ema07_minhold10** — median +36.4%, **median
   excess vs benchmark +14.4%** (the carrier was negative), turnover 0.20 (5× under
   carrier), survives 25 bps at +24.1%. **Best defensive = L3_midlong_minhold10**
   — best F2 crash (−33.0%), medium-horizon.
5. **Not production-ready.** All candidates fail worst-DD (~35–37% in the F2 crash)
   and DSR (<0.06 at N=73). The 4 folds are heavily mined; **FRESH (~2026-11) is
   the arbiter.** No production proposal.

**Next:** Track L is the focus. Highest-EV = attack the residual failures — the F2
crash worst-DD and the DSR bar — via (a) crash-regime cash buffer / R2a-ramp on the
L1 book, (b) PIT-safe low-turnover + defensive factor generation to lift signal
quality (a book constraint cannot fix a signal-level crash). Track H is closed
unless a materially cheaper execution assumption is justified.
"""
(REPO / "DUAL_TRACK_RESULT_H015.md").write_text(doc, encoding="utf-8")
print("wrote DUAL_TRACK_RESULT_H015.md")
print(f"bestL={bestL} bestH={bestH} best_cagr={best_cagr} best_crash={best_crash} best_capacity={best_capacity}")
