#!/usr/bin/env python3
"""H-025 / EXP-025: sourced price-volume factor batch 3 (fu_20260713 mission).

13 a-priori candidates (frozen in HYPOTHESIS_REGISTRY.md H-025 BEFORE this run):
academically sourced families not explicitly spanned by the existing pool
(lottery MAX, realized skew, downside semivol, vol-of-vol, overnight
decomposition, FIP, volume stability, price-volume divergence, quiet volume,
liquidity shock, CGO daily-bar APPROXIMATION) + the batch-1 pre-queued D6
vol-compression re-gate on the medium-turnover track.

CPU-only. Screen window 2023-07-03..2025-08-29 (pre-quarantine, asserted by
the shared loader). Gates are a-priori: batch-1 semantics + novelty gate
(max |corr| vs REF <= 0.85) + medium-turnover track (cap 0.35, cost decay
<= 50%). Same-day close-to-close screen labels (identical to batches 1-2);
ranking use only — never a return claim.

Writes FACTOR_CANDIDATE_LEDGER_batch3.csv (repo root, batch-1/2 convention)
+ runtime/reports/full_universe/fu_20260713/factor_screen_leaderboard.csv.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts"))

from dual_track_factor_batch import (  # noqa: E402
    WIN, _load_panel, eps_add, rss_gib, score_factors,
)
from quantagent.factors import expr as E  # noqa: E402
from quantagent.factors.evaluation import forward_return_labels  # noqa: E402

R1 = E.Returns(E.Close, 1)
LOGV = E.Log(E.Add(E.Volume, E.Constant(1.0)))
_DM = E.Sub(R1, E.TsMean(R1, 20))
_S20 = E.TsStd(R1, 20)
_DOWN = E.Mul(R1, E.Div(E.Sub(E.Constant(1.0), E.Sign(R1)), E.Constant(2.0)))
_VWAP60 = E.Div(E.TsSum(E.Amount, 60), eps_add(E.TsSum(E.Volume, 60)))

FACTORS3 = {
    "M1_max_ret_neg_20": ("defensive", E.Mul(E.Constant(-1.0), E.TsMax(R1, 20))),
    "M2_skew_neg_20": ("defensive", E.Mul(E.Constant(-1.0), E.Div(
        E.TsMean(E.Mul(_DM, E.Mul(_DM, _DM)), 20),
        eps_add(E.Mul(_S20, E.Mul(_S20, _S20)))))),
    "M3_pv_corr_neg_20": ("pv_divergence", E.Mul(E.Constant(-1.0), E.TsCorr(E.Close, E.Volume, 20))),
    "M4_volume_quiet_5_60": ("low_turnover", E.Mul(E.Constant(-1.0), E.Log(
        eps_add(E.Div(E.TsMean(E.Volume, 5), eps_add(E.TsMean(E.Volume, 60))))))),
    "M5_clv_20": ("pv_path", E.TsMean(E.Div(
        E.Sub(E.Mul(E.Constant(2.0), E.Close), E.Add(E.High, E.Low)),
        eps_add(E.Sub(E.High, E.Low))), 20)),
    "M6_overnight_neg_20": ("pv_path", E.Mul(E.Constant(-1.0), E.TsMean(
        E.Sub(E.Div(E.Open, E.Delay(E.Close, 1)), E.Constant(1.0)), 20))),
    "M7_vov_neg_20": ("defensive", E.Mul(E.Constant(-1.0), E.TsStd(E.TsStd(R1, 5), 20))),
    "M8_semivol_neg_20": ("defensive", E.Mul(E.Constant(-1.0), E.TsStd(_DOWN, 20))),
    "M9_liq_shock_neg_20": ("low_turnover", E.Mul(E.Constant(-1.0), E.Div(
        E.Sub(LOGV, E.TsMean(LOGV, 20)), eps_add(E.TsStd(LOGV, 20))))),
    "M10_vol_cv_neg_20": ("pv_stability", E.Mul(E.Constant(-1.0), E.Div(
        E.TsStd(E.Volume, 20), eps_add(E.TsMean(E.Volume, 20))))),
    "M11_fip_20": ("momentum_quality", E.Mul(E.Sign(E.Returns(E.Close, 20)), E.TsMean(E.Sign(R1), 20))),
    "M12_cgo_vwap60_neg": ("chip_approx", E.Mul(E.Constant(-1.0),
        E.Sub(E.Div(E.Close, _VWAP60), E.Constant(1.0)))),
    "D6R_vol_compression_regate": ("defensive_medium", E.Mul(E.Constant(-1.0),
        E.Div(E.TsStd(R1, 5), eps_add(E.TsStd(R1, 60))))),
}

# novelty references (H-025 gate 5): existing survivor proxies + known-strong
# single factors on this universe/window
REF3 = {
    "mom20": E.Returns(E.Close, 20),
    "liq": E.TsMean(E.Amount, 20),
    "lowvol20": E.Mul(E.Constant(-1.0), E.TsStd(R1, 20)),   # materialized survivor D1
    "rev60": E.Mul(E.Constant(-1.0), E.Returns(E.Close, 60)),
    "pv_ret_corr": E.TsCorr(R1, E.Volume, 20),               # existing llm survivor family
}

TURNOVER_CAPS = {"defensive_medium": 0.35}
COST_DECAY_CLASSES = frozenset({"defensive_medium"})
MAX_REF_CORR = 0.85
OUT_LEDGER = REPO / "FACTOR_CANDIDATE_LEDGER_batch3.csv"
OUT_LEADERBOARD = REPO / "runtime/reports/full_universe/fu_20260713/factor_screen_leaderboard.csv"


def build_frame():
    panel = _load_panel()
    for name, (_, ex) in FACTORS3.items():
        panel[name] = ex.evaluate(panel).to_numpy()
    for name, ex in REF3.items():
        panel[name] = ex.evaluate(panel).to_numpy()
    return panel, {n: c for n, (c, _) in FACTORS3.items()}, list(REF3)


def main() -> int:
    t0 = time.time()
    panel, meta, refs = build_frame()
    lab = forward_return_labels(panel, horizons=(10, 20))
    lab = lab[(lab["trade_date"] >= WIN[0]) & (lab["trade_date"] <= WIN[1])].copy()
    df = score_factors(lab, meta, refs, OUT_LEDGER,
                       turnover_caps=TURNOVER_CAPS, max_ref_corr=MAX_REF_CORR,
                       cost_decay_classes=COST_DECAY_CLASSES)
    OUT_LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_LEDGER, OUT_LEADERBOARD)
    print(f"peak RSS {rss_gib():.2f} GiB, {time.time()-t0:.0f}s [batch=3 H-025 N={len(df)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
