#!/usr/bin/env python3
"""Stage 12 Task 1 — seed the PIT feature registry (Line A trainable / Line B forward-only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.registry import FeatureRegistry, FeatureSpec  # noqa: E402

# Line A — historical PIT-safe (trainable now)
LINE_A = [
    FeatureSpec("price_volume", "silver/market_panel OHLCV+amount", "T close", "lag0",
                "v1", "2010-01-01", "none", "2010-01-01", "2010-01-01", "production",
                notes="adjusted close=qfq; vol/amount raw (boardfix)"),
    FeatureSpec("alpha181", "factors/alpha101+alpha_extra on panel", "T close", "lag0",
                "v89", "2015-01-01", "low", "2015-01-01", "2020-06-01", "production",
                notes="cross-sectional; global-vs-cs standardization caveat"),
    FeatureSpec("fundamentals_roe_gm_pb", "silver/market_panel_fund (roe/gross_margin/debt/pb)",
                "report announce_date", "lag_to_announce", "v1", "2016-01-01", "low",
                "2016-01-01", "2026-07-01", "historical_walkforward",
                notes="quarterly-updating; PIT via available_at; Stage11: defensive, no CAGR add in bull"),
    FeatureSpec("earnings_yjbb_yjyg", "akshare stock_yjbb_em/stock_yjyg_em", "announce_date", "lag_to_announce",
                "v1", "2016-01-01", "low", "2016-01-01", "2026-06-30", "historical_walkforward"),
    FeatureSpec("order_labels_hist", "巨潮/cninfo 公告 text (中标/合同/客户认证/产能)", "announce_date", "lag0",
                "v1", "2016-01-01", "medium", "2016-01-01", "2026-07-01", "candidate",
                notes="historical reconstruction needed (Task2); currently forward live only"),
    FeatureSpec("sw1_sw2_industry", "tickflow sector_map (current snapshot)", "as_of snapshot", "lag0",
                "v1", "2018-01-01", "medium", "2018-01-01", "2026-05-31", "production",
                notes="LEAKAGE: current-snapshot membership -> survivorship+reclassification; Stage8/9 null"),
    FeatureSpec("market_style_beta", "computed from returns (all-A/CSI/size/value)", "T close", "lag0",
                "v1", "2010-01-01", "none", "2010-01-01", "2026-07-01", "historical_walkforward",
                notes="Stage11 beta_decomposition"),
    FeatureSpec("v89_composite_score", "retrain_plus7 ensemble_composite (walk-forward)", "T close", "lag0",
                "v89_plus7", "2024-08-09", "none", "2024-08-09", "2024-08-09", "production",
                notes="OOS by walk-forward construction; PRIMARY BASELINE; beta0.91 alpha+12.9%(phase-avg)"),
]

# Line B — forward-only concept features (train only AFTER 60-90d forward PIT accrues)
LINE_B = [
    FeatureSpec(n, "stage10 daily PIT snapshot", "snapshot as_of", "lag0", "v1",
                "2099-01-01", "forward_only", "2026-07-01", "2026-07-01", "forward_paper",
                notes="NO current-membership backfill; train after >=60-90 forward days")
    for n in ("concept_membership", "concept_strength", "concept_fundflow", "concept_breadth",
              "concept_limitup_diffusion", "stock_hardness", "order_labels_concept",
              "revenue_exposure", "concept_chain_state")
]


def main():
    reg = FeatureRegistry()
    for spec in LINE_A + LINE_B:
        reg.add(spec)
    errs = reg.validate_all()
    if errs:
        print("VALIDATION ERRORS:", errs); return 1
    reg.save()
    print(f"[registry] {len(reg.features)} features -> {reg.REGISTRY_PATH if hasattr(reg,'REGISTRY_PATH') else 'configs/feature_registry.json'}")
    tr = reg.trainable("2024-08-09")
    print(f"  trainable as of 2024-08-09 (Line A, excl forward-only): {len(tr)}")
    for f in tr:
        print(f"    {f.feature_name:<26} stage={f.lifecycle_stage:<22} leakage={f.leakage_risk} hash={f.schema_hash}")
    fo = [f for f in reg.features if f.leakage_risk == "forward_only"]
    print(f"  forward-only (Line B, gated): {len(fo)} -> {[f.feature_name for f in fo]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
