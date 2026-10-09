"""Post-cost factor pruning is sign-aware and horizon-consistent (round-29 R3-F04).

A factor whose decile spread is negative is traded the other way round, so its
net is |gross| - cost; netting cost against the signed gross made a costlier
negative factor rank *higher*.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _write_panel(root: Path) -> None:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2018-01-01", periods=700)
    symbols = [f"{i:06d}.SZ" for i in range(120)]
    base_score = rng.normal(size=len(symbols))
    rows, a_vals, b_vals = [], [], []
    for d in dates:
        churny = base_score + rng.normal(scale=3.0, size=len(symbols))
        label = 0.01 * base_score + rng.normal(scale=0.02, size=len(symbols))
        for i, s in enumerate(symbols):
            rows.append((s, d, label[i], True))
            a_vals.append(-base_score[i])  # stable ranking: low churn
            b_vals.append(-churny[i])      # same sign, high churn
    gold = root / "runtime/data/gold/full_universe"
    gold.mkdir(parents=True)
    base = pd.DataFrame(rows, columns=["symbol", "trade_date", "forward_return_5d", "entry_feasible"])
    base.to_parquet(gold / "dataset.parquet")
    keys = base[["symbol", "trade_date"]]
    keys.assign(neg_stable=a_vals).to_parquet(gold / "factors_alpha101.parquet")
    keys.assign(neg_churny=b_vals).to_parquet(gold / "factors_gtja191.parquet")


def test_negative_factors_are_netted_sign_aware_and_low_churn_ranks_first(tmp_path):
    _write_panel(tmp_path)
    out = tmp_path / "report.json"
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}"}
    subprocess.run(
        [sys.executable, str(REPO / "scripts/prune_and_cost_price_factors.py"),
         "--cost-bps", "5", "--corr-dates", "20", "--output", str(out)],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    report = json.loads(out.read_text())
    top = report["top_by_net_spread"]
    assert [row["factor"] for row in top] == ["neg_stable", "neg_churny"]
    assert all(row["direction"] == -1.0 for row in top)
    assert all(row["net_spread"] > 0 for row in top)
    assert report["net_positive_count"] == 2
    assert report["horizon_sessions"] == 5
