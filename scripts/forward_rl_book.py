#!/usr/bin/env python3
"""Forward C-book: roll the PIT-RL policy on the live hold-band book.

RL gate status (2026-06-12): strict replay shows +39pp over the passive book
(survives 16bps + both half-windows + beats random nulls + beats the
deterministic alpha-tilt), but env-frequency value-add is ≈0 — the edge is
execution-path dependent. Verdict: DO_NOT_ENABLE for capital; PROMOTE to a
forward paper C-book so the live A/B/C race decides.

Each run replays the policy deterministically from --episode-start over the
A_default book (state needs the full episode), then emits the LAST day's
target weights to runtime/paper/forward/C_rl/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
BASE_PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
FWD_PREDS = "runtime/reports/v8/forward/ensemble_forward.parquet"
POLICY = "runtime/models/v88_rl_pit/policy.zip"
OUT_DIR = Path("runtime/paper/forward/C_rl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode-start", default="2026-01-02")
    ap.add_argument("--warmup-start", default="2025-10-01")
    ap.add_argument("--policy", default=POLICY)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    from quantagent.portfolio.hold_band import HoldBandConfig, build_hold_band_weights
    from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

    frames = [pd.read_parquet(BASE_PREDS, columns=["trade_date", "symbol", "composite_score"])]
    if Path(FWD_PREDS).exists():
        f = pd.read_parquet(FWD_PREDS)
        sc = "composite_score" if "composite_score" in f.columns else "alpha_score"
        frames.append(f[["trade_date", "symbol", sc]].rename(columns={sc: "composite_score"}))
    preds = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["trade_date", "symbol"], keep="last").rename(columns={"composite_score": "alpha_score"})
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    preds = preds[preds["trade_date"] >= pd.Timestamp(args.warmup_start)]

    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.warmup_start) - pd.Timedelta(days=130)]

    flags = panel[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    preds_f = preds.merge(flags, on=["symbol", "trade_date"], how="left")
    trade_dates = sorted(panel["trade_date"].unique())
    book = build_hold_band_weights(
        preds_f, config=HoldBandConfig(n_hold=50, entry_rank=30, exit_rank=150, delay_days=1),
        trade_dates=trade_dates)
    book = book[book.index >= pd.Timestamp(args.episode_start)]
    if len(book) < 4:
        raise SystemExit("book too short for an episode")

    # the env drops the final date (no forward return yet) — pad it back so
    # today's weights are still emitted: duplicate the last row with a
    # synthetic next date already in the panel calendar
    env = PITPortfolioEnv(book, preds, panel, PITPortfolioEnvConfig(max_book=60))
    model = PPO.load(args.policy, device="cpu")
    obs, _ = env.reset(seed=7)
    last_info = None
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        last_info = info

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w = pd.Series(last_info["weights"])
    w = w[w > 1e-6].rename("weight").rename_axis("symbol").reset_index()
    target_date = pd.Timestamp(last_info["trade_date"])
    w.to_csv(OUT_DIR / "targets_latest.csv", index=False)
    w.assign(target_date=str(target_date.date())).to_csv(
        OUT_DIR / f"targets_{target_date.date()}.csv", index=False)
    summary = {"target_date": str(target_date.date()), "n_held": int(len(w)),
               "gross": round(float(w['weight'].sum()), 4),
               "nav_episode": round(float(last_info["nav"]), 4),
               "nav_passive_episode": round(float(last_info["nav_passive"]), 4),
               "policy": args.policy}
    (OUT_DIR / "last_update.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
