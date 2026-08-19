"""How many of the price factors are independent, and which survive costs.

The round-23 screen established that these factors are not noise in the
tradable domain. It did not establish two things that decide whether they are
worth wiring into anything:

**Independence.** 144 distinct columns is not 144 bets. Alpha101 and GTJA-191
are both built from the same OHLCV primitives, so many are near-copies of each
other. Counting a cluster of eight correlated momentum variants as eight
factors overstates breadth by exactly the amount that matters for
diversification and for any multiple-testing correction. Exact duplicates were
already removed; this measures the near-duplicates.

**Cost.** Rank IC is gross. A factor whose ranking reshuffles every session
earns its IC by trading constantly, and at 12 bps of turnover it can be
strongly predictive and still lose money. The decile spread here is reported
both gross and net, and the net number is the one that decides anything.

Method notes that matter:

* Correlations are **cross-sectional Spearman, averaged over sampled dates** --
  never pooled across time, which would measure co-movement of factor levels
  rather than co-movement of their orderings.
* Turnover is measured on the **decile membership**, not the raw factor value:
  what costs money is a name entering or leaving the book, not the factor
  wobbling within a decile.
* Everything is restricted to `entry_feasible`, so nothing is earned on names
  that could not be bought.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOLD = Path("runtime/data/gold/full_universe")


def _load(family: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(GOLD / f"factors_{family}.parquet", columns=columns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="forward_return_5d")
    ap.add_argument("--cost-bps", type=float, default=12.0)
    ap.add_argument("--quantiles", type=int, default=10)
    ap.add_argument("--corr-dates", type=int, default=250)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--output", type=Path, default=GOLD / "factor_pruning_report.json")
    args = ap.parse_args()

    for family in ("alpha101", "gtja191"):
        if not (GOLD / f"factors_{family}.parquet").exists():
            print(f"BLOCKED_BY_DATA: factors_{family}.parquet missing", file=sys.stderr)
            return 2

    base = pd.read_parquet(
        GOLD / "dataset.parquet",
        columns=["symbol", "trade_date", args.label, "entry_feasible"],
    )
    base = base[base["entry_feasible"].astype(bool)]
    base = base[base[args.label].notna()].reset_index(drop=True)
    print(f"[domain] tradable labelled rows={len(base):,}", flush=True)

    import pyarrow.parquet as pq

    frames = []
    for family in ("alpha101", "gtja191"):
        names = [
            c for c in pq.ParquetFile(GOLD / f"factors_{family}.parquet").schema_arrow.names
            if c not in {"symbol", "trade_date"}
        ]
        piece = _load(family)
        # Drop the structural placeholders: an all-NaN column is not a factor and
        # would otherwise pad every count in this report.
        keep = [c for c in names if piece[c].notna().any()]
        piece = piece[["symbol", "trade_date", *keep]]
        for c in keep:
            piece[c] = piece[c].astype("float32")
        frames.append(piece)
        print(f"[load] {family} usable={len(keep)}", flush=True)

    merged = base.merge(frames[0], on=["symbol", "trade_date"], how="inner")
    merged = merged.merge(frames[1], on=["symbol", "trade_date"], how="inner")
    del frames
    factors = [
        c for c in merged.columns
        if c not in {"symbol", "trade_date", args.label, "entry_feasible"}
    ]
    print(f"[merged] rows={len(merged):,} factors={len(factors)}", flush=True)

    # ---- decile spread, gross and net -------------------------------------
    label = merged[args.label].to_numpy(dtype=float)
    # Decile membership must be compared by SYMBOL across dates. Comparing row
    # indices makes the intersection empty by construction -- the same name has
    # a different row index every session -- which reports churn as exactly 1.0
    # for every factor, a plausible-looking number measuring nothing.
    symbol_codes = merged["symbol"].astype("category").cat.codes.to_numpy()
    results = []
    grouped = merged.groupby("trade_date", sort=True)
    date_index = {d: i for i, d in enumerate(sorted(merged["trade_date"].unique()))}

    for n, factor in enumerate(factors, 1):
        values = merged[factor].to_numpy(dtype=float)
        top_ret, bot_ret, prev_top, prev_bot, turn = [], [], None, None, []
        for _, idx in grouped.indices.items():
            v = values[idx]
            ok = np.isfinite(v)
            if ok.sum() < 50:
                continue
            sub = idx[ok]
            v = values[sub]
            cut = max(1, int(len(v) / args.quantiles))
            order = np.argsort(v)
            bot_rows = sub[order[:cut]]
            top_rows = sub[order[-cut:]]
            top_ret.append(label[top_rows].mean())
            bot_ret.append(label[bot_rows].mean())
            bot = set(symbol_codes[bot_rows])
            top = set(symbol_codes[top_rows])
            if prev_top is not None:
                # One-sided membership churn on each leg, averaged.
                churn_t = 1.0 - len(top & prev_top) / max(len(top), 1)
                churn_b = 1.0 - len(bot & prev_bot) / max(len(bot), 1)
                turn.append((churn_t + churn_b) / 2.0)
            prev_top, prev_bot = top, bot

        if len(top_ret) < 100:
            continue
        gross = float(np.mean(top_ret) - np.mean(bot_ret))
        churn = float(np.mean(turn)) if turn else float("nan")
        # Both legs trade, so the round-trip cost per rebalance is 2 x churn.
        cost = 2.0 * churn * args.cost_bps / 10_000.0
        results.append(
            {"factor": factor, "gross_spread": gross, "decile_churn": churn,
             "cost_per_period": cost, "net_spread": gross - cost,
             "periods": len(top_ret)}
        )
        if n % 20 == 0 or n == len(factors):
            print(f"[spread] {n}/{len(factors)}", flush=True)

    spread = pd.DataFrame(results)
    spread["abs_net"] = spread["net_spread"].abs()
    spread = spread.sort_values("abs_net", ascending=False).reset_index(drop=True)

    # ---- cross-sectional correlation clustering ---------------------------
    all_dates = sorted(merged["trade_date"].unique())
    sample = all_dates[:: max(1, len(all_dates) // args.corr_dates)][: args.corr_dates]
    acc = np.zeros((len(factors), len(factors)))
    used = 0
    for d in sample:
        block = merged.loc[merged["trade_date"] == d, factors]
        if len(block) < 50:
            continue
        c = block.rank().corr(method="pearson").to_numpy()  # Spearman via ranks
        np.nan_to_num(c, copy=False)
        acc += c
        used += 1
    corr = acc / max(used, 1)

    # Greedy: keep the strongest net-spread factor, drop everything correlated
    # with it above the threshold, repeat. This counts INDEPENDENT bets, which
    # is the number that should feed any breadth or multiple-testing claim.
    pos = {f: i for i, f in enumerate(factors)}
    kept, dropped = [], {}
    for _, row in spread.iterrows():
        f = row["factor"]
        i = pos[f]
        clash = next((k for k in kept if abs(corr[i, pos[k]]) >= args.corr_threshold), None)
        if clash is None:
            kept.append(f)
        else:
            dropped[f] = {"absorbed_by": clash, "rho": round(float(corr[i, pos[clash]]), 4)}

    payload = {
        "label": args.label,
        "cost_bps": args.cost_bps,
        "domain": "entry_feasible only",
        "rows": int(len(merged)),
        "factors_evaluated": int(len(spread)),
        "correlation_dates_used": int(used),
        "correlation_threshold": args.corr_threshold,
        "independent_factors": len(kept),
        "absorbed_factors": len(dropped),
        "net_positive_count": int((spread["net_spread"] > 0).sum()),
        "median_decile_churn": round(float(spread["decile_churn"].median()), 4),
        "top_by_net_spread": spread.head(15).round(6).to_dict(orient="records"),
        "independent_set": kept,
        "absorbed": dropped,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in payload.items()
               if k not in {"top_by_net_spread", "independent_set", "absorbed"}}
    print(json.dumps(summary, ensure_ascii=False))
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
