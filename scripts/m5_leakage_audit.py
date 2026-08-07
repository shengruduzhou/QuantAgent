#!/usr/bin/env python3
"""Evidence for M5-02: look-ahead audits that decline, and labels that start too early.

Regenerates the measurements behind DEF-025 and DEF-026.

DEF-025 — the DEF-023 gate hardening was defeated at the producer boundary. Making
a gate report `unknown` on a missing measurement accomplishes nothing while the
layer below fabricates one. Four fabrications did exactly that: `_pit_violations`
returned `0` for a frame it could not read, `_uses_mock_or_synthetic` returned
`False` for a frame with no provenance column, `_aggregate_metrics` wrote
`uses_mock_or_synthetic: False` as a constant, and the CLI injected
`pit_violation_count = 0`. The two gates this programme most needs to earn —
"no look-ahead" and "no synthetic data" — were recorded as *measured* passes.

DEF-026 — nothing had ever compared a row's availability stamp against its own
label window. The v7 builder stamped `available_at = next trading row` while
`v7_label_builder` opens the label window at `close(trade_date)`. Consequences
measured below: 100% of rows misaligned, and a third-party feature published
during the label window joining onto the row through `merge_pit_features` at rank
IC +1.0000.

Section C compares the repository's two live label conventions on real A-share
data — `v7_label_builder`'s `close(t+h)/close(t)-1` against
`gold_bridge.LABEL_CONVENTION`'s `close(t+1+h)/close(t+1)-1` — because which one
is in force decides how much of a measured IC is reachable. That comparison is an
open decision, not a defect, and it is recorded as such.

Usage:
    python scripts/m5_leakage_audit.py                    # synthetic sections only
    python scripts/m5_leakage_audit.py --with-real-panel  # adds section C (~2 min)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantagent.data.v7_dataset_builder import (  # noqa: E402
    build_market_features,
    merge_pit_features,
)
from quantagent.data.v7_label_builder import build_forward_return_labels  # noqa: E402
from quantagent.data.v7_quality_gates import (  # noqa: E402
    V7ModelAcceptanceGateConfig,
    audit_pit,
    audit_provenance,
    evaluate_data_quality_gates,
    evaluate_label_alignment,
    evaluate_model_acceptance_gates,
)

REAL_PANEL = PROJECT_ROOT / "runtime/data/v7/silver/market_panel/market_panel.parquet"
OUTPUT = PROJECT_ROOT / "docs/architecture/m5_leakage_audit.json"

#: Fixed so the synthetic sections are byte-identical across runs. An evidence
#: script that does not reproduce itself is not evidence.
SEED = 11


def synthetic_panel(days: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=days)
    rng = np.random.default_rng(SEED)
    rows = []
    for symbol in ("000001.SZ", "000002.SZ", "600000.SH", "600519.SH"):
        price = 10.0
        for date in dates:
            price = max(1.0, price * (1 + float(rng.normal(0, 0.02))))
            rows.append(
                {
                    "trade_date": date, "symbol": symbol,
                    "open": price * 0.99, "high": price * 1.01, "low": price * 0.98,
                    "close": price, "volume": 1_000_000.0, "amount": price * 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def build_dataset(market: pd.DataFrame) -> pd.DataFrame:
    features = build_market_features(market)
    labels = build_forward_return_labels(market, horizons=(1, 5)).frame
    carry = [
        column for column in labels.columns
        if column.startswith("forward_return_") or column.startswith("label_end_")
    ]
    return features.merge(
        labels[["symbol", "trade_date", *carry]], on=["symbol", "trade_date"], how="inner"
    )


def stale_stamp(dataset: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the stamp the builder used to write: the next trading row."""
    data = dataset.sort_values(["symbol", "trade_date"]).copy()
    data["available_at"] = data.groupby("symbol", sort=False)["trade_date"].shift(-1).values
    data["available_at"] = data["available_at"].fillna(data["trade_date"] + pd.Timedelta(days=1))
    return data


def per_date_rank_ic(frame: pd.DataFrame, feature: str, label: str, min_names: int = 2) -> float | None:
    valid = frame.dropna(subset=[feature, label])
    if valid.empty:
        return None
    per_date = valid.groupby("trade_date").apply(
        lambda f: f[feature].rank().corr(f[label].rank()) if len(f) >= min_names else np.nan,
        include_groups=False,
    ).dropna()
    return None if per_date.empty else round(float(per_date.mean()), 6)


def section_a_audits_that_decline() -> dict:
    """DEF-025: what the audits say when they have nothing to read."""
    bare = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "symbol": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.1],
            "forward_return_1d": [0.01, -0.01],
        }
    )
    pit = audit_pit(bare)
    provenance = audit_provenance(bare)

    # What the training path handed the gate before the fix.
    fabricated = {"uses_mock_or_synthetic": False, "pit_violation_count": 0}
    before = evaluate_model_acceptance_gates(fabricated, V7ModelAcceptanceGateConfig())
    after = evaluate_model_acceptance_gates({}, V7ModelAcceptanceGateConfig())

    def status_of(report, name: str) -> str:
        return next(gate["status"] for gate in report.gates if gate["name"] == name)

    return {
        "frame_columns": list(bare.columns),
        "pit_audit": {"status": pit.status, "violations": pit.violations},
        "provenance_audit": {
            "status": provenance.status,
            "uses_mock_or_synthetic": provenance.uses_mock_or_synthetic,
        },
        "gate_status_with_fabricated_producer_values": {
            "no_pit_violations": status_of(before, "no_pit_violations"),
            "no_mock_or_synthetic": status_of(before, "no_mock_or_synthetic"),
        },
        "gate_status_without_them": {
            "no_pit_violations": status_of(after, "no_pit_violations"),
            "no_mock_or_synthetic": status_of(after, "no_mock_or_synthetic"),
        },
        "measured_passes_on_an_unaudited_run": sum(
            1 for gate in after.gates if gate["status"] == "pass"
        ),
    }


def section_b_label_alignment(market: pd.DataFrame) -> dict:
    """DEF-026: the stamp against the label window, before and after."""
    dataset = build_dataset(market)
    stale = stale_stamp(dataset)

    fixed_report = evaluate_label_alignment(dataset)
    stale_report = evaluate_label_alignment(stale)

    labelled = dataset.dropna(subset=["forward_return_1d"])
    stale_labelled = stale.dropna(subset=["forward_return_1d"])

    # The leak channel: an extra feature knowable only at each session's close,
    # carrying that session's return. Honestly stamped — the dishonesty was never
    # in the input.
    extras = []
    for symbol, frame in market.sort_values(["symbol", "trade_date"]).groupby("symbol"):
        frame = frame.reset_index(drop=True)
        returns = frame["close"].pct_change()
        for index in range(1, len(frame)):
            extras.append(
                {
                    "symbol": symbol,
                    "available_at": frame.loc[index, "trade_date"],
                    "news_score": float(returns.iloc[index]),
                }
            )
    extras_frame = pd.DataFrame(extras)
    labels = build_forward_return_labels(market, horizons=(1,)).frame[
        ["symbol", "trade_date", "forward_return_1d"]
    ]

    features = build_market_features(market)
    fixed_merged = merge_pit_features(features, extras_frame, prefix="").merge(
        labels, on=["symbol", "trade_date"], how="inner"
    )
    stale_features = features.copy()
    stale_features["available_at"] = (
        stale_features.sort_values(["symbol", "trade_date"])
        .groupby("symbol", sort=False)["trade_date"].shift(-1).values
    )
    stale_features["available_at"] = stale_features["available_at"].fillna(
        stale_features["trade_date"] + pd.Timedelta(days=1)
    )
    stale_merged = merge_pit_features(stale_features, extras_frame, prefix="").merge(
        labels, on=["symbol", "trade_date"], how="inner"
    )

    quality = evaluate_data_quality_gates(dataset)
    return {
        "stale_stamp_next_trading_row": {
            "status": stale_report.status,
            "violation_rows": stale_report.violation_rows,
            "labelled_rows_1d": int(len(stale_labelled)),
            "share_of_1d_rows_misaligned": round(
                float(
                    (
                        pd.to_datetime(stale_labelled["available_at"])
                        > pd.to_datetime(stale_labelled["trade_date"])
                    ).mean()
                ), 6,
            ),
            "rank_ic_of_extra_published_during_label_window": per_date_rank_ic(
                stale_merged, "news_score", "forward_return_1d"
            ),
        },
        "fixed_stamp_own_close": {
            "status": fixed_report.status,
            "violation_rows": fixed_report.violation_rows,
            "labelled_rows_1d": int(len(labelled)),
            "share_of_1d_rows_misaligned": round(
                float(
                    (
                        pd.to_datetime(labelled["available_at"])
                        > pd.to_datetime(labelled["trade_date"])
                    ).mean()
                ), 6,
            ),
            "rank_ic_of_extra_published_during_label_window": per_date_rank_ic(
                fixed_merged, "news_score", "forward_return_1d"
            ),
            "entry_basis": fixed_report.entry_basis,
        },
        "dataset_quality_report": {
            "pit_violation_count": quality.metrics["pit_violation_count"],
            "uses_mock_or_synthetic": quality.metrics["uses_mock_or_synthetic"],
            "unknown_gates": [gate["name"] for gate in quality.unknowns],
        },
    }


def section_c_convention_cost() -> dict:
    """The open decision: how much of a measured IC survives a delayed entry.

    Not a defect. Both conventions are defensible and both are in this repository;
    the point of measuring is that the choice is not free, and its cost is not
    uniform across features.
    """
    if not REAL_PANEL.exists():
        return {"status": "skipped_no_real_panel", "path": str(REAL_PANEL)}

    panel = pd.read_parquet(REAL_PANEL, columns=["trade_date", "symbol", "close", "amount"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel.dropna(subset=["trade_date", "symbol", "close"])
    end = panel["trade_date"].max()
    start = end - pd.Timedelta(days=730)
    window = panel[panel["trade_date"].between(start, end)].copy()
    liquidity = window.groupby("symbol")["amount"].median().sort_values(ascending=False)
    window = window[window["symbol"].isin(set(liquidity.head(800).index))]
    window = window.sort_values(["symbol", "trade_date"])

    grouped = window.groupby("symbol", sort=False)
    window["return_1d"] = grouped["close"].pct_change()
    window["momentum_5d"] = grouped["close"].pct_change(5)
    window["momentum_20d"] = grouped["close"].pct_change(20)
    window["volatility_20d"] = grouped["return_1d"].transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    grouped = window.groupby("symbol", sort=False)
    window["undelayed_1d"] = grouped["close"].shift(-1) / window["close"] - 1.0
    window["delay1_1d"] = grouped["close"].shift(-2) / grouped["close"].shift(-1) - 1.0

    by_feature = {}
    for feature in ("momentum_5d", "momentum_20d", "volatility_20d", "return_1d"):
        undelayed = per_date_rank_ic(window, feature, "undelayed_1d", min_names=20)
        delayed = per_date_rank_ic(window, feature, "delay1_1d", min_names=20)
        share = (
            round(1.0 - delayed / undelayed, 6)
            if undelayed not in (None, 0) and delayed is not None else None
        )
        by_feature[feature] = {
            "rank_ic_entry_at_close_t": undelayed,
            "rank_ic_entry_at_close_t_plus_1": delayed,
            "share_of_ic_lost_to_the_delay": share,
        }
    return {
        "status": "measured",
        "panel_rows": int(len(window)),
        "panel_symbols": int(window["symbol"].nunique()),
        "panel_start": str(start.date()),
        "panel_end": str(end.date()),
        "by_feature": by_feature,
    }


def section_d_survivorship_on_the_shipped_gold() -> dict:
    """DEF-027, against the artifact that carries the FULL_UNIVERSE_GOLD_READY claim.

    Two findings live here. DEF-027: `build_masks` consulted a `status` column to
    avoid reading a missing delisting date as "confidently never delisted"
    (DEF-024); U0's H-032C master has no such column, so against *that* master the
    fix answered UNKNOWN for every name — it records the distinction in `source` and
    `status_end_blocked` instead.

    DEF-028: the audit then counted delisted names as those carrying a
    `mask_post_delisting == TRUE` row, i.e. a bar *after* the security stopped
    existing. A panel that correctly includes dead names and correctly stops each at
    its delisting date has none of those, so the verdict inverted on exactly the
    panels that are right. Both numbers are reported below so the reading is not
    left to interpretation.
    """
    # The master `build_u0_full_universe_gold.py` actually reads. U0 has two, and
    # the other one (`historical_security_master.parquet`, H-032C) carries neither
    # `status` nor any delisting date — judging the shipped artifact against it
    # produced a wrong verdict once already (DEF-028).
    master_path = PROJECT_ROOT / "runtime/data/u0/security_master.parquet"
    gold_path = PROJECT_ROOT / "runtime/data/gold/full_universe/dataset.parquet"
    if not master_path.exists() or not gold_path.exists():
        return {"status": "skipped_no_u0_artifacts"}

    from quantagent.data.ashare.gold_bridge import (  # noqa: PLC0415
        MASK_FALSE, MASK_UNKNOWN, build_masks, resolve_listing_status,
    )
    from quantagent.data.v7_quality_gates import evaluate_survivorship  # noqa: PLC0415

    master = pd.read_parquet(master_path)
    resolved, basis = resolve_listing_status(master)
    counts = resolved.value_counts().to_dict()

    panel = pd.read_parquet(gold_path, columns=["symbol", "trade_date", "mask_post_delisting"])
    shipped_mask = panel["mask_post_delisting"].value_counts().to_dict()
    rebuilt = build_masks(panel.drop(columns=["mask_post_delisting"]), master=master)
    report = evaluate_survivorship(rebuilt)

    dead = set(resolved[resolved == "delisted"].index)
    present = panel[panel["symbol"].astype(str).isin(dead)]
    dated = (
        pd.to_datetime(master.set_index(master["symbol"].astype(str))["delisting_date"],
                       errors="coerce")
        if "delisting_date" in master.columns else pd.Series(dtype="datetime64[ns]")
    )
    last_bar = present.groupby(present["symbol"].astype(str))["trade_date"].max()
    overrun = int((last_bar > last_bar.index.map(dated)).sum()) if len(last_bar) else 0
    return {
        "status": "measured",
        "master": {
            "rows": int(len(master)),
            "listing_status_basis": basis,
            "resolved": {str(key): int(value) for key, value in counts.items()},
            "delisting_dates_available": int(master["delisting_date"].notna().sum())
            if "delisting_date" in master.columns else None,
        },
        "shipped_gold_artifact": {
            "rows": int(len(panel)),
            "symbols": int(panel["symbol"].nunique()),
            "mask_post_delisting": {str(key): int(value) for key, value in shipped_mask.items()},
            "delisted_names_present": int(present["symbol"].nunique()),
            "sessions_contributed_by_delisted_names": int(len(present)),
            "share_of_panel": round(float(len(present) / len(panel)), 6),
            # The number that decides whether all-FALSE is right or wrong. A panel
            # that stops each dead name on time has zero post-delisting rows *and*
            # an all-FALSE mask; those are the same fact, not a contradiction.
            "delisted_names_with_bars_past_their_delisting_date": overrun,
        },
        "rebuilt_with_the_resolver": {
            "mask_post_delisting": {
                MASK_FALSE: int((rebuilt["mask_post_delisting"] == MASK_FALSE).sum()),
                MASK_UNKNOWN: int((rebuilt["mask_post_delisting"] == MASK_UNKNOWN).sum()),
            },
            "survivorship_status": report.status,
            "survivorship_unknown_sessions": report.unknown_sessions,
            "survivorship_undated_delisted_symbols": report.undated_delisted_symbols,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-real-panel", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    market = synthetic_panel()
    payload = {
        "schemaVersion": 1,
        "defects": {
            "DEF-025": "audits that could not run reported clean values; producers "
                       "fabricated the two measurements the strictest gates read",
            "DEF-026": "the availability stamp fell inside the row's own label window, "
                       "in 100% of rows, and was the as-of join key for third-party features",
            "DEF-027": "the survivorship mask read a `status` column one of U0's two "
                       "masters does not have, so against that master it could not tell the "
                       "358 dead names from the 5,530 live ones",
            "DEF-028": "the audit counted delisted names as those with a bar *after* their "
                       "delisting, which a correctly-stopped panel has none of — so it "
                       "reported the healthiest possible panel as the most suspicious",
        },
        "sectionA_audits_that_decline": section_a_audits_that_decline(),
        "sectionB_label_alignment": section_b_label_alignment(market),
        "sectionC_convention_cost": (
            section_c_convention_cost() if args.with_real_panel
            else {"status": "not_requested"}
        ),
        "sectionD_survivorship_on_the_shipped_gold": (
            section_d_survivorship_on_the_shipped_gold() if args.with_real_panel
            else {"status": "not_requested"}
        ),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwritten: {target}", file=sys.stderr)

    section_b = payload["sectionB_label_alignment"]
    ok = (
        payload["sectionA_audits_that_decline"]["pit_audit"]["violations"] is None
        and section_b["fixed_stamp_own_close"]["status"] == "pass"
        and section_b["stale_stamp_next_trading_row"]["status"] == "fail"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
