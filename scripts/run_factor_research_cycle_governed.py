#!/usr/bin/env python3
"""Governed factor-research cycle with explicit universe semantics.

The underlying research engine remains ``run_factor_research_cycle.py``. This
entry point adds the stock-pool boundary required for auditable historical
research:

* ``point_in_time_membership`` filters every market row through effective-dated
  membership evidence before any factor or label calculation;
* ``research_universe_explicit_static`` allows a deliberately static symbol set
  for limited research, but stamps survivorship/generalisation warnings and can
  never be interpreted as broad-market/PIT evidence.

Universe choice is bound into every saved research report via a deterministic
contract SHA so runs from different membership versions cannot be silently
compared as if they used the same cross-section.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from quantagent.factors.universe_membership import (
    PIT_UNIVERSE_MODE,
    STATIC_UNIVERSE_MODE,
    UniverseMembershipError,
    dataframe_sha256,
    filter_market_by_membership,
    load_universe_membership,
    membership_artifact_for_window,
    static_membership_artifact,
    symbols_for_window,
)
from scripts.run_factor_research_cycle import (
    _load_market_calendar,
    _load_table,
    _research_baostock,
    _research_baostock_calendar,
    build_parser as build_base_parser,
    run_cycle,
)


GOVERNED_RESEARCH_SCHEMA = "factor_research_cycle_v4_pit_universe"


def _json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _static_symbols(args: argparse.Namespace, market_path: Path | None) -> tuple[str, ...]:
    requested = tuple(
        sorted({item.strip() for item in str(args.symbols or "").split(",") if item.strip()})
    )
    if requested:
        return requested
    if market_path is None:
        raise UniverseMembershipError(
            "explicit static research mode requires --symbols unless --market-panel is supplied"
        )
    frame = _load_table(market_path)
    if "symbol" not in frame.columns:
        raise UniverseMembershipError("provided market panel has no symbol column")
    symbols = tuple(sorted(frame["symbol"].dropna().astype(str).str.strip().unique()))
    if not symbols:
        raise UniverseMembershipError("provided market panel has no research symbols")
    return symbols


def _active_symbols(evidence, trade_date: pd.Timestamp) -> set[str]:
    date = pd.Timestamp(trade_date).normalize()
    frame = evidence.frame
    active = (frame["effective_from"] <= date) & (
        frame["effective_to"].isna() | (frame["effective_to"] >= date)
    )
    return set(frame.loc[active, "symbol"].astype(str))


def _validate_pit_market_coverage(
    raw_market: pd.DataFrame,
    evidence,
    sessions,
) -> dict[str, object]:
    """Fail closed when active members disappear because market data is missing."""

    market = raw_market.copy()
    if not {"symbol", "trade_date"}.issubset(market.columns):
        raise UniverseMembershipError(
            "PIT universe coverage validation requires symbol and trade_date market columns"
        )
    market["symbol"] = market["symbol"].astype(str).str.strip()
    market["trade_date"] = pd.to_datetime(
        market["trade_date"], errors="coerce"
    ).dt.normalize()
    if market["trade_date"].isna().any():
        raise UniverseMembershipError(
            "market data contains invalid trade_date before PIT universe coverage validation"
        )

    per_date: dict[str, dict[str, int]] = {}
    missing_samples: list[str] = []
    missing_total = 0
    for session in sessions:
        day = pd.Timestamp(session).normalize()
        expected = _active_symbols(evidence, day)
        if not expected:
            continue
        observed = set(
            market.loc[market["trade_date"] == day, "symbol"].astype(str).unique()
        )
        missing = sorted(expected - observed)
        if missing:
            missing_total += len(missing)
            if len(missing_samples) < 10:
                remaining = 10 - len(missing_samples)
                missing_samples.extend(
                    f"{symbol}@{day.date()}" for symbol in missing[:remaining]
                )
        per_date[str(day.date())] = {
            "expected_members": int(len(expected)),
            "observed_market_rows": int(len(expected & observed)),
            "missing_members": int(len(missing)),
        }
    if missing_total:
        raise UniverseMembershipError(
            "PIT universe market coverage is incomplete; active members cannot silently "
            f"drop out of the research cross-section. missing_members={missing_total}, "
            f"sample={missing_samples}"
        )
    return {
        "complete": True,
        "missing_member_rows": 0,
        "per_date": per_date,
    }


def _prepare_pit_input(
    args: argparse.Namespace,
    *,
    output: Path,
):
    evidence_path = str(args.universe_membership or "").strip()
    if not evidence_path:
        raise UniverseMembershipError(
            "point_in_time_membership requires --universe-membership"
        )
    if str(args.symbols or "").strip():
        raise UniverseMembershipError(
            "do not combine --symbols with PIT membership; membership evidence defines the cross-section"
        )
    evidence = load_universe_membership(
        evidence_path,
        universe_id=str(args.universe_id or "").strip() or None,
    )
    research_symbols = list(
        symbols_for_window(
            evidence,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )

    upstream_source: dict[str, object]
    if args.market_panel:
        original_path = Path(args.market_panel)
        raw_market = _load_table(original_path)
        upstream_source = {
            "mode": "provided_market_panel_before_universe_filter",
            "path": str(original_path),
            "input_sha256": _file_sha256(original_path),
            "production_integrity_certified": False,
        }
        calendar_path = str(getattr(args, "market_calendar", "") or "").strip()
        if not calendar_path:
            raise UniverseMembershipError(
                "PIT universe with --market-panel still requires independent --market-calendar"
            )
        governed_calendar = Path(calendar_path)
        sessions, calendar_meta = _load_market_calendar(
            governed_calendar,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        upstream_source["calendar"] = calendar_meta
    elif args.provider == "baostock":
        raw_market, upstream_source = _research_baostock(
            symbols=research_symbols,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        sessions, calendar_meta = _research_baostock_calendar(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        governed_calendar = output / "universe_input_market_calendar.csv"
        pd.DataFrame({"trade_date": sessions}).to_csv(governed_calendar, index=False)
        upstream_source = dict(upstream_source)
        upstream_source["calendar"] = calendar_meta
    else:
        raise UniverseMembershipError(
            "PIT universe requires --market-panel or --provider baostock"
        )

    coverage = _validate_pit_market_coverage(raw_market, evidence, sessions)
    filtered = filter_market_by_membership(raw_market, evidence)
    governed_market = output / "universe_filtered_market_panel.csv"
    filtered.to_csv(governed_market, index=False)

    artifact = membership_artifact_for_window(
        evidence,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    universe_artifact_path = output / "universe_membership.csv"
    artifact.to_csv(universe_artifact_path, index=False)
    contract_sha = dataframe_sha256(artifact)

    delegated = argparse.Namespace(**vars(args))
    delegated.market_panel = str(governed_market)
    delegated.market_calendar = str(governed_calendar)
    delegated.provider = "none"
    delegated.symbols = ""
    universe_meta = {
        "mode": PIT_UNIVERSE_MODE,
        "universe_id": evidence.universe_id,
        "point_in_time_membership": True,
        "membership_evidence_pit_valid": True,
        "membership_source_path": evidence.source_path,
        "membership_source_sha256": evidence.source_sha256,
        "membership_source_versions": list(evidence.source_versions),
        "membership_contract_sha256": contract_sha,
        "membership_artifact": universe_artifact_path.name,
        "membership_rows": int(len(artifact)),
        "union_symbol_count": int(len(research_symbols)),
        "market_rows_before_membership": int(len(raw_market)),
        "market_rows_after_membership": int(len(filtered)),
        "market_membership_coverage": coverage,
        "per_date_observed_member_rows": {
            str(pd.Timestamp(day).date()): int(count)
            for day, count in filtered.groupby("trade_date")["symbol"].nunique().items()
        },
        "current_constituent_backfill_blocked": True,
        "survivorship_bias_from_current_membership_blocked": True,
        "broad_market_generalization_certified": False,
        "production_integrity_certified": False,
        "production_note": (
            "effective-dated membership semantics and observed market coverage are validated "
            "and version-bound, but this research wrapper does not independently certify "
            "provider authority/completeness"
        ),
    }
    return delegated, universe_meta, upstream_source


def _prepare_static_input(
    args: argparse.Namespace,
    *,
    output: Path,
):
    if str(args.universe_membership or "").strip():
        raise UniverseMembershipError(
            "explicit static research mode must not carry --universe-membership"
        )
    market_path = Path(args.market_panel) if args.market_panel else None
    symbols = _static_symbols(args, market_path)
    artifact = static_membership_artifact(
        symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    artifact_path = output / "universe_membership.csv"
    artifact.to_csv(artifact_path, index=False)
    contract_sha = dataframe_sha256(artifact)
    universe_meta = {
        "mode": STATIC_UNIVERSE_MODE,
        "universe_id": str(args.universe_id or "").strip() or "explicit_static_research_universe",
        "point_in_time_membership": False,
        "membership_evidence_pit_valid": False,
        "membership_contract_sha256": contract_sha,
        "membership_artifact": artifact_path.name,
        "union_symbol_count": int(len(symbols)),
        "symbols": list(symbols),
        "current_constituent_backfill_blocked": True,
        "survivorship_bias_possible": True,
        "broad_market_generalization_certified": False,
        "production_integrity_certified": False,
        "warning": (
            "explicit static research universe; do not interpret as historical broad-market/index "
            "membership and do not use as production/promotion evidence"
        ),
    }
    return argparse.Namespace(**vars(args)), universe_meta, None


def _bind_universe_to_reports(output: Path, universe_meta: dict[str, object]) -> None:
    digest = str(universe_meta["membership_contract_sha256"])
    mode = str(universe_meta["mode"])
    for name in ("factor_validity.json", "factor_lifecycle_diagnostics.json"):
        path = output / name
        payload = _json_read(path)
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected {name} payload shape")
        for row in payload:
            if isinstance(row, dict):
                row["research_universe_mode"] = mode
                row["research_universe_contract_sha256"] = digest
        _json_write(path, payload)


def run_governed_cycle(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mode = str(args.universe_mode or "").strip()
    if mode == PIT_UNIVERSE_MODE:
        delegated, universe_meta, upstream_source = _prepare_pit_input(args, output=output)
    elif mode == STATIC_UNIVERSE_MODE:
        delegated, universe_meta, upstream_source = _prepare_static_input(args, output=output)
    else:
        raise UniverseMembershipError(f"unsupported universe mode {mode!r}")

    manifest = run_cycle(delegated)
    manifest = dict(manifest)
    manifest["schema_version"] = GOVERNED_RESEARCH_SCHEMA
    manifest["research_universe"] = universe_meta
    manifest["research_degrees_of_freedom"] = {
        "universe_choice_counted": True,
        "universe_contract_sha256": universe_meta["membership_contract_sha256"],
        "note": "trying a different universe/version is a separate research choice",
    }
    if upstream_source is not None:
        manifest["universe_pre_filter_source"] = upstream_source
    manifest["economic_live_eligible"] = False
    manifest["automatic_factor_activation"] = False
    manifest["research_only"] = True
    manifest["universe_membership_sha_required_for_comparison"] = True

    _bind_universe_to_reports(output, universe_meta)
    _json_write(output / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = __doc__
    parser.add_argument(
        "--universe-mode",
        choices=(PIT_UNIVERSE_MODE, STATIC_UNIVERSE_MODE),
        default=STATIC_UNIVERSE_MODE,
        help=(
            "point_in_time_membership requires effective-dated evidence; "
            "research_universe_explicit_static is limited-scope research only"
        ),
    )
    parser.add_argument(
        "--universe-membership",
        default="",
        help="PIT membership parquet/csv/jsonl with effective intervals and available_at",
    )
    parser.add_argument(
        "--universe-id",
        default="",
        help="Universe identifier to select when membership evidence contains multiple universes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_governed_cycle(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
