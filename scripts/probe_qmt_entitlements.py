#!/usr/bin/env python3
"""Probe QMT/MiniQMT entitlements and emit the full evidence bundle.

Fail-closed and honest by construction: on a host that cannot reach MiniQMT the
script still writes every artifact, with the **complete capability catalogue**
marked PLATFORM_UNAVAILABLE rather than an empty file. A reader can then see the
scope of what is unknown instead of an absence.

    python scripts/probe_qmt_entitlements.py --output runtime/data/capabilities/qmt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from quantagent.data.providers import qmt_entitlement as ent  # noqa: E402
from quantagent.data.providers import qmt_gateway as gw  # noqa: E402
from quantagent.data.providers import qmt_skills as sk  # noqa: E402

#: Board-representative cohort. A probe that only tries 600519 proves nothing.
DEFAULT_COHORT: tuple[dict[str, str], ...] = (
    {"symbol": "600000.SH", "board": "SH_Main", "role": "liquid main board"},
    {"symbol": "000001.SZ", "board": "SZ_Main", "role": "liquid main board"},
    {"symbol": "300750.SZ", "board": "ChiNext", "role": "liquid ChiNext"},
    {"symbol": "688981.SH", "board": "STAR", "role": "liquid STAR"},
    {"symbol": "920002.BJ", "board": "BSE", "role": "Beijing Stock Exchange"},
)

#: Securities known to have carried ST status. Used as POSITIVE CONTROLS: if
#: these return empty, the dataset is unavailable rather than the market being
#: clean, and no security may be recorded as never-ST.
DEFAULT_ST_CONTROLS: tuple[str, ...] = ("000004.SZ", "000005.SZ", "600149.SH")
#: Large, continuously-listed names not expected to have ST history.
DEFAULT_NEVER_ST: tuple[str, ...] = ("600519.SH", "000651.SZ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runtime/data/capabilities/qmt")
    parser.add_argument("--earliest", default="19900101",
                        help="oldest date to request, to measure real history depth")
    parser.add_argument("--latest", default="20260724")
    args = parser.parse_args()

    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)

    gateway = gw.QmtGateway()
    environment = gateway.environment
    env_path = target / "environment.json"
    env_path.write_text(
        json.dumps(environment.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _, matrix = gw.build_matrix(gateway)
    written = matrix.write(target, stem="entitlement_matrix")

    # Period list: what the client claims to accept. Recorded from the SDK
    # catalogue when the client is unreachable, and labelled as such.
    periods = {
        "source": "xtdata documentation" if environment.verdict != ent.SERVING else "client",
        "periods": ["tick", "1m", "5m", "15m", "30m", "1h", "1d", "1w", "1mon",
                    "1q", "1hy", "1y", "l2quote", "l2quoteaux", "l2order",
                    "l2transaction", "l2transactioncount", "l2orderqueue"],
        "verified_against_client": environment.verdict == ent.SERVING,
        "source_url": "https://dict.thinktrader.net/nativeApi/xtdata.html",
    }
    (target / "period_list.json").write_text(
        json.dumps(periods, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    unavailable = environment.verdict != ent.SERVING
    blocked_reason = (
        environment.detail or environment.import_error
        or environment.connect_error or "client unavailable"
    )

    symbol_rows = [
        {**entry,
         "probe_status": ent.PLATFORM_UNAVAILABLE if unavailable else ent.NOT_PROBED,
         "detail": blocked_reason if unavailable else "pending live probe"}
        for entry in DEFAULT_COHORT
    ]
    pd.DataFrame(symbol_rows).to_parquet(target / "symbol_probe.parquet", index=False)

    download_probe = {
        "requested_earliest": args.earliest,
        "requested_latest": args.latest,
        "status": ent.PLATFORM_UNAVAILABLE if unavailable else ent.NOT_PROBED,
        "detail": blocked_reason if unavailable else "pending live probe",
        "measured_ranges": {},
        "note": (
            "History depth per period (daily / 1m / 5m / tick) MUST be measured "
            "against the real account. Public comparison tables are not evidence "
            "and are deliberately not copied into this artifact."
        ),
    }
    (target / "download_probe.json").write_text(
        json.dumps(download_probe, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if unavailable:
        st_probe = {
            "status": ent.PLATFORM_UNAVAILABLE,
            "detail": blocked_reason,
            "positive_controls": list(DEFAULT_ST_CONTROLS),
            "never_st_controls": list(DEFAULT_NEVER_ST),
            "entitlement_verdict": ent.PLATFORM_UNAVAILABLE,
            "interpretation": (
                "ST history was NOT probed. No security may be recorded as "
                "never-ST from this run, and st_intervals remains a mandatory "
                "unmet PIT field."
            ),
        }
    else:
        st_probe = gateway.probe_st_with_controls(
            known_st=DEFAULT_ST_CONTROLS, never_st=DEFAULT_NEVER_ST
        )
    (target / "st_probe.json").write_text(
        json.dumps(st_probe, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    level2 = {
        "status": ent.PLATFORM_UNAVAILABLE if unavailable else ent.NOT_PROBED,
        "detail": blocked_reason if unavailable else "pending live probe",
        "capabilities": ["l2quote", "l2quoteaux", "l2order", "l2transaction",
                         "l2transactioncount", "l2orderqueue"],
        "documented_requirement": "获取lv2数据时需要数据终端有lv2数据权限",
        "source_url": "https://dict.thinktrader.net/nativeApi/xtdata.html",
        "records_retrieved": 0,
        "note": "Level-2 is never claimed without real records.",
    }
    (target / "level2_probe.json").write_text(
        json.dumps(level2, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (target / "reconciliation.json").write_text(
        json.dumps({
            "status": ent.PLATFORM_UNAVAILABLE if unavailable else ent.NOT_PROBED,
            "detail": blocked_reason if unavailable else "pending QMT daily download",
            "note": "U0 remains the daily source of truth; QMT patches require evidence.",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    registry = sk.SkillRegistry(matrix)
    (target / "skill_inventory.json").write_text(
        json.dumps({
            "skills": registry.inventory(),
            "platform_is_windows": registry.platform_is_windows,
            "note": "A skill runs only when its required capability is SERVING.",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = matrix.summary()
    print(json.dumps({
        "environment_verdict": environment.verdict,
        "os": f"{environment.os_name} {environment.os_release}",
        "xtquant_installed": environment.xtquant_installed,
        "xtdata_importable": environment.xtdata_importable,
        "client_connected": environment.client_connected,
        "capabilities_catalogued": summary["capabilities"],
        "probe_status_counts": summary["probe_status_counts"],
        "permission_class_counts": summary["permission_class_counts"],
        "serving": summary["serving"],
        "st_entitlement_verdict": st_probe.get("entitlement_verdict"),
        "level2_records_retrieved": level2["records_retrieved"],
        "artifacts": sorted(str(p) for p in target.glob("*")),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
