"""Governed QMT skill registry for agents and the workstation.

The rule this registry exists to enforce: **an LLM must not invoke a skill just
because the function name exists.** A skill runs only when a capability
certificate says the entitlement was actually measured as available, on a
platform that can reach it. Everything else is refused with a reason, and the
refusal is auditable.

Each skill declares its input and output schema, entitlement requirement,
platform requirement, network requirement, allowed output directories, timeout,
retry policy, and whether it may write. There is no shell, no arbitrary path,
and no trading permission anywhere in this module -- a skill whose name or
parameters suggest order submission is rejected before dispatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from quantagent.data.providers import qmt_entitlement as ent
from quantagent.data.providers.qmt_gateway import ALLOWED_OUTPUT_ROOTS

PLATFORM_WINDOWS = "WINDOWS"
PLATFORM_ANY = "ANY"

#: Refusal reasons, so a caller can branch on the cause rather than parse prose.
REFUSED_UNKNOWN_SKILL = "UNKNOWN_SKILL"
REFUSED_PLATFORM = "PLATFORM_NOT_SUPPORTED"
REFUSED_ENTITLEMENT = "ENTITLEMENT_NOT_GRANTED"
REFUSED_PARAMETERS = "INVALID_PARAMETERS"
REFUSED_PATH = "OUTPUT_PATH_NOT_ALLOWED"
REFUSED_TRADING = "TRADING_NOT_PERMITTED"

#: Parameter/skill fragments that indicate an order path. Rejected outright.
TRADING_MARKERS: tuple[str, ...] = (
    "order", "trade", "buy", "sell", "cancel", "position", "xttrader",
    "下单", "委托", "撤单", "交易",
)
#: Read-only skills whose names legitimately contain a trading-ish word. Without
#: this, probing the Level-2 *order* feed would be misread as order submission.
TRADING_MARKER_EXEMPT: frozenset[str] = frozenset({
    "qmt_probe_level2", "qmt_download_tick",
})

_SAFE_SYMBOL = re.compile(r"^[0-9]{6}\.(SH|SZ|BJ)$")
_SAFE_DATE = re.compile(r"^\d{8}$")


class SkillRefused(RuntimeError):
    """Raised when a skill may not run. Carries a machine-readable reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    #: Capability from the QMT catalogue that must be SERVING to run this skill.
    required_capability: str | None
    platform: str
    requires_network: bool
    writes_output: bool
    input_schema: Mapping[str, str]
    output_schema: Mapping[str, str]
    allowed_output_roots: tuple[str, ...] = ALLOWED_OUTPUT_ROOTS
    timeout_seconds: int = 300
    max_retries: int = 2
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"trading_permitted": False}


def _symbols(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [s.strip() for s in value.split(",") if s.strip()]
    if not isinstance(value, (list, tuple)):
        raise SkillRefused(REFUSED_PARAMETERS, f"symbols must be a list, got {type(value).__name__}")
    out: list[str] = []
    for symbol in value:
        if not _SAFE_SYMBOL.match(str(symbol)):
            raise SkillRefused(
                REFUSED_PARAMETERS,
                f"{symbol!r} is not a canonical A-share symbol (<6 digits>.SH|SZ|BJ)",
            )
        out.append(str(symbol))
    return out


def _date(value: Any, *, name: str) -> str:
    text = str(value)
    if not _SAFE_DATE.match(text):
        raise SkillRefused(REFUSED_PARAMETERS, f"{name} must be YYYYMMDD, got {value!r}")
    return text


SKILLS: dict[str, SkillSpec] = {
    "qmt_probe_environment": SkillSpec(
        name="qmt_probe_environment",
        description="Report OS, xtquant install state, MiniQMT connection and data dir.",
        required_capability=None, platform=PLATFORM_ANY,
        requires_network=False, writes_output=True,
        input_schema={}, output_schema={"environment": "object"},
    ),
    "qmt_probe_permissions": SkillSpec(
        name="qmt_probe_permissions",
        description="Probe every catalogued capability and emit the entitlement matrix.",
        required_capability=None, platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={}, output_schema={"matrix": "object"}, timeout_seconds=1800,
    ),
    "qmt_list_periods": SkillSpec(
        name="qmt_list_periods",
        description="List the period strings the client accepts.",
        required_capability=None, platform=PLATFORM_WINDOWS,
        requires_network=False, writes_output=True,
        input_schema={}, output_schema={"periods": "array"},
    ),
    "qmt_list_ashare_symbols": SkillSpec(
        name="qmt_list_ashare_symbols",
        description="List A-share instruments with board, listing and delisting dates.",
        required_capability="instrument_master", platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={"sector": "string?"}, output_schema={"symbols": "parquet"},
    ),
    "qmt_download_daily": SkillSpec(
        name="qmt_download_daily",
        description="Download daily bars for a symbol window (raw/front/back).",
        required_capability="daily_raw", platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={"symbols": "array<string>", "start": "YYYYMMDD",
                      "end": "YYYYMMDD", "dividend_type": "none|front|back"},
        output_schema={"partitions": "array<string>"}, timeout_seconds=3600,
    ),
    "qmt_download_minute": SkillSpec(
        name="qmt_download_minute",
        description="Download 1m/5m bars for a symbol window.",
        required_capability="minute_1m", platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={"symbols": "array<string>", "start": "YYYYMMDD",
                      "end": "YYYYMMDD", "period": "1m|5m"},
        output_schema={"partitions": "array<string>"}, timeout_seconds=3600,
    ),
    "qmt_download_tick": SkillSpec(
        name="qmt_download_tick",
        description="Download tick history for a symbol window (read-only).",
        required_capability="tick", platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={"symbols": "array<string>", "start": "YYYYMMDD", "end": "YYYYMMDD"},
        output_schema={"partitions": "array<string>"}, timeout_seconds=3600,
    ),
    "qmt_download_st_history": SkillSpec(
        name="qmt_download_st_history",
        description="Download historical ST/*ST/PT intervals with positive controls.",
        required_capability="st_history", platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={"symbols": "array<string>", "known_st_controls": "array<string>"},
        output_schema={"st_probe": "object"}, timeout_seconds=1800,
    ),
    "qmt_download_financials": SkillSpec(
        name="qmt_download_financials",
        description="Download financial statement tables.",
        required_capability="financial_statements", platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={"symbols": "array<string>", "tables": "array<string>"},
        output_schema={"partitions": "array<string>"}, timeout_seconds=3600,
    ),
    "qmt_probe_level2": SkillSpec(
        name="qmt_probe_level2",
        description="Probe l2quote/l2order/l2transaction/l2orderqueue entitlement (read-only).",
        required_capability=None, platform=PLATFORM_WINDOWS,
        requires_network=True, writes_output=True,
        input_schema={"symbols": "array<string>"}, output_schema={"level2_probe": "object"},
    ),
    "qmt_export_canonical_partitions": SkillSpec(
        name="qmt_export_canonical_partitions",
        description="Export staged QMT data as canonical Parquet plus manifest.",
        required_capability=None, platform=PLATFORM_ANY,
        requires_network=False, writes_output=True,
        input_schema={"staging_path": "string", "output_path": "string"},
        output_schema={"manifest": "object"},
    ),
    "qmt_reconcile_u0": SkillSpec(
        name="qmt_reconcile_u0",
        description="Reconcile QMT daily bars against the verified U0 panel.",
        required_capability=None, platform=PLATFORM_ANY,
        requires_network=False, writes_output=True,
        input_schema={"qmt_path": "string", "trade_date": "YYYY-MM-DD?"},
        output_schema={"reconciliation": "object"}, timeout_seconds=1800,
    ),
    "qmt_report_coverage": SkillSpec(
        name="qmt_report_coverage",
        description="Summarise measured QMT coverage per board and capability.",
        required_capability=None, platform=PLATFORM_ANY,
        requires_network=False, writes_output=True,
        input_schema={}, output_schema={"coverage": "object"},
    ),
}


@dataclass
class SkillAudit:
    """Append-only record of one dispatch decision."""

    skill: str
    decided_at: str
    allowed: bool
    reason: str | None
    platform: str
    required_capability: str | None
    capability_status: str | None
    parameters_accepted: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillRegistry:
    """Dispatches skills only when the capability certificate authorises them."""

    def __init__(
        self,
        matrix: ent.EntitlementMatrix | None = None,
        *,
        platform_is_windows: bool | None = None,
    ) -> None:
        import platform as _platform

        self.matrix = matrix
        self.platform_is_windows = (
            _platform.system().lower().startswith("win")
            if platform_is_windows is None else platform_is_windows
        )
        self.audit: list[SkillAudit] = []

    # -- introspection -----------------------------------------------------
    def inventory(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in SKILLS.values()]

    def capability_status(self, capability: str | None) -> str | None:
        if capability is None or self.matrix is None:
            return None
        for cell in self.matrix.cells:
            if cell.capability == capability:
                return cell.probe_status
        return None

    # -- gating ------------------------------------------------------------
    def _record(self, spec: SkillSpec | None, *, name: str, allowed: bool,
                reason: str | None, params: Mapping[str, Any] | None = None) -> None:
        self.audit.append(SkillAudit(
            skill=name,
            decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            allowed=allowed, reason=reason,
            platform="WINDOWS" if self.platform_is_windows else "NON_WINDOWS",
            required_capability=spec.required_capability if spec else None,
            capability_status=self.capability_status(spec.required_capability) if spec else None,
            parameters_accepted=dict(params or {}),
        ))

    def authorize(self, name: str, parameters: Mapping[str, Any] | None = None) -> SkillSpec:
        """Decide whether a skill may run. Raises :class:`SkillRefused` if not."""
        parameters = parameters or {}

        spec = SKILLS.get(name)
        if spec is None:
            self._record(None, name=name, allowed=False, reason=REFUSED_UNKNOWN_SKILL)
            raise SkillRefused(
                REFUSED_UNKNOWN_SKILL,
                f"{name!r} is not a registered QMT skill; registered: {sorted(SKILLS)}",
            )

        # Trading rejection comes first: no capability or platform state may
        # authorise an order path.
        if name not in TRADING_MARKER_EXEMPT:
            haystack = f"{name} {' '.join(str(k) + ' ' + str(v) for k, v in parameters.items())}".lower()
            hit = next((m for m in TRADING_MARKERS if m in haystack), None)
            if hit:
                self._record(spec, name=name, allowed=False, reason=REFUSED_TRADING)
                raise SkillRefused(
                    REFUSED_TRADING,
                    f"skill or parameters reference {hit!r}; this gateway is read-only "
                    "and no trading path may be dispatched",
                )

        if spec.platform == PLATFORM_WINDOWS and not self.platform_is_windows:
            self._record(spec, name=name, allowed=False, reason=REFUSED_PLATFORM)
            raise SkillRefused(
                REFUSED_PLATFORM,
                f"{name} requires Windows (MiniQMT); this host cannot reach it. "
                "Report NOT_RUN_PLATFORM rather than a failed probe.",
            )

        if spec.required_capability is not None:
            status = self.capability_status(spec.required_capability)
            if status != ent.SERVING:
                self._record(spec, name=name, allowed=False, reason=REFUSED_ENTITLEMENT)
                raise SkillRefused(
                    REFUSED_ENTITLEMENT,
                    f"{name} requires capability {spec.required_capability!r} to be "
                    f"SERVING; the certificate reports {status!r}. A documented API "
                    "is not a granted entitlement.",
                )

        accepted = self.validate_parameters(spec, parameters)
        self._record(spec, name=name, allowed=True, reason=None, params=accepted)
        return spec

    def validate_parameters(
        self, spec: SkillSpec, parameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Typed validation with no path traversal and no unknown keys."""
        accepted: dict[str, Any] = {}
        known = {k.rstrip("?") for k in spec.input_schema}
        unknown = set(parameters) - known
        if unknown:
            raise SkillRefused(
                REFUSED_PARAMETERS, f"unknown parameters for {spec.name}: {sorted(unknown)}"
            )

        if "symbols" in parameters:
            accepted["symbols"] = _symbols(parameters["symbols"])
        for key in ("start", "end"):
            if key in parameters:
                accepted[key] = _date(parameters[key], name=key)
        if "period" in parameters:
            period = str(parameters["period"])
            if period not in {"1m", "5m"}:
                raise SkillRefused(REFUSED_PARAMETERS, f"period must be 1m or 5m, got {period!r}")
            accepted["period"] = period
        if "dividend_type" in parameters:
            dividend = str(parameters["dividend_type"])
            if dividend not in {"none", "front", "back"}:
                raise SkillRefused(
                    REFUSED_PARAMETERS, f"dividend_type must be none/front/back, got {dividend!r}"
                )
            accepted["dividend_type"] = dividend

        for key in ("staging_path", "output_path", "qmt_path"):
            if key in parameters:
                accepted[key] = self._safe_path(spec, str(parameters[key]))
        return accepted

    def _safe_path(self, spec: SkillSpec, candidate: str) -> str:
        """Reject absolute paths and traversal before any filesystem touch."""
        from pathlib import PurePosixPath

        if candidate.startswith("/") or candidate.startswith("\\") or ":" in candidate[:3]:
            raise SkillRefused(
                REFUSED_PATH, f"{candidate!r} is absolute; only repo-relative paths are allowed"
            )
        parts = PurePosixPath(candidate).parts
        if ".." in parts:
            raise SkillRefused(REFUSED_PATH, f"{candidate!r} contains a parent traversal")
        if not any(candidate.startswith(root) for root in spec.allowed_output_roots):
            raise SkillRefused(
                REFUSED_PATH,
                f"{candidate!r} is outside the allowed roots {list(spec.allowed_output_roots)}",
            )
        return candidate

    def audit_records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.audit]
