"""Read-only Windows QMT data gateway.

Architecture: the gateway is a *Windows-side satellite*, not a migration
target. QuantAgent stays on Linux; only the part that physically requires
MiniQMT runs on Windows, and it exports canonical Parquet plus a manifest that
the Linux side ingests.

    Windows: MiniQMT -> xtquant -> probes/downloads -> staging -> canonical export
    Linux:   U0 foundation, PIT validation, features, backtest, training

**Read-only for this mission, structurally.** This module imports
``xtquant.xtdata`` and nothing else from the QMT stack. It never imports
``xtquant.xttrader``, never constructs ``XtQuantTrader``, and exposes no order
path. A test asserts this against the parsed import graph rather than the file
text, so the module can document the rule without tripping its own check.

Two behaviours matter more than the download logic:

**Empty is not absence.** Every fetch returns a :class:`FetchResult` whose
status distinguishes "the account is not entitled", "the client never answered"
and "there genuinely is no data for this key". A permission-denied ``{}`` is
never allowed to become an empty dataset, because that is precisely how a stock
that was ST for three years becomes "never ST".

**Truncation is detected, not trusted.** A vendor that silently clamps a
request to its entitled window returns success with fewer rows. The gateway
compares the returned span against the requested span and reports ``TRUNCATED``
with both, so a one-year basic tier cannot be mistaken for ten years of history.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from quantagent.data.providers import qmt_entitlement as ent

#: Vendor error fragments that mean "not entitled" rather than "no data".
PERMISSION_MARKERS: tuple[str, ...] = (
    "权限", "authoriz", "authoris", "permission", "not allowed", "无权限",
    "未授权", "vip", "lv2", "level2", "level-2",
)

#: Directories the gateway may write to. Anything else is refused, so a
#: mis-specified skill parameter cannot scatter licensed data across the disk.
ALLOWED_OUTPUT_ROOTS: tuple[str, ...] = (
    "runtime/data/capabilities/qmt",
    "runtime/data/qmt_staging",
    "runtime/data/market_events",
)


class QmtUnavailable(RuntimeError):
    """Raised when the QMT client cannot be reached on this host."""


class QmtWriteRefused(RuntimeError):
    """Raised when an export would write outside the allowed roots."""


@dataclass
class GatewayEnvironment:
    """What this host actually is, measured rather than assumed."""

    probed_at: str
    os_name: str
    os_release: str
    is_windows: bool
    python_version: str
    xtquant_installed: bool = False
    xtquant_version: str | None = None
    xtdata_importable: bool = False
    import_error: str | None = None
    native_extension_platforms: list[str] = field(default_factory=list)
    miniqmt_paths_found: list[str] = field(default_factory=list)
    client_connected: bool = False
    connect_error: str | None = None
    data_dir: str | None = None
    authorized_markets: list[str] = field(default_factory=list)
    verdict: str = ent.PLATFORM_UNAVAILABLE
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FetchResult:
    """Outcome of one gateway read, with truncation and permission separated."""

    capability: str
    status: str
    rows: int = 0
    symbols_requested: int = 0
    symbols_returned: int = 0
    requested_start: str | None = None
    requested_end: str | None = None
    actual_start: str | None = None
    actual_end: str | None = None
    fields: list[str] = field(default_factory=list)
    sample_hash: str | None = None
    error: str | None = None
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status == ent.SERVING

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def looks_like_permission_error(message: str | None) -> bool:
    """Whether a vendor message indicates entitlement rather than absence."""
    if not message:
        return False
    lowered = str(message).lower()
    return any(marker.lower() in lowered for marker in PERMISSION_MARKERS)


def classify_empty(*, error: str | None, entitlement_confirmed: bool) -> str:
    """Decide what an empty response means.

    The default is deliberately pessimistic: without confirmed entitlement an
    empty result is ``EMPTY_UNVERIFIED``, which downstream code must not treat
    as data. Only a confirmed-entitled call may report ``EMPTY_VERIFIED``.
    """
    if looks_like_permission_error(error):
        return ent.PERMISSION_DENIED
    if entitlement_confirmed:
        return ent.EMPTY_VERIFIED
    return ent.EMPTY_UNVERIFIED


def detect_truncation(
    *, requested_start: str | None, actual_start: str | None,
    requested_end: str | None, actual_end: str | None,
) -> bool:
    """True when the vendor returned a narrower window than requested."""
    def _norm(value: str | None) -> str | None:
        if not value:
            return None
        digits = re.sub(r"\D", "", str(value))
        return digits[:8] if len(digits) >= 8 else None

    rs, as_, re_, ae = (_norm(requested_start), _norm(actual_start),
                        _norm(requested_end), _norm(actual_end))
    if rs and as_ and as_ > rs:
        return True
    if re_ and ae and ae < re_:
        return True
    return False


def _frame_hash(frame: Any) -> str | None:
    try:
        import pandas as pd
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        payload = frame.head(200).to_csv(index=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]
    except Exception:  # noqa: BLE001 - hashing must never break a probe
        return None


def probe_environment() -> GatewayEnvironment:
    """Establish whether this host can run the QMT client at all."""
    import sys
    import sysconfig

    env = GatewayEnvironment(
        probed_at=_utc_now(),
        os_name=platform.system(),
        os_release=platform.release(),
        is_windows=platform.system().lower().startswith("win"),
        python_version=sys.version.split()[0],
    )

    if not env.is_windows:
        env.verdict = ent.PLATFORM_UNAVAILABLE
        env.detail = (
            "MiniQMT/QMT is a Windows desktop client; xtquant ships Windows-only "
            "native extensions. No QMT entitlement can be measured from this host."
        )

    try:
        import xtquant  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        env.import_error = f"{type(exc).__name__}: {exc}"
        if env.is_windows:
            env.verdict = ent.PLATFORM_UNAVAILABLE
            env.detail = "xtquant is not installed on this Windows host"
        return env

    env.xtquant_installed = True
    env.xtquant_version = getattr(xtquant, "__version__", None)
    package_dir = Path(xtquant.__file__).parent
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    platforms: set[str] = set()
    for entry in package_dir.iterdir():
        if entry.suffix in {".pyd", ".so"}:
            platforms.add("windows" if entry.suffix == ".pyd" else "posix")
    env.native_extension_platforms = sorted(platforms)

    try:
        from xtquant import xtdata  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        env.import_error = f"{type(exc).__name__}: {exc}"
        env.verdict = ent.PLATFORM_UNAVAILABLE
        env.detail = (
            "xtquant imports but its market-data module does not; the wheel's "
            f"native extensions are {env.native_extension_platforms or 'unknown'} "
            f"while this interpreter needs {suffix}"
        )
        return env

    env.xtdata_importable = True
    try:
        env.data_dir = str(xtdata.get_data_dir())
        if env.data_dir:
            env.miniqmt_paths_found.append(env.data_dir)
    except Exception as exc:  # noqa: BLE001
        env.connect_error = f"get_data_dir: {type(exc).__name__}: {exc}"
    try:
        markets = xtdata.get_authorized_market_list()
        env.authorized_markets = [str(m) for m in (markets or [])]
        env.client_connected = True
        env.verdict = ent.SERVING
        env.detail = "MiniQMT answered; per-capability entitlement still requires probing"
    except Exception as exc:  # noqa: BLE001
        env.connect_error = f"get_authorized_market_list: {type(exc).__name__}: {exc}"
        env.verdict = ent.CLIENT_DISCONNECTED
        env.detail = "xtdata imported but no MiniQMT client answered; start and log in to the terminal"
    return env


def assert_allowed_output(path: str | Path, *, repo_root: str | Path = ".") -> Path:
    """Refuse writes outside the allowlisted roots, including via traversal."""
    root = Path(repo_root).resolve()
    target = (root / Path(path)).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    for allowed in ALLOWED_OUTPUT_ROOTS:
        allowed_path = (root / allowed).resolve()
        try:
            target.relative_to(allowed_path)
            return target
        except ValueError:
            continue
    raise QmtWriteRefused(
        f"{target} is outside the allowed QMT output roots {list(ALLOWED_OUTPUT_ROOTS)}"
    )


class QmtGateway:
    """Read-only gateway over ``xtquant.xtdata``.

    Constructing it does not connect. Every read raises
    :class:`QmtUnavailable` when the client is absent, so a caller on a host
    without QMT gets a loud, specific failure rather than an empty frame that
    looks like a quiet market.
    """

    name = "qmt_gateway"

    def __init__(self, *, environment: GatewayEnvironment | None = None,
                 repo_root: str | Path = ".") -> None:
        self._environment = environment
        self.repo_root = Path(repo_root)

    @property
    def environment(self) -> GatewayEnvironment:
        if self._environment is None:
            self._environment = probe_environment()
        return self._environment

    def _xtdata(self) -> Any:
        env = self.environment
        if not env.xtdata_importable:
            raise QmtUnavailable(
                f"xtquant.xtdata is not importable on this host ({env.os_name}): "
                f"{env.import_error}. MiniQMT is a Windows client."
            )
        if not env.client_connected:
            raise QmtUnavailable(
                f"xtdata imported but no MiniQMT client answered: {env.connect_error}"
            )
        from xtquant import xtdata  # type: ignore[import-not-found]

        return xtdata

    # -- generic probing ---------------------------------------------------
    def probe_capability(
        self,
        capability: str,
        fetch: Callable[[], Any],
        *,
        symbols_requested: int = 0,
        requested_start: str | None = None,
        requested_end: str | None = None,
        entitlement_confirmed: bool = False,
        date_column: str | None = None,
    ) -> FetchResult:
        """Run one capability probe and classify the outcome honestly."""
        result = FetchResult(
            capability=capability,
            status=ent.NOT_PROBED,
            symbols_requested=symbols_requested,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        try:
            payload = fetch()
        except Exception as exc:  # noqa: BLE001 - vendor errors are data
            message = f"{type(exc).__name__}: {exc}"
            result.error = message
            result.status = (
                ent.PERMISSION_DENIED if looks_like_permission_error(message) else ent.ERROR
            )
            return result

        import pandas as pd

        frame = payload if isinstance(payload, pd.DataFrame) else None
        if frame is None and isinstance(payload, Mapping):
            result.symbols_returned = len(payload)
            try:
                frame = pd.concat(
                    [v for v in payload.values() if isinstance(v, pd.DataFrame)],
                    ignore_index=True,
                ) if payload else None
            except Exception:  # noqa: BLE001
                frame = None
        if frame is None and isinstance(payload, (list, tuple)):
            result.rows = len(payload)
            result.status = ent.SERVING if payload else classify_empty(
                error=None, entitlement_confirmed=entitlement_confirmed
            )
            return result

        if frame is None or frame.empty:
            result.status = classify_empty(
                error=None, entitlement_confirmed=entitlement_confirmed
            )
            result.detail = (
                "empty response; entitlement not confirmed, so this is NOT "
                "evidence the data does not exist"
                if result.status == ent.EMPTY_UNVERIFIED else "verified empty"
            )
            return result

        result.rows = int(len(frame))
        result.fields = [str(c) for c in frame.columns][:64]
        result.sample_hash = _frame_hash(frame)
        if date_column and date_column in frame.columns:
            dates = frame[date_column].astype(str)
            result.actual_start = str(dates.min())
            result.actual_end = str(dates.max())

        if detect_truncation(
            requested_start=requested_start, actual_start=result.actual_start,
            requested_end=requested_end, actual_end=result.actual_end,
        ):
            result.status = ent.TRUNCATED
            result.detail = (
                f"requested {requested_start}..{requested_end} but received "
                f"{result.actual_start}..{result.actual_end}; the entitled window "
                "is narrower than the request"
            )
            return result

        result.status = ent.SERVING
        return result

    # -- ST history: the U0 PIT blocker ------------------------------------
    def fetch_st_history(self, symbol: str) -> FetchResult:
        """Historical ST/*ST/PT intervals for one security.

        ``get_his_st_data`` returns ``{}`` both when a security was never ST and
        when the account cannot read the ST dataset. Those are opposite facts,
        so this method refuses to report either without an entitlement signal --
        an unconfirmed empty is ``EMPTY_UNVERIFIED``, never "never ST".
        """
        xtdata = self._xtdata()

        def _fetch() -> Any:
            return xtdata.get_his_st_data(symbol)

        result = self.probe_capability(
            "st_history", _fetch, symbols_requested=1, entitlement_confirmed=False
        )
        if result.status == ent.EMPTY_UNVERIFIED:
            result.detail = (
                f"get_his_st_data({symbol!r}) returned empty. This is NOT a claim "
                "that the security was never ST: the same empty value is returned "
                "when the ST dataset is not downloaded or not entitled. Run "
                "download_his_st_data() first and re-probe a known-ST control."
            )
        return result

    def probe_st_with_controls(
        self, *, known_st: Sequence[str], never_st: Sequence[str]
    ) -> dict[str, Any]:
        """Probe ST history against positive and negative controls.

        Without a positive control an empty answer is uninterpretable. If a
        security known to have been ST also returns empty, the dataset is
        unavailable rather than the security being clean -- and this method says
        so instead of writing a universe full of false negatives.
        """
        xtdata = self._xtdata()
        try:
            xtdata.download_his_st_data()
            download_error = None
        except Exception as exc:  # noqa: BLE001
            download_error = f"{type(exc).__name__}: {exc}"

        positives = {s: self.fetch_st_history(s).to_dict() for s in known_st}
        negatives = {s: self.fetch_st_history(s).to_dict() for s in never_st}

        positive_rows = sum(int(r["rows"]) for r in positives.values())
        entitled = positive_rows > 0
        return {
            "download_error": download_error,
            "download_permission_denied": looks_like_permission_error(download_error),
            "positive_controls": positives,
            "negative_controls": negatives,
            "positive_control_rows": positive_rows,
            "entitlement_verdict": (
                ent.SERVING if entitled
                else ent.PERMISSION_DENIED if looks_like_permission_error(download_error)
                else ent.EMPTY_UNVERIFIED
            ),
            "interpretation": (
                "positive controls returned ST intervals, so an empty result for "
                "another security is genuine evidence it was never ST"
                if entitled else
                "a security KNOWN to have been ST also returned empty, so the ST "
                "dataset is unavailable to this account; no security may be "
                "recorded as never-ST from this probe"
            ),
        }

    # -- export ------------------------------------------------------------
    def export_canonical(
        self, frame: Any, *, relative_path: str, capability: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a canonical Parquet partition plus a manifest.

        Refuses to write outside the allowlisted roots, and never logs
        credentials or account identifiers -- the manifest records provider,
        capability, counts and hashes only.
        """
        target = assert_allowed_output(relative_path, repo_root=self.repo_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)

        manifest = {
            "generated": _utc_now(),
            "provider": "qmt_xtdata",
            "capability": capability,
            "path": str(target),
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "content_hash": _frame_hash(frame),
            "provenance": dict(provenance or {}),
            "read_only": True,
            "contains_credentials": False,
        }
        manifest_path = target.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest


def build_matrix(gateway: QmtGateway | None = None) -> tuple[GatewayEnvironment, ent.EntitlementMatrix]:
    """Produce the entitlement matrix for this host.

    On a host without QMT this returns the *full catalogue* with every cell
    marked ``PLATFORM_UNAVAILABLE`` rather than an empty file, so a reader sees
    the scope of what is unknown.
    """
    gateway = gateway or QmtGateway()
    env = gateway.environment
    if env.verdict != ent.SERVING:
        return env, ent.unprobed_matrix(
            platform=env.os_name,
            reason=env.detail or env.import_error or env.connect_error or "client unavailable",
        )

    matrix = ent.EntitlementMatrix()
    for spec in ent.CAPABILITY_CATALOGUE:
        matrix.add(ent.EntitlementCell(
            capability=spec.capability, api=spec.api, documented=spec.documented,
            platform=env.os_name, permission_class=ent.UNKNOWN_UNTIL_PROBED,
            probe_status=ent.NOT_PROBED, source_url=spec.source_url, notes=spec.notes,
        ))
    return env, matrix
