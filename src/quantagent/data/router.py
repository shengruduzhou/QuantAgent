"""MultiSourceDataRouter — unified TickFlow + Fuyao + Qlib + public fallbacks.

Generic capabilities keep their historical routing semantics. A-share daily bars
add an explicit integrity boundary: research can quarantine malformed rows with
evidence, while production can only serve validated real bars and performs
priority-preserving fallback at ``(symbol, trade_date)`` key granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pandas as pd

from quantagent.data.integrity import (
    DailyOHLCVIntegrityPolicy,
    DailyOHLCVIntegrityResult,
    daily_bar_keys,
    expected_daily_keys,
    validate_daily_ohlcv,
)
from quantagent.data.providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
)


class _DailyProvider(Protocol):
    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult: ...


@dataclass(frozen=True)
class RoutedProvider:
    name: str
    provider: Any
    is_paid: bool = False
    capabilities: tuple[str, ...] = ("daily_ohlcv",)
    quality_baseline: float = 0.80


@dataclass(frozen=True)
class RouterConfig:
    daily_priority: tuple[str, ...] = (
        "tickflow",
        "fuyao",
        "qlib",
        "akshare",
        "baostock",
        "tushare",
    )
    minute_priority: tuple[str, ...] = ("tickflow", "akshare", "baostock", "qlib")
    allow_mock_fallback: bool = False
    merge_partial_results: bool = True
    fail_when_all_unavailable: bool = True


@dataclass
class RouterResult:
    frame: pd.DataFrame
    primary_source: str | None
    fallback_chain: list[str] = field(default_factory=list)
    per_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    integrity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_source": self.primary_source,
            "fallback_chain": list(self.fallback_chain),
            "per_source": {k: dict(v) for k, v in self.per_source.items()},
            "warnings": list(self.warnings),
            "row_count": int(len(self.frame)),
            "integrity": dict(self.integrity),
        }


class RouterAllSourcesUnavailable(RuntimeError):
    pass


class RouterDataIntegrityError(RuntimeError):
    pass


_SOURCE_FATAL_CODES = (
    "missing_column:",
    "provider_not_point_in_time",
    "pit_semantics_missing",
    "mock_or_synthetic_source",
    "frequency_metadata_missing",
    "frequency_not_daily:",
    "timezone_metadata_missing",
    "timezone_mismatch:",
    "volume_unit_missing",
    "amount_unit_missing",
    "adjustment_metadata_missing",
    "adjustment_mismatch:",
    "freshness_reference_missing",
    "freshness_reference_outside_request_window",
    "freshness_unverifiable",
    "stale_daily_data:",
    "trade_date_after_freshness_reference",
)


def _fatal_integrity(violations: tuple[str, ...]) -> bool:
    return any(any(code.startswith(prefix) for prefix in _SOURCE_FATAL_CODES) for code in violations)


def _semantic_signature(report: DailyOHLCVIntegrityResult) -> tuple[str, str, str, str, str, str]:
    metadata = report.metadata
    return (
        str(metadata.get("frequency", "")).strip().lower(),
        str(metadata.get("timezone", "")).strip(),
        str(metadata.get("volume_unit", "")).strip().lower(),
        str(metadata.get("amount_unit", "")).strip().upper(),
        str(metadata.get("adjustment", "")).strip().lower(),
        str(metadata.get("pit_semantics", "")).strip(),
    )


def _signature_dict(signature: tuple[str, str, str, str, str, str] | None) -> dict[str, str] | None:
    if signature is None:
        return None
    return dict(zip(
        ("frequency", "timezone", "volume_unit", "amount_unit", "adjustment", "pit_semantics"),
        signature,
        strict=True,
    ))


def _coverage_basis(policy: DailyOHLCVIntegrityPolicy) -> str:
    if policy.expected_symbol_trade_dates:
        return "authoritative_per_symbol_expected_keys"
    if policy.expected_trade_dates:
        return "authoritative_single_symbol_expected_dates"
    return "observed_only"


class MultiSourceDataRouter:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self._providers: dict[str, RoutedProvider] = {}

    def register(self, routed: RoutedProvider) -> None:
        self._providers[routed.name] = routed

    def deregister(self, name: str) -> None:
        self._providers.pop(name, None)

    def list_sources(self) -> list[str]:
        return list(self._providers.keys())

    def daily_ohlcv(
        self,
        request: ProviderRequest,
        *,
        integrity_policy: DailyOHLCVIntegrityPolicy | None = None,
    ) -> RouterResult:
        return self._serve_daily(
            request,
            policy=integrity_policy or DailyOHLCVIntegrityPolicy.research(),
        )

    def minute_ohlcv(
        self,
        request: ProviderRequest,
        *,
        frequency: str = "5",
    ) -> RouterResult:
        def call(provider, req: ProviderRequest) -> ProviderResult:
            return provider.minute_ohlcv(req, frequency=frequency)

        return self._serve(
            request,
            method_name=f"minute_ohlcv_{frequency}",
            priority=self.config.minute_priority,
            invoke=call,
        )

    def _serve_daily(
        self,
        request: ProviderRequest,
        *,
        policy: DailyOHLCVIntegrityPolicy,
    ) -> RouterResult:
        result = RouterResult(frame=pd.DataFrame(), primary_source=None)
        served = pd.DataFrame()
        served_keys: set[tuple[str, str]] = set()
        served_symbols: set[str] = set()
        expected_keys = expected_daily_keys(request, policy)
        quarantined_total = 0
        production_signature: tuple[str, str, str, str, str, str] | None = None

        for src_name in self.config.daily_priority:
            routed = self._providers.get(src_name)
            if routed is None:
                continue
            result.fallback_chain.append(src_name)
            method = getattr(routed.provider, "daily_ohlcv", None)
            if method is None:
                result.per_source[src_name] = {
                    "status": "unavailable",
                    "reason": f"{src_name} does not implement daily_ohlcv",
                    "rows": 0,
                }
                continue
            try:
                res = method(request)
            except ProviderUnavailable as exc:
                result.per_source[src_name] = {"status": "unavailable", "reason": str(exc), "rows": 0}
                continue
            except Exception as exc:  # noqa: BLE001
                result.per_source[src_name] = {"status": "error", "reason": str(exc), "rows": 0}
                continue

            validated = validate_daily_ohlcv(res, request, policy)
            report = validated.report
            quarantined_total += int(report.quarantined_rows)
            result.per_source[src_name] = {
                "status": "invalid" if report.status == "failed" else "ok",
                "rows": int(report.input_rows),
                "valid_rows": int(report.valid_rows),
                "quarantined_rows": int(report.quarantined_rows),
                "quality_score": float(res.quality_score),
                "point_in_time": bool(res.point_in_time),
                "warnings": list(res.warnings),
                "integrity": report.to_dict(),
            }

            if report.hard_violations and policy.mode == "production" and not policy.allow_invalid_primary_fallback:
                raise RouterDataIntegrityError(
                    f"daily integrity failed at {src_name}: {list(report.hard_violations)}"
                )

            if _fatal_integrity(report.hard_violations):
                result.warnings.append(f"daily_integrity_source_rejected:{src_name}")
                continue

            valid_frame = _attribute_source(validated.valid_frame, src_name)
            if valid_frame.empty:
                continue

            if policy.mode == "production":
                candidate_signature = _semantic_signature(report)
                if production_signature is None:
                    production_signature = candidate_signature
                elif candidate_signature != production_signature:
                    result.per_source[src_name]["status"] = "invalid"
                    result.per_source[src_name]["router_semantics_mismatch"] = {
                        "expected": _signature_dict(production_signature),
                        "observed": _signature_dict(candidate_signature),
                    }
                    result.warnings.append(f"daily_cross_source_semantics_mismatch:{src_name}")
                    continue

                frame_keys = daily_bar_keys(valid_frame)
                if not frame_keys:
                    continue
                if served.empty:
                    accepted = valid_frame
                else:
                    key_series = _daily_key_series(valid_frame)
                    accepted = valid_frame.loc[~key_series.isin(served_keys)].copy()
                if not accepted.empty:
                    if result.primary_source is None:
                        result.primary_source = src_name
                    served = pd.concat([served, accepted], ignore_index=True)
                    served_keys |= daily_bar_keys(accepted)
                    served_symbols = set(served["symbol"].astype(str))
                if expected_keys and served_keys >= expected_keys:
                    break
                continue

            if result.primary_source is None:
                result.primary_source = src_name
                served = valid_frame
            elif self.config.merge_partial_results:
                missing_symbols = set(valid_frame["symbol"].astype(str)) - served_symbols
                if missing_symbols:
                    backfill = valid_frame[valid_frame["symbol"].astype(str).isin(missing_symbols)]
                    served = pd.concat([served, backfill], ignore_index=True)
            served = _deduplicate_daily_priority(served)
            served_symbols = set(served.get("symbol", pd.Series(dtype=str)).astype(str))
            if request.symbols and served_symbols >= set(request.symbols):
                break

        if not served.empty:
            served = _deduplicate_daily_priority(served).reset_index(drop=True)
            served_keys = daily_bar_keys(served)

        missing_expected = sorted(expected_keys - served_keys)
        basis = _coverage_basis(policy)
        result.integrity = {
            "capability": "daily_ohlcv",
            "policy_mode": policy.mode,
            "served_key_count": len(served_keys),
            "quarantined_rows": int(quarantined_total),
            "expected_key_count": len(expected_keys) if expected_keys else None,
            "missing_expected_keys": [f"{symbol}@{date}" for symbol, date in missing_expected],
            "coverage_basis": basis,
            "semantic_signature": _signature_dict(production_signature) if policy.mode == "production" else None,
        }
        if basis == "observed_only":
            result.warnings.append("daily_integrity_expected_calendar_not_supplied_observed_only")
        if policy.mode == "production" and missing_expected:
            raise RouterDataIntegrityError(
                "daily integrity coverage incomplete: "
                f"{len(missing_expected)} expected (symbol,trade_date) keys are missing"
            )

        if served.empty:
            if policy.mode == "production":
                raise RouterDataIntegrityError(
                    "no daily rows satisfied the production integrity contract; "
                    f"sources={ {name: row.get('status') for name, row in result.per_source.items()} }"
                )
            if not self.config.allow_mock_fallback and self.config.fail_when_all_unavailable:
                raise RouterAllSourcesUnavailable(
                    "all sources failed for daily_ohlcv: "
                    f"{ {name: row.get('status') for name, row in result.per_source.items()} }"
                )
            result.warnings.append("router_all_sources_empty")
        result.frame = served
        return result

    def _serve(
        self,
        request: ProviderRequest,
        *,
        method_name: str,
        priority: tuple[str, ...],
        invoke: Callable[[Any, ProviderRequest], ProviderResult] | None = None,
    ) -> RouterResult:
        result = RouterResult(frame=pd.DataFrame(), primary_source=None)
        served = pd.DataFrame()
        served_symbols: set[str] = set()
        for src_name in priority:
            routed = self._providers.get(src_name)
            if routed is None:
                continue
            result.fallback_chain.append(src_name)
            try:
                if invoke is not None:
                    res = invoke(routed.provider, request)
                else:
                    method = getattr(routed.provider, method_name, None)
                    if method is None:
                        raise ProviderUnavailable(f"{src_name} does not implement {method_name}")
                    res = method(request)
            except ProviderUnavailable as exc:
                result.per_source[src_name] = {"status": "unavailable", "reason": str(exc), "rows": 0}
                continue
            except Exception as exc:  # noqa: BLE001
                result.per_source[src_name] = {"status": "error", "reason": str(exc), "rows": 0}
                continue

            served_frame = _attribute_source(res.frame, src_name)
            row_count = int(len(served_frame))
            result.per_source[src_name] = {
                "status": "ok",
                "rows": row_count,
                "quality_score": float(res.quality_score),
                "warnings": list(res.warnings),
            }
            if row_count == 0:
                continue
            if result.primary_source is None:
                result.primary_source = src_name
                served = served_frame
            elif self.config.merge_partial_results:
                missing_symbols = set(served_frame.get("symbol", pd.Series(dtype=str)).astype(str)) - served_symbols
                if missing_symbols:
                    backfill = served_frame[served_frame["symbol"].astype(str).isin(missing_symbols)]
                    served = pd.concat([served, backfill], ignore_index=True)
            served_symbols = set(served.get("symbol", pd.Series(dtype=str)).astype(str))
            if request.symbols and served_symbols >= set(request.symbols):
                break

        if served.empty:
            if not self.config.allow_mock_fallback and self.config.fail_when_all_unavailable:
                raise RouterAllSourcesUnavailable(
                    f"all sources failed for {method_name}: "
                    f"{ {name: row.get('status') for name, row in result.per_source.items()} }"
                )
            result.warnings.append("router_all_sources_empty")
        result.frame = served.reset_index(drop=True)
        return result


def _attribute_source(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["source_name"] = source_name
    return out


def _daily_key_series(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date.astype("string")
    return pd.Series(list(zip(frame["symbol"].astype(str), dates.astype(str), strict=False)), index=frame.index)


def _deduplicate_daily_priority(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or not {"symbol", "trade_date"}.issubset(frame.columns):
        return pd.DataFrame() if frame is None else frame
    out = frame.copy()
    out["__router_trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    out = out.drop_duplicates(["symbol", "__router_trade_date"], keep="first")
    return out.drop(columns=["__router_trade_date"])


def build_default_router(
    *,
    tickflow_provider=None,
    fuyao_provider=None,
    qlib_provider=None,
    akshare_provider=None,
    baostock_provider=None,
    tushare_provider=None,
    config: RouterConfig | None = None,
) -> MultiSourceDataRouter:
    router = MultiSourceDataRouter(config=config)
    if tickflow_provider is not None:
        router.register(RoutedProvider(
            name="tickflow", provider=tickflow_provider, is_paid=True, quality_baseline=0.95,
            capabilities=("daily_ohlcv", "adjusted_prices", "tradability", "minute_ohlcv_5", "financials"),
        ))
    if fuyao_provider is not None:
        router.register(RoutedProvider(
            name="fuyao", provider=fuyao_provider, is_paid=True, quality_baseline=0.98,
            capabilities=("daily_ohlcv", "adjusted_prices", "fundamentals", "trading_calendar", "index_daily", "valuations", "corporate_actions"),
        ))
    if qlib_provider is not None:
        router.register(RoutedProvider(name="qlib", provider=qlib_provider, is_paid=False, quality_baseline=0.90, capabilities=("daily_ohlcv", "index_daily")))
    if akshare_provider is not None:
        router.register(RoutedProvider(name="akshare", provider=akshare_provider, is_paid=False, quality_baseline=0.80, capabilities=("daily_ohlcv", "minute_ohlcv_5")))
    if baostock_provider is not None:
        router.register(RoutedProvider(name="baostock", provider=baostock_provider, is_paid=False, quality_baseline=0.85, capabilities=("daily_ohlcv", "minute_ohlcv_5", "minute_ohlcv_15", "minute_ohlcv_30", "minute_ohlcv_60")))
    if tushare_provider is not None:
        router.register(RoutedProvider(name="tushare", provider=tushare_provider, is_paid=True, quality_baseline=0.88, capabilities=("daily_ohlcv", "fundamentals")))
    return router


__all__ = [
    "DailyOHLCVIntegrityPolicy",
    "MultiSourceDataRouter",
    "RouterAllSourcesUnavailable",
    "RouterConfig",
    "RouterDataIntegrityError",
    "RouterResult",
    "RoutedProvider",
    "build_default_router",
]
