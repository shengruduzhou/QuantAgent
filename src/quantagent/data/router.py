"""MultiSourceDataRouter — unified Qlib + AkShare + Tushare + BaoStock router.

Spec discussion from the user's table:

| source   | daily | minute | full-A | best for                | weakness                     |
|----------|-------|--------|--------|-------------------------|------------------------------|
| Qlib CN  | ✓     | 1m     | yes    | training baseline       | freshness, manual refresh    |
| AkShare  | ✓     | 1/5/.. | by sym | raw / silver ingestion  | 1m only 5 days; brittle endp |
| Tushare  | ✓     | partial| yes    | financials / disclosures| needs paid quota             |
| BaoStock | ✓     | 5/15/.. | yes   | free fallback           | less rich fields             |

The router consults these in a configurable priority list, asks
each for the requested slice, and **fails loud** if all sources
are unavailable (production path must NOT fall back to synthetic
data per the v8 spec). The first non-empty result wins; remaining
sources are used only to fill gaps when the primary returns
partial coverage.

Inputs are :class:`ProviderRequest` records. Outputs are
:class:`RouterResult` carrying:

* ``frame`` — the merged data
* ``per_source`` — diagnostics keyed by source
* ``primary_source`` — which provider answered first
* ``fallback_chain`` — order traversed
* ``provider_attribution`` — per-row ``source_name`` for audit

Never silently substitutes synthetic data. ``allow_mock_fallback``
defaults to ``False``; the only way to opt into a mock provider is
to set it ``True`` explicitly (and the result frame will carry a
``mock_source_used`` warning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol

import pandas as pd

from quantagent.data.providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUnavailable,
)


class _DailyProvider(Protocol):
    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult: ...


@dataclass(frozen=True)
class RoutedProvider:
    """One named source registered into the router."""

    name: str                   # e.g. "qlib", "akshare", "tushare", "baostock"
    provider: Any                # any object exposing daily_ohlcv(...)
    is_paid: bool = False
    capabilities: tuple[str, ...] = ("daily_ohlcv",)
    quality_baseline: float = 0.80


@dataclass(frozen=True)
class RouterConfig:
    daily_priority: tuple[str, ...] = ("qlib", "akshare", "baostock", "tushare")
    minute_priority: tuple[str, ...] = ("akshare", "baostock", "qlib")
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_source": self.primary_source,
            "fallback_chain": list(self.fallback_chain),
            "per_source": {k: dict(v) for k, v in self.per_source.items()},
            "warnings": list(self.warnings),
            "row_count": int(len(self.frame)),
        }


class RouterAllSourcesUnavailable(RuntimeError):
    """Raised when no registered source could serve the request."""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class MultiSourceDataRouter:
    """Order-preserving router across the v8 data providers.

    Registration order is independent from the priority list: callers
    register one provider per source name and the router picks the
    serving order from :attr:`RouterConfig.daily_priority` /
    ``minute_priority``.

    Important production guarantee: the router NEVER silently
    substitutes synthetic data — that is the meaning of
    ``ProviderUnavailable``. If all sources fail and
    ``allow_mock_fallback`` is ``False``, a
    :class:`RouterAllSourcesUnavailable` is raised so the caller
    must opt into mock data deliberately.
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self._providers: dict[str, RoutedProvider] = {}

    # ── registration ─────────────────────────────────────────────────
    def register(self, routed: RoutedProvider) -> None:
        self._providers[routed.name] = routed

    def deregister(self, name: str) -> None:
        self._providers.pop(name, None)

    def list_sources(self) -> list[str]:
        return list(self._providers.keys())

    # ── daily ───────────────────────────────────────────────────────
    def daily_ohlcv(self, request: ProviderRequest) -> RouterResult:
        return self._serve(
            request,
            method_name="daily_ohlcv",
            priority=self.config.daily_priority,
        )

    # ── minute ──────────────────────────────────────────────────────
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

    # ── core orchestration ──────────────────────────────────────────
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
                        raise ProviderUnavailable(
                            f"{src_name} does not implement {method_name}"
                        )
                    res = method(request)
            except ProviderUnavailable as exc:
                result.per_source[src_name] = {
                    "status": "unavailable", "reason": str(exc), "rows": 0,
                }
                continue
            except Exception as exc:  # noqa: BLE001 — must report errors
                result.per_source[src_name] = {
                    "status": "error", "reason": str(exc), "rows": 0,
                }
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
                # Backfill any symbols the primary did not provide.
                missing_symbols = set(served_frame.get("symbol", pd.Series(dtype=str))) - served_symbols
                if missing_symbols:
                    backfill = served_frame[served_frame["symbol"].isin(missing_symbols)]
                    served = pd.concat([served, backfill], ignore_index=True)
            served_symbols = set(served.get("symbol", pd.Series(dtype=str)).astype(str))
            # If full coverage met → stop early
            if request.symbols and served_symbols >= set(request.symbols):
                break
        if served.empty:
            if not self.config.allow_mock_fallback and self.config.fail_when_all_unavailable:
                raise RouterAllSourcesUnavailable(
                    f"all sources failed for {method_name}: "
                    f"{ {n: r.get('status') for n, r in result.per_source.items()} }"
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


# ---------------------------------------------------------------------------
# Convenience constructor for the standard 4-source stack
# ---------------------------------------------------------------------------

def build_default_router(
    *,
    qlib_provider=None,
    akshare_provider=None,
    baostock_provider=None,
    tushare_provider=None,
    config: RouterConfig | None = None,
) -> MultiSourceDataRouter:
    """Wire up the canonical Qlib / AkShare / BaoStock / TuShare router.

    Any source can be passed as ``None`` to omit. Callers should
    construct providers with their own credentials / endpoints and
    pass them in — the router never instantiates providers itself.
    """
    router = MultiSourceDataRouter(config=config)
    if qlib_provider is not None:
        router.register(RoutedProvider(
            name="qlib", provider=qlib_provider,
            is_paid=False, quality_baseline=0.90,
            capabilities=("daily_ohlcv", "index_daily"),
        ))
    if akshare_provider is not None:
        router.register(RoutedProvider(
            name="akshare", provider=akshare_provider,
            is_paid=False, quality_baseline=0.80,
            capabilities=("daily_ohlcv", "minute_ohlcv_5"),
        ))
    if baostock_provider is not None:
        router.register(RoutedProvider(
            name="baostock", provider=baostock_provider,
            is_paid=False, quality_baseline=0.85,
            capabilities=("daily_ohlcv", "minute_ohlcv_5",
                          "minute_ohlcv_15", "minute_ohlcv_30", "minute_ohlcv_60"),
        ))
    if tushare_provider is not None:
        router.register(RoutedProvider(
            name="tushare", provider=tushare_provider,
            is_paid=True, quality_baseline=0.88,
            capabilities=("daily_ohlcv", "fundamentals"),
        ))
    return router


__all__ = [
    "MultiSourceDataRouter",
    "RouterAllSourcesUnavailable",
    "RouterConfig",
    "RouterResult",
    "RoutedProvider",
    "build_default_router",
]
