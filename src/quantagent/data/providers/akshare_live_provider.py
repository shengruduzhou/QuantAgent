from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from quantagent.data.ashare.contracts import (
    QUALITY_DERIVED,
    QUALITY_OK,
    QUALITY_SUSPECT,
    QUALITY_UNIT_AMBIGUOUS,
)
from quantagent.data.ashare.units import (
    VolumeScaleVerdict,
    infer_volume_scale,
    tencent_native_volume_prior,
)
from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar


AKSHARE_MARKET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "source_endpoint",
    "retrieved_at",
    "available_at",
    "quality_status",
)

# EastMoney remains the preferred documented A-share daily endpoint and stays
# first. Sina is now the second choice and Tencent the last resort, ordered by
# how complete the payload is rather than by preference alone.
#
# Measured 2026-08-15, akshare 1.18.60, 5 symbols x 44 sessions through this
# provider:
#
#   east_money  stock_zh_a_hist      UNREACHABLE  push2his.eastmoney.com,
#                                    ConnectionError after ~262s per symbol
#   sina        stock_zh_a_daily     220/220 volume (shares) AND 220/220 amount
#                                    (CNY); amount/(volume*close) median 0.9986
#   tencent     stock_zh_a_hist_tx   220/220 volume, 0/220 amount -- the source
#                                    publishes no CNY turnover at all
#
# Tencent ahead of Sina meant that whenever EastMoney was unavailable the
# pipeline silently lost turnover entirely, which disables every ADV, liquidity
# and capacity screen downstream. Sina supplies it, so it must be tried first.
# Note the failure is slow as well as total: EastMoney burns ~262s per symbol
# before giving up, so an unreachable primary is expensive, not just useless.
_DEFAULT_SOURCE_ORDER: tuple[str, ...] = ("east_money", "sina", "tencent")
_CANONICAL_VOLUME_UNIT = "shares"
_CANONICAL_AMOUNT_UNIT = "CNY"
#: Emitted instead of a canonical unit when a source does not supply the
#: column at all. Never reuse a real unit name for a missing measurement.
_UNIT_UNAVAILABLE = "unavailable"
#: Emitted when a volume is present but its unit was not proven by the
#: response itself (see ``quantagent.data.ashare.units``).
_UNIT_UNVERIFIED = "shares_unverified"
#: What each endpoint's DOCUMENTATION claims (akshare.akfamily.xyz, 1.19.1:
#: stock_zh_a_hist 成交量=手, stock_zh_a_daily 成交量=股, stock_zh_a_hist_tx
#: "volume 统一为股"). Used ONLY to scale a response that cannot prove its own
#: unit, and such rows are never certified. The Tencent claim is measurably
#: false for 000xxx.SZ stocks (akshare >= 1.18.69 skips its x100 for "sz000"),
#: which is exactly why the unit is decided per response, not looked up here.
_DOCUMENTED_VOLUME_SCALE_BY_SOURCE: dict[str, float] = {
    "east_money": 100.0,
    "sina": 1.0,
    "tencent": 1.0,
}
#: Only raw (unadjusted) daily bars may enter a canonical/research panel.
#: Vendor qfq/hfq are refused: EastMoney/Tencent "qfq" is SUBTRACTIVE
#: (raw - qfq constant) while Sina's is MULTIPLICATIVE, and every qfq series is
#: re-anchored on each new ex-date, so the same request returns different
#: history depending on the fetch date (R5: 600519 2024 prices moved -2.4%,
#: 000001 -5.6%). Adjustment comes from U0 hfq factors, never from akshare.
_CANONICAL_ADJUST = ""
_VENDOR_ADJUST_CHOICES: frozenset[str] = frozenset({"qfq", "hfq"})
_SOURCE_FUNCTIONS: dict[str, str] = {
    "east_money": "stock_zh_a_hist",
    "tencent": "stock_zh_a_hist_tx",
    "sina": "stock_zh_a_daily",
}
_SOURCE_RELIABILITY: dict[str, float] = {
    "east_money": 0.78,
    "tencent": 0.76,
    "sina": 0.72,
}


@dataclass
class AkShareLiveProvider:
    """Optional AKShare downloader with explicit source/unit/PIT contracts.

    Network access is opt-in. Only raw/unadjusted daily bars are served on the
    canonical path: vendor ``qfq``/``hfq`` are refused because their semantics
    differ by source (subtractive on EastMoney/Tencent, multiplicative on Sina)
    and change with the fetch date. ``research_only_allow_vendor_adjusted=True``
    is the single, explicitly named escape hatch; rows produced that way are
    ``DERIVED``, never PIT-valid and never production-certified.

    Units are decided **per response** (``quantagent.data.ashare.units``): the
    raw volume scale (shares vs 手) is the one candidate under which
    ``amount / volume`` sits inside ``[low, high]`` for the bulk of rows. A
    response that cannot prove its unit (no CNY turnover, or no scale fits) is
    ``UNIT_AMBIGUOUS``; it is never accepted into the result and the next
    source is tried instead. In particular Tencent only participates in the
    failover chain when its response passes this check (akshare 1.18.60's
    Tencent shape has no turnover and therefore never does).

    One symbol is always served by exactly one source per call. When
    ``source_affinity`` names the source that already serves a symbol's stored
    history, that source is tried first, and a switch away from it is reported
    in ``metadata["source_switches"]`` so the caller can write a
    ``SourceBoundary`` or refuse.

    A daily bar is research-PIT valid only when it is raw and its next-session
    availability resolves from an explicit trading calendar. Missing calendar
    coverage fails closed; this adapter never invents sessions with
    ``pandas.BDay``.
    """

    allow_network: bool = False
    adjust: str = ""
    source_order: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_SOURCE_ORDER)
    trading_calendar: TradingCalendar | None = None
    calendar_source: str = ""
    research_only_allow_vendor_adjusted: bool = False
    source_affinity: Mapping[str, str] | None = None

    def health_check(self, request: ProviderRequest | None = None) -> dict[str, object]:
        if not self.allow_network:
            return {"status": "disabled", "reason": "allow_network_false"}
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            return {"status": "unavailable", "reason": f"akshare_unavailable:{type(exc).__name__}"}
        if request is None:
            return {
                "status": "passed",
                "adjust": self.adjust,
                "akshare_version": _akshare_version(ak),
                "canonical_volume_unit": _CANONICAL_VOLUME_UNIT,
                "canonical_amount_unit": _CANONICAL_AMOUNT_UNIT,
                "volume_unit_rule": "per_response_implied_vwap_in_low_high",
                "source_order": list(self.source_order),
            }
        try:
            result = self.daily_ohlcv(request)
        except ProviderUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc)}
        return {
            "status": "passed" if result.quality_score > 0 else "failed",
            "point_in_time": result.point_in_time,
            "warnings": list(result.warnings),
            "schema_report": result.metadata.get("schema_report", {}),
            "akshare_version": result.metadata.get("akshare_version"),
            "source_counts": result.metadata.get("source_counts", {}),
        }

    def _validated_adjust(self) -> str:
        adjust = str(self.adjust or "")
        if adjust == _CANONICAL_ADJUST:
            return adjust
        if adjust not in _VENDOR_ADJUST_CHOICES:
            raise ProviderUnavailable("AkShare adjust must be one of '', 'qfq', or 'hfq'")
        if not self.research_only_allow_vendor_adjusted:
            raise ProviderUnavailable(
                f"AkShare vendor-adjusted daily bars (adjust={adjust!r}) are refused on "
                "canonical paths: qfq is subtractive on EastMoney/Tencent but "
                "multiplicative on Sina, and every qfq history is rewritten on each "
                "new ex-date. Request adjust='' (raw) and apply U0 hfq factors "
                "(scripts/u0_pit_intervals.py) instead. For a throw-away research "
                "comparison only, set research_only_allow_vendor_adjusted=True; such "
                "rows are DERIVED and never PIT-valid."
            )
        return adjust

    def _ordered_sources(self, symbol: str) -> tuple[tuple[str, ...], str | None]:
        affinity = (self.source_affinity or {}).get(str(symbol))
        if affinity and affinity in self.source_order:
            rest = tuple(source for source in self.source_order if source != affinity)
            return (affinity, *rest), affinity
        return tuple(self.source_order), affinity

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable(
                "AkShare live download is disabled; set data.allow_network=true explicitly"
            )
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("akshare is not available; install quantagent[data]") from exc
        if not request.symbols:
            raise ProviderUnavailable("AkShare live daily_ohlcv requires explicit symbols")
        adjust = self._validated_adjust()
        unknown_sources = sorted(set(self.source_order).difference(_SOURCE_FUNCTIONS))
        if unknown_sources:
            raise ProviderUnavailable(
                "unknown AKShare daily source(s): " + ",".join(unknown_sources)
            )
        version = _akshare_version(ak)

        frames: list[pd.DataFrame] = []
        failed_symbols: list[str] = []
        warnings: list[str] = []
        source_counts: dict[str, int] = {}
        source_by_symbol: dict[str, str] = {}
        volume_unit_basis_by_symbol: dict[str, str] = {}
        observed_raw_units: dict[str, set[str]] = {}
        unit_rejections: list[dict[str, object]] = []
        source_switches: list[dict[str, object]] = []
        for symbol in request.symbols:
            ordered, affinity = self._ordered_sources(str(symbol))
            accepted: pd.DataFrame | None = None
            used_source = ""
            for source in ordered:
                try:
                    raw = _fetch_source_daily(
                        ak,
                        source,
                        symbol,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        adjust=adjust,
                    )
                except Exception as exc:  # pragma: no cover - network path
                    warnings.append(f"akshare_{source}_failed:{symbol}:{type(exc).__name__}:{exc}")
                    continue
                if raw is None or raw.empty:
                    warnings.append(f"akshare_{source}_empty:{symbol}")
                    continue
                normalised = _normalize_akshare_daily(
                    raw,
                    symbol,
                    source=source,
                    adjust=adjust,
                    trading_calendar=self.trading_calendar,
                    akshare_version=version,
                )
                if normalised.empty:
                    warnings.append(f"akshare_normalized_empty:{symbol}:{source}")
                    continue
                outside = _rows_outside_request(normalised, request)
                if outside:
                    warnings.append(f"akshare_rows_outside_request:{symbol}:{outside}")
                    normalised = _filter_request_dates(normalised, request)
                if normalised.empty:
                    warnings.append(f"akshare_no_rows_in_request:{symbol}:{source}")
                    continue
                if not _response_unit_acceptable(normalised, adjust):
                    basis = str(normalised["volume_unit_basis"].iloc[0])
                    warnings.append(f"akshare_{source}_unit_ambiguous:{symbol}:{basis}")
                    unit_rejections.append(
                        {"symbol": str(symbol), "source": source, "basis": basis,
                         "rows": int(len(normalised))}
                    )
                    continue
                accepted, used_source = normalised, source
                break
            if accepted is None:
                failed_symbols.append(str(symbol))
                warnings.append(f"akshare_empty_daily_ohlcv:{symbol}")
                continue
            if affinity and used_source != affinity:
                source_switches.append(
                    {"symbol": str(symbol), "provider_before": f"akshare:{affinity}",
                     "provider_after": f"akshare:{used_source}",
                     "first_new_trade_date": str(accepted["trade_date"].min())}
                )
                warnings.append(f"akshare_source_switch:{symbol}:{affinity}->{used_source}")
            frames.append(accepted)
            source_counts[used_source] = source_counts.get(used_source, 0) + 1
            source_by_symbol[str(symbol)] = used_source
            volume_unit_basis_by_symbol[str(symbol)] = str(accepted["volume_unit_basis"].iloc[0])
            observed_raw_units.setdefault(used_source, set()).update(
                accepted["raw_volume_unit"].dropna().astype(str).unique().tolist()
            )

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        schema_report = akshare_market_schema_report(frame)
        warnings.extend(
            f"akshare_schema_missing:{column}" for column in schema_report["missing_columns"]
        )
        if failed_symbols:
            warnings.append(
                "akshare_incomplete_symbol_coverage:" + ",".join(sorted(set(failed_symbols)))
            )
        if adjust:
            warnings.append(
                "akshare_adjusted_history_not_pit_without_vintaged_adjustment_factors"
            )
            warnings.append("akshare_vendor_adjusted_research_only")
        if self.trading_calendar is None or self.trading_calendar.empty:
            warnings.append("akshare_market_calendar_missing:pit_not_certified")
        elif int(
            frame.get("point_in_time_valid", pd.Series(dtype=bool)).fillna(False).sum()
        ) < len(frame):
            warnings.append("akshare_market_calendar_incomplete:some_available_at_unresolved")
        if "tencent" in source_counts:
            warnings.append("akshare_tencent_failover_used")
        if not frame.empty and "quality_status" in frame.columns:
            suspect = int(frame["quality_status"].astype(str).eq(QUALITY_SUSPECT).sum())
            if suspect:
                warnings.append(f"akshare_rows_outside_implied_vwap_band:{suspect}")

        point_in_time = bool(
            not frame.empty
            and not failed_symbols
            and adjust == ""
            and "point_in_time_valid" in frame.columns
            and frame["point_in_time_valid"].fillna(False).astype(bool).all()
        )
        quality_score = 0.78 if not frame.empty and schema_report["status"] == "passed" else 0.0
        if source_counts and set(source_counts) == {"tencent"}:
            quality_score = min(quality_score, 0.76)
        if failed_symbols or not point_in_time:
            quality_score = min(quality_score, 0.60)

        return ProviderResult(
            frame,
            source="akshare_live_provider:multi_source",
            point_in_time=point_in_time,
            quality_score=quality_score,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "frequency": "1d",
                "timezone": "Asia/Shanghai",
                "volume_unit": _CANONICAL_VOLUME_UNIT,
                "amount_unit": _CANONICAL_AMOUNT_UNIT,
                "adjustment": "raw" if adjust == "" else adjust,
                "pit_semantics": "daily_next_session",
                "schema_report": schema_report,
                "function_name": "|".join(_SOURCE_FUNCTIONS[source] for source in self.source_order),
                "source_order": list(self.source_order),
                "source_functions": {
                    source: _SOURCE_FUNCTIONS[source] for source in self.source_order
                },
                "source_counts": source_counts,
                "source_by_symbol": source_by_symbol,
                "source_affinity_used": bool(self.source_affinity),
                "source_switches": source_switches,
                "failed_symbols": failed_symbols,
                "requested_symbol_count": int(len(request.symbols)),
                "fetched_symbol_count": int(sum(source_counts.values())),
                "adjust": adjust,
                "research_only": bool(adjust),
                "adjustment_pit_vintage_bound": False,
                "canonical_volume_unit": _CANONICAL_VOLUME_UNIT,
                "canonical_amount_unit": _CANONICAL_AMOUNT_UNIT,
                "volume_unit_rule": "per_response_implied_vwap_in_low_high",
                # Observed in THIS call, per accepted response -- not a lookup table.
                "raw_volume_unit_by_source": {
                    source: (sorted(units)[0] if len(units) == 1 else sorted(units))
                    for source, units in observed_raw_units.items()
                },
                "volume_unit_basis_by_symbol": volume_unit_basis_by_symbol,
                "unit_rejections": unit_rejections,
                "documented_volume_scale_by_source": dict(_DOCUMENTED_VOLUME_SCALE_BY_SOURCE),
                "calendar_source": self.calendar_source or None,
                "calendar_bound": bool(
                    self.trading_calendar is not None and not self.trading_calendar.empty
                ),
                "calendar_production_certified": False,
                "akshare_version": version,
                "production_integrity_certified": False,
            },
        )


def _response_unit_acceptable(frame: pd.DataFrame, adjust: str) -> bool:
    """A response is usable only if it proved its own volume unit.

    Raw responses must be ``OK``/``SUSPECT`` row-wise (never
    ``UNIT_AMBIGUOUS``). Vendor-adjusted research responses cannot be checked
    against [low, high] (amount is raw CNY, prices are not), so they are only
    accepted from sources whose response carries turnover at all.
    """
    statuses = set(frame["quality_status"].astype(str).unique())
    if adjust:
        return "amount_unit" in frame.columns and bool(
            frame["amount_unit"].astype(str).eq(_CANONICAL_AMOUNT_UNIT).all()
        )
    return QUALITY_UNIT_AMBIGUOUS not in statuses


def _akshare_version(ak: object) -> str:
    return str(getattr(ak, "__version__", "unknown") or "unknown")


def _plain_a_code(symbol: str) -> str:
    text = str(symbol).split(".")[0]
    lower = text.lower()
    for prefix in ("sh", "sz", "bj"):
        if lower.startswith(prefix):
            return text[len(prefix):]
    return text


def _prefixed_a_symbol(symbol: str) -> str:
    """Return sh/sz/bj-prefixed code used by Tencent and Sina endpoints."""
    text = str(symbol).strip()
    upper = text.upper()
    if "." in upper:
        code, exchange = upper.split(".", 1)
        return f"{exchange.lower()}{code.zfill(6)}"
    lower = text.lower()
    if lower.startswith(("sh", "sz", "bj")):
        return f"{lower[:2]}{text[2:].zfill(6)}"
    code = upper.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8", "92")):
        return f"bj{code}"
    return f"sz{code}"


def _fetch_source_daily(
    ak,
    source: str,
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame | None:
    """One raw akshare daily call against one source; exceptions propagate."""
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")
    if source == "east_money":
        return ak.stock_zh_a_hist(
            symbol=_plain_a_code(symbol),
            period="daily",
            start_date=compact_start,
            end_date=compact_end,
            adjust=adjust,
        )
    if source == "tencent":
        api = getattr(ak, "stock_zh_a_hist_tx", None)
        if api is None:
            raise AttributeError("stock_zh_a_hist_tx unavailable")
        return api(
            symbol=_prefixed_a_symbol(symbol),
            start_date=compact_start,
            end_date=compact_end,
            adjust=adjust,
            timeout=15,
        )
    if source == "sina":
        return ak.stock_zh_a_daily(
            symbol=_prefixed_a_symbol(symbol),
            start_date=compact_start,
            end_date=compact_end,
            adjust=adjust,
        )
    raise ValueError(f"unknown AKShare daily source: {source}")


def _fetch_daily_with_fallback(
    ak,
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    adjust: str,
    source_order: tuple[str, ...],
) -> tuple[pd.DataFrame | None, str, list[str]]:
    """Return the first non-empty RAW response (no unit verification).

    Retained for diagnostics only. Canonical callers go through
    :meth:`AkShareLiveProvider.daily_ohlcv`, which additionally rejects any
    response that cannot prove its own volume unit.
    """
    warnings: list[str] = []
    for source in source_order:
        try:
            raw = _fetch_source_daily(
                ak, source, symbol, start_date=start_date, end_date=end_date, adjust=adjust
            )
        except Exception as exc:  # pragma: no cover - network path
            warnings.append(f"akshare_{source}_failed:{symbol}:{type(exc).__name__}:{exc}")
            continue
        if raw is None or raw.empty:
            warnings.append(f"akshare_{source}_empty:{symbol}")
            continue
        return raw, source, warnings
    return None, source_order[-1] if source_order else "unknown", warnings


def _normalize_akshare_daily(
    frame: pd.DataFrame,
    symbol: str,
    *,
    source: str = "east_money",
    adjust: str = "",
    trading_calendar: TradingCalendar | None = None,
    akshare_version: str | None = None,
) -> pd.DataFrame:
    """Normalise EastMoney/Tencent/Sina output to one economic schema.

    Canonical ``volume`` is shares and ``amount`` is CNY turnover. The raw
    volume scale (x1 shares or x100 手) is decided from THIS response by
    :func:`quantagent.data.ashare.units.infer_volume_scale` -- the candidate
    under which ``amount / volume`` lies inside ``[low, high]`` for the bulk of
    rows -- and recorded per row in ``volume_unit_basis`` /
    ``volume_scale_applied``; ``source_endpoint`` carries the akshare version.

    Tencent shapes (``stock_zh_a_hist_tx``):

    * akshare <= 1.18.68: six columns, NO volume column, and the column named
      ``amount`` is native volume (lots on main boards/ChiNext, **shares on
      STAR**). There is no CNY turnover, so the unit cannot be proven: the
      volume is scaled by the measured board prior (STAR x1, else x100) and the
      rows are ``UNIT_AMBIGUOUS``; amount is NaN, never the lot count.
    * akshare >= 1.18.69: ``volume`` + ``turnover`` + ``amount`` (CNY). akshare
      multiplies volume by 100 except for ``sh688``/``sz000``/``sh000``/
      ``sz399`` prefixes, so 000xxx.SZ STOCKS arrive in lots. The VWAP check
      detects this per response (x100 for 000001.SZ, x1 for 688981.SH).

    A response whose unit cannot be proven keeps quality ``UNIT_AMBIGUOUS`` and
    ``volume_unit='shares_unverified'``; rows outside the implied-VWAP band
    under a proven scale are ``SUSPECT``. Neither is ever written as ``OK``.
    """
    rename_map = {
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "date": "trade_date",
    }
    data = frame.rename(columns=rename_map)
    fallback_scale: float | None = _DOCUMENTED_VOLUME_SCALE_BY_SOURCE.get(source)
    fallback_reason = f"{source}_documented_unit"
    tencent_shape = ""
    if source == "tencent":
        # Structural presence of a `volume` column is NOT sufficient evidence of
        # the newer shape: a present-but-empty column carries no measurement,
        # and treating it as such would pass the volume count through as CNY
        # turnover. Require a usable value, not just a header.
        volume_usable = (
            "volume" in data.columns
            and pd.to_numeric(data["volume"], errors="coerce").notna().any()
        )
        if not volume_usable and "amount" in data.columns:
            if "volume" in data.columns:
                data = data.drop(columns=["volume"])
            data = data.rename(columns={"amount": "volume"})
            data["amount"] = float("nan")
            tencent_shape = "tencent_v1_volume_named_amount"
            fallback_scale, fallback_reason = tencent_native_volume_prior(symbol)
        else:
            tencent_shape = "tencent_v2_volume_turnover_amount"
    keep = [
        c
        for c in ("trade_date", "open", "high", "low", "close", "volume", "amount")
        if c in data.columns
    ]
    if "trade_date" not in keep:
        return pd.DataFrame()
    data = data[keep].copy()
    data["symbol"] = symbol
    trade_dates = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data["trade_date"] = trade_dates.dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    raw_prices = adjust == ""
    has_volume = "volume" in data.columns and data["volume"].notna().any()
    has_amount = "amount" in data.columns and data["amount"].notna().any()
    if has_volume and raw_prices:
        verdict = infer_volume_scale(
            data["volume"],
            data["amount"] if "amount" in data.columns else None,
            data.get("low", pd.Series(float("nan"), index=data.index)),
            data.get("high", pd.Series(float("nan"), index=data.index)),
            fallback_scale=fallback_scale,
            fallback_reason=fallback_reason,
        )
    elif has_volume:
        # Vendor-adjusted prices cannot be checked against raw CNY turnover.
        verdict = VolumeScaleVerdict(
            "unverifiable",
            fallback_scale,
            f"unverifiable:vendor_adjusted_prices({adjust}):"
            + (
                f"fallback_scale={fallback_scale:g}({fallback_reason})"
                if fallback_scale is not None
                else "no_fallback_scale"
            ),
        )
    else:
        verdict = VolumeScaleVerdict("unverifiable", None, "no_volume_values")
    if tencent_shape:
        verdict = VolumeScaleVerdict(
            verdict.status,
            verdict.scale,
            f"{tencent_shape}:{verdict.basis}",
            verdict.testable_rows,
            verdict.fit_rates,
            verdict.row_in_band,
        )

    if has_volume:
        if verdict.scale is None:
            data["volume"] = float("nan")
        else:
            data["volume"] = data["volume"] * float(verdict.scale)
    has_volume = "volume" in data.columns and data["volume"].notna().any()
    # Label only what is actually present, converted AND proven. Stamping a unit
    # onto an absent or unproven column is how "unknown" becomes "fine": it is
    # the label, not the number, that downstream consumers trust.
    if not has_volume:
        data["volume_unit"] = _UNIT_UNAVAILABLE
        data["raw_volume_unit"] = _UNIT_UNAVAILABLE
    else:
        data["volume_unit"] = _CANONICAL_VOLUME_UNIT if verdict.verified else _UNIT_UNVERIFIED
        data["raw_volume_unit"] = verdict.raw_unit_name
    data["amount_unit"] = _CANONICAL_AMOUNT_UNIT if has_amount else _UNIT_UNAVAILABLE
    data["volume_unit_basis"] = verdict.basis
    data["volume_scale_applied"] = (
        float(verdict.scale) if verdict.scale is not None and has_volume else float("nan")
    )

    calendar_ok = trading_calendar is not None and not trading_calendar.empty
    if calendar_ok:
        available = pd.Series(
            [trading_calendar.next_trading_day(value, lag_days=1) for value in trade_dates],
            index=data.index,
            dtype="datetime64[ns]",
        )
    else:
        available = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    data["available_at"] = available.dt.strftime("%Y-%m-%d")

    version = str(akshare_version or "unknown")
    data["price_adjustment"] = adjust or "raw"
    data["adjustment_pit_vintage_bound"] = False
    data["source"] = f"akshare:{source}"
    data["source_endpoint"] = f"akshare=={version}:{_SOURCE_FUNCTIONS.get(source, 'unknown')}"
    data["akshare_version"] = version
    data["retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    if not raw_prices:
        quality = pd.Series(QUALITY_DERIVED, index=data.index, dtype="object")
    elif not verdict.verified:
        quality = pd.Series(QUALITY_UNIT_AMBIGUOUS, index=data.index, dtype="object")
    else:
        quality = pd.Series(QUALITY_OK, index=data.index, dtype="object")
        if verdict.row_in_band is not None:
            outside = verdict.row_in_band.reindex(data.index).eq(0.0)
            quality = quality.mask(outside, QUALITY_SUSPECT)
    data["quality_status"] = quality
    data["source_type"] = "market_data"
    data["source_reliability"] = _SOURCE_RELIABILITY.get(source, 0.0)
    data["point_in_time_valid"] = bool(raw_prices and calendar_ok) & available.notna()
    return data


def _filter_request_dates(frame: pd.DataFrame, request: ProviderRequest) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    start = pd.Timestamp(request.start_date).normalize()
    end = pd.Timestamp(request.end_date).normalize()
    return frame.loc[dates.notna() & (dates >= start) & (dates <= end)].reset_index(drop=True)


def _rows_outside_request(frame: pd.DataFrame, request: ProviderRequest) -> int:
    if frame.empty or "trade_date" not in frame.columns:
        return 0
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    start = pd.Timestamp(request.start_date).normalize()
    end = pd.Timestamp(request.end_date).normalize()
    return int((dates.isna() | (dates < start) | (dates > end)).sum())


def akshare_market_schema_report(
    frame: pd.DataFrame, as_of_date: str | None = None
) -> dict[str, object]:
    missing = [column for column in AKSHARE_MARKET_REQUIRED_COLUMNS if column not in frame.columns]
    pit_violations = 0
    if "available_at" in frame.columns:
        available_at = pd.to_datetime(frame["available_at"], errors="coerce")
        trade_date = (
            pd.to_datetime(frame["trade_date"], errors="coerce")
            if "trade_date" in frame.columns
            else None
        )
        pit_violations += int(available_at.isna().sum())
        if trade_date is not None:
            pit_violations += int(
                (available_at.notna() & trade_date.notna() & (available_at <= trade_date)).sum()
            )
        if as_of_date:
            pit_violations += int(
                (available_at.notna() & (available_at > pd.Timestamp(as_of_date))).sum()
            )
    if "point_in_time_valid" in frame.columns:
        pit_violations += int((~frame["point_in_time_valid"].fillna(False).astype(bool)).sum())
    return {
        "status": "passed" if not missing and pit_violations == 0 else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_MARKET_REQUIRED_COLUMNS),
        "missing_columns": missing,
        "pit_violation_count": pit_violations,
        "canonical_volume_unit": _CANONICAL_VOLUME_UNIT,
        "canonical_amount_unit": _CANONICAL_AMOUNT_UNIT,
    }
