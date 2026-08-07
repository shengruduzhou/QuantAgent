"""Strict exhaustive Fuyao synchronizer.

The historical A-share endpoint documents an ``offset`` pagination parameter
without exposing a page-size argument.  This subclass follows pages until an
empty page is observed and guards against a server returning the same page for a
new offset.  The bulk unadjusted dump remains the canonical raw panel; this path
exists to retain every vendor-provided forward/backward adjusted row as well.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json

from quantagent.data.fuyao_full_sync import (
    FuyaoFullSynchronizer,
    FuyaoUniverse,
    SyncEvent,
    _date_ms,
    _items,
)


class ExhaustiveFuyaoSynchronizer(FuyaoFullSynchronizer):
    """Full synchronizer with fail-visible pagination for adjusted stock history."""

    def _sync_adjusted_stock_history(
        self, universe: FuyaoUniverse, start: date, end: date
    ) -> None:
        start_ms, end_ms = _date_ms(start), _date_ms(end)
        endpoint = "/api/a-share/prices/historical"
        for symbol in universe.a_shares:
            for adjust in ("forward", "backward"):
                offset = 0
                seen_pages: set[str] = set()
                # 10 years of daily bars is far below 100 non-empty pages under
                # any plausible vendor page size. A hard ceiling turns a broken
                # pagination contract into an explicit audit event, not a hang.
                for _page_no in range(100):
                    params = {
                        "thscode": symbol,
                        "interval": "1d",
                        "start": start_ms,
                        "end": end_ms,
                        "adjust": adjust,
                        "offset": offset,
                    }
                    data = self._fetch(
                        "a_share.prices_historical",
                        endpoint,
                        params,
                        qualifier=f"{symbol}/{adjust}/offset-{offset}",
                    )
                    if data is None:
                        break
                    rows = _items(data)
                    if not rows:
                        break
                    fingerprint = hashlib.sha256(
                        json.dumps(rows, ensure_ascii=True, sort_keys=True, default=str).encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    if fingerprint in seen_pages:
                        self.events.append(
                            SyncEvent(
                                "a_share.prices_historical",
                                endpoint,
                                "pagination_stalled",
                                params=params,
                                message=(
                                    f"Fuyao repeated a historical page for {symbol} "
                                    f"adjust={adjust} offset={offset}; completeness cannot be proven."
                                ),
                            )
                        )
                        if self.stop_on_error:
                            raise RuntimeError(
                                f"Fuyao historical pagination stalled for {symbol} {adjust}"
                            )
                        break
                    seen_pages.add(fingerprint)
                    offset += len(rows)
                else:
                    self.events.append(
                        SyncEvent(
                            "a_share.prices_historical",
                            endpoint,
                            "pagination_limit_reached",
                            params={
                                "thscode": symbol,
                                "adjust": adjust,
                                "offset": offset,
                            },
                            message="Exceeded 100 non-empty historical pages; completeness cannot be proven.",
                        )
                    )
                    if self.stop_on_error:
                        raise RuntimeError(
                            f"Fuyao historical pagination exceeded safety ceiling for {symbol} {adjust}"
                        )


__all__ = ["ExhaustiveFuyaoSynchronizer"]
