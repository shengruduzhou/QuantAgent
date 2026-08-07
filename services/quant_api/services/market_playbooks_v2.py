from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from services.quant_api.services.market_playbooks import (
    MarketPlaybookService as _BaseMarketPlaybookService,
    PLAYBOOKS,
    _finite,
    _items,
    _perf,
    _records,
)


class MarketPlaybookService(_BaseMarketPlaybookService):
    """Contract-hardened playbooks for examples whose official recipes are deeper."""

    def _full_limit_up_pool(self) -> dict[str, Any]:
        provider = self.market._provider()
        page = 1
        rows: list[dict[str, Any]] = []
        pagination: Mapping[str, Any] = {}
        while True:
            payload = provider.get_capability(
                "/api/a-share/special-data/limit-up-pool",
                {"page": page, "size": 200, "sort_field": "continue_day_cnt", "sort_dir": "desc"},
            )
            rows.extend(_items(payload))
            pagination = payload.get("pagination", {}) if isinstance(payload, Mapping) else {}
            pages = int(pagination.get("pages") or page)
            if page >= pages:
                break
            page += 1
        return {"item": rows, "pagination": dict(pagination), "pagesFetched": page}

    def _intelligence(self, pid: str) -> dict[str, Any]:
        if pid != "04":
            return super()._intelligence(pid)
        provider = self.market._provider()
        pool = self._full_limit_up_pool()
        ladder = provider.get_capability("/api/a-share/special-data/limit-up-ladder", {})
        return {
            "playbookId": "04",
            "limitPool": pool,
            "limitLadder": ladder,
            "metrics": self._limit_up_metrics(_items(pool), _items(ladder)),
            "provenance": {
                "pool": "/api/a-share/special-data/limit-up-pool?page=all&size=200&sort_field=continue_day_cnt&sort_dir=desc",
                "ladder": "/api/a-share/special-data/limit-up-ladder",
            },
            "notes": ["涨停池与近30交易日天梯具有不同时间窗口；非交易日空集不补模拟数据。"],
        }

    def _limit_up_metrics(self, pool: list[dict[str, Any]], ladder: list[dict[str, Any]]) -> dict[str, Any]:
        board_counts = Counter(int(row.get("continue_day_cnt") or 0) for row in pool)
        max_board = max(board_counts, default=0)
        ladder_rows = self._ladder_rows(ladder)
        return {
            "limitUpCount": len(pool),
            "maxBoard": max_board,
            "boardCounts": dict(sorted(board_counts.items())),
            "ladderLatestMaxBoard": ladder_rows[0]["maxBoard"] if ladder_rows else None,
            "highSealMoney": [
                {"thscode": row.get("thscode"), "name": row.get("name"), "sealMoney": row.get("seal_money")}
                for row in sorted(pool, key=lambda row: float(row.get("seal_money") or 0), reverse=True)[:10]
            ],
        }

    @staticmethod
    def _ladder_rows(ladder: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in ladder:
            boards = item.get("boards", {}) if isinstance(item.get("boards"), Mapping) else {}
            entries = [stock for stocks in boards.values() if isinstance(stocks, list) for stock in stocks if isinstance(stock, Mapping)]
            counts = {str(level): len(stocks) for level, stocks in boards.items() if isinstance(stocks, list)}
            rows.append(
                {
                    "date": item.get("date"),
                    "maxBoard": max([int(stock.get("board_num") or 0) for stock in entries] or [0]),
                    "counts": counts,
                    "stocks": [dict(stock) for stock in entries],
                }
            )
        return rows

    def limitup_sentiment(self) -> dict[str, Any]:
        provider = self.market._provider()
        pool_payload = self._full_limit_up_pool()
        pool = _items(pool_payload)
        ladder_payload = provider.get_capability("/api/a-share/special-data/limit-up-ladder", {})
        ladder = _items(ladder_payload)
        ladder_rows = self._ladder_rows(ladder)
        reasons = Counter(str(row.get("limit_up_reason") or "未披露") for row in pool)
        metrics = self._limit_up_metrics(pool, ladder)
        return {
            "playbookId": "12",
            "limitUpCount": metrics["limitUpCount"],
            "maxBoard": metrics["maxBoard"],
            "reasonDistribution": reasons.most_common(20),
            "ladder": ladder_rows,
            "boardCounts": metrics["boardCounts"],
            "highSealMoney": metrics["highSealMoney"],
            "provenance": {
                "pool": "/api/a-share/special-data/limit-up-pool",
                "ladder": "/api/a-share/special-data/limit-up-ladder",
            },
        }

    def price_volume_breakout(self, symbol: str, benchmark: str, cost_bps: float) -> dict[str, Any]:
        bars = self._stock_bars(symbol, 720)
        close = bars["close"].astype(float)
        volume = bars["volume"].astype(float)
        prior_high = close.rolling(20).max().shift(1)
        average_volume = volume.rolling(20).mean().shift(1)
        ma10 = close.rolling(10).mean()
        breakout = (close > prior_high) & (volume > 1.5 * average_volume)
        desired = pd.Series(0.0, index=bars.index)
        events: list[dict[str, Any]] = []
        active = False
        entry_close = float("nan")
        holding = 0
        for i in range(len(bars)):
            if not active and bool(breakout.iloc[i]):
                active = True
                entry_close = float(close.iloc[i])
                holding = 0
                events.append({"signalDate": str(pd.Timestamp(bars["date"].iloc[i]).date()), "action": "enter_next_open", "reason": "close > prior 20d high and volume > 1.5x prior 20d mean"})
            elif active:
                holding += 1
                stop = float(close.iloc[i] / entry_close - 1.0) <= -0.08 if np.isfinite(entry_close) and entry_close > 0 else False
                trend_exit = bool(pd.notna(ma10.iloc[i]) and close.iloc[i] < ma10.iloc[i])
                timeout = holding >= 20
                if stop or trend_exit or timeout:
                    active = False
                    reasons = [name for name, flag in (("8% close stop", stop), ("close below MA10", trend_exit), ("20d max holding", timeout)) if flag]
                    events.append({"signalDate": str(pd.Timestamp(bars["date"].iloc[i]).date()), "action": "exit_next_open", "reason": "; ".join(reasons)})
            desired.iloc[i] = 1.0 if active else 0.0
        out = self._backtest(
            "13", bars, desired, benchmark, cost_bps,
            {"signal": "20d price breakout + 1.5x volume confirmation", "exit": "MA10 / 8% close stop / 20d max holding"},
        )
        out["events"] = events
        out["formation"] = {"priorHighWindow": 20, "volumeWindow": 20, "volumeMultiple": 1.5}
        return out

    def time_series_momentum(self, symbol: str, benchmark: str, cost_bps: float) -> dict[str, Any]:
        bars = self._index_bars(symbol, 900)
        close = bars["close"].astype(float)
        daily_return = close.pct_change(fill_method=None)
        momentum120 = close.pct_change(120, fill_method=None)
        ma120 = close.rolling(120).mean()
        volatility60 = daily_return.rolling(60).std(ddof=1) * np.sqrt(252.0)
        signal = ((momentum120 > 0) & (close > ma120)).astype(float)
        out = self._backtest(
            "14", bars, signal, benchmark, cost_bps,
            {"signal": "120d own momentum > 0 and close > MA120", "cash": "100% when inactive", "weighting": "single-asset state; vol60 exposed for inverse-vol multi-asset extension"},
        )
        state = pd.DataFrame({
            "date": pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d"),
            "momentum120": momentum120,
            "ma120": ma120,
            "volatility60": volatility60,
            "state": np.where(signal > 0, "Active", "Inactive"),
        })
        state_map = {str(row["date"]): row for row in _records(state)}
        for row in out.get("rows", []):
            extra = state_map.get(str(row.get("date")), {})
            row.update({key: extra.get(key) for key in ("momentum120", "ma120", "volatility60", "state")})
        out["metrics"]["latestVolatility60"] = _finite(volatility60.iloc[-1]) if len(volatility60) else None
        out["riskContribution"] = {"latestVolatility60": out["metrics"]["latestVolatility60"], "method": "single active asset; cash contributes zero market volatility"}
        sensitivity: dict[str, Any] = {}
        for window in (60, 120, 180):
            test_signal = ((close.pct_change(window, fill_method=None) > 0) & (close > close.rolling(window).mean())).astype(float)
            sensitivity[str(window)] = self._backtest("14", bars, test_signal, benchmark, cost_bps, {"window": window})["metrics"]
        out["windowSensitivity"] = sensitivity
        return out

    def short_term_reversal(self, index_symbol: str, benchmark: str, cost_bps: float) -> dict[str, Any]:
        provider = self.market._provider()
        constituents = provider.index_constituents(index_symbol.upper())
        symbols = [str(x) for x in constituents.get("thscode", constituents.get("symbol", pd.Series(dtype=str))).dropna()][:60]
        series: list[pd.Series] = []
        for symbol in symbols:
            try:
                s = self._stock_bars(symbol, 480).set_index("date")["close"].astype(float)
                s.name = symbol
                series.append(s)
            except Exception:
                continue
        if len(series) < 10:
            raise ValueError("insufficient current constituent histories for reversal lab")
        close = pd.concat(series, axis=1).sort_index()
        bench = self._index_bars(benchmark, 480).set_index("date")["close"].reindex(close.index).ffill()
        market_state = bench > bench.rolling(120).mean()
        relative5 = close.pct_change(5, fill_method=None).sub(bench.pct_change(5, fill_method=None), axis=0)
        score = (-relative5).where(market_state, np.nan)
        next_return = close.pct_change(fill_method=None).shift(-1)
        ic = score.corrwith(next_return, axis=1, method="spearman")
        percentile = score.rank(axis=1, pct=True)
        quintile_returns: dict[str, pd.Series] = {}
        for q in range(5):
            lo, hi = q / 5.0, (q + 1) / 5.0
            mask = (percentile > lo) & (percentile <= hi)
            quintile_returns[f"Q{q+1}"] = next_return.where(mask).mean(axis=1)
        target = (percentile > 0.8).astype(float)
        weights = target.div(target.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        gross = next_return.where(target > 0).mean(axis=1).fillna(0.0)
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        net = gross - turnover * cost_bps / 10000.0
        quintiles = pd.DataFrame(quintile_returns)
        return {
            "playbookId": "15",
            "metrics": _perf(net),
            "rankIcMean": _finite(ic.mean()),
            "rankIcIr": _finite(ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) else np.nan),
            "quintileMeanReturns": {column: _finite(quintiles[column].mean()) for column in quintiles},
            "rows": _records(pd.DataFrame({"date": net.index, "netReturn": net, "rankIc": ic.reindex(net.index), "marketActive": market_state.reindex(net.index)}).tail(360)),
            "assumptions": {"execution": "T close score; evaluate T+1 return", "costBps": cost_bps, "marketState": "benchmark > MA120", "constituentCaveat": "current constituents only; no historical constituent backfill"},
        }

    def dragon_tiger_topology(self, index_symbol: str) -> dict[str, Any]:
        provider = self.market._provider()
        constituents = provider.index_constituents(index_symbol.upper())
        allowed = set(str(x) for x in constituents.get("thscode", constituents.get("symbol", pd.Series(dtype=str))).dropna())
        dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in self._index_bars(index_symbol, 60)["date"].tail(8)]
        aggregate: dict[str, dict[str, float]] = {}
        for date in dates:
            for board, key in (("all", "net"), ("org", "org"), ("hot_money", "hot")):
                data = provider.get_capability("/api/a-share/special-data/dragon-tiger-list", {"board_type": board, "date": date})
                rows = data.get("stock_items", []) if isinstance(data, Mapping) else []
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    code = str(row.get("thscode") or "")
                    if code not in allowed:
                        continue
                    target = aggregate.setdefault(code, {"net": 0.0, "org": 0.0, "hot": 0.0})
                    target[key] += float(row.get("net_value") or 0.0)
        nodes = [
            {"id": index_symbol.upper(), "kind": "concept", "value": sum(abs(v["net"]) for v in aggregate.values())},
            {"id": "机构", "kind": "capital", "value": sum(abs(v["org"]) for v in aggregate.values())},
            {"id": "游资", "kind": "capital", "value": sum(abs(v["hot"]) for v in aggregate.values())},
        ]
        links: list[dict[str, Any]] = []
        for code, values in sorted(aggregate.items(), key=lambda item: abs(item[1]["net"]), reverse=True)[:30]:
            nodes.append({"id": code, "kind": "stock", "value": values["net"]})
            links.extend([
                {"source": index_symbol.upper(), "target": code, "value": values["net"]},
                {"source": "机构", "target": code, "value": values["org"]},
                {"source": "游资", "target": code, "value": values["hot"]},
            ])
        return {
            "playbookId": "16",
            "nodes": nodes,
            "links": links,
            "dates": dates,
            "provenance": {"dragonTiger": "/api/a-share/special-data/dragon-tiger-list?board_type=all|org|hot_money", "constituents": "/api/a-share-index/constituents/ths-stock-list"},
            "notes": ["机构与游资分别使用对应 board_type 的 net_value；拓扑仅使用所选指数当前成分，不回填历史成分。"],
        }


__all__ = ["MarketPlaybookService", "PLAYBOOKS"]
