from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from services.quant_api.adapters.utils import (
    clean_value,
    iter_json_array,
    page_slice,
    read_csv_columns,
    read_csv_rows,
    read_json,
    read_parquet_rows,
    require_relative_path,
)
from services.quant_api.config import ApiSettings, stable_id
from services.quant_api.runtime_indexer import RuntimeIndexer


class BacktestAdapter:
    def __init__(self, settings: ApiSettings, indexer: RuntimeIndexer) -> None:
        self.settings = settings
        self.indexer = indexer
        self._runs: dict[str, Path] = {}
        self._name_map: dict[str, str] | None = None

    def list(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        self._runs = {}
        seen_directories: set[Path] = set()
        metric_artifacts = [
            item for item in self.indexer.filter(kind="backtest")
            if item["name"] == "metrics.json"
        ]
        for artifact in metric_artifacts:
            metrics_path = self.settings.project_root / artifact["path"]
            directory = metrics_path.parent
            if not ((directory / "nav.csv").exists() or (directory / "trades.csv").exists()):
                continue
            seen_directories.add(directory.resolve())
            relative = require_relative_path(self.settings, directory)
            backtest_id = stable_id("backtest", relative)
            self._runs[backtest_id] = directory
            metrics = read_json(metrics_path, {}) or {}
            run_config = self._nearby_json(directory, "run_config.json")
            initial_cash = self._metric(run_config, "initial_cash")
            name = directory.parent.parent.name if directory.name == "backtest" else directory.name
            horizon = directory.parent.name if directory.name == "backtest" else None
            summaries.append({
                "id": backtest_id,
                "name": name,
                "strategyVersion": self._metric(run_config, "strategy_version"),
                "modelVersion": self._metric(run_config, "model_version"),
                "factorVersion": self._metric(run_config, "feature_policy"),
                "horizon": horizon,
                "startDate": self._metric(metrics, "start_date"),
                "endDate": self._metric(metrics, "end_date"),
                "universeSize": self._metric(metrics, "universe_size"),
                "initialCash": initial_cash,
                "totalReturn": self._metric(metrics, "total_return", "total"),
                "annualReturn": self._metric(metrics, "annualized_return", "annualised_return", "annualized"),
                "maxDrawdown": self._metric(metrics, "max_drawdown", "maxDD"),
                "sharpe": self._metric(metrics, "sharpe"),
                "calmar": self._metric(metrics, "calmar"),
                "volatility": self._metric(metrics, "volatility"),
                "winRate": self._metric(metrics, "win_rate", "hit_rate"),
                "profitFactor": self._metric(metrics, "profit_factor"),
                "turnover": self._metric(metrics, "turnover", "mean_daily_turnover"),
                "tradeCount": self._int_metric(metrics, "n_trades", "trade_count"),
                "fillCount": self._int_metric(metrics, "n_fills"),
                "tTradeCount": self._int_metric(metrics, "t_trade_count", "do_t_trades"),
                "tContribution": self._metric(metrics, "t_contribution", "overlay_total_return_delta"),
                "totalCost": self._metric(metrics, "total_cost"),
                "status": "ready",
                "path": relative,
                "tags": [item for item in (horizon, "strict-v8" if "risk_events.json" in {p.name for p in directory.iterdir()} else None) if item],
                "capabilities": self._capabilities(directory),
            })
        summaries.extend(self._discover_summary_backtests(seen_directories))
        summaries.sort(key=lambda row: row.get("endDate") or "", reverse=True)
        return summaries

    def get(self, backtest_id: str) -> dict[str, Any] | None:
        summary = next((item for item in self.list() if item["id"] == backtest_id), None)
        if summary is None:
            return None
        directory = self._runs[backtest_id]
        return {
            **summary,
            "files": sorted(require_relative_path(self.settings, item) for item in directory.iterdir() if item.is_file()),
            "factorWeights": read_json(directory / "factor_weights.json", {}) or {},
            "runConfig": self._nearby_json(directory, "run_config.json"),
        }

    def equity(self, backtest_id: str) -> list[dict[str, Any]]:
        directory = self._resolve(backtest_id)
        path = directory / "pnl.csv"
        rows = read_csv_rows(path) if path.exists() else read_csv_rows(directory / "nav.csv")
        nav_values: list[float] = []
        points: list[dict[str, Any]] = []
        for row in rows:
            nav = _float(row.get("nav"))
            if nav is None:
                continue
            nav_values.append(nav)
            peak = max(nav_values)
            points.append({
                "datetime": str(row.get("trade_date") or row.get("") or ""),
                "nav": nav,
                "dailyReturn": _float(row.get("daily_return")),
                "drawdown": nav / peak - 1.0 if peak else None,
                "benchmarkNav": _float(row.get("benchmark_nav")),
                "excessNav": _float(row.get("excess_nav")),
            })
        return points

    def trades(
        self,
        backtest_id: str,
        *,
        symbol: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        directory = self._resolve(backtest_id)
        path = directory / "trades.csv"
        if not path.exists():
            return {
                **page_slice([], page, min(page_size, self.settings.max_table_rows)),
                "sourceSchema": None,
                "issues": [{
                    "code": "trade_artifact_missing",
                    "message": "该实验没有标准成交回报文件。",
                    "recoverable": True,
                }],
            }
        columns = set(read_csv_columns(path))
        if not _is_order_blotter(columns):
            return {
                **page_slice([], page, min(page_size, self.settings.max_table_rows)),
                "sourceSchema": "research_event_table",
                "issues": [{
                    "code": "unsupported_trade_schema",
                    "message": "该 trades.csv 是研究事件表，不包含可验证的 side/status/quantity 成交字段；未映射为 Trade。",
                    "path": require_relative_path(self.settings, path),
                    "recoverable": True,
                }],
            }
        rows = read_csv_rows(path)
        realized = self._realized_lookup(directory, symbol)
        positions: dict[str, float] = defaultdict(float)
        cumulative_pnl = 0.0
        records: list[dict[str, Any]] = []
        names = self._stock_names()
        source_path = require_relative_path(self.settings, path)
        for index, row in enumerate(rows):
            row_symbol = str(row.get("symbol") or "")
            if not row_symbol:
                continue
            status = str(row.get("status") or "").lower()
            side = str(row.get("side") or "").lower()
            requested = _float(row.get("quantity")) or 0.0
            filled = _float(row.get("filled_quantity"))
            quantity = filled if filled is not None and filled > 0 else requested
            price = _float(row.get("avg_price"))
            if (price is None or price <= 0) and _float(row.get("reference_price")) is not None:
                price = _float(row.get("reference_price"))
            price = price or 0.0
            if status in {"filled", "partial"}:
                positions[row_symbol] += quantity if side == "buy" else -quantity
            matched_pnl = None
            if side == "sell":
                key = (row_symbol, str(row.get("trade_date") or "")[:10])
                pnl_queue = realized.get(key, [])
                if pnl_queue:
                    matched_pnl = pnl_queue.pop(0)
                    cumulative_pnl += matched_pnl
            record = {
                "id": str(row.get("client_order_id") or stable_id("trade", f"{source_path}:{index}")),
                "datetime": str(row.get("trade_date") or row.get("datetime") or ""),
                "symbol": row_symbol,
                "name": names.get(_bare_code(row_symbol)),
                "action": "BUY" if side == "buy" else ("SELL" if side == "sell" else "UNKNOWN"),
                "price": price,
                "quantity": quantity,
                "amount": price * quantity if price and quantity else None,
                "fee": None,
                "commission": _float(row.get("commission")),
                "slippage": _float(row.get("slippage")),
                "tax": _float(row.get("stamp_duty")),
                "transferFee": _float(row.get("transfer_fee")),
                "impactCost": _float(row.get("impact_cost")),
                "positionAfter": positions[row_symbol] if status in {"filled", "partial"} else None,
                "positionWeightAfter": None,
                "cashAfter": None,
                "signalSource": row.get("signal_source") or "target_weight_reconcile",
                "signalId": row.get("signal_id"),
                "modelVersion": row.get("model_version"),
                "modelScore": _float(row.get("model_score")),
                "factorContributions": None,
                "riskReason": row.get("risk_reason"),
                "pnl": matched_pnl,
                "cumulativePnl": cumulative_pnl if matched_pnl is not None else None,
                "success": status in {"filled", "partial"},
                "failureReason": None if status in {"filled", "partial"} else row.get("last_message"),
                "status": status or None,
                "tPairId": row.get("t_pair_id"),
                "provenance": {
                    "sourcePath": source_path,
                    "sourceRow": index + 2,
                    "derived": ["amount", "positionAfter"] + (["cumulativePnl"] if matched_pnl is not None else []),
                },
            }
            if symbol is None or row_symbol == symbol:
                records.append(record)
        return {
            **page_slice(records, page, min(page_size, self.settings.max_table_rows)),
            "sourceSchema": "order_blotter",
            "issues": [],
        }

    def signals(self, backtest_id: str, symbol: str | None = None) -> list[dict[str, Any]]:
        trades = self.trades(backtest_id, symbol=symbol, page=1, page_size=self.settings.max_table_rows)["items"]
        return [{
            "id": f"signal_{item['id']}",
            "datetime": item["datetime"],
            "symbol": item["symbol"],
            "type": item["action"],
            "price": item["price"],
            "strength": item.get("modelScore"),
            "confidence": None,
            "source": item.get("signalSource"),
            "factors": item.get("factorContributions"),
            "reason": item.get("riskReason") or item.get("failureReason"),
            "riskFlags": [item["riskReason"]] if item.get("riskReason") else [],
            "actionRaw": item["action"],
            "tPairId": item.get("tPairId"),
        } for item in trades]

    def positions(self, backtest_id: str, *, symbol: str | None = None) -> list[dict[str, Any]]:
        directory = self._resolve(backtest_id)
        candidates = [
            directory / "position_history.csv",
            directory / "holdings.csv",
            directory.parent / "holdings_daily.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            rows = read_csv_rows(path)
            output = []
            for row in rows:
                row_symbol = str(row.get("symbol") or "")
                if symbol and row_symbol != symbol:
                    continue
                output.append({
                    "datetime": str(row.get("trade_date") or row.get("date") or ""),
                    "symbol": row_symbol,
                    "shares": _float(row.get("shares")),
                    "availableShares": _float(row.get("available_shares")),
                    "frozenShares": _float(row.get("frozen_shares")),
                    "weight": _float(row.get("weight")),
                    "marketValue": _float(row.get("market_value")),
                })
            return output
        target_path = directory.parent / "target_weights.parquet"
        if symbol and target_path.exists():
            schema = pl.read_parquet_schema(target_path)
            if symbol in schema:
                date_col = "trade_date" if "trade_date" in schema else None
                columns = [symbol] + ([date_col] if date_col else [])
                rows = read_parquet_rows(target_path, columns=columns)
                return [{
                    "datetime": str(row.get(date_col) or ""),
                    "symbol": symbol,
                    "shares": None,
                    "availableShares": None,
                    "frozenShares": None,
                    "weight": _float(row.get(symbol)),
                    "marketValue": None,
                } for row in rows]
        return []

    def kline(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int = 2_000,
    ) -> dict[str, Any]:
        path = self.settings.runtime_root / "data" / "v7" / "silver" / "market_panel" / "market_panel.parquet"
        if not path.exists():
            return {"bars": [], "sampled": False, "originalPoints": 0, "returnedPoints": 0}
        lazy = pl.scan_parquet(path).filter(pl.col("symbol") == symbol)
        if start:
            lazy = lazy.filter(pl.col("trade_date") >= pl.lit(datetime.fromisoformat(start)))
        if end:
            lazy = lazy.filter(pl.col("trade_date") <= pl.lit(datetime.fromisoformat(end)))
        columns = [
            "symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
            "available_at", "is_st", "is_suspended", "is_limit_up", "is_limit_down", "source",
        ]
        schema = lazy.collect_schema()
        frame = lazy.select([name for name in columns if name in schema]).sort("trade_date").collect()
        original = frame.height
        max_points = max(10, min(int(limit), self.settings.max_chart_points))
        sampled = original > max_points
        if sampled:
            frame = frame.tail(max_points)
        bars = []
        for row in frame.to_dicts():
            bars.append({
                "datetime": clean_value(row.get("trade_date")),
                "symbol": symbol,
                "open": _float(row.get("open")) or 0.0,
                "high": _float(row.get("high")) or 0.0,
                "low": _float(row.get("low")) or 0.0,
                "close": _float(row.get("close")) or 0.0,
                "volume": _float(row.get("volume")),
                "amount": _float(row.get("amount")),
                "availableAt": clean_value(row.get("available_at")),
                "isSt": row.get("is_st"),
                "isSuspended": row.get("is_suspended"),
                "isLimitUp": row.get("is_limit_up"),
                "isLimitDown": row.get("is_limit_down"),
                "source": row.get("source"),
            })
        return {"bars": bars, "sampled": sampled, "originalPoints": original, "returnedPoints": len(bars)}

    def risk_events(self, backtest_id: str, *, page: int = 1, page_size: int = 100) -> dict[str, Any]:
        directory = self._resolve(backtest_id)
        path = directory / "risk_events.json"
        if not path.exists():
            return page_slice([], page, page_size)
        start = (max(1, page) - 1) * page_size
        rows = list(iter_json_array(path, start=start, limit=page_size))
        events = []
        source = require_relative_path(self.settings, path)
        for index, row in enumerate(rows):
            events.append({
                "id": stable_id("risk", f"{source}:{start + index}"),
                "datetime": str(row.get("trade_date") or row.get("datetime") or "") or None,
                "symbol": row.get("symbol"),
                "type": str(row.get("event_type") or row.get("type") or "unknown"),
                "severity": _risk_severity(row),
                "reason": row.get("reason") or row.get("last_message"),
                "rule": row.get("rule"),
                "blocked": _risk_blocked(row),
                "detail": row,
                "sourcePath": source,
            })
        return {
            "items": events,
            "total": start + len(events) + (1 if len(events) == page_size else 0),
            "page": page,
            "pageSize": page_size,
            "hasNext": len(events) == page_size,
        }

    def stock_replay(self, backtest_id: str, symbol: str) -> dict[str, Any]:
        trades = self.trades(backtest_id, symbol=symbol, page=1, page_size=self.settings.max_table_rows)["items"]
        positions = self.positions(backtest_id, symbol=symbol)
        bars = self.kline(symbol)["bars"]
        realized = [item["pnl"] for item in trades if item.get("pnl") is not None]
        equity = []
        cumulative = 0.0
        peak = 0.0
        for item in trades:
            if item.get("pnl") is None:
                continue
            cumulative += float(item["pnl"])
            peak = max(peak, cumulative)
            equity.append({
                "datetime": item["datetime"],
                "nav": cumulative,
                "dailyReturn": None,
                "drawdown": cumulative - peak,
                "benchmarkNav": None,
                "excessNav": None,
            })
        return {
            "backtestId": backtest_id,
            "symbol": symbol,
            "name": self._stock_names().get(_bare_code(symbol)),
            "bars": bars,
            "trades": trades,
            "signals": self.signals(backtest_id, symbol),
            "positions": positions,
            "scoreSeries": [],
            "equity": equity,
            "summary": {
                "realizedPnl": sum(realized) if realized else None,
                "tradeCount": len(trades),
                "winRate": (sum(value > 0 for value in realized) / len(realized)) if realized else None,
                "maxDrawdown": min((point["drawdown"] for point in equity), default=None),
                "firstTrade": trades[0]["datetime"] if trades else None,
                "lastTrade": trades[-1]["datetime"] if trades else None,
            },
            "availability": {
                "bars": bool(bars),
                "trades": bool(trades),
                "positions": bool(positions),
                "scoreSeries": False,
                "equity": bool(equity),
            },
            "issues": self.trades(
                backtest_id,
                symbol=symbol,
                page=1,
                page_size=1,
            ).get("issues", []),
        }

    def _resolve(self, backtest_id: str) -> Path:
        if backtest_id not in self._runs:
            self.list()
        path = self._runs.get(backtest_id)
        if path is None:
            raise KeyError(backtest_id)
        return path

    def _nearby_json(self, directory: Path, name: str) -> dict[str, Any]:
        for candidate in (directory / name, directory.parent / name, directory.parent.parent / name):
            if candidate.exists():
                return read_json(candidate, {}) or {}
        return {}

    @staticmethod
    def _metric(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return clean_value(payload[key])
        return None

    @classmethod
    def _int_metric(cls, payload: dict[str, Any], *keys: str) -> int | None:
        value = cls._metric(payload, *keys)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _stock_names(self) -> dict[str, str]:
        if self._name_map is not None:
            return self._name_map
        path = self.settings.runtime_root / "data" / "v7" / "silver" / "code_name_map.parquet"
        self._name_map = {}
        if path.exists():
            for row in read_parquet_rows(path, columns=["code", "name"]):
                self._name_map[str(row.get("code"))] = str(row.get("name"))
        return self._name_map

    def _realized_lookup(self, directory: Path, symbol: str | None) -> dict[tuple[str, str], list[float]]:
        path = directory / "realized_trades.csv"
        rows = read_csv_rows(path)
        result: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            row_symbol = str(row.get("symbol") or "")
            if symbol and row_symbol != symbol:
                continue
            value = _float(row.get("net_pnl"))
            if value is not None:
                result[(row_symbol, str(row.get("sell_date") or "")[:10])].append(value)
        return result

    def _discover_summary_backtests(self, seen_directories: set[Path]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        summary_artifacts = [
            item for item in self.indexer.scan()
            if item.get("name") == "summary.json" and item.get("status") != "error"
        ]
        for artifact in summary_artifacts:
            summary_path = self.settings.project_root / artifact["path"]
            directory = summary_path.parent
            if directory.resolve() in seen_directories:
                continue
            if not any(
                (directory / name).exists()
                for name in ("nav.csv", "trades.csv", "dot_overlay_trades.csv", "holdings_daily.csv")
            ):
                continue
            payload = read_json(summary_path, {}) or {}
            metrics = _best_metrics_payload(payload)
            relative = require_relative_path(self.settings, directory)
            backtest_id = stable_id("backtest", relative)
            self._runs[backtest_id] = directory
            start_date, end_date = _window_dates(payload.get("window"))
            config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
            summaries.append({
                "id": backtest_id,
                "name": directory.name,
                "strategyVersion": config.get("strategy_version"),
                "modelVersion": config.get("model_version"),
                "factorVersion": config.get("factor_version"),
                "horizon": config.get("horizon"),
                "startDate": metrics.get("start_date") or start_date,
                "endDate": metrics.get("end_date") or end_date,
                "universeSize": metrics.get("n_unique_names_held") or payload.get("n_unique_names_held"),
                "initialCash": config.get("initial_cash"),
                "totalReturn": _first_metric(metrics, "total_return", "total"),
                "annualReturn": _first_metric(metrics, "annualized_return", "annualised_return", "annualized", "annualised", "ann"),
                "maxDrawdown": _first_metric(metrics, "max_drawdown", "maxDD"),
                "sharpe": _first_metric(metrics, "sharpe"),
                "calmar": _first_metric(metrics, "calmar"),
                "volatility": _first_metric(metrics, "volatility"),
                "winRate": _first_metric(metrics, "win_rate", "hit_rate"),
                "profitFactor": _first_metric(metrics, "profit_factor"),
                "turnover": _first_metric(metrics, "turnover", "mean_daily_turnover"),
                "tradeCount": _int(_first_metric(metrics, "n_trades", "trade_count", "executed_legs")),
                "fillCount": _int(_first_metric(metrics, "n_fills")),
                "tTradeCount": _int(_first_metric(metrics, "do_t_trades", "executed_legs", "completed_round_trips")),
                "tContribution": _first_metric(metrics, "annualized_uplift", "overlay_total_return_delta"),
                "totalCost": _first_metric(metrics, "total_cost"),
                "status": "ready",
                "path": relative,
                "tags": ["summary-backed", "paper" if "paper" in relative else "research"],
                "capabilities": self._capabilities(directory),
            })
        return summaries

    def _capabilities(self, directory: Path) -> dict[str, bool | str | None]:
        trade_columns = set(read_csv_columns(directory / "trades.csv"))
        has_order_blotter = _is_order_blotter(trade_columns)
        has_research_events = bool(trade_columns) and not has_order_blotter
        return {
            "equity": any((directory / name).exists() for name in ("pnl.csv", "nav.csv")),
            "trades": has_order_blotter,
            "researchEvents": has_research_events,
            "positions": any(
                path.exists()
                for path in (
                    directory / "position_history.csv",
                    directory / "holdings.csv",
                    directory.parent / "holdings_daily.csv",
                    directory.parent / "target_weights.parquet",
                )
            ),
            "riskEvents": (directory / "risk_events.json").exists(),
            "doT": any(
                (directory / name).exists()
                for name in ("dot_overlay_trades.csv", "factor_combo_scored.parquet")
            ),
            "tradeSchema": "order_blotter" if has_order_blotter else (
                "research_event_table" if has_research_events else None
            ),
        }


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _bare_code(symbol: str) -> str:
    return str(symbol).split(".", 1)[0]


def _risk_severity(row: dict[str, Any]) -> str:
    text = f"{row.get('event_type', '')} {row.get('reason', '')}".lower()
    if any(token in text for token in ("kill", "forced", "critical", "limit_down")):
        return "critical"
    if any(token in text for token in ("reject", "partial", "skip", "warning")):
        return "warning"
    return "info"


def _risk_blocked(row: dict[str, Any]) -> bool | None:
    event = str(row.get("event_type") or "").lower()
    if any(token in event for token in ("rejected", "skipped", "blocked", "cancelled")):
        return True
    return None


def _best_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("paper_account", "metrics", "summary", "executable_eqw_all_A"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            merged = dict(payload)
            merged.update(value)
            return merged
    return payload


def _first_metric(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return clean_value(payload[key])
    return None


def _window_dates(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or ".." not in value:
        return None, None
    start, end = value.split("..", 1)
    return start or None, end or None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_order_blotter(columns: set[str]) -> bool:
    return (
        {"symbol", "side", "status"}.issubset(columns)
        and bool(columns & {"quantity", "filled_quantity"})
        and bool(columns & {"avg_price", "reference_price", "price"})
    )
