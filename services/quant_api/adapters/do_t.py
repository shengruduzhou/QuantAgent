from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from services.quant_api.adapters.utils import read_csv_rows, read_json, require_relative_path
from services.quant_api.config import ApiSettings, stable_id


class DoTAdapter:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._sources: dict[str, Path] = {}

    def list_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        self._sources = {}
        for report_path in self.settings.runtime_root.glob("reports/intraday_dot_factor_combo*/factor_combo_report.json"):
            directory = report_path.parent
            source_id = stable_id("dot", require_relative_path(self.settings, directory))
            self._sources[source_id] = directory
            payload = read_json(report_path, {}) or {}
            metrics = payload.get("metrics", {})
            sources.append({
                "id": source_id,
                "name": directory.name,
                "path": require_relative_path(self.settings, directory),
                "verdict": payload.get("verdict"),
                "reason": payload.get("reason"),
                "metrics": {
                    key: metrics.get(key)
                    for key in (
                        "n_legs", "hit_rate", "mean_net_bps", "daily_uplift_bps",
                        "annualized_uplift", "eod_restore_rate", "stop_rate",
                    )
                    if key in metrics
                },
                "modifiedAt": report_path.stat().st_mtime,
            })
        replay = self.settings.runtime_root / "paper" / "replay_2026"
        if (replay / "dot_overlay_summary.json").exists():
            source_id = stable_id("dot", require_relative_path(self.settings, replay))
            self._sources[source_id] = replay
            payload = read_json(replay / "dot_overlay_summary.json", {}) or {}
            sources.append({
                "id": source_id,
                "name": replay.name,
                "path": require_relative_path(self.settings, replay),
                "verdict": "RESEARCH_ONLY",
                "reason": None,
                "metrics": payload,
                "modifiedAt": (replay / "dot_overlay_summary.json").stat().st_mtime,
            })
        return sorted(sources, key=lambda item: item["modifiedAt"], reverse=True)

    def analyze(self, source_id: str | None = None, symbol: str | None = None, limit: int = 500) -> dict[str, Any]:
        if not self._sources:
            sources = self.list_sources()
        else:
            sources = self.list_sources()
        if not sources:
            return self._empty(symbol)
        if source_id is None:
            source_id = sources[0]["id"]
        directory = self._sources.get(source_id)
        if directory is None:
            raise KeyError(source_id)
        scored = directory / "factor_combo_scored.parquet"
        if scored.exists():
            return self._from_factor_combo(source_id, directory, scored, symbol, limit)
        trades = directory / "dot_overlay_trades.csv"
        if trades.exists():
            return self._from_replay(source_id, directory, trades, symbol, limit)
        return self._empty(symbol, source_id)

    def _from_factor_combo(
        self,
        source_id: str,
        directory: Path,
        path: Path,
        symbol: str | None,
        limit: int,
    ) -> dict[str, Any]:
        lazy = pl.scan_parquet(path)
        if symbol:
            lazy = lazy.filter(pl.col("symbol") == symbol)
        columns = [
            "symbol", "trade_date", "mode", "state", "gross_ret", "net_ret",
            "entry_px", "exit_px", "requested_qty", "filled_qty",
            "entry_fill_time", "exit_fill_time", "entry_fill_status", "exit_fill_status",
            "entry_fill_reason", "exit_fill_reason", "pred_net_ret_bps",
            "pred_eod_restore_prob", "entry_mode_adverse_risk",
        ]
        schema = lazy.collect_schema()
        frame = (
            lazy.select([name for name in columns if name in schema])
            .sort("trade_date", descending=True)
            .head(min(max(1, limit), self.settings.max_table_rows))
            .collect()
        )
        pairs = []
        wins = 0
        total_net = 0.0
        high_fail = 0
        low_fail = 0
        for index, row in enumerate(frame.to_dicts()):
            net_ret = _float(row.get("net_ret"))
            state = str(row.get("state") or "")
            mode = str(row.get("mode") or "")
            success = net_ret > 0 if net_ret is not None else None
            wins += int(success is True)
            total_net += net_ret or 0.0
            high_failed = "sell" in mode and ("stop" in state or "restore" in state)
            low_failed = "buy" in mode and ("stop" in state or "breakdown" in state)
            high_fail += int(high_failed)
            low_fail += int(low_failed)
            entry_status = row.get("entry_fill_status")
            exit_status = row.get("exit_fill_status")
            quantity = _float(row.get("filled_qty")) or _float(row.get("requested_qty"))
            pairs.append({
                "id": stable_id("dotpair", f"{source_id}:{index}:{row.get('symbol')}:{row.get('trade_date')}"),
                "symbol": str(row.get("symbol") or ""),
                "tradeDate": _iso(row.get("trade_date")),
                "mode": "SELL_HIGH_BUY_LOW" if "sell" in mode else ("BUY_LOW_SELL_OLD_HIGH" if "buy" in mode else "UNKNOWN"),
                "buyTime": row.get("exit_fill_time") if "sell" in mode else row.get("entry_fill_time"),
                "sellTime": row.get("entry_fill_time") if "sell" in mode else row.get("exit_fill_time"),
                "buyPrice": _float(row.get("exit_px")) if "sell" in mode else _float(row.get("entry_px")),
                "sellPrice": _float(row.get("entry_px")) if "sell" in mode else _float(row.get("exit_px")),
                "quantity": quantity,
                "grossPnl": None,
                "cost": None,
                "netPnl": net_ret,
                "edgePct": _float(row.get("gross_ret")),
                "state": state or None,
                "entryStatus": entry_status,
                "exitStatus": exit_status,
                "highSellFailed": high_failed,
                "lowBuyFailed": low_failed,
                "missedUpsidePct": None,
                "adverseMovePct": _float(row.get("entry_mode_adverse_risk")),
                "drawdownContribution": None,
                "success": success,
                "predictionNetBps": _float(row.get("pred_net_ret_bps")),
                "restoreProbability": _float(row.get("pred_eod_restore_prob")),
                "issues": [],
            })
        report = read_json(directory / "factor_combo_report.json", {}) or {}
        count = len(pairs)
        return {
            "sourceId": source_id,
            "symbol": symbol,
            "summary": {
                "pairCount": count,
                "successRate": wins / count if count else None,
                "failureRate": 1.0 - wins / count if count else None,
                "highSellFailureRate": high_fail / count if count else None,
                "lowBuyFailureRate": low_fail / count if count else None,
                "totalNetPnl": total_net if count else None,
                "returnContribution": _nested_metric(report, "metrics", "annualized_uplift"),
                "drawdownContribution": None,
                "qualityScore": _nested_metric(report, "metrics", "hit_rate"),
            },
            "pairs": pairs,
            "byRegime": report.get("by_regime") or report.get("metrics", {}).get("by_regime") or {},
            "verdict": report.get("verdict"),
            "reason": report.get("reason"),
        }

    def _from_replay(
        self,
        source_id: str,
        directory: Path,
        path: Path,
        symbol: str | None,
        limit: int,
    ) -> dict[str, Any]:
        rows = read_csv_rows(path, limit=limit)
        if symbol:
            rows = [row for row in rows if row.get("symbol") == symbol]
        pairs = []
        for index, row in enumerate(rows):
            net_ret = _float(row.get("net_ret"))
            pairs.append({
                "id": stable_id("dotpair", f"{source_id}:{index}"),
                "symbol": str(row.get("symbol") or ""),
                "tradeDate": str(row.get("trade_date") or ""),
                "mode": "UNKNOWN",
                "buyTime": None,
                "sellTime": None,
                "buyPrice": None,
                "sellPrice": None,
                "quantity": None,
                "grossPnl": _float(row.get("gross_ret")),
                "cost": None,
                "netPnl": net_ret,
                "edgePct": _float(row.get("gross_ret")),
                "state": row.get("state"),
                "entryStatus": None,
                "exitStatus": None,
                "highSellFailed": None,
                "lowBuyFailed": str(row.get("state") or "") == "closed_stop",
                "missedUpsidePct": None,
                "adverseMovePct": None,
                "drawdownContribution": None,
                "success": net_ret > 0 if net_ret is not None else None,
                "issues": [{
                    "code": "daily_only_artifact",
                    "message": "该历史 artifact 没有 minute price/quantity。",
                    "recoverable": False,
                }],
            })
        summary = read_json(directory / "dot_overlay_summary.json", {}) or {}
        count = len(pairs)
        wins = sum(item["success"] is True for item in pairs)
        return {
            "sourceId": source_id,
            "symbol": symbol,
            "summary": {
                "pairCount": count,
                "successRate": wins / count if count else None,
                "failureRate": 1.0 - wins / count if count else None,
                "highSellFailureRate": None,
                "lowBuyFailureRate": None,
                "totalNetPnl": sum(item["netPnl"] or 0.0 for item in pairs) if count else None,
                "returnContribution": summary.get("annualized_uplift"),
                "drawdownContribution": None,
                "qualityScore": summary.get("hit_rate"),
            },
            "pairs": pairs,
            "byRegime": summary.get("by_regime", {}),
            "verdict": "RESEARCH_ONLY",
            "reason": None,
        }

    @staticmethod
    def _empty(symbol: str | None, source_id: str = "") -> dict[str, Any]:
        return {
            "sourceId": source_id,
            "symbol": symbol,
            "summary": {
                "pairCount": 0,
                "successRate": None,
                "failureRate": None,
                "highSellFailureRate": None,
                "lowBuyFailureRate": None,
                "totalNetPnl": None,
                "returnContribution": None,
                "drawdownContribution": None,
                "qualityScore": None,
            },
            "pairs": [],
            "byRegime": {},
        }


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


def _nested_metric(payload: dict[str, Any], group: str, key: str) -> Any:
    value = payload.get(group, {})
    return value.get(key) if isinstance(value, dict) else None
