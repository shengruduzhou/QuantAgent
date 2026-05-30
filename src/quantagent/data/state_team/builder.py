"""State-team inference builder.

Reads three optional public-data inputs and emits inferred buying
events:

1. ``etf_flows`` — wide-form daily ETF net inflow (CNY billions).
   columns = ETF code, index = trade_date.
2. ``top10_holders`` — long-form filings: ``trade_date``, ``symbol``,
   ``holder_name``, ``share_pct`` (% of outstanding).
3. ``benchmark_returns`` — used to detect "post-crash" windows.

Each module can be called independently. The builder concatenates,
normalises, and computes a per-(date, theme) cumulative inferred
strength suitable for use as a model feature.

Compliance posture (hard-coded):
* Every output row carries ``evidence_label = "inferred"``.
* Every row records the evidence type + numeric strength so reviewers
  can audit how the signal was constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Public list of recognised evidence types
EVIDENCE_TYPES: tuple[str, ...] = (
    "etf_concentrated_inflow",
    "top10_holder_appearance",
    "post_crash_index_buying",
    "block_trade_match",
)


INFERENCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "trade_date",
    "available_at",
    "evidence_type",
    "evidence_label",       # always "inferred"
    "evidence_strength",
    "scope",                # "index_wide" / "symbol" / "sector"
    "scope_value",          # ETF code / symbol / sector name
    "description",
    "source",
    "source_version",
    "fetched_at",
)


# Known state-team holder name fragments (Chinese + English transliterations)
STATE_TEAM_HOLDER_KEYWORDS: tuple[str, ...] = (
    "中央汇金",
    "汇金",
    "中国证券金融",
    "证金",
    "国新投资",
    "国新",
    "全国社保",
    "社保基金",
    "中投",
    "CIC",
)


# ---------------------------------------------------------------------------
# Config + Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateTeamInferenceConfig:
    output_root: str | Path = "runtime/data/v7"
    source: str = "manual_local_import"
    source_version: str = "unknown"
    # Inference thresholds
    etf_concentrated_inflow_threshold_cny_bn: float = 5.0  # 5 亿/日 算异常
    post_crash_5d_threshold: float = -0.08
    post_crash_etf_inflow_threshold_cny_bn: float = 10.0  # 10 亿
    # Gate
    min_events: int = 3
    min_mean_strength: float = 0.40


@dataclass
class StateTeamInferenceResult:
    frame: pd.DataFrame
    coverage: dict[str, Any]
    validation: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_ts(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _event_id(evidence_type: str, scope_value: str, trade_date: pd.Timestamp) -> str:
    raw = f"{evidence_type}||{scope_value}||{trade_date.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _wrap_event(
    *,
    trade_date: pd.Timestamp,
    evidence_type: str,
    strength: float,
    scope: str,
    scope_value: str,
    description: str,
    config: StateTeamInferenceConfig,
) -> dict[str, Any]:
    return {
        "event_id": _event_id(evidence_type, scope_value, trade_date),
        "trade_date": trade_date,
        # state-team filings appear with T+1 lag for ETF flows, T+45 for top-10 filings
        # (handled per-evidence below; this is a safe default)
        "available_at": trade_date + pd.tseries.offsets.BDay(1),
        "evidence_type": evidence_type,
        "evidence_label": "inferred",
        "evidence_strength": float(np.clip(strength, 0.0, 1.0)),
        "scope": scope,
        "scope_value": str(scope_value),
        "description": description,
        "source": config.source,
        "source_version": config.source_version,
        "fetched_at": pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)),
    }


# ---------------------------------------------------------------------------
# Evidence-type detectors
# ---------------------------------------------------------------------------

def infer_etf_concentrated_inflow(
    etf_flows: pd.DataFrame | None,
    *,
    config: StateTeamInferenceConfig | None = None,
) -> list[dict[str, Any]]:
    """Detect days when a broad-index ETF has a concentrated net inflow.

    Parameters
    ----------
    etf_flows : DataFrame with index trade_date and columns = ETF
        codes (e.g. "510300.SH" for HuaTai-PineBridge CSI300). Values
        are net inflow in CNY billions.
    """
    cfg = config or StateTeamInferenceConfig()
    if etf_flows is None or etf_flows.empty:
        return []
    df = etf_flows.copy()
    if "trade_date" in df.columns:
        df = df.set_index("trade_date")
    df.index = pd.to_datetime(df.index)
    out: list[dict[str, Any]] = []
    for etf, series in df.items():
        for trade_date, value in series.dropna().items():
            if value >= cfg.etf_concentrated_inflow_threshold_cny_bn:
                # Strength scales from 0.5 (at threshold) to 0.85 (at 5x threshold)
                ratio = value / cfg.etf_concentrated_inflow_threshold_cny_bn
                strength = 0.50 + min(0.35, (ratio - 1.0) * 0.10)
                out.append(
                    _wrap_event(
                        trade_date=trade_date,
                        evidence_type="etf_concentrated_inflow",
                        strength=strength,
                        scope="index_wide",
                        scope_value=str(etf),
                        description=f"{etf} net inflow {float(value):.2f}亿",
                        config=cfg,
                    )
                )
    return out


def infer_post_crash_index_buying(
    benchmark_returns: pd.Series | pd.DataFrame | None,
    etf_flows: pd.DataFrame | None,
    *,
    config: StateTeamInferenceConfig | None = None,
) -> list[dict[str, Any]]:
    """Day after a 5-day benchmark crash with a heavy ETF inflow.

    Historically this pattern coincides with state-team intervention.
    """
    cfg = config or StateTeamInferenceConfig()
    if benchmark_returns is None or etf_flows is None:
        return []
    if isinstance(benchmark_returns, pd.DataFrame):
        if {"trade_date", "ret"}.issubset(benchmark_returns.columns):
            bench = benchmark_returns.set_index("trade_date")["ret"]
        else:
            return []
    else:
        bench = benchmark_returns
    bench = pd.to_numeric(bench, errors="coerce").dropna()
    bench.index = pd.to_datetime(bench.index)
    rolling_5d = (1.0 + bench).rolling(5, min_periods=5).apply(lambda x: x.prod() - 1.0, raw=True)
    crash_dates = rolling_5d[rolling_5d <= cfg.post_crash_5d_threshold].index

    df = etf_flows.copy()
    if "trade_date" in df.columns:
        df = df.set_index("trade_date")
    df.index = pd.to_datetime(df.index)
    out: list[dict[str, Any]] = []
    for crash_dt in crash_dates:
        candidate_dates = pd.bdate_range(
            crash_dt, crash_dt + pd.Timedelta(days=10)
        )[:5]
        for etf, series in df.items():
            for trade_date in candidate_dates:
                if trade_date not in series.index:
                    continue
                value = float(series.loc[trade_date]) if pd.notna(series.loc[trade_date]) else 0.0
                if value >= cfg.post_crash_etf_inflow_threshold_cny_bn:
                    # Strength higher than the plain ETF detector because
                    # the context (crash) is unusual
                    ratio = value / cfg.post_crash_etf_inflow_threshold_cny_bn
                    strength = 0.60 + min(0.30, (ratio - 1.0) * 0.10)
                    out.append(
                        _wrap_event(
                            trade_date=trade_date,
                            evidence_type="post_crash_index_buying",
                            strength=strength,
                            scope="index_wide",
                            scope_value=str(etf),
                            description=(
                                f"5d crash {float(rolling_5d.loc[crash_dt]):.2%} "
                                f"+ {etf} inflow {value:.2f}亿"
                            ),
                            config=cfg,
                        )
                    )
    return out


def infer_top10_holder_appearance(
    top10_holders: pd.DataFrame | None,
    *,
    config: StateTeamInferenceConfig | None = None,
) -> list[dict[str, Any]]:
    """A state-team-name holder appears in a symbol's top-10 list."""
    cfg = config or StateTeamInferenceConfig()
    if top10_holders is None or top10_holders.empty:
        return []
    work = top10_holders.copy()
    required = {"trade_date", "symbol", "holder_name", "share_pct"}
    if not required.issubset(work.columns):
        return []
    work["trade_date"] = work["trade_date"].map(_coerce_ts)
    work = work[work["trade_date"].notna()]
    work["holder_name"] = work["holder_name"].astype(str)
    work["is_state_team"] = work["holder_name"].apply(
        lambda nm: any(kw in nm for kw in STATE_TEAM_HOLDER_KEYWORDS)
    )
    hits = work[work["is_state_team"]]
    out: list[dict[str, Any]] = []
    for _, row in hits.iterrows():
        share = float(row["share_pct"] or 0.0)
        # Strength scales with stake size: 1% → 0.50, 5%+ → 0.95
        strength = 0.50 + min(0.45, max(0.0, share - 1.0) * 0.11)
        evt = _wrap_event(
            trade_date=row["trade_date"],
            evidence_type="top10_holder_appearance",
            strength=strength,
            scope="symbol",
            scope_value=str(row["symbol"]),
            description=f"{row['holder_name']} {share:.2f}%",
            config=cfg,
        )
        # Top-10 filings are quarterly and typically published 45 days after
        # quarter-end. Push the available_at out accordingly.
        evt["available_at"] = row["trade_date"] + pd.tseries.offsets.BDay(45)
        out.append(evt)
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_state_team_inference(
    *,
    etf_flows: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    top10_holders: pd.DataFrame | None = None,
    extra_events: pd.DataFrame | None = None,
    config: StateTeamInferenceConfig | None = None,
) -> StateTeamInferenceResult:
    """Concatenate evidence from every detector + optional extra rows."""
    cfg = config or StateTeamInferenceConfig()
    rows: list[dict[str, Any]] = []
    rows.extend(infer_etf_concentrated_inflow(etf_flows, config=cfg))
    rows.extend(infer_post_crash_index_buying(benchmark_returns, etf_flows, config=cfg))
    rows.extend(infer_top10_holder_appearance(top10_holders, config=cfg))

    if extra_events is not None and not extra_events.empty:
        for _, r in extra_events.iterrows():
            evt = {
                "event_id": str(r.get("event_id") or _event_id(
                    str(r.get("evidence_type") or "extra"),
                    str(r.get("scope_value") or "_"),
                    pd.to_datetime(r.get("trade_date")),
                )),
                "trade_date": _coerce_ts(r.get("trade_date")),
                "available_at": _coerce_ts(r.get("available_at"))
                or (_coerce_ts(r.get("trade_date")) + pd.tseries.offsets.BDay(1)),
                "evidence_type": str(r.get("evidence_type") or "extra"),
                "evidence_label": "inferred",  # hard-coded
                "evidence_strength": float(np.clip(r.get("evidence_strength", 0.5), 0.0, 1.0)),
                "scope": str(r.get("scope") or "symbol"),
                "scope_value": str(r.get("scope_value") or ""),
                "description": str(r.get("description") or ""),
                "source": str(r.get("source") or cfg.source),
                "source_version": str(r.get("source_version") or cfg.source_version),
                "fetched_at": _coerce_ts(r.get("fetched_at")) or pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)),
            }
            rows.append(evt)

    if not rows:
        empty_frame = pd.DataFrame(columns=list(INFERENCE_REQUIRED_COLUMNS))
        return StateTeamInferenceResult(
            frame=empty_frame,
            coverage={
                "n_events": 0,
                "mean_strength": 0.0,
                "type_counts": {},
                "gate": {
                    "state_team_inference_usable_for_features": False,
                    "reason": "no_events",
                },
            },
            validation={"status": "passed", "n": 0, "errors": []},
        )

    frame = pd.DataFrame(rows)
    # De-dup
    frame = frame.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)
    # Hard compliance check: evidence_label must be "inferred" everywhere
    frame["evidence_label"] = "inferred"
    frame = frame[list(INFERENCE_REQUIRED_COLUMNS)]
    frame = frame.sort_values(["trade_date", "evidence_type"]).reset_index(drop=True)

    n = int(len(frame))
    mean_strength = float(frame["evidence_strength"].mean()) if n else 0.0
    type_counts = frame["evidence_type"].value_counts().to_dict()

    gate_open = n >= cfg.min_events and mean_strength >= cfg.min_mean_strength
    reason = "passed" if gate_open else _gate_reason(n, mean_strength, cfg)

    coverage = {
        "n_events": n,
        "mean_strength": mean_strength,
        "type_counts": type_counts,
        "gate": {
            "state_team_inference_usable_for_features": bool(gate_open),
            "reason": reason,
        },
    }
    validation = {"status": "passed", "n": n, "errors": []}
    return StateTeamInferenceResult(frame=frame, coverage=coverage, validation=validation)


def _gate_reason(n: int, mean_strength: float, cfg: StateTeamInferenceConfig) -> str:
    if n < cfg.min_events:
        return f"too_few_events_{n}_lt_{cfg.min_events}"
    if mean_strength < cfg.min_mean_strength:
        return f"mean_strength_{mean_strength:.3f}_below_{cfg.min_mean_strength:.3f}"
    return "unknown"


# ---------------------------------------------------------------------------
# Builder with writer
# ---------------------------------------------------------------------------

class StateTeamInferenceBuilder:
    def __init__(self, config: StateTeamInferenceConfig | None = None) -> None:
        self.config = config or StateTeamInferenceConfig()

    def build(
        self,
        *,
        etf_flows: pd.DataFrame | None = None,
        benchmark_returns: pd.Series | pd.DataFrame | None = None,
        top10_holders: pd.DataFrame | None = None,
        extra_events: pd.DataFrame | None = None,
    ) -> StateTeamInferenceResult:
        return build_state_team_inference(
            etf_flows=etf_flows,
            benchmark_returns=benchmark_returns,
            top10_holders=top10_holders,
            extra_events=extra_events,
            config=self.config,
        )

    def write(self, result: StateTeamInferenceResult) -> StateTeamInferenceResult:
        root = Path(self.config.output_root)
        silver_dir = root / "silver" / "state_team_inference"
        silver_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = silver_dir / "state_team_inference.parquet"
        result.frame.to_parquet(parquet_path, index=False)
        (silver_dir / "coverage_report.json").write_text(
            json.dumps(result.coverage, indent=2, default=str), encoding="utf-8"
        )
        (silver_dir / "validation_report.json").write_text(
            json.dumps(result.validation, indent=2, default=str), encoding="utf-8"
        )
        manifests_dir = root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / "state_team_inference.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "state_team_inference",
                    "rows": int(len(result.frame)),
                    "extra": {"coverage_report": result.coverage},
                    "source_version": self.config.source_version,
                    "compliance_note": "All rows labelled evidence_label='inferred'; consumer UIs must surface this label.",
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        result.output_paths = {
            "state_team_inference": str(parquet_path),
            "coverage_report": str(silver_dir / "coverage_report.json"),
            "validation_report": str(silver_dir / "validation_report.json"),
            "manifest": str(manifest_path),
        }
        return result


# ---------------------------------------------------------------------------
# Feature attach
# ---------------------------------------------------------------------------

def apply_state_team_features(
    panel: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a per-symbol state-team feature to the training panel.

    The feature ``state_team_signal`` is the cumulative sum of
    ``evidence_strength`` for every event with
    ``available_at <= trade_date`` AND (scope='index_wide' OR
    (scope='symbol' AND scope_value == symbol)).

    PIT-safe via merge_asof backward on available_at.
    """
    if panel is None or panel.empty:
        return panel.copy() if panel is not None else pd.DataFrame()
    if events is None or events.empty:
        out = panel.copy()
        out["state_team_signal"] = 0.0
        return out

    ev = events.copy()
    ev["available_at"] = pd.to_datetime(ev["available_at"])
    ev = ev.dropna(subset=["available_at"]).sort_values("available_at")

    panel_out = panel.copy()
    panel_out["trade_date"] = pd.to_datetime(panel_out["trade_date"])
    panel_out["symbol"] = panel_out["symbol"].astype(str)

    # Index-wide events broadcast to every symbol
    idx_wide = ev[ev["scope"] == "index_wide"].copy()
    if not idx_wide.empty:
        idx_wide["cum_strength"] = idx_wide["evidence_strength"].cumsum()
        left = panel_out[["trade_date"]].copy()
        left["__orig_index"] = panel_out.index
        left = left.sort_values("trade_date").reset_index(drop=True)
        merged = pd.merge_asof(
            left,
            idx_wide[["available_at", "cum_strength"]],
            left_on="trade_date",
            right_on="available_at",
            direction="backward",
        )
        idx_signal = pd.Series(0.0, index=panel_out.index, dtype=float)
        idx_signal.loc[merged["__orig_index"].values] = (
            merged["cum_strength"].fillna(0.0).values
        )
    else:
        idx_signal = pd.Series(0.0, index=panel_out.index, dtype=float)

    # Per-symbol events: cumsum within (symbol, available_at)
    sym_events = ev[ev["scope"] == "symbol"].copy()
    sym_signal = pd.Series(0.0, index=panel_out.index, dtype=float)
    if not sym_events.empty:
        sym_events = sym_events.sort_values(["scope_value", "available_at"])
        sym_events["cum_strength"] = sym_events.groupby("scope_value")[
            "evidence_strength"
        ].cumsum()
        sym_events = sym_events.rename(columns={"scope_value": "symbol"})
        for sym, sub in sym_events.groupby("symbol"):
            sub = sub[["available_at", "cum_strength"]].sort_values("available_at")
            mask = panel_out["symbol"] == sym
            if not mask.any():
                continue
            left = panel_out.loc[mask, ["trade_date"]].sort_values("trade_date").copy()
            left["__orig_index"] = left.index
            merged = pd.merge_asof(
                left, sub, left_on="trade_date", right_on="available_at", direction="backward"
            )
            sym_signal.loc[merged["__orig_index"].values] = (
                sym_signal.loc[merged["__orig_index"].values].values
                + merged["cum_strength"].fillna(0.0).values
            )

    panel_out["state_team_signal"] = idx_signal + sym_signal
    panel_out["state_team_evidence_label"] = "inferred"  # compliance trail
    return panel_out


# ---------------------------------------------------------------------------
# Manifest-gated helper
# ---------------------------------------------------------------------------

def state_team_inference_for_features(
    events: pd.DataFrame | None,
    manifest_path: str | Path | None,
) -> pd.DataFrame | None:
    if events is None or len(events) == 0:
        return None
    if manifest_path is None:
        return None
    p = Path(manifest_path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    gate = (
        (payload.get("extra") or {})
        .get("coverage_report", {})
        .get("gate", {})
    )
    if not gate.get("state_team_inference_usable_for_features"):
        return None
    return events
