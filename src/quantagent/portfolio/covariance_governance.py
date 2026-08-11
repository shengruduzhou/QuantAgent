"""Train-window covariance governance for portfolio construction.

The allocator must not receive an unaudited covariance matrix whose estimator,
fit window, missing-data policy or PSD repair is unknown.  This module keeps the
risk model inside the research clock:

* callers provide a return history and an explicit ``train_end``;
* all estimator selection is performed inside that train window only;
* sample, diagonal shrinkage, EWMA and Ledoit-Wolf candidates are compared on a
  chronological calibration tail of the train window;
* the selected matrix is symmetrised and projected to PSD with an explicit
  eigenvalue floor;
* fit-window and data fingerprints are returned for Stage-4 lineage.

The calibration tail is *not* a final holdout.  It only chooses a covariance
estimator.  Portfolio/alpha promotion still requires the repository's governed
OOS/holdout and executable-cost evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.quant_math.covariance import (
    ewma_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
    shrinkage_covariance,
)


@dataclass(frozen=True)
class CovarianceGovernanceConfig:
    method: str = "auto"  # auto | sample | diagonal_shrinkage | ewma | ledoit_wolf
    calibration_fraction: float = 0.25
    min_fit_observations: int = 60
    min_calibration_observations: int = 20
    diagonal_shrinkage: float = 0.20
    ewma_span: int = 60
    annualize: bool = True
    periods_per_year: int = 244
    eigenvalue_floor: float = 1e-10
    max_condition_number: float = 1e8


@dataclass(frozen=True)
class CovarianceFitResult:
    covariance: pd.DataFrame
    report: dict[str, object]


def _canonical_returns(
    returns: pd.DataFrame,
    *,
    train_end: object,
    assets: Iterable[str] | None,
    min_observations: int,
) -> pd.DataFrame:
    if returns is None or returns.empty:
        raise ValueError("covariance returns are empty")
    frame = returns.copy()
    if assets is not None:
        columns = [str(item) for item in assets]
        missing = sorted(set(columns) - set(frame.columns.astype(str)))
        if missing:
            raise ValueError(f"covariance returns missing assets: {missing[:10]}")
        frame.columns = frame.columns.astype(str)
        frame = frame[columns]
    else:
        frame.columns = frame.columns.astype(str)
    if not isinstance(frame.index, pd.DatetimeIndex):
        parsed = pd.to_datetime(frame.index, errors="coerce")
        if parsed.isna().any():
            raise ValueError("covariance returns index must be date-like")
        frame.index = pd.DatetimeIndex(parsed)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    if frame.index.duplicated().any():
        raise ValueError("covariance returns require unique dates")
    frame = frame.sort_index()
    cutoff = pd.Timestamp(train_end)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    cutoff = cutoff.normalize()
    frame = frame.loc[frame.index <= cutoff]
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Complete-case estimation is intentionally conservative. Pairwise sample
    # counts can generate an internally inconsistent/non-PSD covariance matrix.
    frame = frame.dropna(how="any")
    if len(frame) < int(min_observations):
        raise ValueError(
            f"covariance train window has {len(frame)} complete observations; "
            f"requires >= {int(min_observations)}"
        )
    if frame.shape[1] < 2:
        raise ValueError("covariance estimation requires at least two assets")
    return frame


def _fingerprint(frame: pd.DataFrame) -> str:
    payload = bytearray()
    payload.extend("|".join(frame.columns.astype(str)).encode("utf-8"))
    payload.extend(b"\n")
    payload.extend("|".join(frame.index.strftime("%Y-%m-%d")).encode("utf-8"))
    payload.extend(b"\n")
    values = np.ascontiguousarray(frame.to_numpy(dtype="float64"))
    payload.extend(values.tobytes())
    return sha256(bytes(payload)).hexdigest()


def _estimate_raw(frame: pd.DataFrame, method: str, config: CovarianceGovernanceConfig) -> pd.DataFrame:
    annualize = bool(config.annualize)
    periods = int(config.periods_per_year)
    if method == "sample":
        return sample_covariance(frame, annualize=annualize, periods_per_year=periods)
    if method == "diagonal_shrinkage":
        return shrinkage_covariance(
            frame,
            shrinkage=float(config.diagonal_shrinkage),
            annualize=annualize,
            periods_per_year=periods,
        )
    if method == "ewma":
        return ewma_covariance(
            frame,
            span=int(config.ewma_span),
            annualize=annualize,
            periods_per_year=periods,
        )
    if method == "ledoit_wolf":
        return ledoit_wolf_covariance(
            frame,
            annualize=annualize,
            periods_per_year=periods,
        )
    raise ValueError(f"unsupported covariance method: {method}")


def _psd_repair(covariance: pd.DataFrame, config: CovarianceGovernanceConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    matrix = covariance.to_numpy(dtype=float)
    if matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("covariance estimator returned non-finite/non-square matrix")
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    max_eig = float(max(np.max(eigvals), config.eigenvalue_floor))
    floor = max(float(config.eigenvalue_floor), max_eig / float(config.max_condition_number))
    clipped = np.maximum(eigvals, floor)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    repaired = 0.5 * (repaired + repaired.T)
    condition = float(np.max(clipped) / max(np.min(clipped), floor))
    return (
        pd.DataFrame(repaired, index=covariance.index, columns=covariance.columns),
        {
            "raw_min_eigenvalue": float(np.min(eigvals)),
            "psd_floor": float(floor),
            "repaired_min_eigenvalue": float(np.min(clipped)),
            "condition_number": condition,
            "eigenvalues_clipped": int(np.sum(eigvals < floor)),
        },
    )


def _validation_loss(predicted: pd.DataFrame, validation: pd.DataFrame, annualize_factor: float) -> float:
    # Compare a covariance forecast to the realised covariance of the later
    # calibration block. Normalisation keeps scores comparable across volatility
    # regimes while preserving the cross-covariance structure.
    realised = validation.cov() * annualize_factor
    p = predicted.reindex(index=realised.index, columns=realised.columns).to_numpy(dtype=float)
    r = realised.to_numpy(dtype=float)
    denom = float(np.linalg.norm(r, ord="fro"))
    if not np.isfinite(p).all() or not np.isfinite(r).all() or denom <= 1e-16:
        return float("inf")
    return float(np.linalg.norm(p - r, ord="fro") / denom)


def fit_governed_covariance(
    returns: pd.DataFrame,
    *,
    train_end: object,
    assets: Iterable[str] | None = None,
    config: CovarianceGovernanceConfig | None = None,
) -> CovarianceFitResult:
    cfg = config or CovarianceGovernanceConfig()
    allowed = ("sample", "diagonal_shrinkage", "ewma", "ledoit_wolf")
    if cfg.method not in {"auto", *allowed}:
        raise ValueError(f"unsupported covariance governance method: {cfg.method}")
    if not 0.05 <= float(cfg.calibration_fraction) <= 0.50:
        raise ValueError("calibration_fraction must be in [0.05, 0.50]")

    frame = _canonical_returns(
        returns,
        train_end=train_end,
        assets=assets,
        min_observations=max(
            int(cfg.min_fit_observations) + int(cfg.min_calibration_observations),
            int(cfg.min_fit_observations),
        ) if cfg.method == "auto" else int(cfg.min_fit_observations),
    )
    scores: dict[str, float | None] = {name: None for name in allowed}
    rejected: dict[str, str] = {}

    if cfg.method == "auto":
        n_cal = max(
            int(cfg.min_calibration_observations),
            int(round(len(frame) * float(cfg.calibration_fraction))),
        )
        n_cal = min(n_cal, len(frame) - int(cfg.min_fit_observations))
        if n_cal < int(cfg.min_calibration_observations):
            raise ValueError("insufficient train-window calibration observations")
        fit = frame.iloc[:-n_cal]
        validation = frame.iloc[-n_cal:]
        annualize_factor = float(cfg.periods_per_year) if cfg.annualize else 1.0
        for method in allowed:
            try:
                raw = _estimate_raw(fit, method, cfg)
                repaired, _ = _psd_repair(raw, cfg)
                scores[method] = _validation_loss(repaired, validation, annualize_factor)
            except Exception as exc:  # estimator remains auditable, never silently selected
                scores[method] = float("inf")
                rejected[method] = f"{type(exc).__name__}: {exc}"
        finite = {name: value for name, value in scores.items() if value is not None and np.isfinite(value)}
        if not finite:
            raise ValueError(f"all covariance estimators failed: {rejected}")
        selected_method = min(finite, key=finite.get)
        calibration = {
            "fit_start": fit.index.min().date().isoformat(),
            "fit_end": fit.index.max().date().isoformat(),
            "validation_start": validation.index.min().date().isoformat(),
            "validation_end": validation.index.max().date().isoformat(),
            "fit_observations": int(len(fit)),
            "validation_observations": int(len(validation)),
        }
    else:
        selected_method = cfg.method
        calibration = {"status": "fixed_estimator_no_internal_selection"}

    raw_full = _estimate_raw(frame, selected_method, cfg)
    covariance, repair = _psd_repair(raw_full, cfg)
    report: dict[str, object] = {
        "schema": "quantagent.portfolio.covariance_governance.v1",
        "researchOnly": True,
        "productionEligible": False,
        "selected_method": selected_method,
        "candidate_validation_loss": scores,
        "rejected_estimators": rejected,
        "train_start": frame.index.min().date().isoformat(),
        "train_end": frame.index.max().date().isoformat(),
        "train_observations": int(len(frame)),
        "assets": list(frame.columns.astype(str)),
        "asset_count": int(frame.shape[1]),
        "returns_sha256": _fingerprint(frame),
        "complete_case_policy": True,
        "calibration": calibration,
        "psd_repair": repair,
        "config": asdict(cfg),
        "promotion_note": (
            "estimator selection uses only the supplied train window; final portfolio "
            "promotion still requires independent OOS/holdout and executable-cost evidence"
        ),
    }
    return CovarianceFitResult(covariance=covariance, report=report)


__all__ = [
    "CovarianceGovernanceConfig",
    "CovarianceFitResult",
    "fit_governed_covariance",
]
