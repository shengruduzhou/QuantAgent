"""GTJA-191 (国泰君安《基于短周期价量特征的多因子选股体系》) factor library.

Tranche 1: 55 representative formulas re-implemented as FULL-HISTORY
vectorised daily factors (the public reference implementation computes a
single end-date cross-section with deprecated pandas APIs, so formulas were
ported by semantics, not by copying code).

Conventions
-----------
* Input: long frame with symbol/trade_date/open/high/low/close/volume/amount.
* ``vwap`` = amount/volume (per bar), GTJA's ``VWAP``.
* ``SMA(X, n, m)`` is the recursive smoother y_t = (m·x_t + (n−m)·y_{t−1})/n,
  implemented exactly via ``ewm(alpha=m/n, adjust=False)``.
* Benchmark-dependent formulas (REGBETA vs index) are deferred to tranche 2.
* Factors keep GTJA numbering: ``gtja001`` .. ``gtja191`` (gaps = not yet
  ported). Every implemented factor must pass through
  ``scripts/factor_full_judgment.py`` before entering any pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")


# --------------------------------------------------------------------------- #
# per-symbol helpers (long frame, sorted by symbol/trade_date)                 #
# --------------------------------------------------------------------------- #


def _g(data: pd.DataFrame, s: pd.Series):
    return s.groupby(data["symbol"].to_numpy(), sort=False)


def _delay(data, s, n=1):
    return _g(data, s).shift(n)


def _delta(data, s, n=1):
    return s - _g(data, s).shift(n)


def _ts_sum(data, s, n):
    return _g(data, s).rolling(n, min_periods=n).sum().reset_index(level=0, drop=True)


def _ts_mean(data, s, n):
    return _g(data, s).rolling(n, min_periods=n).mean().reset_index(level=0, drop=True)


def _ts_std(data, s, n):
    return _g(data, s).rolling(n, min_periods=n).std().reset_index(level=0, drop=True)


def _ts_min(data, s, n):
    return _g(data, s).rolling(n, min_periods=n).min().reset_index(level=0, drop=True)


def _ts_max(data, s, n):
    return _g(data, s).rolling(n, min_periods=n).max().reset_index(level=0, drop=True)


def _sma(data, s, n, m):
    """GTJA SMA(X, n, m): y = (m·x + (n−m)·y_prev)/n == ewm(alpha=m/n)."""
    return _g(data, s).transform(lambda x: x.ewm(alpha=m / n, adjust=False).mean())


def _rank(data, s):
    """Cross-sectional pct rank per trade_date."""
    return s.groupby(data["trade_date"].to_numpy(), sort=False).rank(pct=True)


def _ts_rank(data, s, n):
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for start, stop in _symbol_blocks(data["symbol"].to_numpy()):
        block = arr[start:stop]
        if len(block) < n:
            continue
        views = np.lib.stride_tricks.sliding_window_view(block, n)
        last = views[:, -1]
        with np.errstate(invalid="ignore"):
            less = (views < last[:, None]).sum(axis=1)
            equal = (views == last[:, None]).sum(axis=1)
        r = (less + (equal + 1.0) / 2.0) / n
        r[np.isnan(views).any(axis=1)] = np.nan
        out[start + n - 1 : stop] = r
    return pd.Series(out, index=s.index)


def _ts_corr(data, x, y, n):
    out = pd.Series(np.nan, index=x.index, dtype=float)
    frame = pd.DataFrame({"x": x.to_numpy(dtype=float), "y": y.to_numpy(dtype=float)}, index=x.index)
    for _, group in frame.groupby(data["symbol"].to_numpy(), sort=False):
        out.loc[group.index] = group["x"].rolling(n, min_periods=n).corr(group["y"])
    return out.replace([np.inf, -np.inf], np.nan)


def _decay_linear(data, s, n):
    arr = s.to_numpy(dtype=float)
    w = np.arange(1.0, n + 1.0)
    w = w / w.sum()
    kernel = w[::-1]
    out = np.full(len(arr), np.nan)
    for start, stop in _symbol_blocks(data["symbol"].to_numpy()):
        block = arr[start:stop]
        if len(block) < n:
            continue
        conv = np.convolve(np.nan_to_num(block), kernel, mode="valid")
        bad = np.convolve(np.isnan(block).astype(float), np.ones(n), mode="valid") > 0
        conv[bad] = np.nan
        out[start + n - 1 : stop] = conv
    return pd.Series(out, index=s.index)


def _count(data, cond: pd.Series, n):
    return _ts_sum(data, cond.astype(float), n)


def _sum_if(data, s, cond, n):
    return _ts_sum(data, s.where(cond, 0.0), n)


def _highday(data, s, n):
    """Bars since the rolling-window max (0 = today is the max)."""
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for start, stop in _symbol_blocks(data["symbol"].to_numpy()):
        block = arr[start:stop]
        if len(block) < n:
            continue
        views = np.lib.stride_tricks.sliding_window_view(block, n)
        pos = np.argmax(views, axis=1)
        vals = (n - 1 - pos).astype(float)
        vals[np.isnan(views).any(axis=1)] = np.nan
        out[start + n - 1 : stop] = vals
    return pd.Series(out, index=s.index)


def _lowday(data, s, n):
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for start, stop in _symbol_blocks(data["symbol"].to_numpy()):
        block = arr[start:stop]
        if len(block) < n:
            continue
        views = np.lib.stride_tricks.sliding_window_view(block, n)
        pos = np.argmin(views, axis=1)
        vals = (n - 1 - pos).astype(float)
        vals[np.isnan(views).any(axis=1)] = np.nan
        out[start + n - 1 : stop] = vals
    return pd.Series(out, index=s.index)


def _symbol_blocks(symbols: np.ndarray):
    n = len(symbols)
    if n == 0:
        return
    start = 0
    for i in range(1, n + 1):
        if i == n or symbols[i] != symbols[start]:
            yield start, i
            start = i


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("symbol", "trade_date", *REQUIRED_COLUMNS) if c not in frame.columns]
    if missing:
        raise KeyError(f"GTJA-191 factors require columns {missing}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["symbol", "trade_date"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    for c in REQUIRED_COLUMNS:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["vwap"] = (data["amount"] / data["volume"].replace(0.0, np.nan))
    data["prev_close"] = _delay(data, data["close"], 1)
    data["ret"] = data["close"] / data["prev_close"].replace(0.0, np.nan) - 1.0
    data["hlc3"] = (data["high"] + data["low"] + data["close"]) / 3.0
    return data


# --------------------------------------------------------------------------- #
# factor formulas (tranche 1)                                                  #
# --------------------------------------------------------------------------- #


def _factor_values(d: pd.DataFrame) -> dict[str, pd.Series]:
    O, H, L, C, V, A = d["open"], d["high"], d["low"], d["close"], d["volume"], d["amount"]
    VWAP, PC, RET = d["vwap"], d["prev_close"], d["ret"]
    HLC3 = d["hlc3"]
    eps = 1e-12
    out: dict[str, pd.Series] = {}

    log_v = np.log(V.replace(0.0, np.nan))
    out["gtja001"] = -_ts_corr(d, _rank(d, _delta(d, log_v, 1)), _rank(d, (C - O) / O.replace(0, np.nan)), 6)
    clv = ((C - L) - (H - C)) / (H - L).replace(0, np.nan)
    out["gtja002"] = -_delta(d, clv, 1)
    cond_up, cond_dn = C > PC, C < PC
    part = (C - np.minimum(PC.where(cond_up), L.where(cond_up)).fillna(0)
            + (C - np.maximum(PC.where(cond_dn), H.where(cond_dn))).fillna(0) * 0)
    t3 = (C - np.minimum(PC, L)).where(cond_up, 0.0) + (C - np.maximum(PC, H)).where(cond_dn, 0.0)
    out["gtja003"] = _ts_sum(d, t3, 6)
    del part
    out["gtja005"] = -_ts_max(d, _ts_corr(d, _ts_rank(d, V, 5), _ts_rank(d, H, 5), 5), 3)
    out["gtja006"] = -_rank(d, np.sign(_delta(d, O * 0.85 + H * 0.15, 4)))
    out["gtja008"] = -_rank(d, _delta(d, (H + L) / 2 * 0.2 + VWAP * 0.8, 4))
    out["gtja009"] = _sma(d, ((H + L) / 2 - (_delay(d, H, 1) + _delay(d, L, 1)) / 2) * (H - L) / V.replace(0, np.nan), 7, 2)
    out["gtja011"] = _ts_sum(d, clv * V, 6)
    out["gtja012"] = _rank(d, O - _ts_mean(d, VWAP, 10)) * (-_rank(d, (C - VWAP).abs()))
    out["gtja013"] = np.sqrt((H * L).clip(lower=0)) - VWAP
    out["gtja014"] = C - _delay(d, C, 5)
    out["gtja015"] = O / PC.replace(0, np.nan) - 1.0
    out["gtja016"] = -_ts_max(d, _rank(d, _ts_corr(d, _rank(d, V), _rank(d, VWAP), 5)), 5)
    out["gtja018"] = C / _delay(d, C, 5).replace(0, np.nan)
    d5 = _delay(d, C, 5)
    out["gtja019"] = np.where(C < d5, (C - d5) / d5, np.where(C == d5, 0.0, (C - d5) / C.replace(0, np.nan)))
    out["gtja019"] = pd.Series(out["gtja019"], index=C.index)
    d6 = _delay(d, C, 6)
    out["gtja020"] = (C - d6) / d6.replace(0, np.nan) * 100
    m6 = _ts_mean(d, C, 6)
    out["gtja022"] = _sma(d, ((C - m6) / m6.replace(0, np.nan)) - _delay(d, (C - m6) / m6.replace(0, np.nan), 3), 12, 1)
    out["gtja024"] = _sma(d, C - _delay(d, C, 5), 5, 1)
    out["gtja025"] = (-_rank(d, _delta(d, C, 7) * (1 - _rank(d, _decay_linear(d, V / _ts_mean(d, V, 20).replace(0, np.nan), 9))))
                      * (1 + _rank(d, _ts_sum(d, RET, 244))))
    lo9, hi9 = _ts_min(d, L, 9), _ts_max(d, H, 9)
    rsv = (C - lo9) / (hi9 - lo9).replace(0, np.nan) * 100
    k3 = _sma(d, rsv, 3, 1)
    out["gtja028"] = 3 * k3 - 2 * _sma(d, k3, 3, 1)
    out["gtja029"] = (C - d6) / d6.replace(0, np.nan) * V
    m12 = _ts_mean(d, C, 12)
    out["gtja031"] = (C - m12) / m12.replace(0, np.nan) * 100
    out["gtja032"] = -_ts_sum(d, _rank(d, _ts_corr(d, _rank(d, H), _rank(d, V), 3)), 3)
    out["gtja034"] = m12 / C.replace(0, np.nan)
    out["gtja036"] = _rank(d, _ts_sum(d, _ts_corr(d, _rank(d, V), _rank(d, VWAP), 6), 2))
    out["gtja038"] = np.where(_ts_sum(d, H, 20) / 20 < H, -_delta(d, H, 2), 0.0)
    out["gtja038"] = pd.Series(out["gtja038"], index=C.index)
    out["gtja040"] = (_sum_if(d, V, C > PC, 26) / _sum_if(d, V, C <= PC, 26).replace(0, np.nan)) * 100
    out["gtja042"] = -_rank(d, _ts_std(d, H, 10)) * _ts_corr(d, H, V, 10)
    out["gtja046"] = (_ts_mean(d, C, 3) + _ts_mean(d, C, 6) + _ts_mean(d, C, 12) + _ts_mean(d, C, 24)) / (4 * C.replace(0, np.nan))
    hi6, lo6 = _ts_max(d, H, 6), _ts_min(d, L, 6)
    out["gtja047"] = _sma(d, (hi6 - C) / (hi6 - lo6).replace(0, np.nan) * 100, 9, 1)
    hd = _delta(d, H, 1)
    ld = -_delta(d, L, 1)
    plus_dm = np.maximum(hd, 0.0).where(hd > ld, 0.0)
    minus_dm = np.maximum(ld, 0.0).where(ld > hd, 0.0)
    tr = np.maximum(H - L, np.maximum((H - PC).abs(), (L - PC).abs()))
    out["gtja049"] = (_ts_sum(d, pd.Series(minus_dm, index=C.index), 12)
                      / (_ts_sum(d, pd.Series(minus_dm, index=C.index), 12) + _ts_sum(d, pd.Series(plus_dm, index=C.index), 12)).replace(0, np.nan))
    out["gtja052"] = (_ts_sum(d, np.maximum(0.0, H - _delay(d, HLC3, 1)), 26)
                      / _ts_sum(d, np.maximum(0.0, _delay(d, HLC3, 1) - L), 26).replace(0, np.nan) * 100)
    out["gtja053"] = _count(d, C > PC, 12) / 12 * 100
    out["gtja057"] = _sma(d, rsv, 3, 1)
    out["gtja058"] = _count(d, C > PC, 20) / 20 * 100
    out["gtja060"] = _ts_sum(d, clv * V, 20)
    out["gtja062"] = -_ts_corr(d, H, _rank(d, V), 5)
    out["gtja066"] = (C - m6) / m6.replace(0, np.nan) * 100
    out["gtja070"] = _ts_std(d, A, 6)
    m24 = _ts_mean(d, C, 24)
    out["gtja071"] = (C - m24) / m24.replace(0, np.nan) * 100
    mthlc = _ts_mean(d, HLC3, 12)
    out["gtja078"] = (HLC3 - mthlc) / (0.015 * _ts_mean(d, (C - mthlc).abs(), 12)).replace(0, np.nan)
    d20 = _delay(d, C, 20)
    out["gtja088"] = (C - d20) / d20.replace(0, np.nan) * 100
    out["gtja093"] = _sum_if(d, np.maximum(O - L, O - _delay(d, O, 1)), O >= _delay(d, O, 1), 20)
    out["gtja096"] = _sma(d, _sma(d, rsv, 3, 1), 3, 1)
    out["gtja097"] = _ts_std(d, V, 10)
    out["gtja100"] = _ts_std(d, V, 20)
    dv = _delta(d, V, 1)
    out["gtja102"] = (_sma(d, np.maximum(dv, 0.0), 6, 1) / _sma(d, dv.abs(), 6, 1).replace(0, np.nan)) * 100
    out["gtja106"] = C - d20
    hl_sma = _sma(d, H - L, 10, 2)
    out["gtja109"] = hl_sma / _sma(d, hl_sma, 10, 2).replace(0, np.nan)
    out["gtja111"] = (_sma(d, V * clv, 11, 2) - _sma(d, V * clv, 4, 2))
    out["gtja118"] = _ts_sum(d, H - O, 20) / _ts_sum(d, O - L, 20).replace(0, np.nan) * 100
    logc = np.log(C.replace(0, np.nan))
    t122 = _sma(d, _sma(d, _sma(d, logc, 13, 2), 13, 2), 13, 2)
    out["gtja122"] = (t122 - _delay(d, t122, 1)) / _delay(d, t122, 1).replace(0, np.nan)
    out["gtja126"] = HLC3
    out["gtja129"] = _ts_sum(d, (C - PC).abs().where(C < PC, 0.0), 12)
    out["gtja133"] = ((20 - _highday(d, H, 20)) / 20 * 100) - ((20 - _lowday(d, L, 20)) / 20 * 100)
    out["gtja150"] = HLC3 * V
    smac15 = _sma(d, C, 15, 2)
    out["gtja158"] = ((H - smac15) - (L - smac15)) / C.replace(0, np.nan)
    out["gtja167"] = _ts_sum(d, np.maximum(C - PC, 0.0), 12)
    out["gtja168"] = -(V / _ts_mean(d, V, 20).replace(0, np.nan))
    out["gtja171"] = (-(L - C) * O.pow(5)) / ((C - H) * C.pow(5)).replace(0, np.nan)
    trs = pd.Series(tr, index=C.index)
    out["gtja175"] = _ts_mean(d, np.maximum(np.maximum(H - L, (PC - H).abs()), (PC - L).abs()), 6)
    out["gtja187"] = _sum_if(d, np.maximum(H - O, O - _delay(d, O, 1)), O > _delay(d, O, 1), 20)
    out["gtja189"] = _ts_mean(d, (C - m6).abs(), 6)
    out["gtja191"] = _ts_corr(d, _ts_mean(d, V, 20), L, 5) + (H + L) / 2 - C
    _ = trs, eps
    return {k: pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# public API (mirrors cicc_ashare80)                                           #
# --------------------------------------------------------------------------- #


def compute_gtja191_factors(
    frame: pd.DataFrame,
    names: list[str] | None = None,
    *,
    wide: bool = False,
) -> pd.DataFrame:
    """Compute tranche-1 GTJA-191 factors (full history, per-symbol windows)."""
    data = _base(frame)
    values = _factor_values(data)
    selected = names or list(values)
    unknown = [n for n in selected if n not in values]
    if unknown:
        raise KeyError(f"unknown / not-yet-ported GTJA factors: {unknown}")
    if wide:
        return pd.DataFrame(
            {"trade_date": data["trade_date"].to_numpy(),
             "symbol": data["symbol"].to_numpy(),
             **{n: values[n].to_numpy(dtype=float) for n in selected}},
        )
    rows = []
    for n in selected:
        piece = data[["trade_date", "symbol"]].copy()
        piece["factor_name"] = n
        piece["factor_value"] = values[n].to_numpy(dtype=float)
        rows.append(piece)
    return pd.concat(rows, ignore_index=True)


def gtja191_names() -> tuple[str, ...]:
    sample = pd.DataFrame(
        {"symbol": ["X"] * 30, "trade_date": pd.date_range("2024-01-01", periods=30, freq="B"),
         "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 100.0}
    )
    return tuple(_factor_values(_base(sample)).keys())


__all__ = ["compute_gtja191_factors", "gtja191_names", "REQUIRED_COLUMNS"]
