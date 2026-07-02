"""Portfolio policy search over OOS predictions — objective = after-cost CAGR.

Given full-universe OOS predictions (``symbol, trade_date, alpha_{h}d``) and the
market panel, this evaluates portfolio *construction* policies (which names to
hold, how to weight, how often to rebalance) and ranks them by **after-cost
absolute annualised return**, while always reporting max drawdown, Sharpe,
turnover and win-rate so a high-CAGR / high-turnover overfit is visible.

It does NOT re-train any model: the predictions are fixed; only the portfolio
policy varies. Tradability is honoured at signal time (ST / suspended /
limit-up excluded from buys) and execution is lagged one day; transaction cost
+ slippage are charged on turnover. This is a fast vectorised search; the
winning policy should be re-confirmed through the full strict simulator
(``scripts/baseline_protocol.py``).

Timeline convention (no look-ahead):
  * signal at date t uses alpha known at close t;
  * execution at close t+lag (lag=1);
  * the selected book earns realised close-to-close returns on days AFTER the
    execution day, until the next rebalance's execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


@dataclass(frozen=True)
class PolicyConfig:
    horizon: int                      # 1 / 5 / 20  -> score column alpha_{h}d
    top_k: int                        # names per side
    rebalance_days: int               # 1 daily / 5 weekly / 20
    side: str = "long_only"           # long_only | long_short
    transform: str = "raw"            # raw | rank | zscore | csrank
    neutralize: str = "none"          # none | industry | size | industry_size
    liquidity_filter: str = "none"    # none | ex_bottom_30pct
    exclude_st: bool = True
    exclude_suspended: bool = True
    exclude_limit_up: bool = True
    cost_bps_per_turnover: float = 13.0   # charged on sum|Δw| (~26 bps per full round-trip)
    lag: int = 1

    def label(self) -> str:
        return (f"h{self.horizon}_k{self.top_k}_rb{self.rebalance_days}_{self.side}"
                f"_{self.transform}_neu-{self.neutralize}_liq-{self.liquidity_filter}")


@dataclass(frozen=True)
class PolicyResult:
    config: PolicyConfig
    metrics: dict
    nav: pd.Series = field(default=None, repr=False)


def prepare_working_frame(preds: pd.DataFrame, panel: pd.DataFrame,
                          sector: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge predictions with realised returns + tradability + liquidity + industry.

    ``ret`` is the close-to-close return realised ON that day (close_t/close_{t-1}-1).
    """
    p = preds[["symbol", "trade_date", "alpha_1d", "alpha_5d", "alpha_20d"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    m = panel[["symbol", "trade_date", "close", "amount", "is_st", "is_suspended", "is_limit_up"]].copy()
    m["trade_date"] = pd.to_datetime(m["trade_date"], errors="coerce")
    m = m.sort_values(["symbol", "trade_date"])
    m["ret"] = m.groupby("symbol", sort=False)["close"].pct_change()
    work = p.merge(m.drop(columns=["close"]), on=["symbol", "trade_date"], how="left")
    if sector is not None and not sector.empty:
        s = sector.rename(columns={c: "industry" for c in sector.columns if c.lower() in ("industry", "sw1", "sector")})
        col = "industry" if "industry" in s.columns else s.columns[-1]
        work = work.merge(s[["symbol", col]].rename(columns={col: "industry"}).drop_duplicates("symbol"),
                          on="symbol", how="left")
    if "industry" not in work.columns:
        work["industry"] = "NA"
    work["industry"] = work["industry"].fillna("NA")
    for c in ("is_st", "is_suspended", "is_limit_up"):
        work[c] = work[c].fillna(False).astype(bool)
    return work


def _residualize_by_date(d: pd.DataFrame, ycol: str, xcol: str) -> pd.Series:
    """Per-day OLS residual of y on a single regressor x (+ intercept)."""
    def f(grp: pd.DataFrame) -> pd.Series:
        y = grp[ycol].to_numpy(dtype=float)
        x = grp[xcol].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        out = y.copy()
        if mask.sum() >= 3:
            xc = x[mask] - x[mask].mean()
            yc = y[mask] - y[mask].mean()
            denom = float(xc @ xc)
            b = float(xc @ yc) / denom if denom > 0 else 0.0
            res = y - b * x
            out = res - np.nanmean(res[mask])
        return pd.Series(out, index=grp.index)
    return d.groupby("trade_date", group_keys=False).apply(f)


def _final_score(d: pd.DataFrame, cfg: PolicyConfig) -> pd.DataFrame:
    g = d.groupby("trade_date")["score"]
    if cfg.transform in ("rank", "csrank"):
        d["fscore"] = g.rank(pct=True)
    elif cfg.transform == "zscore":
        std = g.transform("std").replace(0, np.nan)
        d["fscore"] = (d["score"] - g.transform("mean")) / std
    else:
        d["fscore"] = d["score"]
    if cfg.neutralize in ("industry", "industry_size"):
        d["fscore"] = d["fscore"] - d.groupby(["trade_date", "industry"])["fscore"].transform("mean")
    if cfg.neutralize in ("size", "industry_size"):
        d["_logamt"] = np.log(pd.to_numeric(d["amount"], errors="coerce").clip(lower=1.0))
        d["fscore"] = _residualize_by_date(d, "fscore", "_logamt")
    return d


def annualised_metrics(daily_net: pd.Series) -> dict:
    daily_net = daily_net.dropna()
    n = int(len(daily_net))
    if n < 2:
        return {"cagr": float("nan"), "max_drawdown": float("nan"), "sharpe": float("nan"),
                "win_rate_daily": float("nan"), "n_days": n, "total_return": float("nan")}
    nav = (1.0 + daily_net).cumprod()
    total = float(nav.iloc[-1] - 1.0)
    cagr = float(nav.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
    dd = float((nav / nav.cummax() - 1.0).min())
    sd = float(daily_net.std(ddof=0))
    sharpe = float(daily_net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else float("nan")
    win = float((daily_net > 0).mean())
    return {"cagr": cagr, "max_drawdown": dd, "sharpe": sharpe,
            "win_rate_daily": win, "n_days": n, "total_return": total}


def eligible_scored(work: pd.DataFrame, cfg: PolicyConfig) -> pd.DataFrame:
    """Apply tradability/liquidity filters + score transform/neutralisation.

    Depends only on (horizon, transform, neutralize, liquidity, filters), so the
    result can be cached and reused across top_k / rebalance / side variants.
    """
    score_col = f"alpha_{cfg.horizon}d"
    d = work[["trade_date", "symbol", score_col, "ret", "amount", "industry",
              "is_st", "is_suspended", "is_limit_up"]].rename(columns={score_col: "score"})
    d = d.dropna(subset=["score"]).copy()
    elig = pd.Series(True, index=d.index)
    if cfg.exclude_st:
        elig &= ~d["is_st"]
    if cfg.exclude_suspended:
        elig &= ~d["is_suspended"]
    if cfg.exclude_limit_up:
        elig &= ~d["is_limit_up"]
    if cfg.liquidity_filter == "ex_bottom_30pct":
        thr = d.groupby("trade_date")["amount"].transform(lambda s: s.quantile(0.30))
        elig &= pd.to_numeric(d["amount"], errors="coerce") >= thr
    d = d[elig].copy()
    return _final_score(d, cfg)


def backtest_policy(work: pd.DataFrame, cfg: PolicyConfig,
                    scored: pd.DataFrame | None = None) -> PolicyResult:
    d = eligible_scored(work, cfg) if scored is None else scored

    dates = sorted(d["trade_date"].unique())
    idx_of = {dt: i for i, dt in enumerate(dates)}
    signal_idxs = list(range(0, len(dates), cfg.rebalance_days))

    # Selection per signal date -> {symbol: weight}.
    by_date = {dt: g for dt, g in d.groupby("trade_date")}
    selections: list[tuple[int, dict[str, float]]] = []  # (exec_idx, weights)
    for si in signal_idxs:
        exec_idx = si + cfg.lag
        if exec_idx >= len(dates):
            continue
        grp = by_date[dates[si]].sort_values("fscore", ascending=False)
        longs = grp["symbol"].head(cfg.top_k).tolist()
        weights: dict[str, float] = {}
        if cfg.side == "long_short":
            shorts = grp["symbol"].tail(cfg.top_k).tolist()
            for s in longs:
                weights[s] = 0.5 / cfg.top_k
            for s in shorts:
                weights[s] = weights.get(s, 0.0) - 0.5 / cfg.top_k
        else:
            for s in longs:
                weights[s] = 1.0 / cfg.top_k
        selections.append((exec_idx, weights))

    if not selections:
        return PolicyResult(cfg, annualised_metrics(pd.Series(dtype=float)))

    # Vectorised: build a (earn_date, symbol, weight) book, merge realised returns
    # once, aggregate per day. Cost is charged on the first earning day of each
    # selection. Avoids a per-day Python .loc loop (orders of magnitude faster).
    parts: list[pd.DataFrame] = []
    turnovers: list[float] = []
    cost_by_date: dict[pd.Timestamp, float] = {}
    prev_w: dict[str, float] = {}
    for j, (exec_idx, weights) in enumerate(selections):
        next_exec = selections[j + 1][0] if j + 1 < len(selections) else len(dates)
        earn_idxs = [di for di in range(exec_idx + 1, next_exec + 1) if di < len(dates)]
        all_syms = set(weights) | set(prev_w)
        turnover = float(sum(abs(weights.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_syms))
        turnovers.append(turnover * 0.5)  # one-way
        prev_w = weights
        if not earn_idxs:
            continue
        cost_by_date[dates[earn_idxs[0]]] = cost_by_date.get(dates[earn_idxs[0]], 0.0) \
            + turnover * cfg.cost_bps_per_turnover / 1e4
        syms = list(weights)
        wv = np.array([weights[s] for s in syms], dtype=float)
        earn_dates = [dates[di] for di in earn_idxs]
        parts.append(pd.DataFrame({
            "trade_date": np.repeat(earn_dates, len(syms)),
            "symbol": np.tile(syms, len(earn_dates)),
            "w": np.tile(wv, len(earn_dates)),
        }))

    held = pd.concat(parts, ignore_index=True)
    held = held.merge(d[["trade_date", "symbol", "ret"]], on=["trade_date", "symbol"], how="left")
    held["contrib"] = held["w"] * held["ret"].fillna(0.0)
    gross = held.groupby("trade_date")["contrib"].sum()
    cost_s = pd.Series(cost_by_date, dtype=float)
    net = gross.subtract(cost_s, fill_value=0.0).sort_index()

    metrics = annualised_metrics(net)
    metrics["avg_one_way_turnover"] = float(np.mean(turnovers)) if turnovers else float("nan")
    metrics["annual_turnover"] = float(np.mean(turnovers) * (TRADING_DAYS / cfg.rebalance_days)) if turnovers else float("nan")
    metrics["n_rebalances"] = int(len(selections))
    return PolicyResult(cfg, metrics, nav=(1.0 + net).cumprod())


def universe_benchmark(work: pd.DataFrame, *, exclude_st=True, exclude_suspended=True) -> dict:
    """Equal-weight basket of the tradable universe (same filters, no selection)."""
    d = work[["trade_date", "symbol", "ret", "is_st", "is_suspended"]].dropna(subset=["ret"]).copy()
    if exclude_st:
        d = d[~d["is_st"]]
    if exclude_suspended:
        d = d[~d["is_suspended"]]
    daily = d.groupby("trade_date")["ret"].mean()
    return annualised_metrics(daily)


def search_policies(work: pd.DataFrame, configs: list[PolicyConfig],
                    progress: bool = False) -> pd.DataFrame:
    """Backtest every config; cache the scored universe across topK/rebalance/side."""
    cache: dict[tuple, pd.DataFrame] = {}
    rows = []
    for i, cfg in enumerate(configs):
        key = (cfg.horizon, cfg.transform, cfg.neutralize, cfg.liquidity_filter,
               cfg.exclude_st, cfg.exclude_suspended, cfg.exclude_limit_up)
        if key not in cache:
            cache[key] = eligible_scored(work, cfg)
        res = backtest_policy(work, cfg, scored=cache[key])
        rows.append({"policy": cfg.label(), **{k: getattr(cfg, k) for k in
                     ("horizon", "top_k", "rebalance_days", "side", "transform",
                      "neutralize", "liquidity_filter")}, **res.metrics})
        if progress and (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(configs)}] policies done", flush=True)
    lb = pd.DataFrame(rows)
    if not lb.empty:
        lb = lb.sort_values("cagr", ascending=False).reset_index(drop=True)
    return lb


__all__ = [
    "PolicyConfig", "PolicyResult", "prepare_working_frame", "backtest_policy",
    "universe_benchmark", "search_policies", "annualised_metrics",
]
