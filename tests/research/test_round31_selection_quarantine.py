"""Selection/evaluation paths that read quarantined windows when handed a panel that extends into them."""
import logging
import numpy as np, pandas as pd
from quantagent.fusion.search import FusionSearchConfig, run_fusion_search
from quantagent.factors.factor_synthesis import RDAgentFactorLoopConfig, synthesize_factors_rd_agent

BURN_START = pd.Timestamp("2025-09-01")
logging.disable(logging.WARNING)

def _panel(T=520, N=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-03-01", periods=T)          # ends 2026-02 -> inside burned holdout
    close = 10 * np.exp(np.cumsum(0.02 * rng.standard_normal((T, N)), axis=0))
    df = pd.DataFrame({"trade_date": np.repeat(dates, N), "symbol": np.tile([f"S{i:03d}" for i in range(N)], T),
                       "open": close.ravel(), "high": close.ravel() * 1.01, "low": close.ravel() * 0.99,
                       "close": close.ravel(), "volume": 1e6, "amount": 1e7,
                       "f1": rng.standard_normal(T * N), "f2": rng.standard_normal(T * N),
                       "forward_return_5d": 0.04 * rng.standard_normal(T * N)})
    return df

def test_fusion_search_tests_inside_burned_holdout():
    df = _panel()
    fwd = df[["trade_date", "symbol", "forward_return_5d"]].rename(columns={"forward_return_5d": "forward_return"})
    res = run_fusion_search(factor_panel=df[["trade_date", "symbol", "f1", "f2"]], forward_panel=fwd,
                            config=FusionSearchConfig(factor_names=("f1", "f2"), include_genetic=False, random_controls=1, single_factor_baselines=1))
    windows = res.to_dict()["foldWindows"] if hasattr(res, "to_dict") else res.fold_windows
    starts = [pd.Timestamp(w["testStart"]) for w in windows]
    inside = [str(s.date()) for s in starts if s >= BURN_START]
    print("fusion test windows starting in burned holdout:", inside)
    assert not inside, f"fusion search evaluates inside quarantine: {inside}"

def test_rd_agent_validation_inside_burned_holdout(monkeypatch):
    import quantagent.factors.factor_synthesis as fs
    seen = {}
    orig = fs._chronological_split
    def spy(panel, labels, trade_date, cfg):
        out = orig(panel, labels, trade_date, cfg)
        seen["val_max"] = pd.to_datetime(out[5]).max(); seen["val_min"] = pd.to_datetime(out[5]).min()
        return out
    monkeypatch.setattr(fs, "_chronological_split", spy)
    res = fs.synthesize_factors_rd_agent(_panel(), config=RDAgentFactorLoopConfig(rounds=1, fitness_sample_dates=0, fitness_sample_symbols=0))
    print("rd-agent validation segment", seen["val_min"].date(), "->", seen["val_max"].date(), "accepted", len(res.definitions))
    assert seen["val_max"] < BURN_START, f"rd-agent acceptance IC measured on quarantined dates up to {seen['val_max'].date()}"
