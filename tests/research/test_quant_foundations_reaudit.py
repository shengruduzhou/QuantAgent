from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import typer

from quantagent.cli.fusion import _validate_benchmark_contract
from quantagent.factors.expression_safety import expression_leakage_reasons, validate_feature_expression
from quantagent.quant_math.black_scholes import black_scholes, implied_volatility
from quantagent.research.nonlinear_promotion import NonlinearPromotionConfig, evaluate_nonlinear_promotion


def test_formulaic_alpha_blocks_negative_qlib_ref_and_future_labels() -> None:
    assert "negative_ref_is_future" in expression_leakage_reasons("Ref($close, -1) / $close - 1")
    assert "future_or_label_token" in expression_leakage_reasons("forward_return_20d")
    with pytest.raises(ValueError, match="not PIT-safe"): validate_feature_expression("Rank(Ref($close, -2))")
    assert validate_feature_expression("Rank(Ref($close, 5) / $close)") == "Rank(Ref($close, 5) / $close)"


def test_probabilistic_sharpe_uses_pearson_kurtosis_convention() -> None:
    from statistics import NormalDist
    from quantagent.quant_math.performance import probabilistic_sharpe_ratio, sharpe_ratio
    rng=np.random.default_rng(7); values=pd.Series(rng.standard_t(df=6,size=900)*0.01+0.0006); observed=probabilistic_sharpe_ratio(values); sr=sharpe_ratio(values)/np.sqrt(252.0); skew=float(values.skew()); excess=float(values.kurt()); denom=np.sqrt(max(1.0-skew*sr+((excess+2.0)/4.0)*sr**2,1e-12)); expected=NormalDist().cdf(sr*np.sqrt(len(values)-1)/denom); assert observed==pytest.approx(expected,rel=1e-12,abs=1e-12)


def test_black_scholes_put_call_parity_and_implied_vol_recovery() -> None:
    kwargs=dict(spot=100.0,strike=105.0,maturity=0.75,rate=0.025,volatility=0.31,dividend_yield=0.01); call=black_scholes(option_type="call",**kwargs); put=black_scholes(option_type="put",**kwargs); lhs=call.price-put.price; rhs=kwargs["spot"]*np.exp(-kwargs["dividend_yield"]*kwargs["maturity"])-kwargs["strike"]*np.exp(-kwargs["rate"]*kwargs["maturity"]); assert lhs==pytest.approx(rhs,abs=1e-10); recovered=implied_volatility(call.price,spot=kwargs["spot"],strike=kwargs["strike"],maturity=kwargs["maturity"],rate=kwargs["rate"],dividend_yield=kwargs["dividend_yield"],option_type="call"); assert recovered==pytest.approx(kwargs["volatility"],abs=1e-7); assert call.gamma>0 and call.vega>0


def test_governed_fusion_cli_requires_explicit_benchmark(tmp_path) -> None:
    with pytest.raises(typer.BadParameter,match="explicit benchmark"): _validate_benchmark_contract(None,"")
    path=tmp_path/"benchmark.parquet"; path.write_bytes(b"placeholder")
    with pytest.raises(typer.BadParameter,match="benchmark-symbol"): _validate_benchmark_contract(path,"")
    _validate_benchmark_contract(path,"000300.SH")


def test_nonlinear_promotion_requires_pbo_dsr_and_spa() -> None:
    rng=np.random.default_rng(9); index=pd.date_range("2021-01-04",periods=320,freq="B"); baseline=pd.Series(rng.normal(0.0,0.006,len(index)),index=index); challenger=baseline+pd.Series(rng.normal(0.0018,0.0015,len(index)),index=index)
    report=SimpleNamespace(verdict="production_accepted",champion="gbm",pbo=0.10,dsr_probability=0.99,arms=[SimpleNamespace(name="linear_baseline",status="measured",daily_returns=baseline),SimpleNamespace(name="gbm",status="measured",daily_returns=challenger)])
    gate=evaluate_nonlinear_promotion(report,config=NonlinearPromotionConfig(spa_bootstrap=300)); assert gate.accepted; assert gate.final_verdict=="production_accepted"; assert gate.spa_pvalue<=0.05
    report.pbo=0.30; rejected=evaluate_nonlinear_promotion(report,config=NonlinearPromotionConfig(spa_bootstrap=300)); assert not rejected.accepted; assert any(reason.startswith("pbo=") for reason in rejected.rejection_reasons)
