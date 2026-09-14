"""Future/unrequested cache rows must not select a historical factor set."""
import numpy as np
import pandas as pd
import pytest
from quantagent.data.dataset_builder.v7_training_dataset import _append_cached_factors
from quantagent.fusion.schemes import factor_ic_series


def test_future_cache_rows_do_not_change_past_feature_contract(tmp_path):
    base = pd.DataFrame({'trade_date': pd.to_datetime(['2020-01-02','2020-01-03']), 'symbol':['A','A']})
    small = base.assign(f=[1.,2.])
    path = tmp_path / 'f.csv'
    small.to_csv(path,index=False)
    before, _ = _append_cached_factors(base,str(path),min_finite_ratio=.3)
    future = pd.DataFrame({'trade_date':pd.bdate_range('2020-01-06',periods=8),'symbol':['A']*8,'f':[np.nan]*8})
    pd.concat([small,future]).to_csv(path,index=False)
    after, _ = _append_cached_factors(base,str(path),min_finite_ratio=.3)
    pd.testing.assert_frame_equal(before,after)
    assert 'f' in after


def test_sparse_rank_ic_ranks_only_jointly_finite_pairs():
    factor = pd.DataFrame({'trade_date':[pd.Timestamp('2020-01-02')]*6,
                           'symbol':list('ABCDEF'),'f':[1.,3.,np.nan,np.nan,2.,100.]})
    target = factor[['trade_date','symbol']].assign(forward_return=[1.,2.,3.,4.,5.,np.inf])
    result = factor_ic_series(factor,target,['f'])
    assert result.iloc[0,0] == pytest.approx(.5)
    valid = factor.f.notna() & np.isfinite(target.forward_return)
    assert result.iloc[0,0] == pytest.approx(factor.f[valid].corr(target.forward_return[valid],method='spearman'))
