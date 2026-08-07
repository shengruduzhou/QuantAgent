from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.quant_api.services.market_playbooks_v2 import MarketPlaybookService, PLAYBOOKS


class FakeMarket:
    def financial_health(self, symbol: str, limit: int = 10):
        return {"statements":{"income":[{"fiscal_year":2024,"period_end_ms":1,"report_date_ms":11,"operating_income":100.0,"parent_holder_net_profit":10.0},{"fiscal_year":2025,"period_end_ms":2,"report_date_ms":12,"operating_income":120.0,"parent_holder_net_profit":15.0}],"balance":[{"fiscal_year":2024,"period_end_ms":1,"report_date_ms":11,"assets_total":200.0,"total_debt":80.0},{"fiscal_year":2025,"period_end_ms":2,"report_date_ms":12,"assets_total":240.0,"total_debt":84.0}],"cashflow":[{"fiscal_year":2024,"period_end_ms":1,"report_date_ms":11,"act_cash_flow_net":12.0,"pay_fixed_assets_etc_cash":3.0},{"fiscal_year":2025,"period_end_ms":2,"report_date_ms":12,"act_cash_flow_net":18.0,"pay_fixed_assets_etc_cash":4.0}]},"provenance":{"income":"income","balance":"balance","cashflow":"cashflow"}}
    def index_catalog(self, tag: str): return {"items":[{"thscode":f"I{i:02d}.TI","name":f"行业{i:02d}"} for i in range(12)]}
    def index_overview(self, symbol: str, calendar_days: int = 120): return {"constituentCount":3,"snapshots":[{"changePercent":1.0},{"changePercent":-0.5},{"changePercent":0.2}]}


class SyntheticPlaybooks(MarketPlaybookService):
    def _index_bars(self, symbol: str, days: int) -> pd.DataFrame:
        idx=int(symbol[1:3]) if symbol.startswith("I") else 0; dates=pd.date_range("2024-01-02",periods=260,freq="B"); close=100.0*np.cumprod(np.full(len(dates),1.0+(idx+1)/100000.0)); return pd.DataFrame({"date":dates,"symbol":symbol,"open":close,"high":close,"low":close,"close":close,"volume":1.0})


def test_playbook_registry_is_exactly_16_and_never_calls_mapping_complete_parity() -> None:
    assert [item["id"] for item in PLAYBOOKS]==[f"{i:02d}" for i in range(1,17)]; assert len(PLAYBOOKS)==16; statuses={item["id"]:item["status"] for item in PLAYBOOKS}; assert statuses["06"]=="computed_local"; assert statuses["10"]=="computed"; assert statuses["16"]=="computed"


def test_financial_health_scorecard_uses_disclosure_date_and_four_dimensions() -> None:
    payload=MarketPlaybookService(FakeMarket()).financial_health_scorecard("600519.SH"); assert payload["pitKey"]=="reportDateMs"; assert payload["rows"][-1]["revenueGrowth"]==pytest.approx(.20); assert payload["rows"][-1]["netMargin"]==pytest.approx(.125); assert payload["rows"][-1]["cashConversion"]==pytest.approx(1.2); assert payload["rows"][-1]["debtToAssets"]==pytest.approx(.35)


def test_industry_strength_reports_current_breadth_without_historical_backfill() -> None:
    payload=SyntheticPlaybooks(FakeMarket()).industry_strength_matrix(); assert len(payload["rows"])==12; assert payload["rows"][0]["advancingShare"]==pytest.approx(2/3); assert any("不回填" in note for note in payload["notes"])


def test_time_series_momentum_exposes_official_volatility_and_sensitivity() -> None:
    payload=SyntheticPlaybooks(FakeMarket()).time_series_momentum("000300.SH","000300.SH",8.0); assert payload["metrics"]["latestVolatility60"] is not None; assert set(payload["windowSensitivity"])=={"60","120","180"}; assert payload["rows"][-1]["state"] in {"Active","Inactive"}; assert "volatility60" in payload["rows"][-1]


class FakeFundProvider:
    def __init__(self): self.calls=[]
    def get_capability(self,path: str,params):
        self.calls.append((path,dict(params)))
        if path.endswith('/profile/detail'): return {'item':[{'name':'示例ETF'}]}
        if path.endswith('/portfolio/holdings'): return {'item':[{'thscode':'600000.SH','nav_ratio':5.0,'report_date_ms':1}]}
        if path.endswith('/performance/nav'): return {'item':[{'date_ms':1,'unit_nav':1.0}]}
        if path.endswith('/performance/returns'): return {'item':[{'period':'1y','return_rate':8.0}]}
        if path.endswith('/holders/detail'): return {'item':[{'merge_scope':'merged','report_date_ms':1}]}
        if path.endswith('/market/snapshot'): return {'item':[{'thscode':'510300.SH','last_price':4.0}]}
        if path.endswith('/market/historical'): return {'item':[{'date_ms':1,'close_price':4.0}]}
        raise AssertionError(path)
class FakeFundMarket:
    def __init__(self): self.provider=FakeFundProvider()
    def _provider(self): return self.provider


def test_fund_research_aggregates_disclosed_and_market_contracts() -> None:
    from services.quant_api.services.fund_research import FundResearchService
    market=FakeFundMarket(); payload=FundResearchService(market).overview('510300.SH',fund_type='exchange'); assert payload['fundType']=='exchange'; assert payload['panels']['holdings']['item'][0]['thscode']=='600000.SH'; assert payload['pit']['market']=='etf_only_upstream'; assert payload['issues']==[]; snapshot_call=next(params for path,params in market.provider.calls if path.endswith('/market/snapshot')); assert snapshot_call=={'thscode':'510300.SH'}


def test_local_marketdb_playbook_reads_validated_parquet_and_labels_raw_basis(tmp_path,monkeypatch) -> None:
    dates=pd.date_range('2025-01-02',periods=25,freq='B',tz='Asia/Shanghai'); rows=[]
    for symbol,drift in [('600000.SH',.002),('000001.SZ',-.001)]:
        price=10.0
        for date in dates:
            price*=1.0+drift; rows.append({'thscode':symbol,'date_ms':int(date.timestamp()*1000),'open_price':price,'high_price':price,'low_price':price,'close_price':price,'volume':1000.0,'turnover':1_000_000.0})
    dump=tmp_path/'fuyao-daily-k.parquet'; pd.DataFrame(rows).to_parquet(dump,index=False); monkeypatch.setenv('QUANTAGENT_FUYAO_DUMP_ROOT',str(tmp_path)); payload=MarketPlaybookService(FakeMarket()).local_marketdb_research(); assert payload['priceBasis']=='raw_unadjusted'; assert payload['metrics']['sampleCount']==2; assert payload['rows']


def test_limit_up_ladder_normalizes_board_matrix() -> None:
    ladder=[{'date':'20260807','boards':{'two_board':[{'thscode':'A','board_num':2}], 'five_board':[{'thscode':'B','board_num':5}], 'seven_over':[]}}]; rows=MarketPlaybookService._ladder_rows(ladder); assert rows[0]['maxBoard']==5; assert rows[0]['counts']['two_board']==1; assert rows[0]['counts']['five_board']==1
