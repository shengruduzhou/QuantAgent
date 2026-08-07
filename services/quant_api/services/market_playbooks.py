from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantagent.data.providers.base import ProviderRequest

SHANGHAI = ZoneInfo("Asia/Shanghai")

_TITLES = (
    "单股行情与趋势速览", "单股财务体检", "同花顺概念板块联动", "涨停池与连板天梯",
    "自选股当日异动监控", "本地全市场趋势研究", "市场热度与飙升雷达", "龙虎榜机构与游资观察",
    "行业强度作战矩阵", "现金流质量稽核台", "热榜—股价关系观察台", "涨停情绪市场脉冲屏",
    "价格成交量突破回测台", "时间序列动量回测台", "短期反转回测实验室", "龙虎榜资金流向拓扑台",
)
_STATUS = {"01":"native","03":"native","04":"native","06":"computed_local","07":"native","08":"native"}
_KIND = {"05":"monitor","06":"local-research","10":"evidence","11":"evidence","12":"evidence","13":"backtest","14":"backtest","15":"backtest","16":"flow"}
_NATIVE = {"01":"/stock-replay","03":"/market-intelligence?view=sector","04":"/market-intelligence?view=pulse","07":"/market-intelligence?view=pulse","08":"/market-intelligence?view=pulse"}
PLAYBOOKS = tuple({"id":f"{i:02d}","title":title,"kind":_KIND.get(f"{i:02d}","dashboard"),"status":_STATUS.get(f"{i:02d}","computed"),"route":_NATIVE.get(f"{i:02d}",f"/market-playbooks?id={i:02d}"),"evidence":[]} for i,title in enumerate(_TITLES,1))


class MarketPlaybookService:
    """Fuyao-inspired research workbenches with explicit PIT/T+1 provenance."""

    def __init__(self, market_service: Any) -> None:
        self.market = market_service

    def catalog(self) -> dict[str, Any]:
        return {"count":16,"items":[dict(x) for x in PLAYBOOKS],"source":"hithink_fuyao+quantagent","rule":"functional parity; QuantAgent UI; no synthetic financial data"}

    def run(self, playbook_id: str, *, symbol: str="600519.SH", benchmark: str="000300.SH", index_symbol: str="881101.TI", cost_bps: float=8.0) -> dict[str, Any]:
        pid=str(playbook_id).zfill(2)
        if pid=="01": return {"playbookId":pid,**self.market.stock_overview(symbol,calendar_days=420),"notes":["前复权日K、均线、回撤与成交额由原生单股工作台展示。"]}
        if pid=="02": return self.financial_health_scorecard(symbol)
        if pid=="03": return {"playbookId":pid,**self.market.index_overview(index_symbol,calendar_days=365),"notes":["成分股为当前成分，不回填历史。"]}
        if pid in {"04","08"}: return self._intelligence(pid)
        if pid=="05": return self.watchlist_anomalies(tuple(x.strip().upper() for x in symbol.split(",") if x.strip()))
        if pid=="06": return self.local_marketdb_research()
        if pid=="07": return self._heat(symbol)
        if pid=="09": return self.industry_strength_matrix()
        if pid=="10": return self.cashflow_quality(symbol)
        if pid=="11": return self.attention_price(symbol,benchmark)
        if pid=="12": return self.limitup_sentiment()
        if pid=="13": return self.price_volume_breakout(symbol,benchmark,cost_bps)
        if pid=="14": return self.time_series_momentum(index_symbol,benchmark,cost_bps)
        if pid=="15": return self.short_term_reversal(index_symbol,benchmark,cost_bps)
        if pid=="16": return self.dragon_tiger_topology(index_symbol)
        raise ValueError(f"unknown playbook id: {playbook_id}")

    def financial_health_scorecard(self,symbol: str) -> dict[str,Any]:
        h=self.market.financial_health(symbol,limit=10); groups={k:{r.get("period_end_ms"):r for r in h["statements"].get(k,[])} for k in ("income","balance","cashflow")}; rows=[]; prev=None
        for p in sorted(set().union(*(set(v) for v in groups.values()))):
            inc,bal,cf=(groups[k].get(p,{}) for k in ("income","balance","cashflow")); rev=_num(inc,"operating_income","revenue"); profit=_num(inc,"parent_holder_net_profit","net_profit"); assets=_num(bal,"assets_total","total_assets"); debt=_num(bal,"total_debt","liabilities_total","total_liabilities"); ocf=_num(cf,"act_cash_flow_net")
            row={"fiscalYear":inc.get("fiscal_year") or bal.get("fiscal_year"),"periodEndMs":p,"reportDateMs":max([x for x in (inc.get("report_date_ms"),bal.get("report_date_ms"),cf.get("report_date_ms")) if isinstance(x,(int,float))] or [0]) or None,"revenueGrowth":None if prev in (None,0) or rev is None else rev/prev-1,"netMargin":None if rev in (None,0) or profit is None else profit/rev,"cashConversion":None if profit in (None,0) or ocf is None else ocf/profit,"debtToAssets":None if assets in (None,0) or debt is None else debt/assets}
            rows.append(row); prev=rev if rev is not None else prev
        latest=rows[-1] if rows else {}; return {"playbookId":"02","symbol":symbol.upper(),"rows":rows,"metrics":{k:latest.get(k) for k in ("revenueGrowth","netMargin","cashConversion","debtToAssets")},"pitKey":"reportDateMs","periodKey":"periodEndMs","provenance":h["provenance"],"notes":["历史可见时间按 report_date_ms；字段缺失不插值。"]}

    def watchlist_anomalies(self,symbols: tuple[str,...]) -> dict[str,Any]:
        if not symbols or len(symbols)>50: raise ValueError("watchlist requires 1..50 thscodes")
        p=self.market._provider(); snaps=_records(p.snapshot(symbols)); ev=_items(p.get_capability("/api/a-share/special-data/anomaly-analysis-stock",{"thscodes":",".join(symbols)})); bys={str(r.get("symbol") or r.get("thscode")):r for r in snaps}; bye={s:[] for s in symbols}
        for r in ev:
            code=str(r.get("thscode") or r.get("symbol") or "")
            if code in bye: bye[code].append(r)
        return {"playbookId":"05","rows":[{"symbol":s,"snapshot":bys.get(s),"events":bye[s],"eventCount":len(bye[s])} for s in symbols],"provenance":{"snapshot":"/api/a-share/prices/snapshot","anomaly":"/api/a-share/special-data/anomaly-analysis-stock"}}

    def local_marketdb_research(self) -> dict[str,Any]:
        root=Path(os.getenv("QUANTAGENT_FUYAO_DUMP_ROOT","data/fuyao/full")).expanduser(); candidates=[p for p in root.rglob("*.parquet") if "daily" in p.name.lower() and "10d" not in p.name.lower() and "adjust" not in p.name.lower()] if root.exists() else []
        if not candidates: raise ValueError("no validated full-market daily-K parquet under QUANTAGENT_FUYAO_DUMP_ROOT")
        path=max(candidates,key=lambda p:p.stat().st_mtime); import pyarrow.dataset as ds; dataset=ds.dataset(str(path),format="parquet"); req={"thscode","date_ms","close_price","turnover"}; missing=req-set(dataset.schema.names)
        if missing: raise ValueError(f"local daily-K dump missing {sorted(missing)}")
        f=dataset.to_table(columns=list(req)).to_pandas().sort_values(["thscode","date_ms"]); f=f[f["date_ms"]>=f["date_ms"].max()-45*86_400_000]; g=f.groupby("thscode",sort=False); f["ret1d"]=g["close_price"].pct_change(fill_method=None); f["ret20d"]=g["close_price"].pct_change(20,fill_method=None); f["avgTurnover20d"]=g["turnover"].transform(lambda x:x.rolling(20,min_periods=5).mean()); latest=f[f["date_ms"]==f["date_ms"].max()].copy(); usable=latest.dropna(subset=["ret1d"]); trend=latest.dropna(subset=["ret20d"])
        return {"playbookId":"06","rows":[{"symbol":str(r.thscode),"return20d":_finite(r.ret20d),"turnover20d":_finite(r.avgTurnover20d)} for r in trend.nlargest(80,"ret20d").itertuples()],"metrics":{"latestDateMs":int(f["date_ms"].max()),"sampleCount":int(latest["thscode"].nunique()),"advancingShare":float((usable["ret1d"]>0).mean()) if len(usable) else None,"strongTrendCount20d10pct":int((trend["ret20d"]>=.10).sum())},"dataPath":str(path),"priceBasis":"raw_unadjusted","provenance":{"dailyK":"/dump/market-dumps/daily-k/download-url","increment":"/dump/market-dumps/daily-k-10d/download-url","adjustmentEvents":"/dump/market-dumps/adjustment-factors/download-url"},"notes":["未可靠物化 as-of 复权序列前，只声明 raw_unadjusted；筛选结果不是推荐。"]}

    def industry_strength_matrix(self,limit: int=12) -> dict[str,Any]:
        rows=[]
        for item in self.market.index_catalog("industry").get("items",[])[:max(6,min(20,limit))]:
            code=str(item.get("thscode") or item.get("symbol") or "")
            if not code: continue
            bars=self._index_bars(code,190)
            if len(bars)<25: continue
            c=bars["close"].astype(float); r5=float(c.iloc[-1]/c.iloc[-6]-1); r20=float(c.iloc[-1]/c.iloc[-21]-1); rows.append({"symbol":code,"name":item.get("name") or item.get("index_name"),"return5d":r5,"return20d":r20,"acceleration":r5-r20/4})
        rows.sort(key=lambda r:r["return20d"],reverse=True)
        for row in rows[:3]:
            snap=self.market.index_overview(row["symbol"],calendar_days=120); ch=[float(x["changePercent"]) for x in snap.get("snapshots",[]) if isinstance(x.get("changePercent"),(int,float))]; row["advancingShare"]=float(sum(v>0 for v in ch)/len(ch)) if ch else None; row["constituentCount"]=snap.get("constituentCount")
        return {"playbookId":"09","rows":rows,"provenance":{"catalog":"/api/a-share-index/catalog/ths-index-list","history":"/api/a-share-index/prices/historical","currentConstituents":"/api/a-share-index/constituents/ths-stock-list"},"notes":["当前成分上涨占比只做当前横截面，不回填历史成分。"]}

    def cashflow_quality(self,symbol: str) -> dict[str,Any]:
        h=self.market.financial_health(symbol,limit=10); cash={r.get("period_end_ms"):r for r in h["statements"].get("cashflow",[])}; rows=[]
        for inc in h["statements"].get("income",[]):
            cf=cash.get(inc.get("period_end_ms"),{}); profit=_num(inc,"parent_holder_net_profit","net_profit"); ocf=_num(cf,"act_cash_flow_net"); capex=_num(cf,"pay_fixed_assets_etc_cash"); rows.append({"fiscalYear":inc.get("fiscal_year"),"periodEndMs":inc.get("period_end_ms"),"reportDateMs":inc.get("report_date_ms") or cf.get("report_date_ms"),"netProfit":profit,"operatingCashFlow":ocf,"freeCashFlowProxy":None if ocf is None or capex is None else ocf-capex,"cashConversion":None if not profit or ocf is None else ocf/profit,"fieldCompleteness":sum(v is not None for v in (profit,ocf,capex))/3})
        return {"playbookId":"10","symbol":symbol.upper(),"rows":rows,"pitKey":"reportDateMs","periodKey":"periodEndMs","provenance":h["provenance"],"notes":["FCF proxy=经营现金流-资本开支代理；历史可用性按 report_date_ms。"]}

    def attention_price(self,symbol: str,benchmark: str) -> dict[str,Any]:
        p=self.market._provider(); end=datetime.now(SHANGHAI).date(); start=end-timedelta(days=365); rank=pd.DataFrame(_items(p.get_capability("/api/a-share/special-data/hot-stock-rank-trend",{"thscode":symbol.upper(),"start_date":start.isoformat(),"end_date":end.isoformat()}))); stock=_close_frame(p.historical_prices(ProviderRequest(start.isoformat(),end.isoformat(),(symbol.upper(),)),adjust="forward").frame,"stockClose"); idx=_close_frame(p.index_daily(ProviderRequest(start.isoformat(),end.isoformat(),(benchmark.upper(),))).frame,"benchmarkClose"); j=stock.merge(idx,on="date",how="inner")
        if not rank.empty:
            dc=next((c for c in ("date","trade_date","date_ms") if c in rank),None); rc=next((c for c in ("rank","hot_rank","rank_num") if c in rank),None)
            if dc and rc: rank=rank.assign(date=_to_date(rank[dc]),rank=pd.to_numeric(rank[rc],errors="coerce")); j=j.merge(rank[["date","rank"]].dropna(),on="date",how="inner")
        if "rank" not in j: j["rank"]=np.nan
        j=j.sort_values("date"); j["stockIndex"]=j["stockClose"]/j["stockClose"].iloc[0]*100 if len(j) else np.nan; j["benchmarkIndex"]=j["benchmarkClose"]/j["benchmarkClose"].iloc[0]*100 if len(j) else np.nan; j["relativeReturn"]=j["stockClose"].pct_change()-j["benchmarkClose"].pct_change(); j["rankChange"]=j["rank"].shift(1)-j["rank"]; v=j[["rankChange","relativeReturn"]].dropna(); sp=float(v.corr(method="spearman").iloc[0,1]) if len(v)>=3 else None
        return {"playbookId":"11","symbol":symbol.upper(),"benchmark":benchmark.upper(),"spearman":sp,"sample":len(v),"rows":_records(j.tail(260)),"provenance":{"rank":"/api/a-share/special-data/hot-stock-rank-trend","stock":"/api/a-share/prices/historical?adjust=forward","benchmark":"/api/a-share-index/prices/historical"},"notes":["同日对齐的相关关系不是因果或未来收益结论。"]}

    def limitup_sentiment(self) -> dict[str,Any]:
        p=self.market.market_intelligence().get("panels",{}); pool=_items(p.get("limitPool") or {}); ladder=_items(p.get("limitLadder") or {}); reasons=Counter(str(r.get("reason") or r.get("reason_type") or "未分类") for r in pool); max_board=max([int(r.get("continue_day_cnt") or r.get("board_num") or 0) for r in pool] or [0]); return {"playbookId":"12","limitUpCount":len(pool),"maxBoard":max_board,"reasonDistribution":reasons.most_common(20),"ladder":ladder,"provenance":{"pool":"/api/a-share/special-data/limit-up-pool","ladder":"/api/a-share/special-data/limit-up-ladder"}}

    def price_volume_breakout(self,symbol: str,benchmark: str,cost_bps: float) -> dict[str,Any]:
        b=self._stock_bars(symbol,620); prior=b["close"].rolling(20).max().shift(1); avg=b["volume"].rolling(20).mean().shift(1); return self._backtest("13",b,((b["close"]>prior)&(b["volume"]>1.5*avg)).astype(float),benchmark,cost_bps,{"signal":"20d breakout + volume confirmation"})

    def time_series_momentum(self,symbol: str,benchmark: str,cost_bps: float) -> dict[str,Any]:
        b=self._index_bars(symbol,760); signal=((b["close"].pct_change(120)>0)&(b["close"]>b["close"].rolling(120).mean())).astype(float); out=self._backtest("14",b,signal,benchmark,cost_bps,{"signal":"120d momentum > 0 and close > MA120","cash":"100% when inactive"}); return out

    def short_term_reversal(self,index_symbol: str,benchmark: str,cost_bps: float) -> dict[str,Any]:
        p=self.market._provider(); cons=p.index_constituents(index_symbol.upper()); symbols=[str(x) for x in cons.get("thscode",cons.get("symbol",pd.Series(dtype=str))).dropna()][:40]; series=[]
        for s in symbols:
            try: x=self._stock_bars(s,420).set_index("date")["close"].astype(float); x.name=s; series.append(x)
            except Exception: continue
        if len(series)<10: raise ValueError("insufficient current constituent histories for reversal lab")
        close=pd.concat(series,axis=1).sort_index(); bench=self._index_bars(benchmark,420).set_index("date")["close"].reindex(close.index).ffill(); score=-(close.pct_change(5).sub(bench.pct_change(5),axis=0)).where(close>close.rolling(120).mean()); nxt=close.pct_change().shift(-1); ic=score.corrwith(nxt,axis=1,method="spearman"); mask=score.rank(axis=1,pct=True)>=.8; gross=nxt.where(mask).mean(axis=1); w=mask.astype(float).div(mask.sum(axis=1).replace(0,np.nan),axis=0).fillna(0); net=gross.fillna(0)-w.diff().abs().sum(axis=1).fillna(0)*cost_bps/10000
        return {"playbookId":"15","metrics":_perf(net),"rankIcMean":_finite(ic.mean()),"rankIcIr":_finite(ic.mean()/ic.std(ddof=1) if ic.std(ddof=1) else np.nan),"rows":_records(pd.DataFrame({"date":net.index,"netReturn":net,"rankIc":ic.reindex(net.index)}).tail(320)),"assumptions":{"execution":"signal at T close; evaluate T+1 return","costBps":cost_bps,"constituentCaveat":"current constituents only"}}

    def dragon_tiger_topology(self,index_symbol: str) -> dict[str,Any]:
        p=self.market._provider(); cons=p.index_constituents(index_symbol.upper()); allowed=set(str(x) for x in cons.get("thscode",cons.get("symbol",pd.Series(dtype=str))).dropna()); dates=[pd.Timestamp(x).strftime("%Y-%m-%d") for x in self._index_bars(index_symbol,60)["date"].tail(8)]; agg={}
        for d in dates:
            data=p.get_capability("/api/a-share/special-data/dragon-tiger-list",{"board_type":"all","date":d})
            for r in data.get("stock_items",[]) if isinstance(data,Mapping) else []:
                code=str(r.get("thscode") or "")
                if code not in allowed: continue
                a=agg.setdefault(code,{"net":0.,"org":0.,"hot":0.}); a["net"]+=float(r.get("net_value") or 0); a["org"]+=float(r.get("org_net_value") or 0); a["hot"]+=float(r.get("hot_money_net_value") or 0)
        nodes=[{"id":index_symbol.upper(),"kind":"concept","value":sum(abs(v["net"]) for v in agg.values())},{"id":"机构","kind":"capital","value":sum(abs(v["org"]) for v in agg.values())},{"id":"游资","kind":"capital","value":sum(abs(v["hot"]) for v in agg.values())}]; links=[]
        for code,v in sorted(agg.items(),key=lambda kv:abs(kv[1]["net"]),reverse=True)[:30]: nodes.append({"id":code,"kind":"stock","value":v["net"]}); links.extend([{"source":index_symbol.upper(),"target":code,"value":v["net"]},{"source":"机构","target":code,"value":v["org"]},{"source":"游资","target":code,"value":v["hot"]}])
        return {"playbookId":"16","nodes":nodes,"links":links,"dates":dates,"notes":["拓扑仅使用所选指数当前成分，不回填历史成分。"]}

    def _intelligence(self,pid: str) -> dict[str,Any]:
        x=self.market.market_intelligence(); p=x.get("panels",{}); return ({"playbookId":pid,"limitPool":p.get("limitPool"),"limitLadder":p.get("limitLadder"),"issues":x.get("issues",[])} if pid=="04" else {"playbookId":pid,"dragonAll":p.get("dragonAll"),"dragonOrg":p.get("dragonOrg"),"dragonHotMoney":p.get("dragonHotMoney"),"issues":x.get("issues",[])})
    def _heat(self,symbol: str) -> dict[str,Any]:
        x=self.market.market_intelligence(); p=x.get("panels",{}); provider=self.market._provider(); end=datetime.now(SHANGHAI).date(); start=end-timedelta(days=365); trend=provider.get_capability("/api/a-share/special-data/hot-stock-rank-trend",{"thscode":symbol.upper(),"start_date":start.isoformat(),"end_date":end.isoformat()}); return {"playbookId":"07","hotDay":p.get("hotDay"),"hotHour":p.get("hotHour"),"skyrocketDay":p.get("skyrocketDay"),"skyrocketHour":p.get("skyrocketHour"),"rankTrend":trend,"notes":["热度是观察变量，不作为因果或未来收益声明。"]}
    def _stock_bars(self,symbol: str,days: int) -> pd.DataFrame:
        p=self.market._provider(); end=datetime.now(SHANGHAI).date(); start=end-timedelta(days=days); return _bars(p.historical_prices(ProviderRequest(start.isoformat(),end.isoformat(),(symbol.upper(),)),adjust="forward").frame)
    def _index_bars(self,symbol: str,days: int) -> pd.DataFrame:
        p=self.market._provider(); end=datetime.now(SHANGHAI).date(); start=end-timedelta(days=days); return _bars(p.index_daily(ProviderRequest(start.isoformat(),end.isoformat(),(symbol.upper(),))).frame)
    def _backtest(self,pid: str,b: pd.DataFrame,signal: pd.Series,benchmark: str,cost_bps: float,rules: Mapping[str,Any]) -> dict[str,Any]:
        ret=b["open"].shift(-1)/b["open"]-1; pos=signal.astype(float).clip(0,1).shift(1).fillna(0); turn=pos.diff().abs().fillna(pos.abs()); net=(pos*ret).fillna(0)-turn*cost_bps/10000; rows=pd.DataFrame({"date":b["date"],"close":b["close"],"volume":b["volume"],"position":pos,"netReturn":net,"nav":(1+net).cumprod()}); return {"playbookId":pid,"metrics":_perf(net),"rows":_records(rows.tail(420)),"assumptions":{**dict(rules),"execution":"signal at T close -> position at T+1 open -> open-to-open marking","costBps":cost_bps},"benchmark":benchmark.upper()}


def _num(row: Mapping[str,Any],*keys: str) -> float|None:
    for k in keys:
        v=row.get(k)
        if isinstance(v,(int,float)) and np.isfinite(v): return float(v)
    return None

def _items(data: Mapping[str,Any]) -> list[dict[str,Any]]:
    v=data.get("item",[]) if isinstance(data,Mapping) else []; return [dict(x) for x in v if isinstance(x,Mapping)] if isinstance(v,list) else []
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s,unit="ms",errors="coerce").dt.normalize() if pd.api.types.is_numeric_dtype(s) else pd.to_datetime(s,errors="coerce").dt.normalize()
def _close_frame(f: pd.DataFrame,name: str) -> pd.DataFrame:
    if f.empty:return pd.DataFrame(columns=["date",name])
    dc="trade_date" if "trade_date" in f else "date_ms"; return pd.DataFrame({"date":_to_date(f[dc]),name:pd.to_numeric(f["close"],errors="coerce")}).dropna()
def _bars(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty:return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])
    w=f.copy(); w["date"]=_to_date(w["trade_date"] if "trade_date" in w else w["date_ms"]); w["symbol"]=w.get("symbol","")
    for c in ("open","high","low","close","volume"): w[c]=pd.to_numeric(w[c],errors="coerce")
    return w[["date","symbol","open","high","low","close","volume"]].dropna(subset=["date","open","close"]).sort_values("date").reset_index(drop=True)
def _perf(r: pd.Series) -> dict[str,float|None]:
    x=r.dropna().astype(float)
    if x.empty:return {"totalReturn":None,"annualReturn":None,"sharpe":None,"maxDrawdown":None,"winRate":None}
    nav=(1+x).cumprod(); vol=x.std(ddof=1); dd=nav/nav.cummax()-1; return {"totalReturn":float(nav.iloc[-1]-1),"annualReturn":float(nav.iloc[-1]**(252/max(1,len(x)))-1),"sharpe":float(np.sqrt(252)*x.mean()/vol) if vol>1e-15 else None,"maxDrawdown":float(dd.min()),"winRate":float((x>0).mean())}
def _finite(v: Any) -> float|None:
    try:x=float(v)
    except (TypeError,ValueError):return None
    return x if np.isfinite(x) else None
def _records(f: Any) -> list[dict[str,Any]]:
    if hasattr(f,"frame"):f=f.frame
    if not isinstance(f,pd.DataFrame) or f.empty:return []
    w=f.copy()
    for c in w:
        if pd.api.types.is_datetime64_any_dtype(w[c]):w[c]=w[c].dt.strftime("%Y-%m-%d")
    return w.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict(orient="records")

__all__=["MarketPlaybookService","PLAYBOOKS"]
