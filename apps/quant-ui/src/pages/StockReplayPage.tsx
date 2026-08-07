import { useEffect, useMemo, useState, type FormEvent } from "react";
import { ChartLineUp, DownloadSimple, MagnifyingGlass, ShieldWarning } from "@phosphor-icons/react";
import { useSearchParams } from "react-router-dom";
import type { BacktestSummary, KlineBar, Page, SelectionRun, StockReplay, Trade } from "../api/types";
import { downloadJson } from "../api/client";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { TradeTable } from "../components/TradeTable";
import { useApi } from "../hooks/useApi";
import { formatCompact, formatNumber, formatPercent } from "../utils/format";
import { MarketCandlestickChart } from "../vnext/market/MarketCandlestickChart";
import { buildMarketAnalytics, describeTrend } from "../vnext/market/marketAnalytics";

interface MarketTicker { symbol: string; ticker?: string | null; name?: string | null; exchange?: string | null; assetType?: string | null; currency?: string | null; }
interface MarketSnapshot { symbol: string; ticker?: string | null; lastPrice?: number | null; priceChange?: number | null; changePercent?: number | null; open?: number | null; high?: number | null; low?: number | null; prevClose?: number | null; volume?: number | null; amount?: number | null; asOf?: string | null; }
interface MarketValuation { peTtm?: number | null; peMrq?: number | null; pbMrq?: number | null; psTtm?: number | null; pcfTtm?: number | null; }
interface MarketOverview { symbol: string; ticker?: string | null; name?: string | null; exchange?: string | null; currency: string; source: string; adjustment: "forward" | "none" | "backward"; interval: string; asOf?: string | null; snapshot: MarketSnapshot; valuation: MarketValuation; bars: KlineBar[]; provenance: Record<string, string | number | null>; }
interface DoTPair { id: string; symbol: string; buyTime?: string | null; sellTime?: string | null; buyPrice?: number | null; sellPrice?: number | null; quantity?: number | null; netPnl?: number | null; success?: boolean | null; state?: string | null; }
interface DoTAnalysis { pairs: DoTPair[]; }
interface DecisionChain { gates: Array<{ order: number; name: string; passed: boolean; reason?: string | null; }>; finalDecision?: string | null; issues?: Array<{ message: string }>; }

function pctFromSnapshot(value: number | null | undefined): number | null { return typeof value === "number" && Number.isFinite(value) ? value / 100 : null; }
function toneClass(value: number | null | undefined): string { if (value === null || value === undefined || !Number.isFinite(value) || value === 0) return "tone-neutral"; return value > 0 ? "tone-positive" : "tone-negative"; }
function asDate(value: string | null | undefined): string { return value ? value.slice(0, 10) : "暂无"; }
function MarketMetric({ label, value, detail, tone }: { label: string; value: string; detail?: string; tone?: string }): JSX.Element { return <div className="market-metric-card"><span>{label}</span><strong className={tone ?? ""}>{value}</strong>{detail ? <small>{detail}</small> : null}</div>; }
function DataFact({ label, value }: { label: string; value: string }): JSX.Element { return <div className="market-data-fact"><span>{label}</span><strong>{value}</strong></div>; }

export function StockReplayPage(): JSX.Element {
  const [searchParams] = useSearchParams();
  const requestedSymbol = searchParams.get("symbol")?.trim().toUpperCase() ?? "";
  const backtests = useApi<BacktestSummary[]>(["market-workbench-backtests"], "/backtests");
  const selectionRuns = useApi<SelectionRun[]>(["market-workbench-selection-runs"], "/selection/runs");
  const tradableRuns = (backtests.data?.data ?? []).filter((item) => item.capabilities?.trades);
  const [backtestId, setBacktestId] = useState("");
  const [symbol, setSymbol] = useState(requestedSymbol);
  const [searchInput, setSearchInput] = useState(requestedSymbol);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(null);
  const [tradeFilter, setTradeFilter] = useState("ALL");

  useEffect(() => { if (!backtestId && tradableRuns[0]) setBacktestId(tradableRuns[0].id); }, [backtestId, tradableRuns]);

  const tradePage = useApi<Page<Trade>>(["market-workbench-trades", backtestId], backtestId ? `/backtests/${backtestId}/trades` : null, { pageSize: 1000 });
  const availableSymbols = useMemo(() => [...new Set((tradePage.data?.data.items ?? []).map((trade) => trade.symbol))], [tradePage.data?.data.items]);

  useEffect(() => {
    if (!symbol && availableSymbols[0]) { setSymbol(availableSymbols[0]); setSearchInput(availableSymbols[0]); }
  }, [availableSymbols, symbol]);

  const marketSearch = useApi<MarketTicker[]>(["market-search", searchQuery], searchQuery ? "/market/search" : null, { q: searchQuery, limit: 12 }, { staleTime: 60_000 });
  const market = useApi<MarketOverview>(["market-overview", symbol], symbol ? `/market/stocks/${encodeURIComponent(symbol)}/overview` : null, { calendarDays: 420 }, { staleTime: 30_000 });
  const hasReplaySymbol = Boolean(backtestId && symbol && availableSymbols.includes(symbol));
  const replay = useApi<StockReplay>(["market-replay", backtestId, symbol], hasReplaySymbol ? `/backtests/${backtestId}/stocks/${symbol}` : null);
  const doT = useApi<DoTAnalysis>(["market-dot", backtestId, symbol], hasReplaySymbol ? `/backtests/${backtestId}/stocks/${symbol}/t-analysis` : null);
  const latestSelection = selectionRuns.data?.data[0];
  const chain = useApi<DecisionChain>(["market-decision-chain", latestSelection?.id, symbol], latestSelection && symbol ? `/selection/runs/${latestSelection.id}/stocks/${symbol}/decision-chain` : null);

  const liveData = market.data?.status === "ready" || market.data?.status === "partial" ? market.data.data : null;
  const replayData = replay.data?.data;
  const bars = useMemo(() => liveData?.bars?.length ? liveData.bars : replayData?.bars ?? [], [liveData?.bars, replayData?.bars]);
  const analytics = useMemo(() => buildMarketAnalytics(bars), [bars]);
  const tTrades = useMemo<Trade[]>(() => {
    const output: Trade[] = [];
    for (const pair of doT.data?.data.pairs ?? []) {
      if (pair.buyTime && pair.buyPrice !== null && pair.buyPrice !== undefined) output.push({ id: `${pair.id}-buy`, datetime: pair.buyTime, symbol: pair.symbol, action: "T_BUY", price: pair.buyPrice, quantity: pair.quantity ?? 0, success: pair.success, signalSource: "T+1 intraday overlay", tPairId: pair.id });
      if (pair.sellTime && pair.sellPrice !== null && pair.sellPrice !== undefined) output.push({ id: `${pair.id}-sell`, datetime: pair.sellTime, symbol: pair.symbol, action: "T_SELL", price: pair.sellPrice, quantity: pair.quantity ?? 0, pnl: pair.netPnl, success: pair.success, signalSource: "T+1 intraday overlay", riskReason: pair.state, tPairId: pair.id });
    }
    return output;
  }, [doT.data?.data.pairs]);
  const replayTrades = replayData?.trades ?? [];
  const allMarkers = useMemo(() => [...replayTrades, ...tTrades], [replayTrades, tTrades]);
  const selectedTrade = allMarkers.find((trade) => trade.id === selectedTradeId) ?? allMarkers[0];
  const filteredTrades = replayTrades.filter((trade) => tradeFilter === "ALL" || (tradeFilter === "BUY" && trade.action.includes("BUY")) || (tradeFilter === "SELL" && trade.action.includes("SELL")) || (tradeFilter === "RISK" && (Boolean(trade.riskReason) || trade.action.includes("RISK") || trade.action.includes("STOP"))));
  useEffect(() => { if (!selectedTradeId && selectedTrade) setSelectedTradeId(selectedTrade.id); }, [selectedTrade, selectedTradeId]);

  const submitSearch = (event: FormEvent): void => { event.preventDefault(); const query = searchInput.trim(); if (query) setSearchQuery(query); };
  const chooseTicker = (ticker: MarketTicker): void => { setSymbol(ticker.symbol); setSearchInput(`${ticker.name ?? ""} ${ticker.symbol}`.trim()); setSearchQuery(""); setSelectedTradeId(null); };
  const chooseReplaySymbol = (nextSymbol: string): void => { setSymbol(nextSymbol); setSearchInput(nextSymbol); setSearchQuery(""); setSelectedTradeId(null); };

  const snapshot = liveData?.snapshot;
  const latestPrice = snapshot?.lastPrice ?? analytics.latest?.close ?? null;
  const dailyReturn = pctFromSnapshot(snapshot?.changePercent) ?? analytics.dailyReturn;
  const sourceLabel = liveData ? "同花顺 Fuyao 官方 API" : replayData ? "Persisted backtest artifact" : "暂无数据";
  const adjustmentLabel = liveData?.adjustment === "forward" ? "前复权" : replayData ? "artifact 未声明" : "—";
  const displayName = liveData?.name ?? replayData?.name ?? symbol;
  const asOf = liveData?.asOf ?? analytics.latest?.datetime ?? null;
  const valuation = liveData?.valuation;
  const exportPayload = { symbol, market: liveData, replay: replayData, decisionChain: chain.data?.data ?? null };

  return (
    <div className="page institutional-workbench stock-replay-page market-visualization-page">
      <header className="market-hero">
        <div className="market-hero-main">
          <div className="market-eyebrow"><ChartLineUp size={16} /> QuantAgent · 行情工作台</div>
          <div className="market-title-row">
            <div><h1>{displayName || "A 股行情研究"}</h1><p>{symbol || "搜索代码或名称进入单票行情、估值与 QuantAgent 决策证据。"}</p></div>
            {symbol ? <div className="market-quote-block"><strong>{formatNumber(latestPrice)}</strong><span className={toneClass(dailyReturn)}>{dailyReturn !== null && dailyReturn !== undefined && dailyReturn > 0 ? "+" : ""}{formatPercent(dailyReturn)}</span><small>{snapshot?.priceChange !== null && snapshot?.priceChange !== undefined ? `Δ ${formatNumber(snapshot.priceChange)}` : asDate(asOf)}</small></div> : null}
          </div>
          <div className="market-provenance-row"><StatusBadge status={liveData ? "ready" : replayData ? "partial" : "unavailable"} label={sourceLabel} /><span>截至 {asDate(asOf)}</span><span>日线 · {adjustmentLabel}</span><span>币种 {liveData?.currency ?? "CNY"}</span></div>
        </div>
        <div className="market-toolbar-card">
          <form className="market-search-form" onSubmit={submitSearch}>
            <label htmlFor="market-symbol-search">股票搜索</label>
            <div><input id="market-symbol-search" value={searchInput} onChange={(event) => setSearchInput(event.target.value)} placeholder="600519.SH / 贵州茅台" autoComplete="off" /><button type="submit" className="primary-button"><MagnifyingGlass size={16} /> 查询</button></div>
          </form>
          {tradableRuns.length ? <div className="market-replay-controls">
            <label><span>回测上下文</span><select value={backtestId} onChange={(event) => setBacktestId(event.target.value)}>{tradableRuns.map((run) => <option key={run.id} value={run.id}>{run.name ?? run.id}</option>)}</select></label>
            <label><span>关联标的</span><select value={availableSymbols.includes(symbol) ? symbol : ""} onChange={(event) => chooseReplaySymbol(event.target.value)}><option value="">选择回测标的</option>{availableSymbols.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          </div> : null}
          <button type="button" className="secondary-button market-export" disabled={!symbol} onClick={() => downloadJson(`market-workbench-${symbol || "empty"}.json`, exportPayload)}><DownloadSimple size={16} /> 导出当前证据</button>
        </div>
      </header>

      {searchQuery ? <section className="market-search-results" aria-live="polite">{marketSearch.isLoading ? <span>正在检索…</span> : null}{(marketSearch.data?.data ?? []).map((ticker) => <button key={ticker.symbol} type="button" onClick={() => chooseTicker(ticker)}><strong>{ticker.name ?? ticker.ticker ?? ticker.symbol}</strong><span>{ticker.symbol}</span></button>)}{marketSearch.data?.status === "empty" ? <span>没有匹配的 A 股标的。</span> : null}{marketSearch.data?.status === "unavailable" ? <span>{marketSearch.data.issues[0]?.message ?? "Fuyao 当前不可用。"}</span> : null}</section> : null}
      {symbol && market.data?.status === "unavailable" ? <div className="market-provider-warning"><ShieldWarning size={17} /><span>{market.data.issues[0]?.message ?? "Fuyao 行情不可用。"} {replayData ? "已明确回退到 persisted replay 数据。" : "不会用模拟行情填充。"}</span></div> : null}

      {symbol ? <>
        <section className="market-metric-grid" aria-label="区间行情指标">
          <MarketMetric label="20 日收益" value={formatPercent(analytics.return20)} tone={toneClass(analytics.return20)} detail={liveData ? "前复权收盘价" : "artifact 历史收盘价"} />
          <MarketMetric label="60 日收益" value={formatPercent(analytics.return60)} tone={toneClass(analytics.return60)} detail={liveData ? "前复权收盘价" : "artifact 历史收盘价"} />
          <MarketMetric label="120 日收益" value={formatPercent(analytics.return120)} tone={toneClass(analytics.return120)} detail={liveData ? "前复权收盘价" : "artifact 历史收盘价"} />
          <MarketMetric label="60 日最大回撤" value={formatPercent(analytics.maxDrawdown60)} tone={toneClass(analytics.maxDrawdown60)} detail="仅历史窗口" />
        </section>
        <section className="market-primary-grid">
          <Panel title={`${displayName || symbol} · ${symbol}`} eyebrow="K 线 / 成交量 / MA / QuantAgent 交易事件" className="market-chart-panel kline-panel" actions={<div className="market-panel-badges"><span>MA20 / MA60 / MA120</span><span>{adjustmentLabel}</span></div>}>
            {bars.length ? <MarketCandlestickChart bars={bars} trades={allMarkers} symbol={symbol} selectedTradeId={selectedTradeId} onTradeSelect={setSelectedTradeId} /> : <StateView state={market.isLoading || replay.isLoading ? "loading" : "empty"} detail="没有可展示的历史日线；不会以 mock 数据补齐。" />}
          </Panel>
          <aside className="market-research-sidebar">
            <section className="market-side-section"><header><span>行情快照</span><small>Latest snapshot</small></header><div className="market-fact-grid"><DataFact label="开盘" value={formatNumber(snapshot?.open ?? analytics.latest?.open)} /><DataFact label="最高" value={formatNumber(snapshot?.high ?? analytics.latest?.high)} /><DataFact label="最低" value={formatNumber(snapshot?.low ?? analytics.latest?.low)} /><DataFact label="前收" value={formatNumber(snapshot?.prevClose ?? analytics.previousClose)} /><DataFact label="成交量" value={formatCompact(snapshot?.volume ?? analytics.latest?.volume)} /><DataFact label="成交额" value={formatCompact(snapshot?.amount ?? analytics.latest?.amount)} /></div></section>
            <section className="market-side-section"><header><span>趋势结构</span><small>Derived from history</small></header><p className="market-research-copy">{describeTrend(analytics)}</p><div className="market-fact-grid"><DataFact label="MA20" value={formatNumber(analytics.ma20)} /><DataFact label="MA60" value={formatNumber(analytics.ma60)} /><DataFact label="MA120" value={formatNumber(analytics.ma120)} /><DataFact label="20日年化波动" value={formatPercent(analytics.annualizedVolatility20)} /><DataFact label="250日高" value={formatNumber(analytics.high250)} /><DataFact label="250日低" value={formatNumber(analytics.low250)} /><DataFact label="20日均成交额" value={formatCompact(analytics.averageAmount20)} /></div></section>
            <section className="market-side-section"><header><span>估值快照</span><small>Fuyao valuation</small></header><div className="market-fact-grid"><DataFact label="PE TTM" value={formatNumber(valuation?.peTtm)} /><DataFact label="PE MRQ" value={formatNumber(valuation?.peMrq)} /><DataFact label="PB MRQ" value={formatNumber(valuation?.pbMrq)} /><DataFact label="PS TTM" value={formatNumber(valuation?.psTtm)} /><DataFact label="PCF TTM" value={formatNumber(valuation?.pcfTtm)} /></div>{!liveData ? <p className="market-unavailable-copy">当前不是 Fuyao live view，因此不伪造估值。</p> : null}</section>
          </aside>
        </section>
        <section className="market-evidence-grid">
          <Panel title="QuantAgent 交易证据" eyebrow={hasReplaySymbol ? `${replayTrades.length} 条标准成交 · ${tTrades.length} 个 T+1 leg` : "当前标的未绑定 persisted backtest trade"} className="market-trades-panel" actions={hasReplaySymbol ? <div className="table-filter-tabs">{[["ALL", "全部"], ["BUY", "买入"], ["SELL", "卖出"], ["RISK", "风控"]].map(([value, label]) => <button key={value} className={tradeFilter === value ? "active" : ""} onClick={() => setTradeFilter(value)}>{label}</button>)}</div> : null}>{filteredTrades.length ? <TradeTable trades={filteredTrades} selectedId={selectedTradeId} onSelect={(trade) => setSelectedTradeId(trade.id)} /> : <StateView state="empty" detail="行情数据与策略成交证据分离：没有 persisted trade 时不生成交易标记。" />}</Panel>
          <Panel title="决策链" eyebrow="Persisted selection trace" className="market-decision-panel">{(chain.data?.data.gates ?? []).length ? <div className="market-decision-list">{chain.data?.data.gates.map((gate) => <div key={`${gate.order}-${gate.name}`} className={gate.passed ? "passed" : "failed"}><span>{String(gate.order).padStart(2, "0")}</span><div><strong>{gate.name}</strong><small>{gate.reason ?? "无附加说明"}</small></div><b>{gate.passed ? "PASS" : "BLOCK"}</b></div>)}<div className="market-final-decision"><span>最终决策</span><strong>{chain.data?.data.finalDecision ?? "暂无"}</strong></div></div> : <StateView state="empty" detail={chain.data?.data.issues?.[0]?.message ?? "该标的没有 persisted selection decision chain。"} />}</Panel>
        </section>
        <footer className="market-source-note"><strong>Data contract</strong><span>行情优先：Fuyao snapshot + 约 420 个自然日的前复权日 K；回测成交、T+1 与决策链只来自 QuantAgent persisted artifacts。</span><span>任何上游缺失项显示“暂无”，不补零、不用模拟值冒充真实数据。</span></footer>
      </> : <StateView state={backtests.isLoading ? "loading" : "empty"} title="选择一个 A 股标的" detail="输入完整 thscode、六位代码或中文名称，使用 Fuyao 标的检索后进入行情工作台。" />}
    </div>
  );
}
