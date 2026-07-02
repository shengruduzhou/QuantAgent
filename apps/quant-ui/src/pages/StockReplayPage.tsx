import { useEffect, useMemo, useState } from "react";
import {
  ArrowsClockwise,
  CaretLeft,
  CaretRight,
  Crosshair,
  DownloadSimple,
  ArrowsOut,
  Info,
  ShieldWarning,
} from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { useSearchParams } from "react-router-dom";
import type {
  BacktestSummary,
  Page,
  SelectionRun,
  StockReplay,
  Trade,
} from "../api/types";
import { downloadJson } from "../api/client";
import { useApi } from "../hooks/useApi";
import { CandlestickChart } from "../components/CandlestickChart";
import { EChart } from "../components/EChart";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { TradeTable } from "../components/TradeTable";
import { formatCompact, formatNumber, formatPercent } from "../utils/format";

interface DoTPair {
  id: string;
  symbol: string;
  tradeDate: string;
  buyTime?: string | null;
  sellTime?: string | null;
  buyPrice?: number | null;
  sellPrice?: number | null;
  quantity?: number | null;
  netPnl?: number | null;
  success?: boolean | null;
  state?: string | null;
}

interface DoTAnalysis {
  sourceId: string;
  symbol?: string | null;
  summary: Record<string, number | null>;
  pairs: DoTPair[];
  verdict?: string | null;
  reason?: string | null;
}

interface DecisionChain {
  gates: Array<{
    order: number;
    name: string;
    passed: boolean;
    reason?: string | null;
    detail: Record<string, unknown>;
  }>;
  finalDecision?: string | null;
  issues?: Array<{ message: string }>;
}

export function StockReplayPage(): JSX.Element {
  const backtests = useApi<BacktestSummary[]>(["stock-replay-backtests"], "/backtests");
  const selectionRuns = useApi<SelectionRun[]>(["stock-replay-selection-runs"], "/selection/runs");
  const tradableRuns = (backtests.data?.data ?? []).filter((item) => item.capabilities?.trades);
  const [backtestId, setBacktestId] = useState<string>("");
  const [symbol, setSymbol] = useState<string>("");
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(null);
  const [tradeFilter, setTradeFilter] = useState("ALL");
  const [searchParams] = useSearchParams();
  const requestedSymbol = searchParams.get("symbol")?.toUpperCase() ?? "";

  useEffect(() => {
    if (!backtestId && tradableRuns[0]) setBacktestId(tradableRuns[0].id);
  }, [backtestId, tradableRuns]);

  const tradePage = useApi<Page<Trade>>(
    ["stock-replay-trades", backtestId],
    backtestId ? `/backtests/${backtestId}/trades` : null,
    { pageSize: 1000 },
  );
  const availableSymbols = useMemo(
    () => [...new Set((tradePage.data?.data.items ?? []).map((trade) => trade.symbol))],
    [tradePage.data?.data.items],
  );

  useEffect(() => {
    if (requestedSymbol && availableSymbols.includes(requestedSymbol) && symbol !== requestedSymbol) {
      setSymbol(requestedSymbol);
      setSelectedTradeId(null);
      return;
    }
    if ((!symbol || !availableSymbols.includes(symbol)) && availableSymbols[0]) {
      setSymbol(availableSymbols[0]);
      setSelectedTradeId(null);
    }
  }, [availableSymbols, requestedSymbol, symbol]);

  const replay = useApi<StockReplay>(
    ["stock-replay", backtestId, symbol],
    backtestId && symbol ? `/backtests/${backtestId}/stocks/${symbol}` : null,
  );
  const doT = useApi<DoTAnalysis>(
    ["stock-replay-dot", backtestId, symbol],
    backtestId && symbol ? `/backtests/${backtestId}/stocks/${symbol}/t-analysis` : null,
  );
  const latestSelection = selectionRuns.data?.data[0];
  const chain = useApi<DecisionChain>(
    ["stock-replay-chain", latestSelection?.id, symbol],
    latestSelection && symbol
      ? `/selection/runs/${latestSelection.id}/stocks/${symbol}/decision-chain`
      : null,
  );

  const tTrades = useMemo<Trade[]>(() => {
    const output: Trade[] = [];
    for (const pair of doT.data?.data.pairs ?? []) {
      if (pair.buyTime && pair.buyPrice !== null && pair.buyPrice !== undefined) {
        output.push({
          id: `${pair.id}-buy`,
          datetime: pair.buyTime,
          symbol: pair.symbol,
          action: "T_BUY",
          price: pair.buyPrice,
          quantity: pair.quantity ?? 0,
          pnl: null,
          success: pair.success,
          signalSource: "T+1 intraday overlay",
          tPairId: pair.id,
        });
      }
      if (pair.sellTime && pair.sellPrice !== null && pair.sellPrice !== undefined) {
        output.push({
          id: `${pair.id}-sell`,
          datetime: pair.sellTime,
          symbol: pair.symbol,
          action: "T_SELL",
          price: pair.sellPrice,
          quantity: pair.quantity ?? 0,
          pnl: pair.netPnl,
          success: pair.success,
          signalSource: "T+1 intraday overlay",
          riskReason: pair.state,
          tPairId: pair.id,
        });
      }
    }
    return output;
  }, [doT.data?.data.pairs]);

  const replayTrades = replay.data?.data.trades ?? [];
  const filteredTrades = replayTrades.filter((trade) => {
    if (tradeFilter === "ALL") return true;
    if (tradeFilter === "BUY") return trade.action.includes("BUY");
    if (tradeFilter === "SELL") return trade.action.includes("SELL");
    if (tradeFilter === "RISK") return Boolean(trade.riskReason) || trade.action.includes("RISK") || trade.action.includes("STOP");
    return true;
  });
  const allMarkers = useMemo(() => [...replayTrades, ...tTrades], [replayTrades, tTrades]);
  const selectedTrade = allMarkers.find((trade) => trade.id === selectedTradeId) ?? replayTrades[0] ?? tTrades[0];

  useEffect(() => {
    if (!selectedTradeId && selectedTrade) setSelectedTradeId(selectedTrade.id);
  }, [selectedTrade, selectedTradeId]);

  if (backtests.isLoading) return <StateView state="loading" />;
  if (!tradableRuns.length) return <StateView state="empty" detail="没有发现符合标准 order blotter schema 的回测。" />;

  const data = replay.data?.data;
  const currentIndex = selectedTrade ? allMarkers.findIndex((trade) => trade.id === selectedTrade.id) : -1;
  const moveSelection = (direction: number): void => {
    if (!allMarkers.length) return;
    const next = Math.min(allMarkers.length - 1, Math.max(0, currentIndex + direction));
    setSelectedTradeId(allMarkers[next].id);
  };

  return (
    <div className="page stock-replay-page">
      <section className="workbench-toolbar">
        <label>
          <span>回测引擎</span>
          <select value={backtestId} onChange={(event) => setBacktestId(event.target.value)}>
            {tradableRuns.map((run) => <option key={run.id} value={run.id}>{run.name} · {run.horizon}</option>)}
          </select>
        </label>
        <label className="symbol-selector">
          <span>股票标的</span>
          <select value={symbol} onChange={(event) => setSymbol(event.target.value)}>
            {availableSymbols.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <div className="toolbar-status">
          <StatusBadge status={data?.availability.bars ? "ready" : "partial"} label="日线数据" />
          <StatusBadge status={data?.availability.trades ? "ready" : "partial"} label="标准成交" />
          <StatusBadge status={doT.data?.data.pairs.length ? "ready" : "partial"} label="T+1 做 T" />
        </div>
        <button className="secondary-button" onClick={() => data && downloadJson(`stock-replay-${symbol}.json`, data)}>
          <DownloadSimple size={16} /> 导出
        </button>
      </section>

      {requestedSymbol && availableSymbols.length > 0 && !availableSymbols.includes(requestedSymbol) ? (
        <div className="requested-symbol-note">
          <ShieldWarning size={16} />
          <span>
            <strong>{requestedSymbol}</strong> 在当前回测中没有标准成交，已显示可复盘标的 {symbol}；可切换回测实验继续查找。
          </span>
        </div>
      ) : null}

      <section className="stock-workbench-grid">
        <div className="stock-workbench-main">
          <Panel
            title={`${data?.name ?? symbol ?? "股票"} ${symbol}`}
            eyebrow={`K 线 · 买卖点 · T+1 合规做 T 点`}
            className="kline-panel"
            actions={<div className="chart-toolbar"><span className="status-badge status-success">日线 · 原始复权口径</span><button onClick={() => document.querySelector(".kline-panel")?.requestFullscreen()} aria-label="全屏 K 线"><ArrowsOut size={15} /></button></div>}
          >
            {data?.bars.length ? (
              <CandlestickChart
                bars={data.bars}
                trades={allMarkers}
                selectedTradeId={selectedTradeId}
                onTradeSelect={setSelectedTradeId}
              />
            ) : <StateView state={replay.isLoading ? "loading" : "empty"} />}
          </Panel>

          <div className="replay-strip-grid">
            <SignalStrip label="持仓 Position" value={data?.positions.at(-1)?.weight as number | null} points={data?.positions.map((item) => Number(item.weight ?? item.shares ?? 0)) ?? []} color="#31c99c" />
            <SignalStrip label="模型分 Model Score" value={null} points={[]} color="#35c5b4" />
            <SignalStrip label="因子综合 Factor Score" value={null} points={[]} color="#3e8cff" />
            <SignalStrip label="风险分 Risk Score" value={null} points={[]} color="#e6a23c" />
            <SignalStrip label="单票净值 Equity" value={data?.summary.realizedPnl as number | null} points={data?.equity.map((item) => item.nav) ?? []} color="#30c4a5" />
          </div>

          <Panel
            title="交易记录"
            eyebrow={`Transactions · ${replayTrades.length} 条标准成交 · ${tTrades.length} 个 T+1 leg`}
            className="replay-trades-panel"
            actions={<div className="table-filter-tabs">
              {[
                ["ALL", "全部"],
                ["BUY", "买入"],
                ["SELL", "卖出"],
                ["RISK", "风控退出"],
              ].map(([value, label]) => (
                <button key={value} className={tradeFilter === value ? "active" : ""} onClick={() => setTradeFilter(value)}>{label}</button>
              ))}
            </div>}
          >
            {filteredTrades.length ? (
              <TradeTable trades={filteredTrades} selectedId={selectedTradeId} onSelect={(trade) => setSelectedTradeId(trade.id)} />
            ) : <StateView state="empty" detail={replay.data?.issues?.[0]?.message} />}
          </Panel>
        </div>

        <aside className="decision-inspector">
          <div className="inspector-header">
            <div>
              <strong>决策检查器</strong>
              <span>Decision Inspector</span>
            </div>
            <Info size={17} />
          </div>
          <div className="inspector-nav">
            <span>选中交易</span>
            <div>
              <button onClick={() => moveSelection(-1)} aria-label="上一笔"><CaretLeft size={15} /></button>
              <button onClick={() => moveSelection(1)} aria-label="下一笔"><CaretRight size={15} /></button>
            </div>
          </div>
          {selectedTrade ? (
            <>
              <div className="selected-action">
                <span className={`trade-action action-${selectedTrade.action.toLowerCase()}`}>{selectedTrade.action}</span>
                <strong className="mono">{selectedTrade.datetime.slice(0, 16)}</strong>
              </div>
              <div className="inspector-kv">
                <KeyValue label="价格 Price" value={formatNumber(selectedTrade.price)} />
                <KeyValue label="数量 Quantity" value={formatCompact(selectedTrade.quantity)} />
                <KeyValue label="金额 Amount" value={formatCompact(selectedTrade.amount)} />
                <KeyValue label="成交后仓位" value={formatCompact(selectedTrade.positionAfter)} />
                <KeyValue label="单笔收益" value={formatNumber(selectedTrade.pnl)} tone={(selectedTrade.pnl ?? 0) >= 0 ? "positive" : "negative"} />
              </div>
              <InspectorSection title="交易原因 Rationale">
                <p>{selectedTrade.riskReason ?? selectedTrade.failureReason ?? selectedTrade.signalSource ?? "该 artifact 未记录逐笔 rationale。"}</p>
              </InspectorSection>
              <InspectorSection title="因子贡献 Factor Attribution">
                {selectedTrade.factorContributions ? Object.entries(selectedTrade.factorContributions).map(([factor, value]) => (
                  <div className="factor-bar" key={factor}>
                    <span>{factor}</span>
                    <i><b style={{ width: `${Math.min(100, Math.abs(value) * 100)}%` }} /></i>
                    <strong className={value >= 0 ? "tone-positive" : "tone-negative"}>{value.toFixed(3)}</strong>
                  </div>
                )) : <p className="muted-copy">没有 persisted per-trade factor contribution，不使用 feature importance 冒充。</p>}
              </InspectorSection>
              <InspectorSection title="模型与风险">
                <div className="confidence-meter"><i style={{ width: `${Math.max(0, Math.min(100, (selectedTrade.modelScore ?? 0) * 100))}%` }} /></div>
                <KeyValue label="模型分数" value={formatNumber(selectedTrade.modelScore)} />
                <KeyValue label="成交状态" value={selectedTrade.status ?? "暂无"} />
                <KeyValue label="T+1 Pair" value={selectedTrade.tPairId ?? "非做 T leg"} />
              </InspectorSection>
              <div className={`risk-callout ${selectedTrade.success === false ? "risk-callout-danger" : ""}`}>
                <ShieldWarning size={18} />
                <div>
                  <strong>{selectedTrade.success === false ? "执行失败" : "研究执行记录"}</strong>
                  <span>{selectedTrade.failureReason ?? "No live order · 仅回测/研究"}</span>
                </div>
              </div>
            </>
          ) : <StateView state="empty" />}
        </aside>
      </section>

      <Panel title="决策链" eyebrow="Decision Chain · persisted trace preferred" className="decision-chain-panel">
        <div className="decision-chain">
          {(chain.data?.data.gates.length ? chain.data.data.gates : defaultGates()).map((gate) => (
            <div className={`decision-step ${gate.passed ? "decision-pass" : "decision-missing"}`} key={`${gate.order}-${gate.name}`}>
              <span>{gate.order}</span>
              <div>
                <strong>{gate.name}</strong>
                <small>{gate.reason ?? (gate.passed ? "通过 / available" : "暂无 persisted trace")}</small>
              </div>
            </div>
          ))}
          <div className="decision-step decision-action">
            <span><ArrowsClockwise size={18} /></span>
            <div>
              <strong>执行动作</strong>
              <small>{chain.data?.data.finalDecision ?? selectedTrade?.action ?? "观察"}</small>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
}

interface SignalStripProps {
  label: string;
  value: number | null | undefined;
  points: number[];
  color: string;
}

function SignalStrip({ label, value, points, color }: SignalStripProps): JSX.Element {
  const option = useMemo<EChartsOption>(() => ({
    animation: false,
    grid: { left: 8, right: 8, top: 4, bottom: 4 },
    xAxis: { type: "category", show: false, data: points.map((_, index) => index) },
    yAxis: { type: "value", show: false, scale: true },
    series: [{ type: "line", data: points, showSymbol: false, lineStyle: { color, width: 1.2 }, areaStyle: { color: `${color}12` } }],
  }), [color, points]);
  return (
    <div className="signal-strip">
      <div><span>{label}</span><strong className="mono">{formatNumber(value)}</strong></div>
      {points.length ? <EChart option={option} className="signal-strip-chart" /> : <span className="strip-empty">暂无序列</span>}
    </div>
  );
}

interface KeyValueProps {
  label: string;
  value: string;
  tone?: "positive" | "negative";
}

function KeyValue({ label, value, tone }: KeyValueProps): JSX.Element {
  return <div className="key-value"><span>{label}</span><strong className={tone ? `tone-${tone}` : "mono"}>{value}</strong></div>;
}

function InspectorSection({ title, children }: { title: string; children: React.ReactNode }): JSX.Element {
  return <section className="inspector-section"><h3>{title}</h3>{children}</section>;
}

function defaultGates(): DecisionChain["gates"] {
  return [
    { order: 1, name: "初始池 Initial Pool", passed: false, reason: "股票未出现在最新 selection run", detail: {} },
    { order: 2, name: "流动性过滤 Liquidity", passed: false, reason: "暂无 persisted trace", detail: {} },
    { order: 3, name: "风险过滤 Risk", passed: false, reason: "暂无 persisted trace", detail: {} },
    { order: 4, name: "因子打分 Factor", passed: false, reason: "暂无 persisted trace", detail: {} },
    { order: 5, name: "模型打分 Model", passed: false, reason: "暂无 persisted trace", detail: {} },
    { order: 6, name: "最终排名 Final Rank", passed: false, reason: "暂无 persisted trace", detail: {} },
  ];
}
