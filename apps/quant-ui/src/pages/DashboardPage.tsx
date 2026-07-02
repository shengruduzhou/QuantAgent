import {
  ArrowSquareOut,
  ChartLineUp,
  Coins,
  Gauge,
  Pulse,
  ShieldWarning,
  TrendDown,
} from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import type {
  BacktestSummary,
  EquityPoint,
  Page,
  RiskOverview,
  SelectionRun,
  SystemOverview,
  Trade,
} from "../api/types";
import { useApi } from "../hooks/useApi";
import { EquityChart } from "../components/EquityChart";
import { MetricCard } from "../components/MetricCard";
import { Panel } from "../components/Panel";
import { RiskRadar } from "../components/RiskRadar";
import { SelectionFunnel } from "../components/SelectionFunnel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { TradeTable } from "../components/TradeTable";
import { formatBytes, formatCompact, formatNumber, formatPercent } from "../utils/format";

interface RiskStock {
  symbol: string;
  netPnl?: number | null;
  winRate?: number | null;
  tradeCount?: number | null;
  riskScore?: number | null;
}

interface FunnelStage {
  stage: string;
  count: number | null;
  reason?: string | null;
}

export function DashboardPage(): JSX.Element {
  const overview = useApi<SystemOverview>(["dashboard-overview"], "/system/overview");
  const backtests = useApi<BacktestSummary[]>(["dashboard-backtests"], "/backtests");
  const data = overview.data?.data;
  const latestBacktest = data?.latestBacktest ?? backtests.data?.data[0];
  const latestSelection = data?.latestSelection as SelectionRun | null | undefined;
  const equity = useApi<EquityPoint[]>(
    ["dashboard-equity", latestBacktest?.id],
    latestBacktest ? `/backtests/${latestBacktest.id}/equity` : null,
  );
  const trades = useApi<Page<Trade>>(
    ["dashboard-trades", latestBacktest?.id],
    latestBacktest ? `/backtests/${latestBacktest.id}/trades` : null,
    { pageSize: 8 },
  );
  const risk = useApi<RiskOverview>(
    ["dashboard-risk", latestBacktest?.id],
    "/risk/overview",
    { backtestId: latestBacktest?.id },
  );
  const riskStocks = useApi<RiskStock[]>(
    ["dashboard-risk-stocks", latestBacktest?.id],
    "/risk/stocks",
    { backtestId: latestBacktest?.id },
  );
  const funnel = useApi<FunnelStage[]>(
    ["dashboard-funnel", latestSelection?.id],
    latestSelection ? `/selection/runs/${latestSelection.id}/funnel` : null,
  );

  if (overview.isLoading) return <StateView state="loading" />;
  if (overview.isError || !data) return <StateView state="error" detail={overview.error?.message} />;

  const topContributors = [...(riskStocks.data?.data ?? [])]
    .filter((item) => item.netPnl !== null && item.netPnl !== undefined)
    .sort((left, right) => (right.netPnl ?? 0) - (left.netPnl ?? 0))
    .slice(0, 6);
  const contributorScale = Math.max(
    1,
    ...topContributors.map((item) => Math.abs(item.netPnl ?? 0)),
  );
  const riskData = risk.data?.data ?? data.risk;

  return (
    <div className="page dashboard-page">
      <section className="metric-grid metric-grid-7">
        <MetricCard label="组合总收益" value={formatPercent(latestBacktest?.totalReturn)} delta={latestBacktest?.totalReturn} icon={ChartLineUp} />
        <MetricCard label="年化收益" value={formatPercent(latestBacktest?.annualReturn)} delta={latestBacktest?.annualReturn} icon={Coins} />
        <MetricCard label="最大回撤" value={formatPercent(latestBacktest?.maxDrawdown)} delta={latestBacktest?.maxDrawdown ? -Math.abs(latestBacktest.maxDrawdown) : null} icon={TrendDown} />
        <MetricCard label="Sharpe" value={formatNumber(latestBacktest?.sharpe)} detail={`Calmar ${formatNumber(latestBacktest?.calmar)}`} icon={Gauge} />
        <MetricCard label="年化换手" value={formatPercent(latestBacktest?.turnover)} tone="warning" detail={`${formatCompact(latestBacktest?.tradeCount)} trades`} icon={Pulse} />
        <MetricCard label="当前候选池" value={formatCompact(data.stockPoolCount)} detail={`初始 ${formatCompact(data.candidateCount)}`} icon={Pulse} />
        <MetricCard label="风险状态" value={data.riskStatus === "normal" ? "正常" : "需关注"} tone={data.riskStatus === "normal" ? "positive" : "warning"} detail={`${Object.keys(data.risk.eventCounts ?? {}).length} 类事件`} icon={ShieldWarning} />
      </section>

      <section className="dashboard-main-grid">
        <Panel
          title="组合净值与回撤"
          eyebrow={`${latestBacktest?.name ?? "暂无回测"} · ${latestBacktest?.horizon ?? "multi-horizon"}`}
          className="dashboard-equity"
          actions={<Link className="text-link" to="/backtests">进入回测实验 <ArrowSquareOut size={14} /></Link>}
        >
          {equity.data?.data.length ? <EquityChart points={equity.data.data} /> : <StateView state="empty" />}
        </Panel>

        <Panel title="风险雷达" eyebrow="Risk Radar · 相对阈值" className="dashboard-radar">
          {risk.data?.data ? <RiskRadar risk={risk.data.data} /> : <StateView state={risk.isLoading ? "loading" : "empty"} />}
        </Panel>

        <Panel title="系统健康" eyebrow="System Health" className="dashboard-health">
          <div className="health-list">
            <HealthRow label="Runtime Index" status="ready" latency="snapshot" quality={`${data.runtime.artifactCount.toLocaleString()} items`} />
            <HealthRow label="Model Registry" status={data.modelStatus} latency="metadata" quality={data.latestModel?.version ?? "暂无版本"} />
            <HealthRow label="Backtest Adapter" status={latestBacktest ? "ready" : "partial"} latency="on-demand" quality={`${backtests.data?.data.length ?? 0} runs`} />
            <HealthRow label="Selection Pipeline" status={latestSelection ? "ready" : "partial"} latency="on-demand" quality={`${data.stockPoolCount ?? 0} names`} />
            <HealthRow label="Risk Engine" status={data.riskStatus} latency="streamed" quality={`${Object.values(data.risk.eventCounts ?? {}).reduce((sum, value) => sum + value, 0)} events`} />
          </div>
          <div className="health-summary">
            <span>索引体积</span>
            <strong>{formatBytes(data.runtime.totalSizeBytes)}</strong>
          </div>
        </Panel>

        <Panel title="透明选股漏斗" eyebrow="Selection Funnel" className="dashboard-funnel">
          {funnel.data?.data.length ? <SelectionFunnel stages={funnel.data.data} /> : <StateView state="empty" detail="当前 selection run 没有 persisted funnel。" />}
        </Panel>

        <Panel title="风险暴露" eyebrow="Risk Exposure · relative intensity" className="dashboard-exposure">
          <div className="exposure-list">
            <ExposureRow label="组合回撤" value={riskData.maxDrawdown} />
            <ExposureRow label="流动性风险" value={riskData.liquidityRisk} />
            <ExposureRow label="跌停风险" value={riskData.limitDownRisk} />
            <ExposureRow label="停牌风险" value={riskData.suspensionRisk} />
            <ExposureRow label="波动率暴露" value={riskData.volatilityExposure} />
          </div>
        </Panel>

        <Panel title="Top 贡献标的" eyebrow="Realized PnL Attribution" className="dashboard-contributors">
          {topContributors.length ? (
            <div className="contributor-list">
              {topContributors.map((stock, index) => (
                <div key={stock.symbol} className="contributor-row">
                  <span className="rank-number">{index + 1}</span>
                  <strong>{stock.symbol}</strong>
                  <div className="contributor-bar"><i style={{ width: `${Math.max(6, (Math.abs(stock.netPnl ?? 0) / contributorScale) * 100)}%` }} /></div>
                  <span className={(stock.netPnl ?? 0) >= 0 ? "tone-positive mono" : "tone-negative mono"}>{formatNumber(stock.netPnl)}</span>
                </div>
              ))}
            </div>
          ) : <StateView state="empty" detail="profit_by_stock.csv 不存在或没有 realized PnL。" />}
        </Panel>

        <Panel title="最新交易明细" eyebrow="Execution Blotter · 标准成交回报" className="dashboard-trades">
          {trades.data?.data.items.length ? (
            <TradeTable trades={trades.data.data.items} compact />
          ) : <StateView state="empty" detail={trades.data?.issues[0]?.message} />}
        </Panel>
      </section>
    </div>
  );
}

function ExposureRow({ label, value }: { label: string; value?: number | null }): JSX.Element {
  const intensity = Math.min(100, Math.abs(value ?? 0) * 100);
  return (
    <div className="exposure-row">
      <span>{label}</span>
      <div><i style={{ width: `${intensity}%` }} /></div>
      <strong className="mono">{formatPercent(value)}</strong>
    </div>
  );
}

interface HealthRowProps {
  label: string;
  status: string;
  latency: string;
  quality: string;
}

function HealthRow({ label, status, latency, quality }: HealthRowProps): JSX.Element {
  return (
    <div className="health-row">
      <strong>{label}</strong>
      <StatusBadge status={status} />
      <span>{latency}</span>
      <span className="mono">{quality}</span>
    </div>
  );
}
